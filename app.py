from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
import threading
import time
from datetime import datetime

from models.crowd_detection import CrowdDetector
from models.prediction import CrowdPredictor
from models.anomaly_detection import AnomalyDetector
from models.evacuation import EvacuationRouter

app = Flask(__name__)
app.secret_key = 'super_secret_control_room_key_123'

USERS = {
    'admin': 'password',
    'operator': 'password'
}

# Incident logs
MAX_LOGS = 100
incident_logs = []
active_alerts = []

def add_log(message, level):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {'time': timestamp, 'message': message, 'level': level}
    # Avoid duplicate rapid logging for exactly same message
    if len(incident_logs) == 0 or incident_logs[0]['message'] != message:
        incident_logs.insert(0, log_entry)
        if len(incident_logs) > MAX_LOGS:
            incident_logs.pop()

system_state = {
    'counts': {'total': 0, 'grids': {}},
    'predictions': {'total': 0, 'grids': {}, 'overflows': []},
    'grid_predictions': {},
    'growth_rates': {'total': 0, 'grids': {}},
    'alerts': active_alerts,
    'fire_grids': [],
    'fallen_grids': [],
    'logs': incident_logs,
    'risk_scores': {},
    'fps': 0,
    'system_status': 'Monitoring',
    'latency': 100,
    'cameras_online': '2/2',
    'model_info': {'name': 'YOLOv8', 'status': 'Active', 'accuracy': '~85%'}
}

system_settings = {
    'detection_active': True,
    'show_heatmap': False,
    'show_grid': True,
    'camera_source': 'test_video1.mp4.mp4',
    'threshold': 15,
    'sound_alerts': True,
    'simulation_state': 'playing'
}

# Grid mapping utility (4x4)
def get_grid_name(r, c):
    row_char = chr(65 + r) # A, B, C, D
    col_idx = c + 1
    return f"{row_char}{col_idx}"

class ThreadedCamera:
    """Read frames in a background thread to prevent buffer overflow."""
    def __init__(self):
        self.stream = None
        self.started = False
        self.thread = None
        self.grabbed = False
        self.frame = None
        self.fps = 30.0
        self.current_source = None
        self.loading = False

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            src = system_settings['camera_source']
            if src == 'demo':
                time.sleep(0.05)
                continue

            # Handle source switching
            if self.current_source != src:
                self.loading = True
                try:
                    if self.stream is not None:
                        self.stream.release()
                    
                    if isinstance(src, str) and src.isdigit():
                        import sys
                        if sys.platform == 'win32':
                            self.stream = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
                        else:
                            self.stream = cv2.VideoCapture(int(src))
                    else:
                        self.stream = cv2.VideoCapture(src)
                        
                    if self.stream.isOpened():
                        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
                    if not self.fps or self.fps <= 0:
                        self.fps = 30.0
                    self.current_source = src
                    self.grabbed = False
                    self.frame = None
                except Exception as e:
                    print(f"Hardware init error: {e}")
                finally:
                    self.loading = False
            
            if self.stream is not None and self.stream.isOpened():
                if isinstance(src, str) and not src.isdigit() and not self.grabbed:
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.grabbed, self.frame = self.stream.read()
                
                if not self.grabbed:
                    time.sleep(0.1)
                elif isinstance(src, str) and not src.isdigit():
                    time.sleep(0.8 / self.fps)
            else:
                time.sleep(0.1)

    def read(self):
        if system_settings['camera_source'] == 'demo':
            return True, None
        return self.grabbed, self.frame

    def stop(self):
        self.started = False
        if self.thread is not None:
            self.thread.join()
        if self.stream is not None:
            self.stream.release()

def generate_frames():
    global system_state
    
    detector = CrowdDetector(model_path='yolov8n.pt', grid_rows=4, grid_cols=4)
    predictor = CrowdPredictor(history_len=30, predict_steps=30, grid_threshold=10, total_threshold=system_settings['threshold'])
    anomaly_detector = AnomalyDetector(fps=15, fallen_duration_sec=3)
    
    cam = ThreadedCamera().start()
    time.sleep(1.0)
    
    frame_skip = 2
    frame_counter = 0
    read_fail_count = 0

    annotated_frame = None
    counts = {'total': 0, 'grids': {}}
    fire_grids, fire_alerts = [], []
    fallen_grids, fallen_alerts = [], []
    last_time = time.time()
    current_fps = 0

    # Synthetic variables for demo mode
    synthetic_persons = [{"id": i, "x": np.random.randint(50, 590), "y": np.random.randint(50, 430), 
                          "dx": np.random.choice([-2, 2]), "dy": np.random.choice([-2, 2])} for i in range(15)]

    while True:
        src = system_settings['camera_source']
        success, img = cam.read()
        
        # Calculate real fps
        frame_counter += 1
        curr_time = time.time()
        if curr_time - last_time >= 1.0:
            current_fps = frame_counter
            system_state['fps'] = current_fps
            frame_counter = 0
            last_time = curr_time

        # Wait if the camera is currently hardware switching
        if getattr(cam, 'loading', False):
            read_fail_count = 0
            time.sleep(0.1)
            continue

        # Handle stream failure gracefully by switching to demo mode
        if not success or (img is None and src != 'demo'):
            if src != 'demo':
                read_fail_count += 1
                if read_fail_count < 40:  # Allow 4 seconds for slow webcams to initialize
                    time.sleep(0.1)
                    continue
                # Generate a NO SIGNAL frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO SIGNAL - SWITCHING TO DEMO MODE", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(2.0)
                system_settings['camera_source'] = 'demo'
                src = 'demo'
            else:
                time.sleep(0.05)
                continue
        else:
            read_fail_count = 0
            
        if src == 'demo':
            # Dynamic crowd simulation
            if system_settings['simulation_state'] != 'paused':
                # Randomly spawn or remove persons over time to simulate dynamic surges
                if np.random.rand() < 0.15 and len(synthetic_persons) < 40:
                    i = np.random.randint(100, 999)
                    # Cluster in bottom-right grid often to trigger risk indicators
                    if np.random.rand() < 0.6:
                        nx, ny = np.random.randint(400, 590), np.random.randint(300, 430)
                    else:
                        nx, ny = np.random.randint(50, 590), np.random.randint(50, 430)
                    synthetic_persons.append({"id": i, "x": nx, "y": ny, "dx": np.random.choice([-3, -2, 2, 3]), "dy": np.random.choice([-3, -2, 2, 3])})
                elif np.random.rand() < 0.05 and len(synthetic_persons) > 5:
                    synthetic_persons.pop(np.random.randint(0, len(synthetic_persons)))

                # Add dummy background logs periodically
                if np.random.rand() < 0.05:
                    test_logs = ["Routine Check: Central Hub clear", "Minor crowding detected near Exit B", "Camera 2 calibrated", "Gate A flow normalized"]
                    add_log(np.random.choice(test_logs), "yellow")

            # Generate Demo Frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (20, 25, 30) # Dark bg
            
            # Draw realistic demo background layout
            cv2.rectangle(frame, (10, 10), (630, 470), (50, 60, 70), 2)
            cv2.line(frame, (320, 10), (320, 470), (50, 60, 70), 1)
            cv2.line(frame, (10, 240), (630, 240), (50, 60, 70), 1)

            persons = []
            counts = {'total': 0, 'grids': {}}
            for p in synthetic_persons:
                if system_settings['simulation_state'] != 'paused':
                    p['x'] += p['dx']
                    p['y'] += p['dy']
                    if p['x'] < 20 or p['x'] > 620: p['dx'] *= -1
                    if p['y'] < 20 or p['y'] > 460: p['dy'] *= -1
                
                # grid math
                c = int(p['x'] / 160)
                r = int(p['y'] / 120)
                c = min(c, 3); r = min(r, 3)
                g_id = f"{r}-{c}"
                counts['grids'][g_id] = counts['grids'].get(g_id, 0) + 1
                counts['total'] += 1
                persons.append({'id': p['id'], 'grid': (r,c), 'box': (p['x'], p['y'], p['x']+20, p['y']+50)})
                
                # Highlight pulsing dense grids directly in the video frame if heatmap is on
                if system_settings.get('show_heatmap') and counts['grids'][g_id] > 5:
                    pulse_radius = int(25 + 5 * np.sin(time.time() * 5))
                    cv2.circle(frame, (p['x']+10, p['y']+25), pulse_radius, (0, 100, 255), -1)

                cv2.rectangle(frame, (p['x'], p['y']), (p['x']+20, p['y']+50), (200, 200, 200), 2)
                # Crowd Flow Arrows
                cv2.arrowedLine(frame, (p['x']+10, p['y']+25), (p['x']+10+p['dx']*5, p['y']+25+p['dy']*5), (0, 255, 255), 2, tipLength=0.3)
            
            # Apply alpha blending if heatmap was drawn
            if system_settings.get('show_heatmap'):
                overlay = frame.copy()
                frame = cv2.addWeighted(overlay, 0.6, np.full((480, 640, 3), (20, 25, 30), dtype=np.uint8), 0.4, 0)

            annotated_frame = frame.copy()
            fire_grids, fire_alerts = [], []
            fallen_grids, fallen_alerts = [], []
            
            # Randomly trigger fire in demo mode rarely
            if np.random.rand() < 0.005 and system_settings['simulation_state'] != 'paused': 
                fire_grids.append("2-2")
                fire_alerts.append("Fire Detected at Gate C")
                cv2.rectangle(annotated_frame, (320, 240), (480, 360), (0, 0, 255), 4)

        else:
            frame = cv2.resize(img, (640, 480))
            if system_settings['detection_active']:
                if frame_counter % frame_skip == 0:
                    annotated_frame, counts, persons = detector.process_frame(frame.copy(), 
                                                show_grid=system_settings['show_grid'], 
                                                show_heatmap=system_settings['show_heatmap'])
                    # Fallback on anomalies
                    fallen_grids, fallen_alerts = anomaly_detector.detect_fallen_persons(persons)
                    fire_grids, fire_alerts, annotated_frame = anomaly_detector.detect_fire(annotated_frame, grid_rows=4, grid_cols=4)
                elif annotated_frame is None:
                    annotated_frame = frame.copy()
            else:
                annotated_frame = frame.copy()
                fire_grids, fire_alerts = [], []
                fallen_grids, fallen_alerts = [], []

        if system_settings['detection_active'] or src == 'demo':
            system_state['counts'] = counts
            predictor.total_threshold = system_settings['threshold']
            predictions, growth_rates = predictor.update_and_predict(counts)
            
            system_state['predictions'] = predictions
            system_state['growth_rates'] = growth_rates
            system_state['fallen_grids'] = fallen_grids
            system_state['fire_grids'] = fire_grids
            
            grid_preds_dict = {}
            for grid_id, ct in counts.get('grids', {}).items():
                g_r, g_c = [int(x) for x in grid_id.split('-')]
                loc = get_grid_name(g_r, g_c)
                pred_c = predictions['grids'].get(grid_id, ct)
                
                # Assess localized risk
                stat = "SAFE"
                t_high = 5 if src == 'demo' else 10
                t_warn = 3 if src == 'demo' else 7
                if pred_c >= t_high:
                    stat = "HIGH"
                elif pred_c >= t_warn:
                    stat = "WARNING"
                    
                grid_preds_dict[loc] = {"current": ct, "forecast": pred_c, "risk": stat}
            
            system_state['grid_predictions'] = grid_preds_dict
            
            alerts = []
            for a in fire_alerts: 
                alerts.append(f"CRITICAL: {a}")
                add_log(a, "red")
            for a in fallen_alerts: 
                alerts.append(f"WARNING: {a}")
                add_log(a, "yellow")
                
            for g in predictions.get('overflows', []):
                if g == 'Global':
                    alerts.append(f"GLOBAL OVERFLOW: Expected {predictions['total']} people")
                    add_log(f"Global overflow predicted: {predictions['total']}", "red")
                else:
                    g_r, g_c = [int(x) for x in g.replace('Grid ', '').split('-')]
                    loc = get_grid_name(g_r, g_c)
                    pred_v = predictions['grids'].get(g.replace('Grid ', ''), 0)
                    alerts.append(f"CRITICAL: Predicted overflow in Grid {loc} (Forecast: {pred_v})")
                    add_log(f"Zone {loc} predicted overload", "red")
            
            system_state['alerts'] = alerts

        # Telemetry & UI Overlay
        system_state['latency'] = np.random.randint(80, 150)
        
        # Draw realistic camera and zone overlays
        cam_name = "CAM-1 (MAIN HALL)" if "test_video1" in str(src) else "CAM-2 (EXIT ROW)" if "test_video" in str(src) else "CAM-0 (WEBCAM)"
        if src == 'demo': cam_name = "SIM-1 (DEMO ZONE)"
        
        cv2.putText(annotated_frame, f"{cam_name} | {current_fps} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"STATUS: {'ACTIVE' if system_settings['detection_active'] else 'PAUSED'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if not system_settings['detection_active']:
            cv2.putText(annotated_frame, "DETECTION ENGINE OFFLINE", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ROUTES #################################

@app.before_request
def require_login():
    allowed_routes = ['login', 'static']
    if request.endpoint not in allowed_routes and 'logged_in' not in session:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')
        if USERS.get(user) == pwd:
            session['logged_in'] = True
            session['role'] = user
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid Credentials. Unauthorized access logged.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    return render_template('index.html', role=session.get('role', 'operator').upper())

@app.route('/architecture')
def architecture():
    return render_template('architecture.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def api_data():
    return jsonify(system_state)

@app.route('/api/simulation_control', methods=['POST'])
def simulation_control():
    data = request.json
    action = data.get('action')
    if action in ['play', 'pause']:
        system_settings['simulation_state'] = 'paused' if action == 'pause' else 'playing'
        add_log(f"Simulation {action}d by {session.get('role')}", "yellow")
    elif action == 'reset':
        system_settings['simulation_state'] = 'playing'
        add_log(f"Simulation reset by {session.get('role')}", "yellow")
    return jsonify({"status": "success", "state": system_settings['simulation_state']})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    for key in set(data.keys()).intersection(system_settings.keys()):
        system_settings[key] = data[key]
    add_log(f"System settings updated by {session.get('role')}", "yellow")
    return jsonify({"status": "success", "settings": system_settings})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
