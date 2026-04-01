import cv2
import numpy as np
from collections import defaultdict

class AnomalyDetector:
    def __init__(self, fps=30, fallen_duration_sec=3):
        # Tracking IDs that show a fallen aspect ratio
        self.fallen_tracker = defaultdict(int)
        
        # Frames needed to trigger
        self.fallen_frame_threshold = fps * fallen_duration_sec

    def detect_fallen_persons(self, persons):
        fallen_grids = []
        alerts = []
        
        # We need to decay/clean tracker for IDs no longer seen or no longer fallen
        current_ids = [p['id'] for p in persons]
        
        # Remove old IDs
        for key in list(self.fallen_tracker.keys()):
            if key not in current_ids:
                del self.fallen_tracker[key]
                
        for person in persons:
            w = person['box'][2] - person['box'][0]
            h = person['box'][3] - person['box'][1]
            
            aspect_ratio = w / h if h > 0 else 0
            
            # Unconscious or fallen if wider than tall
            if aspect_ratio > 1.2:
                self.fallen_tracker[person['id']] += 1
                
                # Check threshold
                if self.fallen_tracker[person['id']] > self.fallen_frame_threshold:
                    r, c = person['grid']
                    fallen_grids.append(f"{r}-{c}")
                    alerts.append(f"Person Fallen at Grid {r}-{c}")
            else:
                self.fallen_tracker[person['id']] = 0
                
        return list(set(fallen_grids)), list(set(alerts))

    def detect_fire(self, frame, grid_rows=2, grid_cols=2):
        # Simple OpenCV HSV color thresholding for Fire/Smoke
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Balanced Lower and upper bound for fire colors
        lower_bound = np.array([12, 100, 150]) 
        upper_bound = np.array([35, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Applying some morph operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_grids = []
        alerts = []
        h, w = frame.shape[:2]
        cell_h = h / grid_rows
        cell_w = w / grid_cols
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300: # Lowered threshold so phones/pictures of fire trigger it
                x, y, cw, ch = cv2.boundingRect(contour)
                cx = x + cw / 2
                cy = y + ch / 2
                grid_c = min(int(cx / cell_w), grid_cols - 1)
                grid_r = min(int(cy / cell_h), grid_rows - 1)
                
                fire_grids.append(f"{grid_r}-{grid_c}")
                alerts.append(f"Fire Detected at Grid {grid_r}-{grid_c}")
                
                # Draw fire rect
                cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 0, 255), 3)
                cv2.putText(frame, "FIRE WARNING", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        return list(set(fire_grids)), list(set(alerts)), frame
