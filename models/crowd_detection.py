import cv2
import numpy as np
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, model_path='yolov8n.pt', grid_rows=4, grid_cols=4):
        self.model = YOLO(model_path)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def process_frame(self, frame, show_grid=True, show_heatmap=False):
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        
        counts = {'total': 0, 'grids': {}}
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                counts['grids'][f"{r}-{c}"] = 0

        h, w = frame.shape[:2]
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols
        
        persons = []

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                grid_c = min(int(cx / cell_w), self.grid_cols - 1)
                grid_r = min(int(cy / cell_h), self.grid_rows - 1)
                
                counts['grids'][f"{grid_r}-{grid_c}"] += 1
                counts['total'] += 1
                
                persons.append({
                    'id': int(track_id),
                    'box': (x1, y1, x2, y2),
                    'grid': (grid_r, grid_c),
                    'aspect_ratio': (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Drawing options for "Control Room" mode
        heatmap_overlay = frame.copy() if show_heatmap else None

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y = int(r * cell_h)
                x = int(c * cell_w)
                g_id = f"{r}-{c}"
                density = counts['grids'][g_id]
                alpha_id = f"{chr(65+r)}{c+1}"

                if show_grid:
                    cv2.rectangle(frame, (x, y), (int((c+1)*cell_w), int((r+1)*cell_h)), (0, 255, 255), 1)
                    cv2.putText(frame, alpha_id, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if show_heatmap and density > 0:
                    intensity = min(255, density * 30) # Simple density scaling
                    color = (0, 0, intensity) # BGR (Red Heatmap)
                    cv2.rectangle(heatmap_overlay, (x, y), (int((c+1)*cell_w), int((r+1)*cell_h)), color, -1)

        if show_heatmap:
            cv2.addWeighted(heatmap_overlay, 0.4, frame, 0.6, 0, frame)

        return frame, counts, persons
