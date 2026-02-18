import cv2
import numpy as np
from ultralytics import YOLO

class SemanticDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load optimized YOLOv8-Nano model
        self.model = YOLO(model_path)
        self.classes = self.model.names
        self.prev_bbox = None
        # High smoothing factor to eliminate the "flickering" box effect
        self.smoothing = 0.85 

    def detect(self, frame):
        """Runs inference and returns padded, smoothed detections."""
        results = self.model(frame, conf=0.4, verbose=False)[0]
        scored_detections = []
        
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            
            # 1. Apply Temporal Smoothing (EMA)
            if self.prev_bbox is not None:
                coords = (self.smoothing * self.prev_bbox) + ((1 - self.smoothing) * coords)
            self.prev_bbox = coords

            # 2. Add padding for the Go stitcher's feathering algorithm
            pad = 25 
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = coords.astype(int)
            
            # Ensure boundaries don't go outside the frame
            final_coords = [max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)]
            
            scored_detections.append({
                'bbox': final_coords,
                'class': self.classes[int(box.cls[0])],
                'priority': int(float(box.conf[0]) * 255)
            })
            
        return scored_detections