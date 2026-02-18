import numpy as np
from ultralytics import YOLO

class SemanticDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.classes = self.model.names
        self.prev_bbox = None
        
        # High smoothing factor (1.0 = locked, 0.0 = no smoothing)
        self.smoothing = 0.85 
        
        # Deadzone: Bounding box must move by at least 15 pixels to "unlock" and shift
        self.deadzone_threshold = 15.0 

    def _apply_deadzone_and_smoothing(self, new_coords):
        if self.prev_bbox is None:
            self.prev_bbox = new_coords
            return new_coords

        # Calculate maximum pixel movement on any axis
        movement = np.max(np.abs(new_coords - self.prev_bbox))

        # 1. Deadzone Lock: Ignore micro-jitter entirely
        if movement < self.deadzone_threshold:
            return self.prev_bbox

        # 2. Smooth Movement: Glide to the new position if movement is significant
        smoothed = (self.smoothing * self.prev_bbox) + ((1 - self.smoothing) * new_coords)
        self.prev_bbox = smoothed
        return smoothed

    def detect(self, frame):
        """Runs YOLO inference and applies stabilization logic."""
        results = self.model(frame, conf=0.4, verbose=False)[0]
        scored_detections = []
        
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            
            # Apply Deadzone Lock & Smoothing
            stable_coords = self._apply_deadzone_and_smoothing(coords)

            # Add large padding to give the Go stitcher plenty of room to apply the 45-pixel fade
            pad = 40 
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = stable_coords.astype(int)
            
            # Prevent padded boundaries from extending outside the physical frame
            final_coords = [max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)]
            
            scored_detections.append({
                'bbox': final_coords,
                'class': self.classes[int(box.cls[0])],
                'priority': int(float(box.conf[0]) * 255)
            })
            
        return scored_detections