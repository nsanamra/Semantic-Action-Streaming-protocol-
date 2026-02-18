import cv2
import numpy as np
from ultralytics import YOLO

class SemanticDetector:
    def __init__(self, model_path='yolov8n-seg.pt'):
        # Automatically downloads the Instance Segmentation model
        self.model = YOLO(model_path)
        self.classes = self.model.names
        self.prev_bbox = None
        self.smoothing = 0.85
        self.deadzone_threshold = 15.0

    def _apply_deadzone_and_smoothing(self, new_coords):
        if self.prev_bbox is None:
            self.prev_bbox = new_coords
            return new_coords

        movement = np.max(np.abs(new_coords - self.prev_bbox))
        if movement < self.deadzone_threshold:
            return self.prev_bbox

        smoothed = (self.smoothing * self.prev_bbox) + ((1 - self.smoothing) * new_coords)
        self.prev_bbox = smoothed
        return smoothed

    def detect(self, frame):
        """Runs segmentation inference and outputs transparent RGBA ROIs"""
        results = self.model(frame, conf=0.4, verbose=False)[0]
        scored_detections = []
        
        # If no objects or no masks are found, return empty
        if results.masks is None:
            return scored_detections

        h, w = frame.shape[:2]
        
        for box, mask in zip(results.boxes, results.masks):
            # 1. Extract the raw pixel silhouette mask
            raw_mask = mask.data[0].cpu().numpy()
            full_mask = cv2.resize(raw_mask, (w, h))
            
            # 2. Feather the edges of the silhouette natively!
            full_mask = (full_mask * 255).astype(np.uint8)
            feathered_mask = cv2.GaussianBlur(full_mask, (21, 21), 0)
            
            # 3. Apply Deadzone to the crop box to stop jitter
            coords = box.xyxy[0].cpu().numpy()
            stable_coords = self._apply_deadzone_and_smoothing(coords)
            
            # 4. Define crop boundaries with padding
            pad = 20
            x1, y1, x2, y2 = stable_coords.astype(int)
            fx1, fy1 = max(0, x1-pad), max(0, y1-pad)
            fx2, fy2 = min(w, x2+pad), min(h, y2+pad)
            
            # 5. Crop the color image and the feathered mask
            roi_rgb = frame[fy1:fy2, fx1:fx2]
            roi_alpha = feathered_mask[fy1:fy2, fx1:fx2]
            
            # 6. Merge into a transparent RGBA image
            b, g, r = cv2.split(roi_rgb)
            roi_rgba = cv2.merge((b, g, r, roi_alpha))
            
            scored_detections.append({
                'bbox': [fx1, fy1, fx2, fy2], # Exact X, Y coordinates
                'roi_rgba': roi_rgba,         # The transparent PNG payload
                'class': self.classes[int(box.cls[0])],
                'priority': int(float(box.conf[0]) * 255)
            })
            
        return scored_detections