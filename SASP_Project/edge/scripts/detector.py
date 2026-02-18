import cv2
import numpy as np
from ultralytics import YOLO

class SemanticDetector:
    def __init__(self, model_path='yolov8n-seg.pt'):
        self.model = YOLO(model_path)
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
        results = self.model(frame, conf=0.4, verbose=False)[0]
        
        # If no objects are found, return None immediately
        if results.masks is None or len(results.boxes) == 0:
            return None

        h, w = frame.shape[:2]
        
        # 1. Create a blank canvas to merge all silhouettes into
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Variables to track the "Global Bounding Box" that fits everyone
        min_x, min_y = w, h
        max_x, max_y = 0, 0

        # 2. Loop through all detected people and merge them
        for box, mask in zip(results.boxes, results.masks):
            # Merge the pixel silhouettes
            raw_mask = mask.data[0].cpu().numpy()
            full_mask = cv2.resize(raw_mask, (w, h))
            full_mask = (full_mask * 255).astype(np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, full_mask) # Adds person to canvas
            
            # Expand the global bounding box to include this person
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords.astype(int)
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

        # 3. Feather the edges of the COMBINED mask natively
        feathered_mask = cv2.GaussianBlur(combined_mask, (21, 21), 0)

        # 4. Apply Anti-Jitter Deadzone to the Global Bounding Box
        current_bbox = np.array([min_x, min_y, max_x, max_y])
        stable_bbox = self._apply_deadzone_and_smoothing(current_bbox)
        
        # 5. Crop the color image and feathered mask using the stabilized coordinates
        pad = 30
        fx1, fy1, fx2, fy2 = stable_bbox.astype(int)
        fx1, fy1 = max(0, fx1-pad), max(0, fy1-pad)
        fx2, fy2 = min(w, fx2+pad), min(h, fy2+pad)

        roi_rgb = frame[fy1:fy2, fx1:fx2]
        roi_alpha = feathered_mask[fy1:fy2, fx1:fx2]
        
        # 6. Merge into a single transparent RGBA image payload
        b, g, r = cv2.split(roi_rgb)
        roi_rgba = cv2.merge((b, g, r, roi_alpha))
        
        return {
            'bbox': [fx1, fy1, fx2, fy2],
            'roi_rgba': roi_rgba
        }