import cv2
import numpy as np
from ultralytics import YOLO

class SemanticDetector:
    def __init__(self, model_path='yolo26s-seg.pt'):
        self.model = YOLO(model_path)
        self.prev_bbox = None
        self.smoothing = 0.85
        self.deadzone_threshold = 15.0

        # Create a Sharpening Filter to enhance the subject's visual quality
        self.sharpen_kernel = np.array([[0, -1, 0],
                                        [-1, 5,-1],
                                        [0, -1, 0]])

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
        
        if results.masks is None or len(results.boxes) == 0:
            return None

        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        min_x, min_y = w, h
        max_x, max_y = 0, 0

        for box, mask in zip(results.boxes, results.masks):
            raw_mask = mask.data[0].cpu().numpy()
            full_mask = cv2.resize(raw_mask, (w, h))
            full_mask = (full_mask * 255).astype(np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, full_mask)
            
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords.astype(int)
            min_x, min_y = min(min_x, x1), min(min_y, y1)
            max_x, max_y = max(max_x, x2), max(max_y, y2)

        feathered_mask = cv2.GaussianBlur(combined_mask, (21, 21), 0)

        current_bbox = np.array([min_x, min_y, max_x, max_y])
        stable_bbox = self._apply_deadzone_and_smoothing(current_bbox)
        
        pad = 30
        fx1, fy1, fx2, fy2 = stable_bbox.astype(int)
        fx1, fy1 = max(0, fx1-pad), max(0, fy1-pad)
        fx2, fy2 = min(w, fx2+pad), min(h, fy2+pad)

        # Extract color ROI
        roi_rgb = frame[fy1:fy2, fx1:fx2]
        
        # APPLY SHARPENING: Enhances edge contrast for maximum clarity
        roi_rgb = cv2.filter2D(roi_rgb, -1, self.sharpen_kernel)

        roi_alpha = feathered_mask[fy1:fy2, fx1:fx2]
        
        b, g, r = cv2.split(roi_rgb)
        roi_rgba = cv2.merge((b, g, r, roi_alpha))
        
        return {
            'bbox': [fx1, fy1, fx2, fy2],
            'roi_rgba': roi_rgba
        }