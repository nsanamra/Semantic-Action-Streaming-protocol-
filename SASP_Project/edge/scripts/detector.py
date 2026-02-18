import cv2
import torch
from ultralytics import YOLO

class SemanticDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load optimized YOLOv8-Nano model
        self.model = YOLO(model_path)
        self.classes = self.model.names
        
        # Priority mapping: 0-10 scale
        self.priority_map = {
            'person': 10,
            'car': 7,
            'truck': 7,
            'motorcycle': 6,
            'bicycle': 5
        }

    def calculate_score(self, class_name, confidence, bbox, frame_shape):
        """Assigns an importance score to a detection"""
        base_priority = self.priority_map.get(class_name, 2)
        
        # Factors: High confidence and larger size = higher importance
        h, w = frame_shape[:2]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        size_factor = min((bbox_area / (w * h)) * 10, 1.5) # Max 1.5x boost for size
        
        final_score = base_priority * confidence * size_factor
        return min(final_score, 10.0)

    def detect(self, frame):
        """Runs inference and returns scored detections"""
        results = self.model(frame, conf=0.4, verbose=False)[0]
        scored_detections = []
        
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.classes[cls_id]
            
            score = self.calculate_score(class_name, conf, coords, frame.shape)
            
            scored_detections.append({
                'bbox': coords.astype(int),
                'class': class_name,
                'score': score,
                'priority': int(score * 25) # Scale to 0-255 for SASP priority
            })
            
        return scored_detections