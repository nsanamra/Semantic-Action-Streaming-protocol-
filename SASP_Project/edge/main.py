import cv2
import socket
import struct
import time
import numpy as np
from scripts.detector import SemanticDetector

# SASP Protocol Constants
MAGIC = b"SASP"
VERSION = 1
MTU_SIZE = 1400  # Safe for most networks

class SASPTransmitter:
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (server_ip, server_port)
        self.frame_id = 0

    def pack_header(self, frame_type, seq_num, total_parts, obj_class=0, priority=10):
        """Packs the 24-byte SASP Header"""
        timestamp = int(time.time_ns())
        header = struct.pack("!4sBB IHH Q BB", 
            MAGIC, VERSION, frame_type, self.frame_id, 
            seq_num, total_parts, timestamp, obj_class, priority
        )
        return header

    def send_data(self, data, frame_type, priority):
        """Chunks data and sends over UDP with priority staggered timing"""
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        
        for i in range(total_parts):
            start = i * MTU_SIZE
            end = min(start + MTU_SIZE, len(data))
            chunk = data[start:end]
            
            header = self.pack_header(frame_type, i, total_parts, priority=priority)
            packet = header + chunk
            self.sock.sendto(packet, self.server_addr)
            
            # Artificial delay for background to prioritize ROI
            if frame_type == 0:  
                time.sleep(0.0005)

def run_sasp_edge():
    detector = SemanticDetector()
    transmitter = SASPTransmitter(server_ip="127.0.0.1", server_port=5000)
    cap = cv2.VideoCapture(0)

    print("SASP Edge Transmitter Started...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Detect and Score
        detections = detector.detect(frame)

        # 2. PRE-BLUR BACKGROUND: Removes noise to allow extreme compression without ugly artifacts
        blurred_bg = cv2.GaussianBlur(frame, (31, 31), 0)
        
        # 3. Dual-Path Encoding: Encode Background at 10% Quality
        _, bg_bytes = cv2.imencode('.jpg', blurred_bg, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        transmitter.send_data(bg_bytes.tobytes(), frame_type=0, priority=10)

        # 4. Dual-Path Encoding: Encode ROI at 95% Quality
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                _, roi_bytes = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                transmitter.send_data(roi_bytes.tobytes(), 
                                     frame_type=1, 
                                     priority=det['priority'])

        transmitter.frame_id += 1
        time.sleep(0.03) # ~30 FPS throttle

    cap.release()

if __name__ == "__main__":
    run_sasp_edge()