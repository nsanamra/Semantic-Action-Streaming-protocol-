import cv2
import socket
import struct
import time
from scripts.detector import SemanticDetector

# SASP Protocol Constants
MAGIC = b"SASP"
VERSION = 1
MTU_SIZE = 1400  # MTU-safe size to prevent fragmentation

class SASPTransmitter:
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (server_ip, server_port)
        self.frame_id = 0

    def pack_header(self, frame_type, seq_num, total_parts, obj_class=0, priority=10):
        """Packs the 24-byte binary SASP Header"""
        timestamp = int(time.time_ns())
        header = struct.pack("!4sBB IHH Q BB", 
            MAGIC, VERSION, frame_type, self.frame_id, 
            seq_num, total_parts, timestamp, obj_class, priority
        )
        return header

    def send_data(self, data, frame_type, priority):
        """Chunks encoded byte data into UDP packets based on MTU"""
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        
        for i in range(total_parts):
            start = i * MTU_SIZE
            end = min(start + MTU_SIZE, len(data))
            chunk = data[start:end]
            
            header = self.pack_header(frame_type, i, total_parts, priority=priority)
            packet = header + chunk
            self.sock.sendto(packet, self.server_addr)
            
            # Artificial micro-delay for background packets to prioritize ROI transit
            if frame_type == 0:  
                time.sleep(0.0005)

def run_sasp_edge():
    detector = SemanticDetector()
    transmitter = SASPTransmitter(server_ip="127.0.0.1", server_port=5000)
    
    # 0 = Default Webcam
    cap = cv2.VideoCapture(0)
    print("SASP Edge Transmitter Started. Initiating live stream...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Detect objects and retrieve locked/smoothed coordinates
        detections = detector.detect(frame)

        # 2. Pre-Blur Background: Obliterates image entropy so 10% compression doesn't look glitchy
        blurred_bg = cv2.GaussianBlur(frame, (31, 31), 0)
        
        # 3. Dual-Path Encoding: Encode the blurred background aggressively
        _, bg_bytes = cv2.imencode('.jpg', blurred_bg, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        transmitter.send_data(bg_bytes.tobytes(), frame_type=0, priority=10)

        # 4. Dual-Path Encoding: Crop and encode the crystal-clear ROI
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                # 95% High Quality for semantic mission targets
                _, roi_bytes = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                transmitter.send_data(roi_bytes.tobytes(), 
                                     frame_type=1, 
                                     priority=det['priority'])

        transmitter.frame_id += 1
        time.sleep(0.03) # Cap to ~30 FPS

    cap.release()

if __name__ == "__main__":
    run_sasp_edge()