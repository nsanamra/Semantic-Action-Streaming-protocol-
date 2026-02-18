import cv2
import socket
import struct
import time
from scripts.detector import SemanticDetector

MAGIC = b"SASP"
VERSION = 1
MTU_SIZE = 1400

class SASPTransmitter:
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (server_ip, server_port)
        self.frame_id = 0

    def pack_header(self, frame_type, seq_num, total_parts, obj_class=0, priority=10, x=0, y=0):
        timestamp = int(time.time_ns())
        header = struct.pack("!4sBB IHH Q BB HH", 
            MAGIC, VERSION, frame_type, self.frame_id, 
            seq_num, total_parts, timestamp, obj_class, priority, int(x), int(y)
        )
        return header

    def send_data(self, data, frame_type, priority, x=0, y=0, obj_class=0):
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        
        for i in range(total_parts):
            start = i * MTU_SIZE
            end = min(start + MTU_SIZE, len(data))
            chunk = data[start:end]
            
            header = self.pack_header(frame_type, i, total_parts, priority=priority, x=x, y=y, obj_class=obj_class)
            packet = header + chunk
            self.sock.sendto(packet, self.server_addr)
            
            if frame_type == 0:  
                time.sleep(0.0005)

def run_sasp_edge():
    detector = SemanticDetector()
    transmitter = SASPTransmitter()
    cap = cv2.VideoCapture(0)

    print("SASP Unified-ROI Edge Started...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Returns a SINGLE object containing all people, or None
        combined_roi = detector.detect(frame)

        # 1. Process Background
        blurred_bg = cv2.GaussianBlur(frame, (31, 31), 0)
        _, bg_bytes = cv2.imencode('.jpg', blurred_bg, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        
        # Tell Go if an ROI is coming (1 = Yes, 0 = No)
        has_objects = 1 if combined_roi else 0
        transmitter.send_data(bg_bytes.tobytes(), frame_type=0, priority=10, obj_class=has_objects)

        # 2. Process the Unified ROI
        if combined_roi:
            roi_rgba = combined_roi['roi_rgba']
            fx1, fy1, _, _ = combined_roi['bbox']
            
            # The transparent gaps between people are compressed massively here
            _, roi_bytes = cv2.imencode('.png', roi_rgba, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            
            transmitter.send_data(roi_bytes.tobytes(), 
                                 frame_type=1, 
                                 priority=200,
                                 x=fx1, y=fy1, obj_class=1)

        transmitter.frame_id += 1
        time.sleep(0.03)

    cap.release()

if __name__ == "__main__":
    run_sasp_edge()