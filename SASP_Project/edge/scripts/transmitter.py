import cv2
import socket
import struct
import time
import numpy as np

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
        """
        Packs the 24-byte SASP Header 
        Structure: Magic(4), Ver(1), Type(1), FrameID(4), Seq(2), Total(2), 
                   TS(8), Class(1), Conf(1). (Simplified for Phase 1)
        """
        timestamp = int(time.time_ns())
        # Format: 4s (Magic), B (Ver), B (Type), I (FrameID), H (Seq), H (Total), Q (TS), B (Class), B (Priority)
        # Total: 4 + 1 + 1 + 4 + 2 + 2 + 8 + 1 + 1 = 24 bytes
        header = struct.pack("!4sBB IHH Q BB", 
            MAGIC, VERSION, frame_type, self.frame_id, 
            seq_num, total_parts, timestamp, obj_class, priority
        )
        return header

    def send_data(self, data, frame_type, priority):
        """Chunks data and sends over UDP [cite: 73, 74]"""
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        
        for i in range(total_parts):
            start = i * MTU_SIZE
            end = min(start + MTU_SIZE, len(data))
            chunk = data[start:end]
            
            header = self.pack_header(frame_type, i, total_parts, priority=priority)
            packet = header + chunk
            self.sock.sendto(packet, self.server_addr)
            
            # Artificial delay for background to prioritize ROI [cite: 28, 109]
            if frame_type == 0:  # Background
                time.sleep(0.0005)

    def run(self):
        cap = cv2.VideoCapture(0) # Use 0 for webcam
        print(f"SASP Transmitter started. Sending to {self.server_addr}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. Mock ROI (Let's pretend a person is in the center) [cite: 105, 106]
            h, w, _ = frame.shape
            roi_rect = (int(w*0.4), int(h*0.4), int(w*0.2), int(h*0.2)) # x, y, w, h
            x, y, rw, rh = roi_rect
            roi_frame = frame[y:y+rh, x:x+rw]

            # 2. Dual-Path Encoding [cite: 9, 25, 90]
            # Background: Low quality (5%)
            _, bg_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
            # ROI: High quality (95%)
            _, roi_encoded = cv2.imencode('.jpg', roi_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # 3. Transmit via SASP [cite: 10, 91]
            # Send Background (Type 0, Priority 10)
            self.send_data(bg_encoded.tobytes(), frame_type=0, priority=10)
            # Send ROI (Type 1, Priority 200)
            self.send_data(roi_encoded.tobytes(), frame_type=1, priority=200)

            self.frame_id += 1
            time.sleep(0.03) # ~30 FPS [cite: 69]

        cap.release()

if __name__ == "__main__":
    transmitter = SASPTransmitter()
    transmitter.run()