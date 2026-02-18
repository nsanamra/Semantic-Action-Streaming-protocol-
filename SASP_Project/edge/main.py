import cv2
import socket
import struct
import time
from scripts.detector import SemanticDetector

# SASP Protocol Constants
MAGIC = b"SASP"
VERSION = 1
MTU_SIZE = 1400  # MTU-safe size to prevent UDP fragmentation

class SASPTransmitter:
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        # Initialize UDP socket for rapid, connectionless transmission
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (server_ip, server_port)
        self.frame_id = 0

    def pack_header(self, frame_type, seq_num, total_parts, obj_class=0, priority=10, x=0, y=0):
        """Packs the 28-byte binary SASP Header to include X, Y segmentation coordinates."""
        timestamp = int(time.time_ns())
        
        # Format: Magic(4s), Version(1B), Type(1B), FrameID(4B), SeqNum(2B), TotalParts(2B), 
        #         Timestamp(8B), Class(1B), Priority(1B), X(2B), Y(2B)
        header = struct.pack("!4sBB IHH Q BB HH", 
            MAGIC, VERSION, frame_type, self.frame_id, 
            seq_num, total_parts, timestamp, obj_class, priority, int(x), int(y)
        )
        return header

    def send_data(self, data, frame_type, priority, x=0, y=0, obj_class=0):
        """Chunks encoded byte data into MTU-sized UDP packets and transmits them."""
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE
        
        for i in range(total_parts):
            start = i * MTU_SIZE
            end = min(start + MTU_SIZE, len(data))
            chunk = data[start:end]
            
            header = self.pack_header(frame_type, i, total_parts, priority=priority, x=x, y=y, obj_class=obj_class)
            packet = header + chunk
            self.sock.sendto(packet, self.server_addr)
            
            # Artificial micro-delay for background packets to prevent them from choking 
            # the router buffer, prioritizing the transit of ROI packets.
            if frame_type == 0:  
                time.sleep(0.0005)

def run_sasp_edge():
    detector = SemanticDetector()
    transmitter = SASPTransmitter()
    
    # 0 = Default Webcam. Change to IP string for external RTSP cameras if needed.
    cap = cv2.VideoCapture(0)
    
    # ---------------------------------------------------------
    # ZERO-LAG OPTIMIZATIONS (Hardware & Buffer Tuning)
    # ---------------------------------------------------------
    # 1. Kill Buffer Bloat: Force the camera to discard old frames and only keep the 1 most recent frame.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    # 2. Cap Native Resolution: Prevents the CPU from wasting time blurring/encoding 1080p pixels 
    #    when the YOLO detector natively scales to 640 anyway.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # ---------------------------------------------------------

    print("SASP Edge Transmitter Started.")
    print("Mode: Unified Instance Segmentation | Anti-Buffer Bloat Active")

    while cap.isOpened():
        # Start timer to measure processing cost for accurate FPS throttling
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret: 
            break

        # Returns a SINGLE payload containing all detected people, or None
        combined_roi = detector.detect(frame)

        # =========================================================
        # PATH 1: BACKGROUND PROCESSING (10% JPEG)
        # =========================================================
        # Blur obliterates high-frequency image entropy so 10% compression looks smooth, not broken
        blurred_bg = cv2.GaussianBlur(frame, (31, 31), 0)
        _, bg_bytes = cv2.imencode('.jpg', blurred_bg, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        
        # Tell the Go backend whether it should wait for a corresponding ROI packet (1 = Yes, 0 = No)
        has_objects = 1 if combined_roi else 0
        transmitter.send_data(bg_bytes.tobytes(), frame_type=0, priority=10, obj_class=has_objects)

        # =========================================================
        # PATH 2: UNIFIED ROI PROCESSING (PNG)
        # =========================================================
        if combined_roi:
            roi_rgba = combined_roi['roi_rgba']
            fx1, fy1, _, _ = combined_roi['bbox']
            
            # Use fastest PNG compression (Level 1) to preserve alpha channel transparency 
            # with the absolute minimum CPU latency cost.
            _, roi_bytes = cv2.imencode('.png', roi_rgba, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            
            transmitter.send_data(roi_bytes.tobytes(), 
                                 frame_type=1, 
                                 priority=200,
                                 x=fx1, y=fy1, obj_class=1)

        # Increment protocol sequence tracking
        transmitter.frame_id += 1
        
        # =========================================================
        # DYNAMIC THROTTLING
        # =========================================================
        # Sleep ONLY if the frame processed faster than 33ms (targeting a stable 30fps).
        # This prevents the loop from needlessly resting if the CPU is running hot.
        processing_time = time.time() - start_time
        sleep_time = max(0, 0.033 - processing_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()

if __name__ == "__main__":
    run_sasp_edge()