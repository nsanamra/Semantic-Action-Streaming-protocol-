from scripts.detector import SemanticDetector
from scripts.transmitter import SASPTransmitter
import cv2

def run_sasp_edge():
    detector = SemanticDetector()
    transmitter = SASPTransmitter(server_ip="127.0.0.1", server_port=5000)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Step 1: Detect and Score
        detections = detector.detect(frame)

        # Step 2: Background Encoding (Low Quality: 10%)
        _, bg_bytes = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        transmitter.send_data(bg_bytes.tobytes(), frame_type=0, priority=10)

        # Step 3: ROI Encoding (High Quality: 90%)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            roi = frame[y1:y2, x1:x2]
            
            # Send only if valid ROI
            if roi.size > 0:
                _, roi_bytes = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                # Pass detection metadata to header
                transmitter.send_data(roi_bytes.tobytes(), 
                                     frame_type=1, 
                                     priority=det['priority'])

    cap.release()

if __name__ == "__main__":
    run_sasp_edge()