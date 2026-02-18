"""
main.py — SASP Edge Transmitter

Improvements over v1:
  • Per-person ROI transmission: each detected subject is sent as its own
    ROI packet (Type 1) with individual X/Y coordinates, eliminating the
    "giant bbox spanning the whole frame" problem.
  • Background blur is cached in the detector — recomputed only when the
    scene changes (saves ~2 ms/frame on static shots).
  • frame_id wraps explicitly at 2^32 to match Go's uint32 and avoid any
    possibility of struct packing overflow.
  • Header is always 28 bytes (added X/Y to transmitter.py parity).
  • Graceful shutdown on SIGINT / camera failure.
  • Resolution, FPS target, and network endpoint configurable at the top.
"""

import signal
import sys
import cv2
import socket
import struct
import time
from scripts.detector import SemanticDetector

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

SERVER_IP    = "127.0.0.1"
SERVER_PORT  = 5000
CAMERA_INDEX = 0        # or an RTSP URL string

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30
FRAME_BUDGET = 1.0 / TARGET_FPS   # seconds per frame

MTU_SIZE    = 1400      # Safe UDP payload (avoids IP fragmentation on most LANs)
FRAME_ID_MAX = 2 ** 32  # uint32 wrap-around point

# SASP Protocol
MAGIC   = b"SASP"
VERSION = 1

# Frame types
TYPE_BACKGROUND = 0
TYPE_ROI        = 1

# ─────────────────────────────────────────────────────────────────────────────
#  Transmitter
# ─────────────────────────────────────────────────────────────────────────────

class SASPTransmitter:
    """
    Packs and sends SASP packets over UDP.

    28-byte header layout (big-endian):
        Offset  Size  Field
        0       4     Magic  "SASP"
        4       1     Version
        5       1     Type   (0=BG, 1=ROI)
        6       4     FrameID  (uint32, wraps at 2^32)
        10      2     SeqNum
        12      2     TotalParts
        14      8     Timestamp  (nanoseconds)
        22      1     Class  (0=no objects, 1=objects present)
        23      1     Priority
        24      2     X  (ROI canvas X coordinate)
        26      2     Y  (ROI canvas Y coordinate)
    Total: 28 bytes
    """

    def __init__(self, server_ip: str = SERVER_IP, server_port: int = SERVER_PORT):
        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (server_ip, server_port)
        self.frame_id    = 0

    def _pack_header(
        self,
        frame_type: int,
        seq_num: int,
        total_parts: int,
        obj_class: int = 0,
        priority: int  = 10,
        x: int         = 0,
        y: int         = 0,
    ) -> bytes:
        timestamp = time.time_ns()
        return struct.pack(
            "!4sBB IHH Q BB HH",
            MAGIC,
            VERSION,
            frame_type,
            self.frame_id,
            seq_num,
            total_parts,
            timestamp,
            obj_class,
            priority,
            max(0, int(x)),
            max(0, int(y)),
        )

    def send(
        self,
        data: bytes,
        frame_type: int,
        priority: int,
        x: int = 0,
        y: int = 0,
        obj_class: int = 0,
    ) -> None:
        """Chunk *data* into MTU-sized UDP datagrams and transmit them."""
        total_parts = (len(data) + MTU_SIZE - 1) // MTU_SIZE

        for i in range(total_parts):
            chunk  = data[i * MTU_SIZE : (i + 1) * MTU_SIZE]
            header = self._pack_header(
                frame_type, i, total_parts,
                obj_class=obj_class, priority=priority, x=x, y=y,
            )
            self.sock.sendto(header + chunk, self.server_addr)

            # Micro-delay on background chunks: lets high-priority ROI packets
            # pass through the loopback / router buffer first.
            if frame_type == TYPE_BACKGROUND:
                time.sleep(0.0005)

    def close(self) -> None:
        self.sock.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Edge Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_sasp_edge() -> None:
    detector    = SemanticDetector()
    transmitter = SASPTransmitter()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # ── Hardware / buffer tuning ──────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # kill buffer bloat — always grab latest frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    running = True

    def _shutdown(sig, frame):
        nonlocal running
        print("\n[sasp] Shutting down…")
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if not cap.isOpened():
        print("[sasp] ERROR: Could not open camera.")
        sys.exit(1)

    print("════════════════════════════════════════════")
    print("  SASP Edge Transmitter — Per-Person ROI Mode")
    print(f"  Target: {SERVER_IP}:{SERVER_PORT}  |  {TARGET_FPS} FPS")
    print("════════════════════════════════════════════")

    while running and cap.isOpened():
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[sasp] Camera read failed — exiting.")
            break

        # ── Detection ─────────────────────────────────────────────────────────
        detections = detector.detect(frame)   # list of per-person dicts

        # ── Background ───────────────────────────────────────────────────────
        # Blurring is cached inside the detector — free when scene is static.
        blurred_bg = detector.get_background(frame)

        _, bg_bytes = cv2.imencode(
            '.jpg', blurred_bg,
            [int(cv2.IMWRITE_JPEG_QUALITY), 10]
        )

        has_objects = 1 if detections else 0
        transmitter.send(
            bg_bytes.tobytes(),
            frame_type=TYPE_BACKGROUND,
            priority=10,
            obj_class=has_objects,
        )

        # ── Per-person ROI ────────────────────────────────────────────────────
        # Each subject gets its own PNG packet with precise X/Y placement.
        # This is far more efficient than a single giant bbox spanning the frame.
        for det in detections:
            fx1, fy1, _, _ = det['bbox']
            roi_rgba = det['roi_rgba']

            # PNG level 1 = fastest compression that still preserves alpha
            _, roi_bytes = cv2.imencode(
                '.png', roi_rgba,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
            )

            transmitter.send(
                roi_bytes.tobytes(),
                frame_type=TYPE_ROI,
                priority=200,
                x=fx1, y=fy1,
                obj_class=1,
            )

        # ── Frame ID wrap ─────────────────────────────────────────────────────
        transmitter.frame_id = (transmitter.frame_id + 1) % FRAME_ID_MAX

        # ── Dynamic throttle ──────────────────────────────────────────────────
        elapsed = time.time() - frame_start
        sleep_for = max(0.0, FRAME_BUDGET - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    cap.release()
    transmitter.close()
    print("[sasp] Transmitter stopped.")


if __name__ == "__main__":
    run_sasp_edge()