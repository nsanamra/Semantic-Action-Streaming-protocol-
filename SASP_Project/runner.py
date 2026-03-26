import subprocess
import time
import requests
import json
import os
from datetime import datetime

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.join(BASE_DIR, "backend")
EDGE_DIR = os.path.join(BASE_DIR, "edge")
MJPEG_PATH = os.path.join(BASE_DIR, "mjpeg.py")

SASP_GO_CMD = ["go", "run", "main.go"]
SASP_EDGE_CMD = ["python3", "main.py"]
FFMPEG_CMD = ["python3", MJPEG_PATH]

SASP_API = "http://localhost:8080/api/metrics"
FFMPEG_API = "http://localhost:8081/metrics"

INTERVAL = 1  # seconds between API calls

# ---------- HELPERS ----------
def start_process(cmd, cwd=None, name=""):
    print(f"[STARTING] {name}")
    return subprocess.Popen(cmd, cwd=cwd)

def fetch_metrics(url):
    try:
        res = requests.get(url, timeout=2)
        return res.json()
    except:
        return None

def stop_process(proc, name):
    if proc and proc.poll() is None:
        print(f"[STOPPING] {name}")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()

def record_metrics(url, process_list, label):
    print(f"\n[RECORDING] {label} metrics continuously...\n")

    data_series = []

    try:
        while True:
            # stop if ALL processes ended
            if all(p.poll() is not None for p in process_list):
                print(f"[INFO] {label} processes ended")
                break

            metrics = fetch_metrics(url)

            if metrics:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics
                }
                data_series.append(entry)
                print(f"[{label}] captured @ {entry['timestamp']}")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Stopping {label} recording...")

    return data_series

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[SAVED] {filename}")

# ---------- MAIN ----------
def main():
    processes = []

    try:
        # ===== SASP =====
        go_proc = start_process(SASP_GO_CMD, cwd=BACKEND_DIR, name="SASP BACKEND")
        edge_proc = start_process(SASP_EDGE_CMD, cwd=EDGE_DIR, name="SASP EDGE")

        sasp_series = record_metrics(
            SASP_API,
            [go_proc, edge_proc],
            "SASP"
        )

        save_json(sasp_series, "sasp_metrics_series.json")

        stop_process(go_proc, "SASP BACKEND")
        stop_process(edge_proc, "SASP EDGE")

        time.sleep(2)

        # ===== FFMPEG =====
        ffmpeg_proc = start_process(FFMPEG_CMD, name="FFMPEG")

        ffmpeg_series = record_metrics(
            FFMPEG_API,
            [ffmpeg_proc],
            "FFMPEG"
        )

        save_json(ffmpeg_series, "ffmpeg_metrics_series.json")

        stop_process(ffmpeg_proc, "FFMPEG")

        print("\n✅ DONE — Time series captured!")

    finally:
        for p in processes:
            stop_process(p, "Cleanup")

# ---------- ENTRY ----------
if __name__ == "__main__":
    main()