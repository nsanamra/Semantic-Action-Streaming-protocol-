import os
import time
import argparse
import sys
import cv2

from detector import SemanticDetector

def main():
    parser = argparse.ArgumentParser(description="Generate Compute vs Network Tradeoff Data")
    parser.add_argument("--image", type=str, required=True, help="Path to a test image")
    parser.add_argument("--iterations", type=int, default=10, help="Number of frames to benchmark")
    parser.add_argument("--bandwidth_kbps", type=float, default=1000.0, help="Simulated Network Bandwidth (KB/s)")
    args = parser.parse_args()

    raw_frame = cv2.imread(args.image)
    if raw_frame is None:
        print(f"Could not load image {args.image}")
        return

    detector = SemanticDetector()
    
    # Warmup
    print(f"Warming up YOLO model on {args.image}...")
    _ = detector.detect(raw_frame)
    
    total_infer_time = 0.0
    sasp_bytes = 0
    trad_bytes = 0
    
    TRAD_QUALITY = 15
    BG_QUALITY = 22
    
    print(f"\n--- Running Benchmark over {args.iterations} iterations ---")
    
    for i in range(args.iterations):
        # 1. Traditional Encode Time
        _, trad_buf = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, TRAD_QUALITY])
        trad_bytes += len(trad_buf)
        
        # 2. SASP Compute Time
        t0 = time.perf_counter()
        detections = detector.detect(raw_frame)
        bg_blurred = detector.get_background(raw_frame)
        total_infer_time += (time.perf_counter() - t0)
        
        _, sasp_bg_buf = cv2.imencode('.jpg', bg_blurred, [cv2.IMWRITE_JPEG_QUALITY, BG_QUALITY])
        sasp_bytes += len(sasp_bg_buf)
        
        if detections:
            roi_rgba = detections[0]['roi_rgba']
            _, roi_png_buf = cv2.imencode('.png', roi_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            sasp_bytes += len(roi_png_buf)
            
    avg_infer_ms = (total_infer_time / args.iterations) * 1000
    avg_trad_kb = (trad_bytes / args.iterations) / 1024.0
    avg_sasp_kb = (sasp_bytes / args.iterations) / 1024.0
    
    # Calculate Transmission Time = (Size MB) / (Bandwidth MB/s)
    trad_transmit_ms = (avg_trad_kb / args.bandwidth_kbps) * 1000
    sasp_transmit_ms = (avg_sasp_kb / args.bandwidth_kbps) * 1000
    
    print(f"\n[Averages per frame]")
    print(f"Traditional Size: {avg_trad_kb:.1f} KB")
    print(f"SASP Size:        {avg_sasp_kb:.1f} KB")
    print(f"SASP Edge Compute Latency: {avg_infer_ms:.1f} ms\n")
    
    print(f"[Simulated Bottleneck @ {args.bandwidth_kbps} KB/s]")
    print(f"Traditional Transmission Time: {trad_transmit_ms:.1f} ms")
    print(f"SASP Transmission Time:        {sasp_transmit_ms:.1f} ms\n")
    
    print(f"[Total Pipeline Latency (Compute + Transmit)]")
    trad_total = trad_transmit_ms # Compute time is ~0 for raw encode
    sasp_total = avg_infer_ms + sasp_transmit_ms
    
    print(f"Traditional Total: {trad_total:.1f} ms")
    print(f"SASP Total:        {sasp_total:.1f} ms")
    
    if sasp_total < trad_total:
        saved = trad_total - sasp_total
        print(f"\n=> CONCLUSION: Spending {avg_infer_ms:.1f}ms on YOLO edge compute SAVES {saved:.1f}ms of total pipeline latency over a {args.bandwidth_kbps} KB/s network!")
    else:
        print(f"\n=> CONCLUSION: At {args.bandwidth_kbps} KB/s, the network is fast enough that the {avg_infer_ms:.1f}ms YOLO compute overhead makes SASP slower. This proves why SASP falls back to Traditional!")

if __name__ == "__main__":
    main()
