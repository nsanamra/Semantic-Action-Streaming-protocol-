#!/usr/bin/env python3
"""
compare.py — Side-by-side SASP vs Traditional metrics
Run in its own terminal while both backends are running.
"""

import time
import urllib.request
import json

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            return json.load(r)
    except Exception:
        return None

def fmt_bw(kbs):
    if kbs >= 1024:
        return f"{kbs/1024:.2f} MB/s"
    return f"{kbs:.1f} KB/s"

def bar(value, max_value, width=24):
    if max_value == 0:
        return "░" * width
    filled = int(min(value / max_value, 1.0) * width)
    return "█" * filled + "░" * (width - filled)

W = 66

def run():
    while True:
        sasp = fetch("http://localhost:8080/api/metrics")
        trad = fetch("http://localhost:8081/stats")

        lines = []
        lines.append("")
        lines.append("=" * W)
        lines.append("  SASP vs TRADITIONAL — Live Comparison".center(W))
        lines.append("=" * W)

        if sasp is None and trad is None:
            lines.append("")
            lines.append("  Waiting for both backends to start...")
            lines.append("")
        else:
            sd = sasp.get("data", {}) if sasp else {}
            td = trad.get("data", {}) if trad else {}

            sasp_bw_out  = sd.get("bandwidth_out_kbps", 0)
            trad_bw_in   = td.get("bandwidth_in_kbs", 0)
            trad_bw_out  = td.get("bandwidth_out_kbs", 0)
            sasp_fps     = sd.get("fps_out", 0)
            trad_fps     = td.get("fps_out", 0)
            sasp_p50     = sd.get("latency_p50_ms", 0)
            sasp_p95     = sd.get("latency_p95_ms", 0)
            trad_p50     = td.get("latency_p50_ms", 0)
            trad_p95     = td.get("latency_p95_ms", 0)
            sasp_drop    = sd.get("dropped_frames", 0)
            trad_drop    = td.get("dropped_frames", 0)
            sasp_persons = sd.get("persons_per_frame", 0)
            sasp_clients = sd.get("active_clients", 0)
            trad_clients = td.get("active_clients", 0)

            L = 20
            C = 18

            lines.append(f"\n  {'':>{L}}  {'SASP':^{C}}  {'TRADITIONAL':^{C}}")
            lines.append(f"  {'':>{L}}  {'localhost:8080':^{C}}  {'localhost:8081':^{C}}")
            lines.append(f"  {'-'*L}  {'-'*C}  {'-'*C}")

            rows = [
                ("FPS out",
                 f"{sasp_fps:.1f}",
                 f"{trad_fps:.1f}"),
                ("Wire BW (UDP in)",
                 "semantic only",
                 fmt_bw(trad_bw_in)),
                ("Client BW (WS out)",
                 fmt_bw(sasp_bw_out),
                 fmt_bw(trad_bw_out)),
                ("Latency P50",
                 f"{sasp_p50:.1f} ms",
                 f"{trad_p50:.1f} ms"),
                ("Latency P95",
                 f"{sasp_p95:.1f} ms",
                 f"{trad_p95:.1f} ms"),
                ("Dropped frames",
                 str(sasp_drop),
                 str(trad_drop)),
                ("Person detection",
                 f"{sasp_persons:.1f} / frame",
                 "none"),
                ("Clients",
                 str(sasp_clients),
                 str(trad_clients)),
            ]

            for label, sv, tv in rows:
                lines.append(f"  {label:>{L}}  {sv:^{C}}  {tv:^{C}}")

            max_bw = max(sasp_bw_out, trad_bw_out, 1)
            lines.append("")
            lines.append(f"  {'BW to clients':>{L}}  (WebSocket output — apples to apples)")
            lines.append(f"  {'SASP':>{L}}  {bar(sasp_bw_out, max_bw)}  {fmt_bw(sasp_bw_out)}")
            lines.append(f"  {'TRAD':>{L}}  {bar(trad_bw_out, max_bw)}  {fmt_bw(trad_bw_out)}")

            lines.append("")
            if sasp_bw_out > 0 and trad_bw_out > 0:
                ratio = trad_bw_out / sasp_bw_out
                saved = (1 - sasp_bw_out / trad_bw_out) * 100
                lines.append(f"  {'SASP sends':>{L}}  {ratio:.1f}x less data to clients")
                lines.append(f"  {'Saving':>{L}}  {saved:.0f}% bandwidth vs Traditional")
            elif sasp is None:
                lines.append("  SASP backend not reachable — is 'go run main.go' running?")
            elif trad is None:
                lines.append("  Traditional not reachable — is 'go run traditional_backend.go' running?")

            lines.append("")
            lines.append(f"  Updated: {time.strftime('%H:%M:%S')}   Ctrl+C to stop")

        lines.append("=" * W)

        # Single atomic print — no flicker
        import os
        os.system("clear")
        print("\n".join(lines), flush=True)

        time.sleep(1)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nStopped.")