import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- LOAD ----------
with open("ffmpeg_metrics_series.json") as f:
    ffmpeg = json.load(f)

with open("sasp_metrics_series.json") as f:
    sasp = json.load(f)

# ---------- PARSE MJPEG ----------
ffmpeg_rows = []
for entry in ffmpeg:
    ts = datetime.fromisoformat(entry["timestamp"])
    d = entry["metrics"]["data"]

    ffmpeg_rows.append({
        "time": ts,
        "fps": d["fps"],
        "bw": d["bandwidth_kbps"],
        "bytes_per_frame": d["bytes_per_frame"],
        "total_mb": d["total_mb_sent"]
    })

df_ff = pd.DataFrame(ffmpeg_rows)

# ---------- PARSE SASP ----------
sasp_rows = []
for entry in sasp:
    ts = datetime.fromisoformat(entry["timestamp"])
    d = entry["metrics"]["data"]

    # keep all non-zero meaningful data
    if d["fps_out"] == 0:
        continue

    sasp_rows.append({
        "time": ts,
        "fps": d["fps_out"],
        "bw": d["bandwidth_out_kbps"],
        "lat_p50": d["latency_p50_ms"],
        "lat_p95": d["latency_p95_ms"],
        "lat_p99": d["latency_p99_ms"],
        "persons": d["persons_per_frame"]
    })

df_sasp = pd.DataFrame(sasp_rows)

# ---------- CONVERT TO RELATIVE TIME ----------
df_ff["t"] = (df_ff["time"] - df_ff["time"].iloc[0]).dt.total_seconds()
df_sasp["t"] = (df_sasp["time"] - df_sasp["time"].iloc[0]).dt.total_seconds()

# ---------- STYLE ----------
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (10, 5),
    "axes.grid": True
})

# =========================================================
# 1️⃣ BANDWIDTH vs TIME
# =========================================================
plt.figure()
plt.plot(df_ff["t"], df_ff["bw"], label="MJPEG")
plt.plot(df_sasp["t"], df_sasp["bw"], label="SASP")
plt.xlabel("Time (s)")
plt.ylabel("Bandwidth (kbps)")
plt.title("Bandwidth Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("bandwidth_vs_time.png", dpi=300)
plt.close()

# =========================================================
# 2️⃣ FPS COMPARISON
# =========================================================
plt.figure()
plt.plot(df_ff["t"], df_ff["fps"], label="MJPEG")
plt.plot(df_sasp["t"], df_sasp["fps"], label="SASP")
plt.xlabel("Time (s)")
plt.ylabel("FPS")
plt.title("FPS Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("fps.png", dpi=300)
plt.close()

# =========================================================
# 3️⃣ BANDWIDTH SAVING %
# (INTERPOLATE SASP TO MJPEG TIME)
# =========================================================
sasp_interp = pd.Series(df_sasp["bw"].values, index=df_sasp["t"])
sasp_interp = sasp_interp.reindex(df_ff["t"], method="nearest")

saving = (df_ff["bw"] - sasp_interp.values) / df_ff["bw"] * 100

plt.figure()
plt.plot(df_ff["t"], saving)
plt.xlabel("Time (s)")
plt.ylabel("Saving (%)")
plt.title("Bandwidth Saving")
plt.tight_layout()
plt.savefig("saving.png", dpi=300)
plt.close()

# =========================================================
# 4️⃣ LATENCY
# =========================================================
plt.figure()
plt.plot(df_sasp["t"], df_sasp["lat_p50"], label="p50")
plt.plot(df_sasp["t"], df_sasp["lat_p95"], label="p95")
plt.plot(df_sasp["t"], df_sasp["lat_p99"], label="p99")
plt.xlabel("Time (s)")
plt.ylabel("Latency (ms)")
plt.title("Latency (SASP)")
plt.legend()
plt.tight_layout()
plt.savefig("latency.png", dpi=300)
plt.close()

# =========================================================
# 5️⃣ BYTES PER FRAME (CORRECTED)
# =========================================================
# kbps → bytes/sec = (kbps * 1000) / 8
df_sasp["bytes_per_frame"] = (df_sasp["bw"] * 1000 / 8) / df_sasp["fps"]

plt.figure()
plt.plot(df_ff["t"], df_ff["bytes_per_frame"], label="MJPEG")
plt.plot(df_sasp["t"], df_sasp["bytes_per_frame"], label="SASP")
plt.xlabel("Time (s)")
plt.ylabel("Bytes per Frame")
plt.title("Compression Efficiency")
plt.legend()
plt.tight_layout()
plt.savefig("bytes.png", dpi=300)
plt.close()

# =========================================================
# 6️⃣ PERSONS vs BANDWIDTH
# =========================================================
plt.figure()
plt.scatter(df_sasp["persons"], df_sasp["bw"])
plt.xlabel("Persons per Frame")
plt.ylabel("Bandwidth (kbps)")
plt.title("Adaptive Streaming Behavior")
plt.tight_layout()
plt.savefig("persons.png", dpi=300)
plt.close()

# =========================================================
# 7️⃣ CUMULATIVE DATA
# =========================================================
df_sasp["cum_mb"] = (df_sasp["bw"].cumsum() / 8000)

plt.figure()
plt.plot(df_ff["t"], df_ff["total_mb"], label="MJPEG")
plt.plot(df_sasp["t"], df_sasp["cum_mb"], label="SASP")
plt.xlabel("Time (s)")
plt.ylabel("Total Data (MB)")
plt.title("Cumulative Data Sent")
plt.legend()
plt.tight_layout()
plt.savefig("cumulative.png", dpi=300)
plt.close()

print("✅ FIXED plots generated")