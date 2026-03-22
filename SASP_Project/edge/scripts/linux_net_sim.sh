#!/bin/bash
# Linux Network Emulator for SASP Research Paper Testing
# Run this on the Linux machine hosting the Go Server to simulate a bad network.

if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo."
  exit 1
fi

echo "[1/3] Installing iproute2 (Traffic Control tool)..."
apt-get update -y
apt-get install -y iproute2

INTERFACE="lo" # Assuming you are testing on localhost. Change to eth0 or wlan0 if testing across devices!
PORT=5000      # The UDP port the Go server listens on

echo "[2/3] Resetting any existing traffic rules on $INTERFACE..."
tc qdisc del dev $INTERFACE root 2>/dev/null

if [ "$1" == "stop" ]; then
    echo "Traffic control rules cleared. Network is back to normal!"
    exit 0
fi

echo "[3/3] Emulating Bad Network on $INTERFACE (Port $PORT)..."

# 1. Add a root queuing discipline (htb = hierarchical token bucket)
tc qdisc add dev $INTERFACE root handle 1: htb default 10

# 2. Limit the bandwidth to 500 KB/s (4000 kbit)
tc class add dev $INTERFACE parent 1: classid 1:1 htb rate 4000kbit

# 3. Add 50ms latency (jitter 10ms) and 5% packet loss to the class
tc qdisc add dev $INTERFACE parent 1:1 handle 10: netem delay 50ms 10ms loss 5%

# 4. Filter only traffic going to exactly UDP Port 5000
tc filter add dev $INTERFACE protocol ip parent 1:0 prio 1 u32 match ip dport $PORT 0xffff flowid 1:1

echo "======================================================"
echo "    SASP NETWORK EMULATION ACTIVE!"
echo "    Interface: $INTERFACE"
echo "    Target:    UDP Port $PORT"
echo "    Limit:     500 KB/s BANDWIDTH"
echo "    Chaos:     50ms Latency | 5% Packet Loss"
echo "======================================================"
echo "Run your Python Edge script now! It should be forced"
echo "to drop into SASP Semantic Mode to survive."
echo "------------------------------------------------------"
echo "To disable this limit and return to normal, run:"
echo "sudo ./linux_net_sim.sh stop"
