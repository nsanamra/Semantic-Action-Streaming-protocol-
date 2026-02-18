
---

# **Roadmap.md: SASP Implementation Strategy**

## **Phase 1: The Skeleton (Basic Connectivity)**

*Goal: Establish a raw UDP video stream between Python (Drone) and Go (Server).*

1. **Python Side:** Capture video using OpenCV and send raw, low-resolution frames over standard UDP.


2. **Go Side:** Create a basic UDP listener and a WebSocket server to pipe those raw frames to a simple HTML Canvas.


3. **Micro-Check:** * Run the Python script and the Go server.
* **Success:** You can see a live (albeit low quality/laggy) video feed in your browser dashboard.


## **Phase 2: Edge Intelligence (Semantic Analysis)**

*Goal: Implement the "Brain" on the Drone to identify what matters.*

1. **Inference Pipeline:** Integrate YOLOv8-Nano. Implement the "Detection Skipping" logic based on a simulated "Battery/Bandwidth" variable.


2. **Scoring Engine:** Implement the `calculate_semantic_importance` function to generate importance scores ().


3. **Micro-Check:** * Run the detector. Print the bounding boxes and their importance scores to the console.
* **Success:** The system correctly identifies a "Person" as higher priority than a "Tree."



## **Phase 3: Dual-Path Encoding & SASP Protocol**

*Goal: Break the frame apart and wrap it in your custom protocol.*

1. **The Splitter:** Implement the logic to crop ROIs and compress the background aggressively (5-10% quality).


2. **Protocol Header:** Build the 24-byte **SASP Header** in both Python (struct pack) and Go (binary Read).


3. **Priority Transmission:** Implement the staggered sending logic (Background packets have a small `time.sleep`, ROI packets are sent instantly).


4. **Micro-Check:** * Save a sample Background and ROI packet to disk.
* **Success:** The ROI image is crisp ( quality) and the Background is highly pixelated, and the Go server can print the `FrameID` and `ObjectType` from the SASP header.



## **Phase 4: High-Performance Backend (Reconstruction)**

*Goal: Use Go’s concurrency to stitch the "Islands" back into a "Sea."*

1. **Worker Pools:** Initialize the `PacketWorker` and `StitchWorker` pools using Goroutines.


2. **The Stitcher:** Implement the feathering/blending algorithm to overlay ROIs onto the background without "harsh edges".


3. **Selective Reliability:** Implement the ACK mechanism where the Go server tells the Python client, "I got the ROI, don't worry about retransmitting".


4. **Micro-Check:** * Stream a frame with a moving object.
* **Success:** The dashboard shows a coherent video where the person is clear, the background is blurry, and there are no flickering black boxes where the ROIs should be.



## **Phase 5: The Optimization Loop (Q-Learning)**

*Goal: Enable the system to learn and adapt autonomously.*

1. **State/Action Space:** Define the Q-Table in the Go backend.


2. **Reward Function:** Implement the logic that penalizes high bandwidth usage and rewards high detection accuracy.


3. **Control Channel:** Use gRPC or WebSockets to send "Action" commands (e.g., "Drop FPS to 15," "Set BG Quality to 5%") back to the drone.


4. **Micro-Check:** * Simulate 50% packet loss.
* **Success:** After a few minutes, the Q-Learning agent should automatically command the drone to lower its frame rate and increase ROI compression to maintain stability.



## **Phase 6: Dashboard & Metrics**

*Goal: Visualize the "Semantic Advantage."*

1. **Comparison View:** Create a toggle on the dashboard to show "Traditional Streaming" vs. "SASP Streaming."
2. **Metrics Overlay:** Graph the real-time Bandwidth Savings () and Latency.


3. **Micro-Check:** * Final stress test with simulated low-bandwidth.
* **Success:** The "Traditional" side freezes/buffers, while the "SASP" side continues to show clear, identifiable targets.



---
