"""
3D Print Failure Detection Server
----------------------------------
Runs YOLOv8 inference on webcam frames sent from the browser,
streams detection results back via Server-Sent Events (SSE).

Requirements:
    pip install flask flask-cors ultralytics opencv-python numpy

Usage:
    python server.py --model path/to/your/best.pt --port 5050
"""

import argparse
import base64
import json
import time
import threading
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ── Globals ──────────────────────────────────────────────────────────────────
model = None
latest_detections = []          # shared state
detection_lock = threading.Lock()
clients = []                    # SSE client queues
clients_lock = threading.Lock()

# ── SSE helpers ──────────────────────────────────────────────────────────────

def push_event(data: dict):
    """Broadcast a detection event to all SSE clients."""
    msg = f"data: {json.dumps(data)}\n\n"
    with clients_lock:
        dead = []
        for q in clients:
            try:
                q.append(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            clients.remove(q)


def event_stream(q):
    """Generator that yields SSE messages from a per-client list."""
    while True:
        if q:
            yield q.pop(0)
        else:
            time.sleep(0.05)


@app.route("/events")
def events():
    q = []
    with clients_lock:
        clients.append(q)
    return Response(event_stream(q),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Inference endpoint ────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    global model
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image"}), 400

    # Decode base64 → numpy BGR
    header, encoded = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()          # [x1,y1,x2,y2]
        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [round(v, 1) for v in xyxy],
            "timestamp": time.strftime("%H:%M:%S"),
        })

    if detections:
        push_event({"detections": detections, "timestamp": time.strftime("%H:%M:%S")})

    return jsonify({"detections": detections})


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=r"C:\Users\punith p\OneDrive\Desktop\runs\detect\3d_printing_training\yolov8s_custom-3\weights\best.pt",
                        help="Path to your fine-tuned YOLOv8 .pt file")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model = YOLO(args.model)
    print("Model loaded ✓")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)