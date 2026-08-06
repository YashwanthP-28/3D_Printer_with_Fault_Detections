import streamlit as st
import cv2
import tempfile
import os
import time
import subprocess
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3D Print Fault Detection",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Class Names & Colors ──────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Spaghetti", 1: "Stringing", 2: "Warping", 3: "Over Extrusion",
    4: "Under Extrusion", 5: "Cracks", 6: "Blobs", 7: "Zits",
    8: "Printer Gantry", 9: "Fail",
}

CLASS_COLORS_HEX = {
    "Spaghetti": "#ff6347", "Stringing": "#ffa500", "Warping": "#ffd700",
    "Over Extrusion": "#00c864", "Under Extrusion": "#1e90ff", "Cracks": "#9400d3",
    "Blobs": "#ff1493", "Zits": "#00ced1", "Printer Gantry": "#a9a9a9", "Fail": "#ff4b6e",
}

CLASS_COLORS_BGR = {
    "Spaghetti": (71, 99, 255), "Stringing": (0, 165, 255), "Warping": (0, 215, 255),
    "Over Extrusion": (100, 200, 0), "Under Extrusion": (255, 144, 30), "Cracks": (211, 0, 148),
    "Blobs": (147, 20, 255), "Zits": (209, 206, 0), "Printer Gantry": (169, 169, 169), "Fail": (110, 75, 255),
}

DEFECT_DESCRIPTIONS = {
    "Spaghetti": "Filament extruded mid-air forming tangled strands",
    "Stringing": "Fine threads of filament between printed parts",
    "Warping": "Edges lifting from the build plate",
    "Over Extrusion": "Too much filament causing blobs or rough surfaces",
    "Under Extrusion": "Too little filament causing gaps or weak layers",
    "Cracks": "Layer delamination or fracture lines",
    "Blobs": "Localised excess material on surfaces",
    "Zits": "Tiny pimple-like artifacts on the surface",
    "Printer Gantry": "Gantry obstructing the camera view",
    "Fail": "Catastrophic general print failure",
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background-color: #080a0f; color: #e2e4f0; }
section[data-testid="stSidebar"] { background: #0b0d12 !important; border-right: 1px solid #1a1f2e; }

.pg-header { display: flex; align-items: center; gap: 14px; padding: 1.2rem 0 0.2rem; border-bottom: 1px solid #1a1f2e; margin-bottom: 1.4rem; }
.pg-logo { font-size: 1.8rem; background: #111827; border: 1px solid #1e2840; border-radius: 10px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; }
.pg-title { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.03em; color: #fff; line-height: 1.1; }
.pg-sub { font-size: 0.78rem; color: #4e5472; margin-top: 2px; }

.stButton > button {
    background: linear-gradient(135deg, #3d5afe, #1a3aff) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: 0.65rem 2rem !important; font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    box-shadow: 0 4px 20px rgba(61,90,254,0.3) !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #5c77ff, #3d5afe) !important; transform: translateY(-1px) !important; }

.stop-btn button {
    background: linear-gradient(135deg, #ff4b6e, #d6304f) !important;
    box-shadow: 0 4px 20px rgba(255,75,110,0.3) !important;
}
.stop-btn button:hover { background: linear-gradient(135deg, #ff6a8a, #ff4b6e) !important; }

.ghost-btn button {
    background: transparent !important;
    border: 1px solid #1e2840 !important;
    color: #c0c4d8 !important;
    box-shadow: none !important;
}
.ghost-btn button:hover { border-color: #3d5afe !important; color: #fff !important; }

.sec-label { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase; color: #2e3558; border-bottom: 1px solid #111827; padding-bottom: 0.4rem; margin-bottom: 0.9rem; }

video, img[data-testid="stImage"] { width: 100% !important; min-height: 460px !important; border-radius: 10px !important; background: #000 !important; object-fit: contain !important; }

.stat-row { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
.stat-card { flex: 1; background: #0e1118; border: 1px solid #1a1f2e; border-radius: 12px; padding: 0.85rem 1rem; }
.stat-card-label { font-size: 0.6rem; letter-spacing: 0.12em; text-transform: uppercase; color: #3a4060; margin-bottom: 0.25rem; }
.stat-card-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #fff; line-height: 1; }

.live-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(255,75,110,0.1); border: 1px solid rgba(255,75,110,0.3); color: #ff4b6e; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; padding: 3px 10px; border-radius: 999px; margin-top: 0.6rem; }
.live-dot { width: 6px; height: 6px; background: #ff4b6e; border-radius: 50%; animation: pulse 1.2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

.log-box { background: #0a0c12; border: 1px solid #131825; border-radius: 10px; padding: 0.6rem 0.8rem; max-height: 380px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; }
.log-entry { display: flex; justify-content: space-between; align-items: center; padding: 0.45rem 0.4rem; border-bottom: 1px solid #111522; font-size: 0.74rem; }
.log-entry:last-child { border-bottom: none; }
.log-time { color: #3a4060; font-size: 0.68rem; }
.log-label { font-weight: 600; }
.log-conf { color: #4e5472; }

.defect-row { margin-bottom: 0.7rem; padding: 0.75rem 0.9rem; background: #0a0c12; border: 1px solid #131825; border-radius: 10px; }
.defect-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.defect-name { font-size: 0.82rem; font-weight: 600; color: #d0d4f0; }
.defect-count { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 600; }
.defect-bar-bg { height: 5px; background: #131825; border-radius: 3px; margin-bottom: 0.3rem; overflow: hidden; }
.defect-bar-fill { height: 100%; border-radius: 3px; }
.defect-meta { font-size: 0.66rem; color: #2e3558; }

.empty-state { border: 2px dashed #131825; border-radius: 14px; padding: 3rem 2rem; text-align: center; color: #2e3558; margin-top: 0.5rem; }
.empty-icon { font-size: 2.6rem; margin-bottom: 0.5rem; }
.empty-title { font-size: 1rem; font-weight: 600; color: #3a4060; margin-bottom: 0.3rem; }
.empty-sub { font-size: 0.78rem; color: #252a40; }
.class-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.8rem; }
.class-card { background: #0a0c12; border: 1px solid #131825; border-radius: 10px; padding: 0.85rem 1rem; }
.class-card-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: middle; }
.class-card-name { font-size: 0.8rem; font-weight: 600; color: #b0b4d0; }
.class-card-desc { font-size: 0.7rem; color: #2e3558; margin-top: 0.2rem; line-height: 1.5; }

/* ── Mode select cards ── */
.hero-block { padding: 0.5rem 0 2rem; text-align: center; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: #3d5afe;
    margin-bottom: 0.7rem;
}
.hero-headline {
    font-size: 1.8rem; font-weight: 700; letter-spacing: -0.03em;
    color: #fff; margin-bottom: 0.5rem;
}
.hero-tagline { font-size: 0.92rem; color: #4e5472; max-width: 480px; margin: 0 auto; }

.mode-card-wrap {
    background: #0c0f16;
    border: 1px solid #1a1f2e;
    border-radius: 16px;
    padding: 1.8rem 1.6rem 1.6rem;
    height: 100%;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
    display: flex; flex-direction: column;
}
.mode-card-wrap:hover {
    border-color: #2a3358;
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.35);
}
.mode-card-icon-wrap {
    width: 52px; height: 52px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; margin-bottom: 1rem;
}
.mode-card-icon-wrap.cam  { background: rgba(255,75,110,0.1); border: 1px solid rgba(255,75,110,0.25); }
.mode-card-icon-wrap.up   { background: rgba(61,90,254,0.1); border: 1px solid rgba(61,90,254,0.25); }
.mode-card-title { font-size: 1.05rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem; letter-spacing: -0.01em; }
.mode-card-desc { font-size: 0.8rem; color: #4e5472; margin-bottom: 1.3rem; line-height: 1.55; flex-grow: 1; }
.mode-card-tag {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: #3a4060; background: #111522; border: 1px solid #1a1f2e;
    padding: 2px 8px; border-radius: 5px; margin-bottom: 0.9rem;
}

/* file_uploader styling to feel embedded, not boxy */
[data-testid="stFileUploaderDropzone"] {
    background: #0a0c12 !important;
    border: 1.5px dashed #1e2840 !important;
    border-radius: 10px !important;
    padding: 0.6rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #3d5afe !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: #4e5472 !important; }

.mode-grid-spacer { margin-bottom: 0.9rem; }

.vinfo { background: #0e1118; border: 1px solid #1a1f2e; border-radius: 10px; padding: 0.8rem 1.2rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #4e5472; margin-bottom: 1.2rem; display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: center; }
.vinfo b { color: #c8cadf; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pg-header">
        <div class="pg-title">3D Print Fault Detection</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**MODEL**")
    model_path = st.text_input(
        "Weights path",
        value=r"C:\Users\punith p\OneDrive\Desktop\runs\detect\3d_printing_training\yolov8s_custom-3\weights\best.pt",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**CAMERA** *(webcam mode only)*")
    camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1,
                                    help="0 is usually the default built-in/USB webcam")
    st.markdown("---")
    st.markdown("**DETECTION**")
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.05)
    iou_threshold  = st.slider("IoU threshold (NMS)",  0.10, 0.95, 0.45, 0.05)
    st.markdown("---")
    st.markdown("**PLAYBACK** *(video mode only)*")
    frame_skip = st.slider("Process every N frames", 1, 10, 2, help="Higher = faster, fewer detections")
    realtime_speed = st.toggle("Match real playback speed", value=True)
    st.markdown("---")
    st.markdown("**OUTPUT**")
    show_labels = st.toggle("Show labels on video", value=True)
    show_conf   = st.toggle("Show confidence %",    value=True)
    st.markdown("---")
    st.caption("YOLOv8s fine-tuned · 10 defect classes · RTX 3050")

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

if not os.path.exists(model_path):
    st.error(f"Model weights not found at `{model_path}`. Update the path in the sidebar.")
    st.stop()

with st.spinner("Loading model weights…"):
    model = load_model(model_path)

# ─── Session state ────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = None        # "webcam" | "upload"
if "scanning" not in st.session_state:
    st.session_state.scanning = False
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None

def reset_to_landing():
    st.session_state.mode = None
    st.session_state.scanning = False
    st.session_state.uploaded_video_path = None

# ════════════════════════════════════════════════════════════════════════════
# LANDING — mode selection
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.mode is None:

    st.markdown("""
    <div class="hero-block">
        <div class="hero-eyebrow">Choose an input source</div>
        <div class="hero-headline">How do you want to scan?</div>
        <div class="hero-tagline">Monitor a live print over webcam, or upload a recording to analyze frame-by-frame.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=False):
            st.markdown("""
            <div class="mode-card-wrap">
                <div class="mode-card-icon-wrap cam">🎥</div>
                <div class="mode-card-tag">Real-time</div>
                <div class="mode-card-title">Live Webcam</div>
                <div class="mode-card-desc">
                    Point a webcam at your printer for continuous monitoring —
                    detections, stats, and the live log update as it runs.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="mode-grid-spacer"></div>', unsafe_allow_html=True)
            if st.button("▶  Start Webcam Scan", use_container_width=True, key="start_webcam"):
                st.session_state.mode = "webcam"
                st.session_state.scanning = True
                st.rerun()

    with col2:
        with st.container(border=False):
            st.markdown("""
            <div class="mode-card-wrap">
                <div class="mode-card-icon-wrap up">📂</div>
                <div class="mode-card-tag">Pre-recorded</div>
                <div class="mode-card-title">Upload Video</div>
                <div class="mode-card-desc">
                    Upload a recording of a print to scan it frame-by-frame
                    at its natural playback speed.
                </div>
            </div>
            """, unsafe_allow_html=True)
            uploaded = st.file_uploader("upload_video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
            if uploaded is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded.read())
                tfile.flush()
                tfile.close()
                st.session_state.uploaded_video_path = tfile.name
                st.session_state.uploaded_video_name = uploaded.name
                st.session_state.uploaded_video_size = uploaded.size
                if st.button("▶  Start Video Scan", use_container_width=True, key="start_upload"):
                    st.session_state.mode = "upload"
                    st.session_state.scanning = True
                    st.rerun()
            else:
                st.markdown('<div class="mode-grid-spacer"></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2.2rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Detectable defect classes</div>', unsafe_allow_html=True)
    st.markdown('<div class="class-grid">', unsafe_allow_html=True)
    for name, desc in DEFECT_DESCRIPTIONS.items():
        color = CLASS_COLORS_HEX.get(name, "#888")
        st.markdown(f"""
        <div class="class-card">
            <div><span class="class-card-dot" style="background:{color}"></span>
            <span class="class-card-name">{name}</span></div>
            <div class="class-card-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# LIVE SCAN — shared UI for both modes
# ════════════════════════════════════════════════════════════════════════════
mode = st.session_state.mode

if mode == "upload":
    vp = st.session_state.uploaded_video_path
    cap_probe = cv2.VideoCapture(vp)
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap_probe.get(cv2.CAP_PROP_FPS) or 25
    src_w        = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    st.markdown(f"""
    <div class="vinfo">
        <span>📹 <b>{st.session_state.uploaded_video_name}</b></span>
        <span>Frames: <b>{total_frames}</b></span>
        <span>FPS: <b>{src_fps:.0f}</b></span>
        <span>Resolution: <b>{src_w}×{src_h}</b></span>
        <span>Size: <b>{st.session_state.uploaded_video_size/1e6:.1f} MB</b></span>
    </div>
    """, unsafe_allow_html=True)

header_col, stop_col, back_col = st.columns([4, 1, 1])
with header_col:
    badge_text = "LIVE · WEBCAM" if mode == "webcam" else "SCANNING · VIDEO FILE"
    st.markdown(f'<span class="live-badge"><span class="live-dot"></span>{badge_text}</span>', unsafe_allow_html=True)
with stop_col:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    stop_clicked = st.button("■ Stop", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with back_col:
    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
    back_clicked = st.button("← Back", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if back_clicked:
    st.session_state.scanning = False
    if st.session_state.uploaded_video_path:
        try: os.unlink(st.session_state.uploaded_video_path)
        except: pass
    reset_to_landing()
    st.rerun()

col_vid, col_log = st.columns([5, 2], gap="large")
with col_vid:
    st.markdown('<div class="sec-label">Live Feed</div>', unsafe_allow_html=True)
    frame_slot = st.empty()
    prog_slot  = st.empty()

with col_log:
    st.markdown('<div class="sec-label">Live Stats</div>', unsafe_allow_html=True)
    stats_slot = st.empty()
    st.markdown('<div class="sec-label" style="margin-top:1rem">Detection Log</div>', unsafe_allow_html=True)
    log_slot = st.empty()

st.markdown('<div class="sec-label" style="margin-top:1.5rem">Defect Breakdown</div>', unsafe_allow_html=True)
breakdown_slot = st.empty()

result_slot = st.empty()

# ─── Open source ────────────────────────────────────────────────────────────────
if mode == "webcam":
    cap = cv2.VideoCapture(int(camera_index))
    src_label = "camera"
else:
    cap = cv2.VideoCapture(st.session_state.uploaded_video_path)
    src_label = "video file"

if not cap.isOpened():
    st.error(f"Could not open {src_label}.")
    st.session_state.scanning = False
    if st.button("← Back to start"):
        reset_to_landing()
        st.rerun()
    st.stop()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps    = cap.get(cv2.CAP_PROP_FPS) or 20
total_frames_loop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "upload" else None
frame_time = 1.0 / fps if fps > 0 else 0.033

avi_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
avi_path = avi_tmp.name
avi_tmp.close()
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out    = cv2.VideoWriter(avi_path, fourcc, max(fps, 10), (width, height))

defect_counts = defaultdict(int)
frame_defects = defaultdict(int)
log_entries   = []
processed     = 0
frame_idx     = 0

# ─── Live loop ──────────────────────────────────────────────────────────────────
while st.session_state.scanning:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break  # camera disconnect OR video ended

    do_infer = (frame_idx % frame_skip == 0) if mode == "upload" else True

    if do_infer:
        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]

        seen = set()
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = CLASS_NAMES.get(cls_id, str(cls_id))
            color  = CLASS_COLORS_BGR.get(label, (255, 255, 255))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                text = f"{label} {conf:.0%}" if show_conf else label
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
                cv2.putText(frame, text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            defect_counts[label] += 1
            seen.add(label)
            log_entries.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "label": label, "conf": conf,
                "color": CLASS_COLORS_HEX.get(label, "#888"),
            })
        log_entries = log_entries[:25]

        for cls in seen:
            frame_defects[cls] += 1
        processed += 1

        # ── Update live feed ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(rgb, use_container_width=True)

        # ── Update progress (upload mode only) ──
        if mode == "upload" and total_frames_loop:
            prog_slot.progress(min(frame_idx / total_frames_loop, 1.0), text=f"Frame {frame_idx} / {total_frames_loop}")

        # ── Update live stats ──
        total = sum(defect_counts.values())
        stats_slot.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-card-label">Detections</div>
                <div class="stat-card-value">{total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-label">Types</div>
                <div class="stat-card-value">{len(defect_counts)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Update log ──
        if log_entries:
            log_html = '<div class="log-box">'
            for e in log_entries:
                log_html += f"""
                <div class="log-entry">
                    <span class="log-time">{e['time']}</span>
                    <span class="log-label" style="color:{e['color']}">{e['label']}</span>
                    <span class="log-conf">{e['conf']:.0%}</span>
                </div>"""
            log_html += '</div>'
            log_slot.markdown(log_html, unsafe_allow_html=True)
        else:
            log_slot.markdown('<div class="log-box" style="color:#2e3558;font-size:0.75rem;padding:1rem;">No detections yet…</div>', unsafe_allow_html=True)

        # ── Update breakdown ──
        if defect_counts:
            sorted_d = sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)
            max_c = sorted_d[0][1]
            bhtml = ""
            for label, count in sorted_d:
                color = CLASS_COLORS_HEX.get(label, "#3d5afe")
                pct   = count / max_c
                fpct  = (frame_defects[label] / processed * 100) if processed else 0
                desc  = DEFECT_DESCRIPTIONS.get(label, "")
                bhtml += f"""
                <div class="defect-row">
                    <div class="defect-top">
                        <span class="defect-name"><span style="display:inline-block;width:8px;height:8px;background:{color};border-radius:50%;margin-right:7px;vertical-align:middle;"></span>{label}</span>
                        <span class="defect-count" style="color:{color}">{count}×</span>
                    </div>
                    <div class="defect-bar-bg"><div class="defect-bar-fill" style="width:{pct*100:.1f}%;background:{color};opacity:0.85"></div></div>
                    <div class="defect-meta">{desc} · {fpct:.1f}% of frames</div>
                </div>"""
            breakdown_slot.markdown(bhtml, unsafe_allow_html=True)

    out.write(frame)
    frame_idx += 1

    if mode == "upload" and realtime_speed:
        elapsed = time.time() - loop_start
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    if stop_clicked:
        st.session_state.scanning = False
        break

cap.release()
out.release()
if mode == "upload" and total_frames_loop:
    prog_slot.empty()

if mode == "upload" and st.session_state.uploaded_video_path:
    try: os.unlink(st.session_state.uploaded_video_path)
    except: pass

# ─── Re-encode and offer download ──────────────────────────────────────────────
result_slot.info("Finalizing recording…")

mp4_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
mp4_path = mp4_tmp.name
mp4_tmp.close()

try:
    subprocess.run(
        ["ffmpeg", "-y", "-i", avi_path, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p", mp4_path],
        capture_output=True, check=True
    )
    with open(mp4_path, "rb") as f:
        video_bytes = f.read()
    dl_ext, dl_mime = ".mp4", "video/mp4"
except Exception:
    with open(avi_path, "rb") as f:
        video_bytes = f.read()
    dl_ext, dl_mime = ".avi", "video/x-msvideo"

for p in [avi_path, mp4_path]:
    try: os.unlink(p)
    except: pass

result_slot.empty()
st.markdown("---")
st.markdown('<div class="sec-label">Session Recording</div>', unsafe_allow_html=True)
st.video(video_bytes)
st.download_button(
    "⬇  Download annotated recording",
    data=video_bytes,
    file_name=f"printguard_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{dl_ext}",
    mime=dl_mime,
)
st.success(f"✅ Scan ended — {sum(defect_counts.values())} detections across {len(defect_counts)} defect types.")

if st.button("↻  Start New Scan"):
    reset_to_landing()
    st.rerun()
