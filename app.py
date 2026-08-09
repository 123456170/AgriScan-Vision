"""
AgriScan Vision - Live Demo
----------------------------
A single-page Streamlit app that runs real-time fruit detection + tracking
on an uploaded video clip or a webcam feed. No external API keys are used;
all models are downloaded once (via ultralytics) and run locally.

WHAT'S REAL vs WHAT'S SIMULATED (see in-app caption too):
  REAL       -> Object detection uses a pretrained YOLO model (COCO weights).
                Multi-object tracking uses ByteTrack (via ultralytics .track()).
  SIMULATED  -> Ripeness (unripe/ripe/overripe) is an HSV color heuristic,
                not a fine-tuned agriculture model.
                The NDVI panel is a pseudo-NDVI derived from RGB only
                (no NIR sensor), for visual effect.
                Fruit size/mask estimation (SAM2) is omitted in this
                lightweight build to keep the demo fast and dependency-light.
"""

import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="AgriScan Vision - Live Demo", layout="wide")

FRUIT_CLASSES = {"apple", "orange", "banana"}

RIPENESS_COLORS = {
    "unripe": (0, 200, 0),      # green, BGR
    "ripe": (0, 165, 255),      # orange, BGR
    "overripe": (19, 69, 139),  # brown, BGR
    "unknown": (200, 200, 200),
}


@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov10n.pt")
    except Exception:
        model = YOLO("yolov8n.pt")
    return model


def classify_ripeness(roi):
    """Very simple HSV heuristic. Not a trained classifier - clearly simulated."""
    if roi is None or roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    if v_mean < 60 or (s_mean < 40 and v_mean < 95):
        return "overripe"
    if 35 <= h_mean <= 85:
        return "unripe"
    return "ripe"


def pseudo_ndvi(frame_bgr):
    """Green-Red Vegetation Index used as a stand-in for true NDVI (no NIR band)."""
    b, g, r = cv2.split(frame_bgr.astype(np.float32))
    grvi = (g - r) / (g + r + 1e-6)
    grvi_norm = cv2.normalize(grvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(grvi_norm, cv2.COLORMAP_JET)


def draw_box(frame, xyxy, track_id, ripeness, conf):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    color = RIPENESS_COLORS.get(ripeness, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID{track_id} {ripeness} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


st.title("AgriScan Vision - Live Demo")
st.caption(
    "REAL: YOLO object detection (pretrained COCO weights) + ByteTrack multi-object tracking. "
    "SIMULATED: ripeness label is an HSV color heuristic (not a fine-tuned agri model); "
    "the NDVI panel is a pseudo-NDVI computed from RGB only (no NIR sensor); "
    "fruit size/mask estimation is omitted in this lightweight build."
)

with st.sidebar:
    st.header("Settings")
    source_type = st.radio("Video source", ["Upload video", "Webcam"])
    conf_thresh = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
    max_frames = st.slider("Max frames to process", 50, 2000, 400, 50)
    frame_width = st.slider("Processing width (px)", 320, 960, 640, 80)
    run = st.checkbox("Run live demo")

video_file = None
if source_type == "Upload video":
    video_file = st.file_uploader("Upload a short orchard/field clip", type=["mp4", "mov", "avi", "m4v"])

col_video, col_stats = st.columns([2, 1])
with col_video:
    st.subheader("Live annotated feed")
    frame_placeholder = st.empty()
    st.subheader("Pseudo-NDVI panel (simulated)")
    ndvi_placeholder = st.empty()
with col_stats:
    st.subheader("Live stats")
    fps_placeholder = st.empty()
    count_placeholder = st.empty()
    breakdown_placeholder = st.empty()

if run:
    model = load_model()
    fruit_ids = [cid for cid, name in model.names.items() if name in FRUIT_CLASSES]

    cap = None
    if source_type == "Webcam":
        cap = cv2.VideoCapture(0)
    elif video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        st.warning("Upload a video clip or switch to Webcam, then check 'Run live demo'.")

    if cap is not None and cap.isOpened():
        unique_ids = set()
        ripeness_counts = {"unripe": 0, "ripe": 0, "overripe": 0}
        seen_ripeness_for_id = {}
        frame_count = 0
        prev_time = time.time()

        while frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            h, w = frame.shape[:2]
            scale = frame_width / w
            frame = cv2.resize(frame, (frame_width, int(h * scale)))

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=fruit_ids if fruit_ids else None,
                conf=conf_thresh,
                verbose=False,
            )

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for tid, xyxy, conf in zip(ids, xyxys, confs):
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    ripeness = classify_ripeness(roi)
                    draw_box(frame, xyxy, tid, ripeness, conf)

                    if tid not in unique_ids:
                        unique_ids.add(tid)
                        ripeness_counts[ripeness] = ripeness_counts.get(ripeness, 0) + 1
                        seen_ripeness_for_id[tid] = ripeness
                    elif seen_ripeness_for_id.get(tid) != ripeness:
                        old = seen_ripeness_for_id[tid]
                        if old in ripeness_counts:
                            ripeness_counts[old] = max(0, ripeness_counts[old] - 1)
                        ripeness_counts[ripeness] = ripeness_counts.get(ripeness, 0) + 1
                        seen_ripeness_for_id[tid] = ripeness

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True
            )
            ndvi_placeholder.image(
                cv2.cvtColor(pseudo_ndvi(frame), cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True
            )

            fps_placeholder.metric("FPS", f"{fps:.1f}")
            count_placeholder.metric("Unique fruit tracked", len(unique_ids))
            breakdown_placeholder.write(
                {
                    "unripe": ripeness_counts.get("unripe", 0),
                    "ripe": ripeness_counts.get("ripe", 0),
                    "overripe": ripeness_counts.get("overripe", 0),
                }
            )

        cap.release()
        st.success(f"Finished processing {frame_count} frames.")
    elif cap is not None:
        st.error("Could not open video source.")
else:
    st.info("Configure a source in the sidebar, then check 'Run live demo' to start.")