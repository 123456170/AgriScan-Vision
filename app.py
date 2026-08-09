import math
import os
import random
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="AgriScan Vision Live Demo",
    page_icon="🌾",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Small Streamlit helper
# -----------------------------------------------------------------------------

def rerun():
    """Compatibility helper for Streamlit rerun."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def show_image(img, caption=None, width=None, container=False):
    """Compatibility helper for st.image options."""
    try:
        if container:
            st.image(img, caption=caption, use_container_width=True)
        else:
            st.image(img, caption=caption, width=width)
    except TypeError:
        if container:
            st.image(img, caption=caption, use_column_width=True)
        else:
            st.image(img, caption=caption, width=width)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SYNTH_SOURCE = "Synthetic orchard demo (live generated)"
WEBCAM_SOURCE = "Webcam (local)"
UPLOAD_SOURCE = "Uploaded video"

RIPENESS_CLASSES = ["unripe", "ripe", "overripe"]

# BGR colors for drawing
RIPENESS_COLORS = {
    "unripe": (0, 220, 0),
    "ripe": (0, 210, 255),
    "overripe": (80, 80, 220),
}

SYNTH_COLORS = {
    "unripe": (0, 190, 0),
    "ripe": (0, 210, 255),
    "overripe": (40, 70, 120),
}

# COCO fruit-ish class ids: banana, apple, orange
FRUIT_CLASS_IDS = {46, 47, 49}


# -----------------------------------------------------------------------------
# Simple ByteTrack-inspired tracker
# -----------------------------------------------------------------------------

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


class Track:
    def __init__(self, track_id, det):
        self.id = track_id
        self.bbox = det["bbox"]
        self.conf = float(det.get("conf", 0.0))
        self.ripeness = det.get("ripeness", "ripe")
        self.frames_since_update = 0
        self.hits = 1
        self.age = 1
        self.counted = False


class SimpleByteTracker:
    """
    A lightweight ByteTrack-inspired tracker.

    - High-confidence detections create new tracks.
    - Lower-confidence detections can update existing tracks.
    - Unique confirmed tracks are counted once.
    """

    def __init__(
        self,
        iou_threshold=0.25,
        max_age=25,
        min_hits=2,
        high_thresh=0.45,
        low_thresh=0.10,
    ):
        self.tracks = []
        self.next_id = 1
        self.confirmed_count = 0

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.confirmed_count = 0

    def _associate(self, detections, tracks, thresh):
        if not detections or not tracks:
            return (
                [],
                list(range(len(detections))),
                list(range(len(tracks))),
            )

        pairs = []
        for di, det in enumerate(detections):
            for ti, track in enumerate(tracks):
                val = iou(det["bbox"], track.bbox)
                if val >= thresh:
                    pairs.append((val, di, ti))

        pairs.sort(key=lambda x: x[0], reverse=True)

        matched = []
        unmatched_d = set(range(len(detections)))
        unmatched_t = set(range(len(tracks)))

        for _, di, ti in pairs:
            if di in unmatched_d and ti in unmatched_t:
                matched.append((di, ti))
                unmatched_d.remove(di)
                unmatched_t.remove(ti)

        return matched, list(unmatched_d), list(unmatched_t)

    def _update_track(self, track, det):
        track.bbox = det["bbox"]
        track.conf = float(det.get("conf", track.conf))
        track.ripeness = det.get("ripeness", track.ripeness)
        track.frames_since_update = 0
        track.hits += 1
        track.age += 1

    def update(self, detections):
        high = [
            d
            for d in detections
            if float(d.get("conf", 0.0)) >= self.high_thresh
        ]
        low = [
            d
            for d in detections
            if self.low_thresh <= float(d.get("conf", 0.0)) < self.high_thresh
        ]

        # First association: high-confidence detections.
        matched_high, unmatched_high_d, unmatched_high_t = self._associate(
            high, self.tracks, self.iou_threshold
        )

        for di, ti in matched_high:
            self._update_track(self.tracks[ti], high[di])

        remaining_tracks = [self.tracks[i] for i in unmatched_high_t]

        # Second association: lower-confidence detections.
        matched_low, _, unmatched_low_t = self._associate(
            low, remaining_tracks, self.iou_threshold
        )

        for di, ti in matched_low:
            self._update_track(remaining_tracks[ti], low[di])

        # Create tracks only from unmatched high-confidence detections.
        for di in unmatched_high_d:
            det = high[di]
            self.tracks.append(Track(self.next_id, det))
            self.next_id += 1

        # Age unmatched tracks.
        for ti in unmatched_low_t:
            remaining_tracks[ti].frames_since_update += 1
            remaining_tracks[ti].age += 1

        # Remove dead tracks.
        self.tracks = [
            t for t in self.tracks if t.frames_since_update <= self.max_age
        ]

        # Count confirmed unique tracks once.
        for t in self.tracks:
            if t.hits >= self.min_hits and not t.counted:
                t.counted = True
                self.confirmed_count += 1

        active = [t for t in self.tracks if t.frames_since_update == 0]
        return active

    def ripeness_counts(self):
        counts = {k: 0 for k in RIPENESS_CLASSES}
        for t in self.tracks:
            if t.counted and t.ripeness in counts:
                counts[t.ripeness] += 1
        return counts


# -----------------------------------------------------------------------------
# Ripeness heuristic
# -----------------------------------------------------------------------------

def classify_ripeness(crop):
    """
    HSV-based ripeness heuristic.

    This is intentionally lightweight and demo-friendly. In production, this
    would be replaced by a fine-tuned ripeness classifier or a multimodal
    ripeness model calibrated per crop.
    """
    if crop is None or crop.size == 0:
        return "ripe", 0.30

    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return "ripe", 0.30

    small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array([36, 50, 50]), np.array([85, 255, 255]))

    red1 = cv2.inRange(hsv, np.array([0, 70, 60]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([168, 70, 60]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)

    yellow = cv2.inRange(hsv, np.array([13, 70, 80]), np.array([35, 255, 255]))

    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 100]))

    total = 32 * 32
    green_ratio = float(np.count_nonzero(green)) / total
    red_ratio = float(np.count_nonzero(red)) / total
    yellow_ratio = float(np.count_nonzero(yellow)) / total
    dark_ratio = float(np.count_nonzero(dark)) / total

    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    # Very dark / low-vitality regions are treated as overripe or decayed.
    if dark_ratio > 0.45 or v_mean < 80:
        return "overripe", 0.60

    # Strong green signal -> unripe.
    if green_ratio > 0.35 and green_ratio >= red_ratio and green_ratio >= yellow_ratio:
        return "unripe", float(np.clip(0.40 + green_ratio, 0.0, 0.95))

    # Dull red/brownish signal -> overripe.
    if v_mean < 130 and h_mean < 25 and (red_ratio > 0.15 or s_mean > 80):
        return "overripe", 0.60

    # Strong red/yellow signal -> ripe.
    if red_ratio > 0.22 or yellow_ratio > 0.22:
        return "ripe", float(np.clip(0.40 + max(red_ratio, yellow_ratio), 0.0, 0.95))

    if h_mean < 13 or h_mean > 167:
        return "ripe", 0.55

    if 13 <= h_mean <= 35:
        return "ripe", 0.55

    if 36 <= h_mean <= 85:
        return "unripe", 0.55

    return "ripe", 0.35


# -----------------------------------------------------------------------------
# Detection helpers
# -----------------------------------------------------------------------------

def nms(dets, thresh=0.40):
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [
            d
            for d in dets
            if iou(best["bbox"], d["bbox"]) < thresh
        ]

    return keep


def detect_color(
    frame,
    min_area=500,
    min_circularity=0.30,
    min_conf=0.35,
):
    """
    Heuristic color/blob detector used as a fallback and for the synthetic demo.

    This is not a production detector. It exists so the demo can always show
    live detection/tracking even without downloaded YOLO weights.
    """
    h, w = frame.shape[:2]

    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    ranges = [
        # red
        ((0, 70, 60), (12, 255, 255)),
        ((168, 70, 60), (179, 255, 255)),
        # yellow/orange
        ((13, 70, 70), (35, 255, 255)),
        # green
        ((36, 50, 50), (85, 255, 255)),
        # brownish/dull red
        ((8, 50, 60), (25, 230, 170)),
    ]

    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    if mask is None:
        return []

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    dets = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > 0.65 * h * w:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 12 or bh < 12:
            continue

        aspect = float(bw) / float(max(1, bh))
        if aspect < 0.20 or aspect > 5.0:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue

        circularity = 4.0 * math.pi * area / (perim * perim)
        if circularity < min_circularity:
            continue

        crop = frame[y : y + bh, x : x + bw]
        ripeness, ripeness_conf = classify_ripeness(crop)

        conf = float(
            np.clip(
                0.25 + 0.45 * circularity + 0.30 * ripeness_conf,
                0.0,
                0.95,
            )
        )

        if conf < min_conf:
            continue

        dets.append(
            {
                "bbox": [x, y, x + bw, y + bh],
                "conf": conf,
                "class": "fruit",
                "ripeness": ripeness,
                "source": "color",
            }
        )

    return nms(dets, 0.40)


@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """
    Try to load a small pre-trained YOLO model.

    The demo first tries YOLOv10n, then falls back to YOLOv8n if needed.
    If weights cannot be downloaded/local are unavailable, the app continues
    with heuristic detection.
    """
    try:
        from ultralytics import YOLO
    except Exception:
        return None, "ultralytics not installed"

    candidates = ["yolov10n.pt", "yolov8n.pt"]

    for name in candidates:
        try:
            model = YOLO(name)
            return model, name
        except Exception:
            continue

    return None, "no YOLO weights available locally/download failed"


def detect_yolo(frame, model, conf_thresh):
    """
    Detect fruit-like objects with a pre-trained YOLO model.

    Ripeness is still estimated by HSV heuristic on the crop because generic
    COCO weights do not include unripe/ripe/overripe classes.
    """
    dets = []

    try:
        results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=0.45,
            imgsz=480,
            verbose=False,
            device="cpu",
        )
    except TypeError:
        try:
            results = model.predict(
                source=frame,
                conf=conf_thresh,
                iou=0.45,
                imgsz=480,
                verbose=False,
            )
        except Exception:
            return []
    except Exception:
        return []

    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []

    try:
        if len(boxes) == 0:
            return []
    except TypeError:
        pass

    h, w = frame.shape[:2]

    for box in boxes:
        try:
            cls_id = int(float(box.cls[0]))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        except Exception:
            continue

        cls_name = ""
        try:
            names = model.names
            cls_name = str(names[cls_id])
        except Exception:
            pass

        is_fruit = cls_id in FRUIT_CLASS_IDS or any(
            k in cls_name.lower()
            for k in ["apple", "banana", "orange", "fruit"]
        )

        if not is_fruit:
            continue

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        ripeness, _ = classify_ripeness(crop)

        dets.append(
            {
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "class": cls_name or "fruit",
                "ripeness": ripeness,
                "source": "yolo",
            }
        )

    return dets


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def draw_annotations(frame, tracks, show_mask=True):
    """
    Draw boxes, labels, and simulated SAM2-style mask ellipses.
    """
    out = frame.copy()

    if show_mask and tracks:
        overlay = out.copy()

        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ax = max(4, (x2 - x1) // 2)
            ay = max(4, (y2 - y1) // 2)

            color = RIPENESS_COLORS.get(t.ripeness, (0, 255, 255))
            cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, color, -1)

        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        if x2 <= x1 or y2 <= y1:
            continue

        color = RIPENESS_COLORS.get(t.ripeness, (0, 255, 255))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"#{t.id} {t.ripeness} {t.conf:.2f}"

        cv2.putText(
            out,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def pseudo_ndvi_panel(frame):
    """
    Simulated RGB vegetation index panel.

    This is NOT true NDVI because no NIR band is available. It uses a
    green-red style index for demo visualization only.
    """
    small = cv2.resize(frame, (220, 124))
    b = small[:, :, 0].astype(np.float32)
    g = small[:, :, 1].astype(np.float32)
    r = small[:, :, 2].astype(np.float32)

    index = (g - r) / (g + r + 1e-6)
    index = np.clip((index + 1.0) * 127.5, 0, 255).astype(np.uint8)

    heat = cv2.applyColorMap(index, cv2.COLORMAP_JET)

    cv2.putText(
        heat,
        "Simulated RGB vegetation index",
        (5, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        heat,
        "No NIR band - not true NDVI",
        (5, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return heat


# -----------------------------------------------------------------------------
# Synthetic orchard scene generator
# -----------------------------------------------------------------------------

class SyntheticScene:
    """
    Generates a live synthetic orchard video with moving fruit blobs.

    This allows the demo to show live video immediately without requiring
    a bundled video file, webcam permission, or internet access.
    """

    def __init__(self, width=640, height=360, count=6):
        self.w = width
        self.h = height
        self.frame_idx = 0
        self.fruits = []

        self.bg = np.full((height, width, 3), (58, 56, 52), dtype=np.uint8)

        # Subtle row lines.
        for y in range(0, height, 36):
            cv2.line(self.bg, (0, y), (width, y), (46, 46, 46), 4)

        # Small leaf-like dots. Kept small so the blob detector mostly ignores them.
        for _ in range(45):
            radius = random.randint(3, 8)
            x = random.randint(radius, width - radius)
            y = random.randint(radius, height - radius)
            color = (
                random.randint(18, 35),
                random.randint(70, 110),
                random.randint(18, 35),
            )
            cv2.circle(self.bg, (x, y), radius, color, -1)

        stages = RIPENESS_CLASSES

        for i in range(count):
            radius = random.randint(20, 34)
            self.fruits.append(
                {
                    "x": float(random.randint(radius, width - radius)),
                    "y": float(random.randint(radius, height - radius)),
                    "vx": random.uniform(-1.6, 1.6) or 1.0,
                    "vy": random.uniform(-1.2, 1.2) or 0.8,
                    "r": radius,
                    "ripeness": stages[i % len(stages)],
                    "next_change": random.randint(400, 1200),
                }
            )

    def read(self):
        self.frame_idx += 1
        frame = self.bg.copy()

        for f in self.fruits:
            # Slowly change ripeness to simulate a live ripening process.
            if self.frame_idx >= f["next_change"]:
                order = RIPENESS_CLASSES
                idx = order.index(f["ripeness"])
                f["ripeness"] = order[(idx + 1) % len(order)]
                f["next_change"] = self.frame_idx + random.randint(700, 1600)

            # Move fruit.
            f["x"] += f["vx"]
            f["y"] += f["vy"]

            if f["x"] - f["r"] <= 0 or f["x"] + f["r"] >= self.w:
                f["vx"] *= -1.0

            if f["y"] - f["r"] <= 0 or f["y"] + f["r"] >= self.h:
                f["vy"] *= -1.0

            f["x"] = max(f["r"], min(self.w - f["r"], f["x"]))
            f["y"] = max(f["r"], min(self.h - f["r"], f["y"]))

            ix = int(f["x"])
            iy = int(f["y"])
            color = SYNTH_COLORS[f["ripeness"]]

            # Shadow/border.
            cv2.circle(frame, (ix, iy), f["r"], (25, 25, 25), -1)

            # Fruit body.
            cv2.circle(frame, (ix, iy), max(4, f["r"] - 2), color, -1)

            # Small highlight.
            cv2.circle(
                frame,
                (ix - f["r"] // 3, iy - f["r"] // 3),
                max(2, f["r"] // 5),
                (170, 170, 170),
                -1,
            )

        return True, frame


# -----------------------------------------------------------------------------
# Uploaded video helper
# -----------------------------------------------------------------------------

def get_uploaded_path(uploaded_file):
    if uploaded_file is None:
        return None

    if "upload_paths" not in st.session_state:
        st.session_state.upload_paths = {}

    try:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    except Exception:
        file_id = uploaded_file.name

    if file_id not in st.session_state.upload_paths:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.close()
        st.session_state.upload_paths[file_id] = tmp.name

    return st.session_state.upload_paths[file_id]


def read_frame(source, webcam_index, uploaded_file):
    """
    Read one frame from the selected source.
    """
    if source == SYNTH_SOURCE:
        if st.session_state.get("scene") is None:
            st.session_state.scene = SyntheticScene()
        return st.session_state.scene.read()

    if source == WEBCAM_SOURCE:
        cap = st.session_state.get("cap")

        if cap is None:
            cap = cv2.VideoCapture(webcam_index)
            if not cap.isOpened():
                return False, None

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            st.session_state.cap = cap

        ret, frame = cap.read()
        if not ret or frame is None:
            return False, None

        return True, frame

    # Uploaded video.
    path = get_uploaded_path(uploaded_file)
    if not path or not os.path.exists(path):
        return False, None

    cap = st.session_state.get("cap")

    if cap is None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False, None
        st.session_state.cap = cap

    ret, frame = cap.read()

    # Loop uploaded video.
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, None

    return True, frame


# -----------------------------------------------------------------------------
# Session state defaults
# -----------------------------------------------------------------------------

if "tracker" not in st.session_state:
    st.session_state.tracker = SimpleByteTracker()

if "running" not in st.session_state:
    st.session_state.running = True

if "scene" not in st.session_state:
    st.session_state.scene = None

if "cap" not in st.session_state:
    st.session_state.cap = None

if "source_key" not in st.session_state:
    st.session_state.source_key = None

if "fps" not in st.session_state:
    st.session_state.fps = 0.0

if "last_time" not in st.session_state:
    st.session_state.last_time = None


# -----------------------------------------------------------------------------
# Sidebar UI
# -----------------------------------------------------------------------------

st.sidebar.title("AgriScan controls")

source = st.sidebar.selectbox(
    "Input source",
    [SYNTH_SOURCE, WEBCAM_SOURCE, UPLOAD_SOURCE],
)

uploaded_file = None
webcam_index = 0

if source == WEBCAM_SOURCE:
    webcam_index = st.sidebar.number_input(
        "Webcam index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )

if source == UPLOAD_SOURCE:
    uploaded_file = st.sidebar.file_uploader(
        "Upload orchard/field video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
    )

try:
    uploaded_id = (
        f"{uploaded_file.name}_{uploaded_file.size}"
        if uploaded_file is not None
        else "none"
    )
except Exception:
    uploaded_id = uploaded_file.name if uploaded_file is not None else "none"

source_key = f"{source}|{webcam_index if source == WEBCAM_SOURCE else uploaded_id}"

# Reset capture/tracker when source changes.
if st.session_state.source_key != source_key:
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass

    st.session_state.cap = None
    st.session_state.scene = None
    st.session_state.tracker.reset()
    st.session_state.source_key = source_key
    st.session_state.running = True

st.sidebar.markdown("---")

c1, c2 = st.sidebar.columns(2)

if c1.button("Start / Resume"):
    st.session_state.running = True

if c2.button("Stop"):
    st.session_state.running = False

if st.sidebar.button("Reset tracker / counts"):
    st.session_state.tracker.reset()
    rerun()

st.sidebar.markdown("---")

conf_thresh = st.sidebar.slider(
    "Confidence threshold",
    0.10,
    0.90,
    0.35,
    0.05,
)

min_area = st.sidebar.slider(
    "Fallback blob min area",
    100,
    5000,
    500,
    50,
)

min_circularity = st.sidebar.slider(
    "Fallback min circularity",
    0.00,
    0.90,
    0.30,
    0.05,
)

target_fps = st.sidebar.slider(
    "Target FPS",
    1,
    30,
    12,
)

use_yolo = st.sidebar.checkbox(
    "Use YOLO detector when possible",
    value=True,
)

fallback_if_empty = st.sidebar.checkbox(
    "Use heuristic fallback if YOLO finds nothing",
    value=True,
)

show_mask = st.sidebar.checkbox(
    "Show simulated SAM2-style masks",
    value=True,
)

show_ndvi = st.sidebar.checkbox(
    "Show simulated RGB vegetation index",
    value=True,
)

if source == SYNTH_SOURCE:
    st.sidebar.info(
        "Synthetic mode uses a heuristic detector so the demo always runs live. "
        "Switch to Webcam or Uploaded video to try YOLO."
    )
elif use_yolo:
    st.sidebar.info(
        "First YOLO use may download small pre-trained weights once."
    )


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

model = None
model_info = "YOLO skipped in synthetic demo mode"

if source != SYNTH_SOURCE and use_yolo:
    model, model_info = load_yolo_model()


# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------

st.title("AgriScan Vision — Live Demo")
st.caption(
    "Demo focus: live fruit detection, ripeness labeling, tracking, unique counting, "
    "and simulated vegetation-index visualization. Production would use fine-tuned "
    "YOLOv10, SAM2, ByteTrack, GPS-aware mapping, and true RGB+NIR indices."
)

left_col, right_col = st.columns([3, 2])

ret, frame = read_frame(source, int(webcam_index), uploaded_file)

if not ret:
    left_col.warning(
        "No frame available. If using webcam, check camera permissions/index. "
        "If using uploaded video, upload a valid video file. "
        "You can always run the synthetic live demo."
    )

    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

    st.session_state.running = False

else:
    detections = []
    detector_used = "No detector"

    # Synthetic mode always uses heuristic detector for guaranteed live behavior.
    if source == SYNTH_SOURCE:
        detections = detect_color(
            frame,
            min_area=min_area,
            min_circularity=min_circularity,
            min_conf=conf_thresh,
        )
        detector_used = "Heuristic color detector (synthetic live demo)"

    else:
        if model is not None:
            detections = detect_yolo(frame, model, conf_thresh)
            if detections:
                detector_used = f"YOLO detector ({model_info})"

        if not detections and (fallback_if_empty or model is None):
            detections = detect_color(
                frame,
                min_area=min_area,
                min_circularity=min_circularity,
                min_conf=conf_thresh,
            )

            if model is not None:
                detector_used = "Heuristic fallback (YOLO found no fruit)"
            else:
                detector_used = f"Heuristic color detector ({model_info})"

        if not detections and model is None:
            detector_used = f"No detections ({model_info})"

    tracker = st.session_state.tracker
    tracker.high_thresh = max(0.35, conf_thresh)
    tracker.low_thresh = max(0.05, conf_thresh - 0.25)

    active_tracks = tracker.update(detections)

    annotated = draw_annotations(frame, active_tracks, show_mask=show_mask)

    status = (
        f"LIVE | {detector_used} | unique tracked: {tracker.confirmed_count}"
    )

    cv2.putText(
        annotated,
        status,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        status,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 230, 0),
        2,
        cv2.LINE_AA,
    )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    with left_col:
        show_image(
            annotated_rgb,
            caption="Live annotated video",
            container=True,
        )

        if show_ndvi:
            show_image(
                pseudo_ndvi_panel(frame),
                caption="Simulated vegetation-index panel",
                width=260,
            )

    counts = tracker.ripeness_counts()
    avg_conf = float(np.mean([d["conf"] for d in detections])) if detections else 0.0

    now = time.time()
    last = st.session_state.last_time
    fps = st.session_state.fps

    if last is not None:
        inst = 1.0 / max(1e-6, now - last)
        fps = inst if fps <= 0 else (0.85 * fps + 0.15 * inst)

    st.session_state.fps = fps
    st.session_state.last_time = now

    with right_col:
        st.subheader("Live stats")

        m1, m2 = st.columns(2)
        m1.metric("Unique fruit tracked", tracker.confirmed_count)
        m2.metric("Active detections", len(active_tracks))

        m3, m4 = st.columns(2)
        m3.metric("Avg confidence", f"{avg_conf:.2f}")
        m4.metric("FPS", f"{fps:.1f}")

        st.markdown(f"**Detector:** `{detector_used}`")

        st.markdown("### Ripeness counts (unique tracked)")
        total_count = max(1, sum(counts.values()))

        for cls in RIPENESS_CLASSES:
            val = counts[cls]
            st.markdown(f"**{cls}**: {val}")
            st.progress(float(val / total_count))

        st.markdown("### Active tracks")
        lines = []

        for t in active_tracks[:12]:
            lines.append(
                f"#{t.id:<3} | {t.ripeness:<8} | conf {t.conf:.2f} | age {t.age}"
            )

        st.code("\n".join(lines) if lines else "No active tracks")

        if st.session_state.running:
            st.success("Live video running. Use Stop to pause.")
        else:
            st.info("Stopped. Use Start / Resume to run again.")

        if source == SYNTH_SOURCE:
            fidelity = (
                "Fidelity note: synthetic live video. Detection uses RGB color "
                "heuristics, not YOLO. Ripeness is HSV heuristic. Masks are simulated "
                "SAM2-style ellipses. Vegetation index is simulated from RGB only."
            )
        elif model is not None:
            fidelity = (
                f"Fidelity note: detection uses pre-trained {model_info}. Ripeness is "
                "HSV heuristic because generic weights do not include ripeness classes. "
                "Masks are simulated SAM2-style ellipses. Vegetation index is simulated "
                "from RGB only."
            )
        else:
            fidelity = (
                "Fidelity note: YOLO unavailable, so detection uses heuristic fallback. "
                "Ripeness is HSV heuristic. Masks are simulated SAM2-style ellipses. "
                "Vegetation index is simulated from RGB only."
            )

        st.caption(fidelity)

    # -------------------------------------------------------------------------
    # Live rerun loop
    # -------------------------------------------------------------------------
    if st.session_state.running:
        elapsed = time.time() - now
        sleep_time = max(0.0, (1.0 / float(target_fps)) - elapsed)

        if sleep_time > 0:
            time.sleep(sleep_time)

        rerun()