import math
import os
import random
import tempfile
import time

import streamlit as st

st.set_page_config(
    page_title="AgriScan Vision Live Demo",
    page_icon="🌾",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Core dependencies
# -----------------------------------------------------------------------------

try:
    import numpy as np
except Exception as e:
    st.error(f"NumPy import failed: {e}")
    st.stop()

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    st.error(f"Pillow import failed: {e}")
    st.stop()

# Optional OpenCV support.
# The cloud-safe demo does NOT require OpenCV.
CV2_AVAILABLE = False
cv2 = None

try:
    import cv2 as _cv2

    cv2 = _cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None


try:
    FONT = ImageFont.load_default()
except Exception:
    FONT = None


# -----------------------------------------------------------------------------
# Streamlit helpers
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
WEBCAM_SOURCE = "Webcam (local, requires OpenCV)"
UPLOAD_SOURCE = "Uploaded video (requires OpenCV)"

RIPENESS_CLASSES = ["unripe", "ripe", "overripe"]

# RGB colors for drawing
RIPENESS_COLORS_RGB = {
    "unripe": (0, 220, 0),
    "ripe": (255, 210, 0),
    "overripe": (220, 80, 80),
}

SYNTH_COLORS_RGB = {
    "unripe": (0, 190, 0),
    "ripe": (255, 210, 0),
    "overripe": (120, 70, 40),
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
# Optional OpenCV-based ripeness heuristic
# -----------------------------------------------------------------------------

def classify_ripeness_cv2(crop):
    """
    HSV-based ripeness heuristic used only when OpenCV is available.
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

    if dark_ratio > 0.45 or v_mean < 80:
        return "overripe", 0.60

    if green_ratio > 0.35 and green_ratio >= red_ratio and green_ratio >= yellow_ratio:
        return "unripe", float(np.clip(0.40 + green_ratio, 0.0, 0.95))

    if v_mean < 130 and h_mean < 25 and (red_ratio > 0.15 or s_mean > 80):
        return "overripe", 0.60

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
# Optional OpenCV detection helpers
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


def detect_color_cv2(
    frame,
    min_area=500,
    min_circularity=0.30,
    min_conf=0.35,
):
    """
    Optional OpenCV color/blob detector.
    Used only if OpenCV is installed locally.
    """
    if not CV2_AVAILABLE:
        return []

    h, w = frame.shape[:2]

    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    ranges = [
        ((0, 70, 60), (12, 255, 255)),
        ((168, 70, 60), (179, 255, 255)),
        ((13, 70, 70), (35, 255, 255)),
        ((36, 50, 50), (85, 255, 255)),
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
        ripeness, ripeness_conf = classify_ripeness_cv2(crop)

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
def load_yolo_model(allow_download=False):
    """
    Optional YOLO loader.

    This is deliberately optional because YOLO/ultralytics/torch are heavy
    dependencies and can cause deployment/installation failures.
    """
    try:
        from ultralytics import YOLO
    except Exception:
        return None, "ultralytics not installed"

    candidates = ["yolov10n.pt", "yolov8n.pt"]

    for name in candidates:
        try:
            if os.path.exists(name) or allow_download:
                model = YOLO(name)
                return model, name
        except Exception:
            continue

    return None, "no local YOLO weights found"


def detect_yolo(frame, model, conf_thresh):
    """
    Optional YOLO detector.
    Used only if ultralytics is installed and enabled.
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
        ripeness, _ = classify_ripeness_cv2(crop)

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
# PIL visualization helpers
# -----------------------------------------------------------------------------

def draw_annotations(img, tracks, show_mask=True):
    """
    Draw boxes, labels, and simulated SAM2-style mask ellipses using Pillow.
    """
    img = img.convert("RGB")

    if show_mask and tracks:
        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)

        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            if x2 <= x1 or y2 <= y1:
                continue

            color = RIPENESS_COLORS_RGB.get(t.ripeness, (255, 255, 0))
            od.ellipse(
                [x1, y1, x2, y2],
                fill=color + (70,),
                outline=color + (120,),
                width=2,
            )

        img = Image.alpha_composite(base, overlay).convert("RGB")

    draw = ImageDraw.Draw(img)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        if x2 <= x1 or y2 <= y1:
            continue

        color = RIPENESS_COLORS_RGB.get(t.ripeness, (255, 255, 0))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"#{t.id} {t.ripeness} {t.conf:.2f}"
        ty = max(2, y1 + 2)

        draw.text((x1 + 1, ty + 1), label, fill=(0, 0, 0), font=FONT)
        draw.text((x1, ty), label, fill=color, font=FONT)

    return img


def add_status_text(img, text):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    draw.text((6, 6), text, fill=(0, 0, 0), font=FONT)
    draw.text((5, 5), text, fill=(0, 230, 0), font=FONT)

    return img


def colorize_index(idx):
    """
    Simple blue-green-red colorization for a normalized index.
    """
    x = idx.astype(np.float32) / 255.0

    r = np.interp(x, [0.0, 0.5, 1.0], [0.0, 0.0, 255.0]).astype(np.uint8)
    g = np.interp(x, [0.0, 0.5, 1.0], [0.0, 255.0, 0.0]).astype(np.uint8)
    b = np.interp(x, [0.0, 0.5, 1.0], [255.0, 0.0, 0.0]).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def pseudo_ndvi_panel(img):
    """
    Simulated RGB vegetation index panel.

    This is NOT true NDVI because no NIR band is available.
    """
    small = img.resize((220, 124)).convert("RGB")
    arr = np.asarray(small, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]

    index = (g - r) / (g + r + 1e-6)
    index = np.clip((index + 1.0) * 127.5, 0, 255).astype(np.uint8)

    heat = colorize_index(index)
    panel = Image.fromarray(heat)

    draw = ImageDraw.Draw(panel)

    draw.text((4, 4), "Simulated RGB vegetation index", fill=(0, 0, 0), font=FONT)
    draw.text((3, 3), "Simulated RGB vegetation index", fill=(255, 255, 255), font=FONT)

    draw.text((4, 18), "No NIR band - not true NDVI", fill=(0, 0, 0), font=FONT)
    draw.text((3, 17), "No NIR band - not true NDVI", fill=(255, 255, 255), font=FONT)

    return panel


# -----------------------------------------------------------------------------
# Synthetic orchard scene generator
# -----------------------------------------------------------------------------

class SyntheticScene:
    """
    Generates a live synthetic orchard video with moving fruit blobs.

    This allows the demo to show live video immediately without requiring
    OpenCV, heavy ML packages, bundled video files, webcam permission, or
    internet access.
    """

    def __init__(self, width=640, height=360, count=6):
        self.w = width
        self.h = height
        self.frame_idx = 0
        self.fruits = []

        self.bg = Image.new("RGB", (width, height), (58, 56, 52))
        draw = ImageDraw.Draw(self.bg)

        # Subtle row lines.
        for y in range(0, height, 36):
            draw.line((0, y, width, y), fill=(46, 46, 46), width=4)

        # Small leaf-like dots.
        for _ in range(45):
            radius = random.randint(3, 8)
            x = random.randint(radius, width - radius)
            y = random.randint(radius, height - radius)
            color = (
                random.randint(18, 35),
                random.randint(70, 110),
                random.randint(18, 35),
            )
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
            )

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

        img = self.bg.copy()
        draw = ImageDraw.Draw(img)

        detections = []

        for f in self.fruits:
            # Slowly change ripeness.
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

            x = int(f["x"])
            y = int(f["y"])
            r = int(f["r"])

            color = SYNTH_COLORS_RGB[f["ripeness"]]

            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=color,
                outline=(25, 25, 25),
                width=3,
            )

            # Small highlight.
            hr = max(2, r // 5)
            hx = x - r // 3
            hy = y - r // 3

            draw.ellipse(
                [hx - hr, hy - hr, hx + hr, hy + hr],
                fill=(190, 190, 190),
            )

            detections.append(
                {
                    "bbox": [x - r, y - r, x + r, y + r],
                    "conf": 0.88,
                    "class": "synthetic-fruit",
                    "ripeness": f["ripeness"],
                    "source": "synthetic",
                }
            )

        return img, detections


# -----------------------------------------------------------------------------
# Optional OpenCV video reader
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


def read_cv2_frame(source, webcam_index, uploaded_file):
    """
    Read one BGR frame using OpenCV.
    Only used when OpenCV is available.
    """
    if not CV2_AVAILABLE:
        return False, None

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

    if source == UPLOAD_SOURCE:
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

    return False, None


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

st.sidebar.caption(
    "Cloud-safe build: the core live demo runs without OpenCV, YOLO, or torch. "
    "Webcam/upload support is enabled automatically only if OpenCV is installed."
)

source_options = [SYNTH_SOURCE]

if CV2_AVAILABLE:
    source_options.extend([WEBCAM_SOURCE, UPLOAD_SOURCE])
else:
    st.sidebar.info(
        "OpenCV is not installed, so webcam/upload are disabled. "
        "The synthetic live demo is running instead."
    )

source = st.sidebar.selectbox(
    "Input source",
    source_options,
)

uploaded_file = None
webcam_index = 0

if source == WEBCAM_SOURCE and CV2_AVAILABLE:
    webcam_index = st.sidebar.number_input(
        "Webcam index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )

if source == UPLOAD_SOURCE and CV2_AVAILABLE:
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
    if st.session_state.cap is not None and CV2_AVAILABLE:
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
    "Blob min area",
    100,
    5000,
    500,
    50,
)

min_circularity = st.sidebar.slider(
    "Min circularity",
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

use_yolo = False
allow_yolo_download = False

if CV2_AVAILABLE:
    use_yolo = st.sidebar.checkbox(
        "Use optional YOLO detector if installed",
        value=False,
    )

    if use_yolo:
        allow_yolo_download = st.sidebar.checkbox(
            "Allow YOLO weights download if missing",
            value=False,
        )

show_mask = st.sidebar.checkbox(
    "Show simulated SAM2-style masks",
    value=True,
)

show_ndvi = st.sidebar.checkbox(
    "Show simulated RGB vegetation index",
    value=True,
)


# -----------------------------------------------------------------------------
# Optional model loading
# -----------------------------------------------------------------------------

model = None
model_info = "YOLO disabled (lightweight mode)"

if source != SYNTH_SOURCE and CV2_AVAILABLE and use_yolo:
    model, model_info = load_yolo_model(allow_yolo_download)


# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------

st.title("AgriScan Vision — Live Demo")
st.caption(
    "Live demo: fruit detection, ripeness labeling, tracking, unique counting, "
    "and simulated vegetation-index visualization. This cloud-safe version avoids "
    "heavy dependencies by default."
)

left_col, right_col = st.columns([3, 2])

ret = True
img = None
detections = []
detector_used = "No detector"

if source == SYNTH_SOURCE:
    if st.session_state.get("scene") is None:
        st.session_state.scene = SyntheticScene()

    img, detections = st.session_state.scene.read()
    detector_used = "Synthetic ground-truth detector (cloud-safe demo)"

else:
    if not CV2_AVAILABLE:
        ret = False
        left_col.warning(
            "Webcam/upload requires OpenCV. Install opencv-python locally to enable it."
        )
        st.session_state.running = False
    else:
        ret, bgr_frame = read_cv2_frame(source, int(webcam_index), uploaded_file)

        if ret:
            img = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

            if model is not None:
                detections = detect_yolo(bgr_frame, model, conf_thresh)
                if detections:
                    detector_used = f"YOLO detector ({model_info})"

            if not detections:
                detections = detect_color_cv2(
                    bgr_frame,
                    min_area=min_area,
                    min_circularity=min_circularity,
                    min_conf=conf_thresh,
                )

                if model is not None:
                    detector_used = "Heuristic fallback (YOLO found no fruit)"
                else:
                    detector_used = f"Heuristic detector ({model_info})"

            if not detections and model is None:
                detector_used = f"No detections ({model_info})"
        else:
            left_col.warning(
                "No frame available. Check webcam permissions/index or upload a valid video."
            )

            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                except Exception:
                    pass
                st.session_state.cap = None

            st.session_state.running = False

if ret and img is not None:
    tracker = st.session_state.tracker
    tracker.high_thresh = max(0.35, conf_thresh)
    tracker.low_thresh = max(0.05, conf_thresh - 0.25)

    active_tracks = tracker.update(detections)

    annotated = draw_annotations(img, active_tracks, show_mask=show_mask)

    status = (
        f"LIVE | {detector_used} | unique tracked: {tracker.confirmed_count}"
    )

    annotated = add_status_text(annotated, status)

    with left_col:
        show_image(
            annotated,
            caption="Live annotated video",
            container=True,
        )

        if show_ndvi:
            show_image(
                pseudo_ndvi_panel(img),
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
                "Fidelity note: synthetic live video. Detections are generated from the "
                "synthetic scene to guarantee a cloud-safe live demo. Ripeness labels are "
                "part of the simulation. Masks are simulated SAM2-style ellipses. "
                "Vegetation index is simulated from RGB only."
            )
        elif model is not None:
            fidelity = (
                f"Fidelity note: detection uses pre-trained {model_info}. Ripeness is "
                "HSV heuristic because generic weights do not include ripeness classes. "
                "Masks are simulated SAM2-style ellipses. Vegetation index is simulated "
                "from RGB only."
            )
        else:
            if use_yolo:
                fidelity = (
                    f"Fidelity note: YOLO unavailable ({model_info}). Using lightweight "
                    "heuristic detection. Ripeness is HSV heuristic. Masks are simulated "
                    "SAM2-style ellipses. Vegetation index is simulated from RGB only."
                )
            else:
                fidelity = (
                    "Fidelity note: YOLO disabled. Using lightweight heuristic detection. "
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