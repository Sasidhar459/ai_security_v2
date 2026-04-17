"""
security_system.py

Pipeline:
  YOLO detection → padded crop → MTCNN alignment → FaceNet embedding
  → Dual Verification Gate (with high-confidence override)
  → Smoothed per-track voting → known / intruder decision
  → Telegram alert on confirmed intruder
"""

import os
import sys
import cv2
import torch
import numpy as np
import pickle
import time
import json
import logging
import threading
import csv
import argparse
import atexit
import re
import signal
from datetime import datetime, date
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from PIL import Image
from collections import defaultdict, deque
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

from config import (
    BASE_DIR, MODEL_DIR,
    YOLO_FACE_MODEL, MODEL_PATH, SCALER_PATH, CENTROIDS_PATH,
    INTRUDER_DIR, INTRUDER_DB_PATH,
    CONFIDENCE_THRESHOLD, CONFIDENCE_OVERRIDE, CENTROID_ACCEPT_THRESHOLD,
    CENTROID_RESCUE_THRESHOLD, CENTROID_RESCUE_CONFIDENCE,
    DEPARTURE_TIMEOUT, CAMERA_INDEX,
    SMOOTHING_FRAMES, MIN_KNOWN_VOTES, MIN_UNKNOWN_VOTES,
    INTRUDER_CONFIRM_SECS, FRAME_SKIP,
    KNOWN_DISTANCE_THRESHOLD, UNKNOWN_IMMEDIATE_THRESHOLD,
    INTRUDER_MAX_ALERTS, INTRUDER_ALERT_GAP, INTRUDER_EMBED_DIST,
    CROP_PADDING, USE_FAISS,
    ALERT_ON_UNIDENTIFIED, UNIDENTIFIED_CONFIRM_SECS,
    MIN_TRACK_AGE_FOR_INTRUDER,
)
from telegram_send import send_telegram_async

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("security.log"), logging.StreamHandler()],
)
log = logging.getLogger("security_system")
sys.modules.setdefault("security_system", sys.modules[__name__])

# -------------------------
# CSV logs for analysis
# -------------------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

DETECTION_LOG = os.path.join(LOG_DIR, "detection_log.csv")
PERFORMANCE_LOG = os.path.join(LOG_DIR, "performance_log.csv")
FRAME_LOG = os.path.join(LOG_DIR, "frame_log.csv")
RUNTIME_STATE_FILE = os.path.join(LOG_DIR, "runtime_state.json")
TEST_RUN_NAME = "default"
TEST_RUN_DIR = LOG_DIR
TEST_FILE_LOG_HANDLER = None
_INITIALIZED_CSV_HEADERS: set[str] = set()
_RUNTIME_STATE_LOCK = threading.Lock()

DETECTION_FIELDS = [
    "timestamp", "frame_id", "face_id", "confidence", "true_label",
    "predicted_label", "processing_time", "top_prediction",
    "verify_reason", "centroid_distance", "nearest_centroid",
]
PERFORMANCE_FIELDS = ["timestamp", "fps", "avg_latency_ms"]
FRAME_FIELDS = [
    "timestamp", "frame_id", "processed", "face_count", "predicted_labels",
    "true_label", "true_people", "processing_time",
]
latest_frame = None
latest_alert = None
latest_frame_updated_at = None
latest_alert_updated_at = None


def read_runtime_state() -> dict:
    if not os.path.exists(RUNTIME_STATE_FILE):
        return {}
    try:
        with open(RUNTIME_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        log.exception("Failed to read runtime state file.")
        return {}


def write_runtime_state(active: bool, reason: str,
                        timestamp: datetime | None = None,
                        last_heartbeat: datetime | None = None):
    now = timestamp or datetime.now()
    heartbeat = last_heartbeat or now
    payload = {
        "active": active,
        "pid": os.getpid(),
        "reason": reason,
        "updated_at": now.isoformat(),
        "last_heartbeat": heartbeat.isoformat(),
    }
    with _RUNTIME_STATE_LOCK:
        with open(RUNTIME_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def recover_unclean_shutdown(close_all_inside_attendance):
    state = read_runtime_state()
    if not state or not state.get("active"):
        return

    stamp_text = state.get("last_heartbeat") or state.get("updated_at")
    recovered_time = datetime.now()
    if isinstance(stamp_text, str) and stamp_text:
        try:
            recovered_time = datetime.fromisoformat(stamp_text)
        except ValueError:
            pass

    log.warning(
        "Previous run appears to have ended uncleanly. Closing open attendance rows at %s.",
        recovered_time.isoformat(timespec="seconds"),
    )
    close_all_inside_attendance(recovered_time)
    try:
        write_runtime_state(False, "recovered_unclean_shutdown", recovered_time, recovered_time)
    except Exception:
        log.exception("Failed to write recovered runtime state.")


def ensure_csv_header(path: str, fields: list[str]):
    if path in _INITIALIZED_CSV_HEADERS:
        return

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(fields)
        _INITIALIZED_CSV_HEADERS.add(path)
        return

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if rows and rows[0] == fields:
        _INITIALIZED_CSV_HEADERS.add(path)
        return

    if len(rows) > 1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{path}.bak_{timestamp}"
        os.replace(path, backup_path)
        log.warning("CSV header changed; backed up old log to %s", backup_path)

    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(fields)
    _INITIALIZED_CSV_HEADERS.add(path)


def append_csv_row(path: str, fields: list[str], row: list):
    if path not in _INITIALIZED_CSV_HEADERS:
        ensure_csv_header(path, fields)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def slugify_test_name(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned or cleaned == "?":
        cleaned = "unlabeled"
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unlabeled"


def make_test_run_name(value: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slugify_test_name(value)}_{stamp}"


def configure_test_log_paths(test_run_name: str):
    global DETECTION_LOG, PERFORMANCE_LOG, FRAME_LOG, TEST_RUN_NAME, TEST_RUN_DIR, TEST_FILE_LOG_HANDLER

    TEST_RUN_NAME = slugify_test_name(test_run_name)
    TEST_RUN_DIR = os.path.join(LOG_DIR, "test_runs", TEST_RUN_NAME)
    os.makedirs(TEST_RUN_DIR, exist_ok=True)

    DETECTION_LOG = os.path.join(TEST_RUN_DIR, f"detection_log_{TEST_RUN_NAME}.csv")
    PERFORMANCE_LOG = os.path.join(TEST_RUN_DIR, f"performance_log_{TEST_RUN_NAME}.csv")
    FRAME_LOG = os.path.join(TEST_RUN_DIR, f"frame_log_{TEST_RUN_NAME}.csv")
    _INITIALIZED_CSV_HEADERS.discard(DETECTION_LOG)
    _INITIALIZED_CSV_HEADERS.discard(PERFORMANCE_LOG)
    _INITIALIZED_CSV_HEADERS.discard(FRAME_LOG)

    ensure_csv_header(DETECTION_LOG, DETECTION_FIELDS)
    ensure_csv_header(PERFORMANCE_LOG, PERFORMANCE_FIELDS)
    ensure_csv_header(FRAME_LOG, FRAME_FIELDS)

    if TEST_FILE_LOG_HANDLER is not None:
        log.removeHandler(TEST_FILE_LOG_HANDLER)
        TEST_FILE_LOG_HANDLER.close()
    test_log_path = os.path.join(TEST_RUN_DIR, f"security_{TEST_RUN_NAME}.log")
    TEST_FILE_LOG_HANDLER = logging.FileHandler(test_log_path, encoding="utf-8")
    TEST_FILE_LOG_HANDLER.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(TEST_FILE_LOG_HANDLER)

    log.info("Test logs for this run: %s", TEST_RUN_DIR)

# Set by --test-recording / --true-label / --true-people.
REALTIME_TRUE_LABEL = "?"
REALTIME_TRUE_PEOPLE = "?"

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Using device: %s", device)

# -------------------------
# Models
# -------------------------
log.info("Loading YOLO face model...")
yolo = YOLO(YOLO_FACE_MODEL)
try:
    yolo.to(device)
except Exception:
    pass

log.info("Loading FaceNet...")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

log.info("Loading MTCNN aligner...")
mtcnn_aligner = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=True,
    device=device,
)

_fallback_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# -------------------------
# Classifier + scaler
# -------------------------
log.info("Loading classifier and scaler...")
classifier  = None
scaler      = None
known_names = []

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        classifier = pickle.load(f)
    known_names = list(getattr(classifier, "classes_", []))
    if not known_names:
        log.warning("Classifier has no classes_ attribute.")
else:
    log.error("Classifier not found at %s", MODEL_PATH)
    raise FileNotFoundError(MODEL_PATH)

if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        log.info("Scaler loaded.")
    except Exception:
        log.exception("Failed to load scaler.")
else:
    log.info("No scaler found; using raw embeddings.")

# -------------------------
# Centroids
# -------------------------
centroid_labels: list       = []
centroid_matrix: np.ndarray = np.zeros((0, 512), dtype=np.float32)
faiss_index                 = None

if os.path.exists(CENTROIDS_PATH):
    try:
        with open(CENTROIDS_PATH, "rb") as f:
            centroids = pickle.load(f)
        centroid_labels = list(centroids.keys())
        centroid_vectors = np.stack([centroids[l] for l in centroid_labels], axis=0).astype(np.float32)
        if scaler is not None:
            centroid_vectors = scaler.transform(centroid_vectors).astype(np.float32)
        centroid_matrix = normalize(
            centroid_vectors,
            axis=1,
        )
        log.info("Loaded %d centroids.", len(centroid_labels))

        if USE_FAISS and centroid_labels:
            import faiss as _faiss
            d           = centroid_matrix.shape[1]
            faiss_index = _faiss.IndexFlatIP(d)
            faiss_index.add(centroid_matrix)
            log.info("FAISS index built (%d vectors).", faiss_index.ntotal)
    except Exception:
        log.exception("Failed to load centroids; fast-path disabled.")
        centroid_labels = []
        centroid_matrix = np.zeros((0, 512), dtype=np.float32)
else:
    log.warning(
        "No centroids file found at %s. "
        "Run generate_centroids.py to build it. "
        "Falling back to classifier-only mode (less accurate).",
        CENTROIDS_PATH,
    )

# -------------------------
# Tracker
# -------------------------
from utils.tracker import FaceTracker
tracker = FaceTracker(timeout=DEPARTURE_TIMEOUT)

# -------------------------
# IntruderDB
# -------------------------
class IntruderDB:
    def __init__(self, path: str):
        self.path  = path
        self.lock  = threading.Lock()
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"next_id": 1, "records": {}}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _day_reset(self, rec: dict):
        today = str(date.today())
        if rec.get("alert_date") != today:
            rec["alert_count"] = 0
            rec["alert_date"]  = today

    def find_or_create(self, embedding: np.ndarray) -> str:
        with self.lock:
            best_id, best_dist = None, float("inf")
            for iid, rec in self._data["records"].items():
                d = cosine(embedding, np.array(rec["embedding"]))
                if d < best_dist:
                    best_dist, best_id = d, iid
            if best_id and best_dist < INTRUDER_EMBED_DIST:
                return best_id
            iid = f"INTRUDER_{self._data['next_id']:04d}"
            self._data["next_id"] += 1
            self._data["records"][iid] = {
                "embedding"    : embedding.tolist(),
                "first_seen"   : datetime.now().isoformat(),
                "alert_count"  : 0,
                "alert_date"   : str(date.today()),
                "last_alert_ts": 0.0,
                "last_seen_ts" : 0.0,
            }
            self._save()
            log.info("New intruder registered: %s", iid)
            return iid

    def can_alert(self, iid: str) -> bool:
        with self.lock:
            rec = self._data["records"].get(iid)
            if not rec:
                return False
            self._day_reset(rec)
            return (
                rec["alert_count"] < INTRUDER_MAX_ALERTS
                and time.time() - rec["last_alert_ts"] >= INTRUDER_ALERT_GAP
            )

    def record_alert(self, iid: str):
        with self.lock:
            rec = self._data["records"].get(iid)
            if rec:
                self._day_reset(rec)
                rec["alert_count"]  += 1
                rec["last_alert_ts"] = time.time()
                rec["last_seen_ts"]  = time.time()
                self._save()

    def get_alert_count(self, iid: str) -> int:
        with self.lock:
            rec = self._data["records"].get(iid)
            if not rec:
                return 0
            self._day_reset(rec)
            return rec["alert_count"]


intruder_db = IntruderDB(INTRUDER_DB_PATH)

# -------------------------
# Per-track state
# -------------------------
class TrackState:
    def __init__(self):
        self.votes          = deque(maxlen=SMOOTHING_FRAMES)
        self.total_frames   = 0
        self.first_unk_time = None
        self.intruder_id    = None
        self.last_embedding = None
        self.unknown_streak = 0

    def push(self, label: str, embedding: np.ndarray, now: float):
        self.total_frames += 1
        self.votes.append(label)
        self.last_embedding = embedding
        if label == "Unknown":
            self.unknown_streak += 1
            if self.first_unk_time is None:
                self.first_unk_time = now
        else:
            self.unknown_streak = 0
            self.first_unk_time = None

    @property
    def stable_label(self) -> str:
        return max(set(self.votes), key=self.votes.count) if self.votes else "Unknown"

    @property
    def known_votes(self) -> int:
        return sum(1 for v in self.votes if v != "Unknown")

    @property
    def unknown_votes(self) -> int:
        return sum(1 for v in self.votes if v == "Unknown")

    def is_confirmed_known(self) -> bool:
        return len(self.votes) >= SMOOTHING_FRAMES and self.known_votes >= MIN_KNOWN_VOTES

    def is_confirmed_intruder(self, now: float, fast_path: bool = False) -> bool:
        required_votes = MIN_UNKNOWN_VOTES
        confirm_secs = INTRUDER_CONFIRM_SECS

        if fast_path and ALERT_ON_UNIDENTIFIED:
            required_votes = min(MIN_UNKNOWN_VOTES, 3)
            confirm_secs = min(INTRUDER_CONFIRM_SECS, UNIDENTIFIED_CONFIRM_SECS)

        min_track_age = max(1, min(MIN_TRACK_AGE_FOR_INTRUDER, SMOOTHING_FRAMES))
        return (
            len(self.votes) >= required_votes
            and self.total_frames >= min_track_age
            and self.unknown_votes >= required_votes
            and self.unknown_streak >= max(2, required_votes - 1)
            and self.first_unk_time is not None
            and (now - self.first_unk_time) >= confirm_secs
        )


track_states: dict = defaultdict(TrackState)

# -------------------------
# Embedding helpers
# -------------------------
def get_padded_crop(frame: np.ndarray,
                    x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w   = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * CROP_PADDING), int(bh * CROP_PADDING)
    return frame[max(0, y1-py):min(h, y2+py),
                 max(0, x1-px):min(w, x2+px)]


def get_embedding(face_bgr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    try:
        aligned = mtcnn_aligner(img)
    except Exception:
        aligned = None
    if aligned is None:
        aligned = _fallback_transform(img)
    img_t = aligned.unsqueeze(0).to(device)
    with torch.no_grad():
        if device.startswith("cuda"):
            from torch.cuda.amp import autocast
            with autocast():
                emb = facenet(img_t)
        else:
            emb = facenet(img_t)
    return emb.cpu().numpy()[0]


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    if scaler is None:
        return emb
    try:
        return scaler.transform([emb])[0]
    except Exception:
        log.exception("Scaler transform failed.")
        return emb


def get_centroid_distance(emb_scaled: np.ndarray) -> tuple:
    """Return (cosine_distance, nearest_label) or (None, None) if no centroids."""
    if centroid_matrix.shape[0] == 0:
        return None, None
    emb_norm = emb_scaled / (np.linalg.norm(emb_scaled) + 1e-10)
    if USE_FAISS and faiss_index is not None:
        D, I = faiss_index.search(
            np.expand_dims(emb_norm.astype(np.float32), axis=0), 1
        )
        return float(1.0 - D[0][0]), centroid_labels[int(I[0][0])]
    sims     = centroid_matrix.dot(emb_norm)
    best_idx = int(np.argmax(sims))
    return float(1.0 - sims[best_idx]), centroid_labels[best_idx]


# ============================================================
# DUAL VERIFICATION GATE
#
# Three possible outcomes:
#
#  1. HIGH-CONFIDENCE OVERRIDE
#     classifier_prob >= CONFIDENCE_OVERRIDE (e.g. 0.96)
#     → accept immediately, centroid check skipped
#     → handles real-time lighting/angle variation for very
#       confident predictions
#
#  2. NORMAL DUAL GATE
#     classifier_prob >= CONFIDENCE_THRESHOLD (e.g. 0.85)
#     AND centroid_dist <= CENTROID_ACCEPT_THRESHOLD (e.g. 0.72)
#     AND centroid nearest name == classifier name
#     → accept as known
#
#  3. REJECT → "Unknown"
#     anything else
#
# TUNING:
#   Known person wrongly flagged as intruder:
#     → raise CENTROID_ACCEPT_THRESHOLD (e.g. 0.75, 0.80)
#     → or lower CONFIDENCE_OVERRIDE (e.g. 0.93)
#   Unknown slipping through as known:
#     → lower CENTROID_ACCEPT_THRESHOLD (e.g. 0.65)
#     → or raise CONFIDENCE_OVERRIDE (e.g. 0.98)
# ============================================================
def dual_verify(emb_scaled: np.ndarray,
                classifier_name: str,
                classifier_prob: float) -> tuple:
    """Return (verified_label, reason_string)."""

    # --- Gate 0: High-confidence override ---
    # If classifier is extremely confident, trust it without centroid check.
    # This handles cases where real-time lighting shifts the embedding away
    # from the training centroid for genuine known faces.
    if classifier_prob >= CONFIDENCE_OVERRIDE:
        return classifier_name, f"override(prob={classifier_prob:.2f})"

    # --- Gate 1: Centroid distance ---
    dist, nearest = get_centroid_distance(emb_scaled)

    if dist is None:
        # No centroids available — fall back to classifier alone.
        # Less secure but still functional.
        log.debug("No centroids; classifier-only for %s (prob=%.2f)",
                  classifier_name, classifier_prob)
        return classifier_name, f"no_centroids,classifier_only(prob={classifier_prob:.2f})"

    # --- Gate 2: Centroid rescue ---
    # Live camera probability can be modest even when FaceNet embedding distance
    # and classifier identity agree strongly.
    if is_centroid_rescue_candidate(classifier_name, classifier_prob, dist, nearest):
        return classifier_name, (
            f"centroid_rescue(prob={classifier_prob:.2f},"
            f"dist={dist:.2f},nearest={nearest})"
        )

    # --- Gate 3: Minimum classifier confidence ---
    if classifier_prob < CONFIDENCE_THRESHOLD:
        return "Unknown", f"low_prob({classifier_prob:.2f})"

    if dist > CENTROID_ACCEPT_THRESHOLD:
        return "Unknown", (
            f"centroid_too_far(dist={dist:.2f},"
            f"nearest={nearest},threshold={CENTROID_ACCEPT_THRESHOLD})"
        )

    # --- Gate 4: Identity agreement ---
    if nearest != classifier_name:
        return "Unknown", (
            f"identity_mismatch(clf={classifier_name},"
            f"centroid={nearest},dist={dist:.2f})"
        )

    # All gates passed
    return classifier_name, f"verified(prob={classifier_prob:.2f},dist={dist:.2f})"
# ============================================================


# -------------------------
# Intruder alert
# -------------------------
def handle_intruder_alert(track_id: str, state: TrackState, frame: np.ndarray):
    global latest_alert, latest_alert_updated_at
    if state.last_embedding is None:
        return
    if state.intruder_id is None:
        state.intruder_id = intruder_db.find_or_create(state.last_embedding)
    iid = state.intruder_id
    count         = intruder_db.get_alert_count(iid)
    readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_repeat = count > 0
    title = "Intruder Still Present" if is_repeat else "Intruder Detected"
    msg = (
        f"Unknown person detected near the camera. Alert ID: {iid}."
        if not is_repeat
        else f"Unknown person is still visible near the camera. Alert ID: {iid}."
    )
    latest_alert = {
        "id": f"{iid}:{count + 1}",
        "type": "intruder",
        "severity": "warning" if is_repeat else "critical",
        "title": title,
        "name": iid,
        "message": msg,
        "text": f"{title} - {iid}",
        "timestamp": readable_time,
    }
    latest_alert_updated_at = datetime.now().isoformat()
    if not intruder_db.can_alert(iid):
        return
    ts            = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_path      = os.path.join(INTRUDER_DIR, f"{iid}_{ts}.jpg")
    cv2.imwrite(img_path, frame)
    send_telegram_async(msg, img_path)
    intruder_db.record_alert(iid)
    log.info("[ALERT %d/%d] %s", count + 1, INTRUDER_MAX_ALERTS, msg)


# -------------------------
# Drawing
# -------------------------
def draw_label_box(frame: np.ndarray,
                   x1: int, y1: int, x2: int, y2: int,
                   display: str, color: tuple):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y0), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, display, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# -------------------------
# Frame processing
# -------------------------
_frame_count    = 0
_last_frame_out = None
_frame_times = deque(maxlen=30)
_last_perf_log_time = time.time()
_perf_frame_count = 0
_frame_id = 0
_display_fps = 0.0


def normalize_people_label(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "?"
    if cleaned.lower() in {"none", "unknown", "intruder", "unauthorized"}:
        return "unauthorized"
    if "," in cleaned:
        return "authorized"
    return cleaned


def split_people_names(value: str) -> list[str]:
    if value.strip().lower() in {"", "?", "none", "unknown", "intruder", "unauthorized"}:
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


def labels_to_text(labels: list[str]) -> str:
    return "|".join(labels) if labels else ""


def log_detection(frame_id: int, face_id: str, confidence: float,
                  predicted_label: str, processing_time_ms: float,
                  top_prediction: str = "", verify_reason: str = "",
                  centroid_distance: float | None = None,
                  nearest_centroid: str = ""):
    append_csv_row(
        DETECTION_LOG,
        DETECTION_FIELDS,
        [
            datetime.now().isoformat(),
            frame_id,
            face_id,
            confidence,
            REALTIME_TRUE_LABEL,
            predicted_label,
            processing_time_ms,
            top_prediction,
            verify_reason,
            "" if centroid_distance is None else centroid_distance,
            nearest_centroid,
        ],
    )


def log_frame(frame_id: int, processed: bool, face_count: int,
              predicted_labels: list[str], processing_time_ms: float):
    append_csv_row(
        FRAME_LOG,
        FRAME_FIELDS,
        [
            datetime.now().isoformat(),
            frame_id,
            int(processed),
            face_count,
            labels_to_text(predicted_labels),
            REALTIME_TRUE_LABEL,
            REALTIME_TRUE_PEOPLE,
            processing_time_ms,
        ],
    )


def is_centroid_rescue_candidate(classifier_name: str, classifier_prob: float,
                                 dist: float | None, nearest: str | None) -> bool:
    return (
        dist is not None
        and nearest == classifier_name
        and dist <= CENTROID_RESCUE_THRESHOLD
        and classifier_prob >= CENTROID_RESCUE_CONFIDENCE
    )


def draw_status_overlay(frame: np.ndarray, fps: float, face_count: int):
    text = f"FPS: {fps:.1f} | Faces: {face_count}"
    if REALTIME_TRUE_PEOPLE != "?":
        text += f" | Test: {REALTIME_TRUE_PEOPLE}"
    cv2.rectangle(frame, (10, 10), (10 + 12 * len(text), 44), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )


def publish_latest_frame(frame: np.ndarray | None):
    global latest_frame, latest_frame_updated_at
    if frame is None or getattr(frame, "size", 0) == 0:
        return
    latest_frame = frame.copy()
    latest_frame_updated_at = datetime.now().isoformat()


def process_frame(frame: np.ndarray) -> np.ndarray:
    global _frame_count, _last_frame_out, _last_perf_log_time, _perf_frame_count, _frame_id, _display_fps
    if frame is None or frame.size == 0:
        return frame

    frame_start = time.time()
    _frame_count += 1
    _frame_id += 1
    if _frame_count % FRAME_SKIP != 0:
        log_frame(
            _frame_id,
            False,
            0,
            [],
            (time.time() - frame_start) * 1000,
        )
        out = _last_frame_out if _last_frame_out is not None else frame
        publish_latest_frame(out)
        return out

    now            = time.time()
    detected_faces = []
    active_ids     = set()
    frame_predictions = []

    # ---- YOLO detection ----
    try:
        results = yolo(frame, device=device, verbose=False)
    except Exception as e:
        log.exception("YOLO inference failed: %s", e)
        publish_latest_frame(frame)
        return frame

    face_entries = []
    for r in results:
        for box in r.boxes:
            try:
                conf = float(box.conf[0])
            except Exception:
                conf = 0.0
            if conf < 0.35:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
            face_crop = get_padded_crop(frame, x1, y1, x2, y2)
            if face_crop.size == 0:
                continue
            face_entries.append((x1, y1, x2, y2, face_crop))

    for (x1, y1, x2, y2, face_crop) in face_entries:
        cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
        track_id = f"{cx // 40}_{cy // 40}"
        active_ids.add(track_id)
        state    = track_states[track_id]

        # ---- Embedding ----
        try:
            face_start = time.time()
            embedding = get_embedding(face_crop)
            embedding_ms = (time.time() - face_start) * 1000
        except Exception:
            log.exception("Embedding failed.")
            continue
        emb_scaled = normalize_embedding(embedding)

        # ---- Fast-path: centroid distance way too large ----
        dist, nearest = get_centroid_distance(emb_scaled)
        if dist is not None and dist > KNOWN_DISTANCE_THRESHOLD:
            state.push("Unknown", embedding, now)
            reason = f"distance_too_far(dist={dist:.3f},nearest={nearest})"
            if state.is_confirmed_intruder(now, fast_path=True):
                if state.intruder_id is None:
                    state.intruder_id = intruder_db.find_or_create(embedding)
                label_text = state.intruder_id
                color = (0, 0, 255)
                log.info("Confirmed intruder after smoothing: %s track=%s", reason, track_id)
                handle_intruder_alert(track_id, state, frame)
            else:
                label_text = "Checking..."
                color = (0, 165, 255)
            draw_label_box(frame, x1, y1, x2, y2,
                           f"{label_text} (dist:{dist:.2f})", color)
            log_detection(
                _frame_id, track_id, -1.0, label_text, embedding_ms,
                top_prediction="", verify_reason=reason,
                centroid_distance=dist, nearest_centroid=nearest,
            )
            frame_predictions.append(label_text)
            continue

        # ---- Classifier ----
        try:
            classify_start = time.time()
            proba    = classifier.predict_proba([emb_scaled])[0]
            max_prob = float(np.max(proba))
            top_name = classifier.classes_[np.argmax(proba)]
            classify_ms = (time.time() - classify_start) * 1000
        except Exception:
            log.exception("Classifier failed.")
            max_prob, top_name = 0.0, "Unknown"
            classify_ms = 0.0

        # Fast-path: extremely low classifier confidence
        if (
            max_prob < UNKNOWN_IMMEDIATE_THRESHOLD
            and not is_centroid_rescue_candidate(top_name, max_prob, dist, nearest)
        ):
            state.push("Unknown", embedding, now)
            reason = f"low_prob({max_prob:.3f})"
            if state.is_confirmed_intruder(now, fast_path=True):
                if state.intruder_id is None:
                    state.intruder_id = intruder_db.find_or_create(embedding)
                label_text = state.intruder_id
                color = (0, 0, 255)
                log.info("Confirmed intruder after smoothing: %s track=%s", reason, track_id)
                handle_intruder_alert(track_id, state, frame)
            else:
                label_text = "Checking..."
                color = (0, 165, 255)
            draw_label_box(frame, x1, y1, x2, y2,
                           f"{label_text} (p:{max_prob:.2f})", color)
            log_detection(
                _frame_id, track_id, max_prob, label_text, embedding_ms + classify_ms,
                top_prediction=top_name, verify_reason=reason,
                centroid_distance=dist, nearest_centroid=nearest or "",
            )
            frame_predictions.append(label_text)
            continue

        # ---- Dual verification gate ----
        verified_label, reason = dual_verify(emb_scaled, top_name, max_prob)
        log.debug("dual_verify → %s | %s", verified_label, reason)

        # ---- Smoothed voting ----
        state.push(verified_label, embedding, now)
        stable = state.stable_label
        fast_intruder_path = verified_label == "Unknown" and ALERT_ON_UNIDENTIFIED

        if state.is_confirmed_known():
            display = f"{stable} ({max_prob:.2f})"
            color   = (0, 200, 0)
            detected_faces.append({"bbox": (x1, y1, x2, y2), "name": stable})

        elif state.is_confirmed_intruder(now, fast_path=fast_intruder_path):
            if state.intruder_id is None:
                emb_db            = state.last_embedding if state.last_embedding is not None else embedding
                state.intruder_id = intruder_db.find_or_create(emb_db)
            display = f"{state.intruder_id} ({max_prob:.2f})"
            color   = (0, 0, 255)
            handle_intruder_alert(track_id, state, frame)

        else:
            label_text = verified_label if verified_label != "Unknown" else "Checking..."
            display    = f"{label_text} ({max_prob:.2f})"
            color      = (0, 165, 255)

        draw_label_box(frame, x1, y1, x2, y2, display, color)
        predicted_label = display.split(" (", 1)[0]
        log_detection(
            _frame_id, track_id, max_prob, predicted_label, embedding_ms + classify_ms,
            top_prediction=top_name, verify_reason=reason,
            centroid_distance=dist, nearest_centroid=nearest or "",
        )
        frame_predictions.append(predicted_label)

    # ---- Tracker ----
    events, _ = tracker.update_with_ids(detected_faces)
    try:
        from database import log_arrival, log_departure
    except Exception:
        def log_arrival(name):   log.info("log_arrival  fallback: %s", name)
        def log_departure(name): log.info("log_departure fallback: %s", name)

    for ev_type, name in events:
        if ev_type == "ARRIVAL":
            log.info("ARRIVAL  -> %s", name)
            log_arrival(name)
        elif ev_type == "DEPARTURE":
            log.info("DEPARTURE -> %s", name)
            log_departure(name)

    # Purge stale tracks
    for tid in [t for t in list(track_states) if t not in active_ids]:
        del track_states[tid]

    elapsed_ms = (time.time() - frame_start) * 1000
    instant_fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
    _display_fps = instant_fps if _display_fps == 0.0 else (0.85 * _display_fps + 0.15 * instant_fps)
    draw_status_overlay(frame, _display_fps, len(face_entries))
    _last_frame_out = frame.copy()
    publish_latest_frame(_last_frame_out)

    log_frame(_frame_id, True, len(face_entries), frame_predictions, elapsed_ms)
    _frame_times.append(elapsed_ms)
    _perf_frame_count += 1
    now_perf = time.time()
    if now_perf - _last_perf_log_time >= 1.0:
        append_csv_row(
            PERFORMANCE_LOG,
            PERFORMANCE_FIELDS,
            [
                datetime.now().isoformat(),
                _perf_frame_count / (now_perf - _last_perf_log_time),
                float(np.mean(_frame_times)) if _frame_times else elapsed_ms,
            ],
        )
        _perf_frame_count = 0
        _last_perf_log_time = now_perf

    return frame


# -------------------------
# Safe display
# -------------------------
def safe_imshow(window_name: str, frame: np.ndarray, save_preview: bool = True):
    try:
        cv2.imshow(window_name, frame)
    except cv2.error:
        if save_preview:
            path = os.path.join(BASE_DIR, "latest_preview.jpg")
            try:
                cv2.imwrite(path, frame)
            except Exception:
                log.exception("Failed to save preview.")


# -------------------------
# Main
# -------------------------
def main():
    try:
        from database import test_connection, flush_db_queue, close_all_inside_attendance
    except Exception:
        def test_connection(): return True
        def flush_db_queue():  pass
        def close_all_inside_attendance(shutdown_time=None): return 0

    if not test_connection():
        log.error("Database connection failed; aborting.")
        return

    recover_unclean_shutdown(close_all_inside_attendance)
    try:
        write_runtime_state(True, "startup")
    except Exception:
        log.exception("Failed to write startup runtime state.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error("Cannot open camera index %s", CAMERA_INDEX)
        try:
            write_runtime_state(False, "camera_open_failed")
        except Exception:
            log.exception("Failed to write camera_open_failed runtime state.")
        return

    shutdown_lock = threading.Lock()
    shutdown_state = {"done": False}
    signal_handlers_to_restore: list[tuple[int, object]] = []
    last_heartbeat_write = 0.0

    def finalize_shutdown(reason: str, shutdown_time: datetime | None = None):
        with shutdown_lock:
            if shutdown_state["done"]:
                return
            shutdown_state["done"] = True

        when = shutdown_time or datetime.now()
        log.info("Shutdown started (%s) at %s", reason, when.isoformat(timespec="seconds"))

        try:
            flush_db_queue()
        except Exception:
            log.exception("Failed while flushing DB queue during shutdown.")

        try:
            close_all_inside_attendance(when)
        except Exception:
            log.exception("Failed while closing INSIDE attendance rows during shutdown.")

        try:
            write_runtime_state(False, reason, when, when)
        except Exception:
            log.exception("Failed to write shutdown runtime state.")

        try:
            cap.release()
        except Exception:
            log.exception("Failed to release camera during shutdown.")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        for sig_num, previous_handler in signal_handlers_to_restore:
            try:
                signal.signal(sig_num, previous_handler)
            except Exception:
                pass

        log.info("Shutdown complete.")

    atexit.register(finalize_shutdown, "process_exit")

    def handle_stop_signal(sig_num, _frame):
        try:
            sig_name = signal.Signals(sig_num).name
        except Exception:
            sig_name = str(sig_num)
        log.info("Received stop signal: %s", sig_name)
        raise KeyboardInterrupt

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig_num = getattr(signal, sig_name, None)
        if sig_num is None:
            continue
        try:
            signal_handlers_to_restore.append((sig_num, signal.getsignal(sig_num)))
            signal.signal(sig_num, handle_stop_signal)
        except Exception:
            pass

    try:
        from retrain_job import start_scheduler_thread
        start_scheduler_thread()
    except Exception:
        log.debug("Retrain scheduler not available.")

    if REALTIME_TRUE_LABEL != "?":
        log.info("Real-time test recording label: %s | people: %s",
                 REALTIME_TRUE_LABEL, REALTIME_TRUE_PEOPLE)
    log.info("Security system running. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            now_loop = time.time()
            if now_loop - last_heartbeat_write >= 2.0:
                try:
                    write_runtime_state(
                        True,
                        "running",
                        datetime.now(),
                        datetime.now(),
                    )
                    last_heartbeat_write = now_loop
                except Exception:
                    log.exception("Failed to update runtime heartbeat.")
            publish_latest_frame(frame)
            out = process_frame(frame)
            safe_imshow("AI Security System", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        finalize_shutdown("main_loop_exit")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AI security system.")
    parser.add_argument(
        "--test-name",
        default=None,
        help="Optional name for this test run. If omitted, it is built from true people/label plus timestamp.",
    )
    parser.add_argument(
        "--test-recording",
        action="store_true",
        help="Ask who is present and write that ground truth into real-time logs.",
    )
    parser.add_argument(
        "--true-label",
        default=None,
        help="Ground-truth label for the recording, e.g. authorized or unauthorized.",
    )
    parser.add_argument(
        "--true-people",
        default=None,
        help="Names of people present in front of the camera, comma-separated.",
    )
    parser.add_argument(
        "--keep-frame-skip",
        action="store_true",
        help="Do not force FRAME_SKIP=1 during test recording.",
    )
    parser.add_argument(
        "--no-ground-truth-prompt",
        action="store_true",
        help="Start without asking who is present in front of the camera.",
    )
    return parser.parse_args()


def configure_realtime_ground_truth(args):
    global REALTIME_TRUE_LABEL, REALTIME_TRUE_PEOPLE, FRAME_SKIP

    if not args.true_people and not args.true_label and not args.no_ground_truth_prompt:
        try:
            args.true_people = input(
                "Who is present in front of the camera? "
                "Use names separated by comma, type unauthorized/none, "
                "or press Enter to skip: "
            ).strip()
        except EOFError:
            args.true_people = ""

    if (args.test_recording or args.true_people) and not args.keep_frame_skip:
        FRAME_SKIP = 1
        log.info("Test recording enabled; processing every frame (FRAME_SKIP=1).")

    if args.true_people:
        REALTIME_TRUE_PEOPLE = args.true_people.strip()
        REALTIME_TRUE_LABEL = args.true_label or normalize_people_label(args.true_people)
        missing = [
            name for name in split_people_names(REALTIME_TRUE_PEOPLE)
            if name not in known_names
        ]
        if missing:
            log.warning(
                "These test people are not in the current trained model labels: %s",
                ", ".join(missing),
            )
    elif args.true_label:
        REALTIME_TRUE_LABEL = args.true_label.strip()
        REALTIME_TRUE_PEOPLE = args.true_label.strip()

    test_name_source = REALTIME_TRUE_PEOPLE
    if test_name_source == "?":
        test_name_source = REALTIME_TRUE_LABEL
    if args.test_name:
        test_run_name = make_test_run_name(args.test_name)
    else:
        test_run_name = make_test_run_name(test_name_source)
    configure_test_log_paths(test_run_name)


if __name__ == "__main__":
    args = parse_args()
    configure_realtime_ground_truth(args)
    try:
        from flask_app import app
        threading.Thread(
            target=app.run,
            kwargs={"host": "0.0.0.0", "port": 5000, "debug": False},
            daemon=True,
        ).start()
    except Exception:
        log.debug("Flask app not started.")
    main()
