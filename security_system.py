"""
security_system.py

Pipeline:
  YOLO detection → padded crop → MTCNN alignment → FaceNet embedding
  → Dual Verification Gate (with high-confidence override)
  → Smoothed per-track voting → known / intruder decision
  → Telegram alert on confirmed intruder
"""

import os
import cv2
import torch
import numpy as np
import pickle
import time
import json
import logging
import threading
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
    DEPARTURE_TIMEOUT, CAMERA_INDEX,
    SMOOTHING_FRAMES, MIN_KNOWN_VOTES, MIN_UNKNOWN_VOTES,
    INTRUDER_CONFIRM_SECS, FRAME_SKIP,
    KNOWN_DISTANCE_THRESHOLD, UNKNOWN_IMMEDIATE_THRESHOLD,
    INTRUDER_MAX_ALERTS, INTRUDER_ALERT_GAP, INTRUDER_EMBED_DIST,
    CROP_PADDING, USE_FAISS,
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
        centroid_matrix = normalize(
            np.stack([centroids[l] for l in centroid_labels], axis=0).astype(np.float32),
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
        self.first_unk_time = None
        self.intruder_id    = None
        self.last_embedding = None

    def push(self, label: str, embedding: np.ndarray, now: float):
        self.votes.append(label)
        self.last_embedding = embedding
        if label == "Unknown":
            if self.first_unk_time is None:
                self.first_unk_time = now
        else:
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

    def is_confirmed_intruder(self, now: float) -> bool:
        return (
            len(self.votes) >= SMOOTHING_FRAMES
            and self.unknown_votes >= MIN_UNKNOWN_VOTES
            and self.first_unk_time is not None
            and (now - self.first_unk_time) >= INTRUDER_CONFIRM_SECS
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

    # --- Gate 1: Minimum classifier confidence ---
    if classifier_prob < CONFIDENCE_THRESHOLD:
        return "Unknown", f"low_prob({classifier_prob:.2f})"

    # --- Gate 2: Centroid distance ---
    dist, nearest = get_centroid_distance(emb_scaled)

    if dist is None:
        # No centroids available — fall back to classifier alone.
        # Less secure but still functional.
        log.debug("No centroids; classifier-only for %s (prob=%.2f)",
                  classifier_name, classifier_prob)
        return classifier_name, f"no_centroids,classifier_only(prob={classifier_prob:.2f})"

    if dist > CENTROID_ACCEPT_THRESHOLD:
        return "Unknown", (
            f"centroid_too_far(dist={dist:.2f},"
            f"nearest={nearest},threshold={CENTROID_ACCEPT_THRESHOLD})"
        )

    # --- Gate 3: Identity agreement ---
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
    if state.last_embedding is None:
        return
    if state.intruder_id is None:
        state.intruder_id = intruder_db.find_or_create(state.last_embedding)
    iid = state.intruder_id
    if not intruder_db.can_alert(iid):
        return
    ts            = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_path      = os.path.join(INTRUDER_DIR, f"{iid}_{ts}.jpg")
    cv2.imwrite(img_path, frame)
    count         = intruder_db.get_alert_count(iid)
    readable_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"[ALERT] Unknown person detected! [{iid}] at {readable_time}"
        if count == 0
        else f"[WARN] Intruder [{iid}] still present — {readable_time}"
    )
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


def process_frame(frame: np.ndarray) -> np.ndarray:
    global _frame_count, _last_frame_out
    if frame is None or frame.size == 0:
        return frame

    _frame_count += 1
    if _frame_count % FRAME_SKIP != 0:
        return _last_frame_out if _last_frame_out is not None else frame

    now            = time.time()
    detected_faces = []
    active_ids     = set()

    # ---- YOLO detection ----
    try:
        results = yolo(frame, device=device, verbose=False)
    except Exception as e:
        log.exception("YOLO inference failed: %s", e)
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
            embedding = get_embedding(face_crop)
        except Exception:
            log.exception("Embedding failed.")
            continue
        emb_scaled = normalize_embedding(embedding)

        # ---- Fast-path: centroid distance way too large ----
        dist, nearest = get_centroid_distance(emb_scaled)
        if dist is not None and dist > KNOWN_DISTANCE_THRESHOLD:
            state.push("Unknown", embedding, now)
            if state.intruder_id is None:
                state.intruder_id = intruder_db.find_or_create(embedding)
            if intruder_db.can_alert(state.intruder_id):
                log.info("Fast-path intruder: dist=%.3f nearest=%s track=%s",
                         dist, nearest, track_id)
                handle_intruder_alert(track_id, state, frame)
            draw_label_box(frame, x1, y1, x2, y2,
                           f"{state.intruder_id} (dist:{dist:.2f})", (0, 0, 255))
            continue

        # ---- Classifier ----
        try:
            proba    = classifier.predict_proba([emb_scaled])[0]
            max_prob = float(np.max(proba))
            top_name = classifier.classes_[np.argmax(proba)]
        except Exception:
            log.exception("Classifier failed.")
            max_prob, top_name = 0.0, "Unknown"

        # Fast-path: extremely low classifier confidence
        if max_prob < UNKNOWN_IMMEDIATE_THRESHOLD:
            state.push("Unknown", embedding, now)
            if state.intruder_id is None:
                state.intruder_id = intruder_db.find_or_create(embedding)
            if intruder_db.can_alert(state.intruder_id):
                log.info("Fast-path intruder: prob=%.3f track=%s", max_prob, track_id)
                handle_intruder_alert(track_id, state, frame)
            draw_label_box(frame, x1, y1, x2, y2,
                           f"{state.intruder_id} (p:{max_prob:.2f})", (0, 0, 255))
            continue

        # ---- Dual verification gate ----
        verified_label, reason = dual_verify(emb_scaled, top_name, max_prob)
        log.debug("dual_verify → %s | %s", verified_label, reason)

        # ---- Smoothed voting ----
        state.push(verified_label, embedding, now)
        stable = state.stable_label

        if state.is_confirmed_known():
            display = f"{stable} ({max_prob:.2f})"
            color   = (0, 200, 0)
            detected_faces.append({"bbox": (x1, y1, x2, y2), "name": stable})

        elif state.is_confirmed_intruder(now):
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

    _last_frame_out = frame.copy()
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
        from database import test_connection, flush_db_queue
    except Exception:
        def test_connection(): return True
        def flush_db_queue():  pass

    if not test_connection():
        log.error("Database connection failed; aborting.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error("Cannot open camera index %s", CAMERA_INDEX)
        return

    try:
        from retrain_job import start_scheduler_thread
        start_scheduler_thread()
    except Exception:
        log.debug("Retrain scheduler not available.")

    log.info("Security system running. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            out = process_frame(frame)
            safe_imshow("AI Security System", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        flush_db_queue()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        log.info("Shutdown complete.")


if __name__ == "__main__":
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