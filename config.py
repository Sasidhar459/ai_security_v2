import os
import torch

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def _is_valid_model_file(path):
    return os.path.isfile(path) and os.path.getsize(path) > 1_000_000


def _resolve_yolo_model():
    candidates = [
        os.path.join(BASE_DIR, "model", "yolov8n-face.pt"),
        os.path.join(BASE_DIR, "yolov8n-face.pt"),
    ]
    for candidate in candidates:
        if _is_valid_model_file(candidate):
            return candidate
    # Keep the fallback explicit so startup still fails with a clear path
    # if the model file is missing.
    return candidates[0]


YOLO_FACE_MODEL = _resolve_yolo_model()
MODEL_PATH = os.path.join(MODEL_DIR, "face_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CENTROIDS_PATH = os.path.join(MODEL_DIR, "centroids.pkl")

INTRUDER_DIR = os.path.join(BASE_DIR, "intruder_images")
os.makedirs(INTRUDER_DIR, exist_ok=True)
INTRUDER_DB_PATH = os.path.join(INTRUDER_DIR, "intruder_db.json")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

REPORT_DIR = os.path.join(BASE_DIR, "analysis_report")
os.makedirs(REPORT_DIR, exist_ok=True)

# -------------------------
# Database (SQL Server)
# -------------------------
# Keep env overrides for compatibility with the older files while still
# providing the local defaults used by this project.
DB_CONFIG = {
    "driver": os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server").strip(),
    "server": os.getenv("DB_SERVER", r"localhost\SQLEXPRESS").strip(),
    "database": os.getenv("DB_NAME", "AI_SECURITY_SYSTEM").strip(),
}

# -------------------------
# Camera / runtime
# -------------------------
DEPARTURE_TIMEOUT = 60
CAMERA_INDEX = 0
FRAME_SKIP = 2
YOLO_IMGSZ = 512
YOLO_MAX_DET = 10

# -------------------------
# Face cropping padding
# -------------------------
CROP_PADDING = 0.20

# -------------------------
# Dual-gate thresholds
# -------------------------
CONFIDENCE_THRESHOLD = 0.25
CONFIDENCE_OVERRIDE = 0.80
CENTROID_ACCEPT_THRESHOLD = 0.85

# Legacy rescue gate values still imported by security_system.py.
CENTROID_RESCUE_THRESHOLD = 0.60
CENTROID_RESCUE_CONFIDENCE = 0.12

UNKNOWN_IMMEDIATE_THRESHOLD = 0.12
KNOWN_DISTANCE_THRESHOLD = 0.90

# -------------------------
# Smoothing / voting
# -------------------------
SMOOTHING_FRAMES = 10
MIN_KNOWN_VOTES = 6
MIN_UNKNOWN_VOTES = 8

# -------------------------
# Intruder alerts
# -------------------------
INTRUDER_ALERT_GAP = 300
INTRUDER_MAX_ALERTS = 2
INTRUDER_EMBED_DIST = 0.45
ALERT_ON_UNIDENTIFIED = True
ATTENDANCE_LOG_COOLDOWN_SECS = 60
UNIDENTIFIED_CONFIRM_SECS = 4
INTRUDER_CONFIRM_SECS = 12
RECENT_KNOWN_GRACE_SECS = 20
MIN_TRACK_AGE_FOR_INTRUDER = 20
REALTIME_CLOSEDSET_FALLBACK = True
REALTIME_CLOSEDSET_MIN_CONF = 0.24
REALTIME_CLOSEDSET_MIN_MARGIN = 0.04
REALTIME_LABEL_HOLD_SECS = 15.0
REALTIME_KNOWN_MIN_CONF = 0.14
REALTIME_ALLOWED_NAMES = ["Sasidhar", "Mohan", "Suresh"]
REALTIME_POSE_TTA_ENABLED = True
REALTIME_POSE_TTA_ANGLES = [-10, 10]
REALTIME_POSE_TTA_TRIGGER_CONF = 0.45
REALTIME_POSE_TTA_EVERY_N_FRAMES = 2

# -------------------------
# Identity lock
# -------------------------
IDENTITY_LOCK_ENABLED = True
IDENTITY_LOCK_MIN_CONF = 0.18
IDENTITY_LOCK_HOLD_SECS = 15
IDENTITY_UNLOCK_MISMATCH_FRAMES = 8

# -------------------------
# Tracking
# -------------------------
TRACKING_MAX_DISAPPEARED = 30
TRACKING_MAX_DISTANCE = 60

# -------------------------
# Telegram
# -------------------------
TELEGRAM_BOT_TOKEN = "8489650776:AAE8lV1AtvqzXz7L-X8u6kquk0Cuj5sOtM8"
TELEGRAM_CHAT_ID = "6034901248"

# -------------------------
# Optional
# -------------------------
USE_FAISS = False
TUNE_SVM = True
ANTI_SPOOFING_ENABLED = False

# -------------------------
# CUDA / runtime acceleration
# -------------------------
CUDA_REQUIRED = True
TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# -------------------------
# Automatic live metrics
# -------------------------
AUTO_METRICS_ENABLED = True
AUTO_METRICS_INTERVAL_SECS = 900
LIVE_METRICS_JSON = os.path.join(REPORT_DIR, "live_metrics_latest.json")
LIVE_METRICS_CSV = os.path.join(REPORT_DIR, "live_metrics_history.csv")
FRAME_LEVEL_LOG_PATH = os.path.join(LOG_DIR, "realtime_frame_log.csv")
