import os

# -------------------------
# Paths and config
# -------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

YOLO_FACE_MODEL  = os.path.join(BASE_DIR, "yolov8n-face.pt")
MODEL_PATH       = os.path.join(MODEL_DIR, "face_classifier.pkl")
SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.pkl")
CENTROIDS_PATH   = os.path.join(MODEL_DIR, "centroids.pkl")

INTRUDER_DIR     = os.path.join(BASE_DIR, "intruder_images")
os.makedirs(INTRUDER_DIR, exist_ok=True)
INTRUDER_DB_PATH = os.path.join(INTRUDER_DIR, "intruder_db.json")

# -------------------------
# Dataset folder (for collected face images)
# -------------------------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# -------------------------
# Database
# <- Fill in your SQL Server details here
# -------------------------
DB_CONFIG = {
    "driver"  : "ODBC Driver 17 for SQL Server",  # run: python -c "import pyodbc; print(pyodbc.drivers())"
    "server"  : "localhost\\SQLEXPRESS",               # e.g. "localhost" or "DESKTOP-XXX\\SQLEXPRESS"
    "database": "AI_SECURITY",             # e.g. "SecurityDB"
}

# -------------------------
# Camera / runtime
# -------------------------
DEPARTURE_TIMEOUT = 60
CAMERA_INDEX      = 0
FRAME_SKIP        = 2

# -------------------------
# Verification thresholds
#
# HOW THE DUAL GATE WORKS:
#
#  Case A - High confidence override (classifier alone is enough):
#    classifier_prob >= CONFIDENCE_OVERRIDE  ->  accepted as known
#    (centroid check is skipped entirely)
#
#  Case B - Normal dual gate:
#    classifier_prob >= CONFIDENCE_THRESHOLD
#    AND centroid distance <= CENTROID_ACCEPT_THRESHOLD
#    AND centroid nearest name == classifier name
#    ->  accepted as known
#
#  Anything else -> Unknown / Intruder
#
# TUNING GUIDE:
#   Known people flagged as intruder -> raise CENTROID_ACCEPT_THRESHOLD (e.g. 0.75)
#   Unknown slipping through as known -> lower CENTROID_ACCEPT_THRESHOLD (e.g. 0.60)
# -------------------------
CONFIDENCE_THRESHOLD      = 0.85   # minimum prob for normal dual-gate path
CONFIDENCE_OVERRIDE       = 0.96   # if prob >= this, skip centroid check entirely
CENTROID_ACCEPT_THRESHOLD = 0.72   # max cosine distance to centroid

# Below this prob -> immediate intruder, skip all gates
UNKNOWN_IMMEDIATE_THRESHOLD = 0.35

# Centroid distance above this -> immediate intruder (no classifier needed)
KNOWN_DISTANCE_THRESHOLD  = 0.92

# -------------------------
# Smoothing / voting
# -------------------------
SMOOTHING_FRAMES      = 8
MIN_KNOWN_VOTES       = 5
MIN_UNKNOWN_VOTES     = 4
INTRUDER_CONFIRM_SECS = 3

# -------------------------
# Intruder alerts
#
# INTRUDER_ALERT_GAP   : seconds between repeated alerts for the same intruder
# INTRUDER_MAX_ALERTS  : safety cap on alerts per intruder per day
#                        (user can also stop alerts per-intruder from the dashboard)
# INTRUDER_EMBED_DIST  : cosine distance below which two embeddings = same intruder
# -------------------------
INTRUDER_ALERT_GAP  = 300    # 5 minutes between alerts
INTRUDER_MAX_ALERTS = 100    # effectively unlimited - user stops via dashboard
INTRUDER_EMBED_DIST = 0.45

# -------------------------
# Face crop padding
# -------------------------
CROP_PADDING = 0.20

# -------------------------
# Telegram credentials
# -------------------------
TELEGRAM_BOT_TOKEN = "8489650776:AAE8lV1AtvqzXz7L-X8u6kquk0Cuj5sOtM8"
TELEGRAM_CHAT_ID   = "6034901248"

# -------------------------
# Optional FAISS
# -------------------------
USE_FAISS = False