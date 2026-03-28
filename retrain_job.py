import os, sys, time, pickle, shutil, logging, argparse, threading
import numpy as np, cv2, torch
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# ============================================================
# CONFIG
# ============================================================
DATASET_PATH         = "dataset"
MODEL_DIR            = "model"
MODEL_PATH           = os.path.join(MODEL_DIR, "face_classifier.pkl")
BACKUP_DIR           = os.path.join(MODEL_DIR, "backups")
LOG_PATH             = "logs/retrain.log"
RETRAIN_INTERVAL_HRS = 168       # auto-retrain every N hours
MIN_IMAGES_PER_CLASS = 5        # abort if any class has fewer images
MIN_ACCURACY         = 0.75     # keep old model if new one scores below this
MAX_BACKUPS          = 5        # how many old models to keep

# ============================================================
# LOGGING
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log    = logging.getLogger("retrain")
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    log.info("Loading YOLO and FaceNet...")
    yolo    = YOLO("yolov8n-face.pt")
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    log.info(f"Models ready on: {device}")
    return yolo, facenet


def validate_dataset():
    if not os.path.exists(DATASET_PATH):
        log.error(f"Dataset not found: {DATASET_PATH}")
        return False, {}
    stats = {}
    for person in os.listdir(DATASET_PATH):
        pp = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(pp): continue
        imgs = [f for f in os.listdir(pp)
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        stats[person] = len(imgs)
    if len(stats) < 2:
        log.warning(f"Need >= 2 persons. Found: {len(stats)}")
        return False, stats
    low = {k: v for k, v in stats.items() if v < MIN_IMAGES_PER_CLASS}
    if low:
        log.warning(f"Too few images for: {low}. Add more before retraining.")
        return False, stats
    log.info(f"Dataset OK -- {stats}")
    return True, stats


def extract_embeddings(yolo, facenet):
    embeddings, labels, skipped = [], [], 0
    for person in os.listdir(DATASET_PATH):
        pp = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(pp): continue
        count = 0
        for img_name in os.listdir(pp):
            frame = cv2.imread(os.path.join(pp, img_name))
            if frame is None: skipped += 1; continue
            try:
                res = yolo(frame, verbose=False)[0]
                if len(res.boxes) == 0:
                    face = cv2.resize(frame, (160, 160))
                else:
                    x1, y1, x2, y2 = map(int, res.boxes.xyxy[0])
                    x1=max(0,x1); y1=max(0,y1)
                    x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0: skipped += 1; continue
                    face = cv2.resize(face, (160, 160))
                face   = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_t = torch.tensor(face).permute(2,0,1).float()/255.0
                face_t = face_t.unsqueeze(0).to(device)
                emb    = facenet(face_t).detach().cpu().numpy()[0]
                embeddings.append(emb); labels.append(person); count += 1
            except Exception as e:
                log.warning(f"Error {img_name}: {e}"); skipped += 1
        log.info(f"  {person}: {count} embeddings")
    log.info(f"Total: {len(embeddings)} embeddings | Skipped: {skipped}")
    return embeddings, labels, skipped


def train_classifier(embeddings, labels):
    X, y = np.array(embeddings), np.array(labels)
    log.info("Training SVM...")
    cv     = min(5, len(set(labels)))
    scores = cross_val_score(SVC(kernel="linear", probability=True),
                             X, y, cv=cv, scoring="accuracy")
    acc    = float(np.mean(scores))
    log.info(f"Cross-val accuracy: {acc:.3f} (+/- {np.std(scores):.3f})")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y)
    return clf, acc


def backup_model():
    if not os.path.exists(MODEL_PATH): return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(BACKUP_DIR, f"face_classifier_{ts}.pkl")
    shutil.copy2(MODEL_PATH, dst)
    log.info(f"Backup: {dst}")
    bkps = sorted([os.path.join(BACKUP_DIR, f)
                   for f in os.listdir(BACKUP_DIR) if f.endswith(".pkl")])
    while len(bkps) > MAX_BACKUPS:
        os.remove(bkps.pop(0)); log.info("Old backup pruned.")


def save_model(clf):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f: pickle.dump(clf, f)
        log.info(f"Model saved: {MODEL_PATH}")
        return True
    except Exception as e:
        log.error(f"Save failed: {e}"); return False


def hot_reload_classifier():
    """Swap classifier in running security_system without restart."""
    try:
        import security_system
        with open(MODEL_PATH, "rb") as f:
            security_system.classifier = pickle.load(f)
        log.info("Classifier hot-reloaded in security_system.")
    except ImportError:
        log.info("security_system not in this process -- skipped.")
    except Exception as e:
        log.warning(f"Hot-reload failed: {e}")


def run_retrain(force=False):
    log.info("=" * 55)
    log.info("RETRAIN JOB STARTED")
    log.info("=" * 55)
    valid, stats = validate_dataset()
    if not valid: return False
    yolo, facenet = load_models()
    embeddings, labels, _ = extract_embeddings(yolo, facenet)
    if not embeddings: log.error("No embeddings."); return False
    new_model, acc = train_classifier(embeddings, labels)
    if not force and acc < MIN_ACCURACY:
        log.warning(f"Accuracy {acc:.3f} below {MIN_ACCURACY} -- keeping old model.")
        return False
    backup_model()
    if not save_model(new_model): return False
    hot_reload_classifier()
    log.info(f"Done. Accuracy={acc:.3f} | Persons={list(stats.keys())}")
    return True


def scheduler_loop():
    secs     = RETRAIN_INTERVAL_HRS * 3600
    log.info(f"Scheduler: every {RETRAIN_INTERVAL_HRS}h")
    next_run = time.time() + secs
    while True:
        if time.time() >= next_run:
            log.info("Scheduled retrain triggered.")
            try: run_retrain()
            except Exception as e: log.error(f"Error: {e}")
            next_run = time.time() + secs
        time.sleep(60)


def start_scheduler_thread():
    """
    Add this to security_system.py for automatic retraining:

        from retrain_job import start_scheduler_thread
        start_scheduler_thread()   # call once near the top of main()
    """
    t = threading.Thread(target=scheduler_loop, daemon=True, name="retrain")
    t.start()
    log.info("Retrain scheduler thread started.")
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now",      action="store_true", help="Retrain immediately")
    parser.add_argument("--force",    action="store_true", help="Save even if accuracy low")
    parser.add_argument("--schedule", action="store_true", help="Run as continuous scheduler")
    parser.add_argument("--interval", type=int, default=RETRAIN_INTERVAL_HRS)
    args = parser.parse_args()
    RETRAIN_INTERVAL_HRS = args.interval

    if args.now or args.force:
        sys.exit(0 if run_retrain(force=args.force) else 1)
    elif args.schedule:
        try:
            run_retrain()
            scheduler_loop()
        except KeyboardInterrupt:
            log.info("Stopped.")
    else:
        sys.exit(0 if run_retrain() else 1)