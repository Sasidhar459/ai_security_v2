"""
generate_centroids.py

Run this ONCE (and re-run whenever you add new people to the dataset).
It scans your face dataset, computes a FaceNet embedding for every image,
averages them per person, and saves centroids.pkl next to your other models.

Expected dataset layout (same folder you used for training):
    dataset/
        Alice/
            img1.jpg
            img2.jpg
        Bob/
            img1.jpg
        ...

Usage:
    python generate_centroids.py
    python generate_centroids.py --dataset path/to/dataset --out model/centroids.pkl
"""

import os
import sys
import pickle
import argparse
import logging
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import normalize
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("generate_centroids")

# -------------------------
# Defaults (edit or pass as CLI args)
# -------------------------
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET  = os.path.join(BASE_DIR, "dataset")   # your training images folder
DEFAULT_OUT      = os.path.join(BASE_DIR, "model", "centroids.pkl")
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -------------------------
# Device + models
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Using device: %s", device)

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    post_process=True,
    device=device,
)

_fallback = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def get_embedding(img_path: str) -> np.ndarray | None:
    """Return 512-D FaceNet embedding for a single image, or None on failure."""
    try:
        bgr = cv2.imread(img_path)
        if bgr is None:
            log.warning("Cannot read image: %s", img_path)
            return None
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # Try MTCNN alignment first
        try:
            aligned = mtcnn(img)
        except Exception:
            aligned = None

        if aligned is None:
            # Fallback: plain resize
            aligned = _fallback(img)

        img_t = aligned.unsqueeze(0).to(device)
        with torch.no_grad():
            if device.startswith("cuda"):
                from torch.cuda.amp import autocast
                with autocast():
                    emb = facenet(img_t)
            else:
                emb = facenet(img_t)
        return emb.cpu().numpy()[0]

    except Exception:
        log.exception("Error processing %s", img_path)
        return None


def build_centroids(dataset_dir: str, out_path: str):
    if not os.path.isdir(dataset_dir):
        log.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    person_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not person_dirs:
        log.error("No person subdirectories found in: %s", dataset_dir)
        sys.exit(1)

    log.info("Found %d people: %s", len(person_dirs), person_dirs)

    centroids      = {}
    all_embeddings = defaultdict(list)
    total_images   = 0
    failed_images  = 0

    for person in person_dirs:
        person_dir = os.path.join(dataset_dir, person)
        images     = [
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        ]

        if not images:
            log.warning("No images found for %s — skipping.", person)
            continue

        log.info("Processing %-20s — %d images", person, len(images))
        for img_file in images:
            emb = get_embedding(os.path.join(person_dir, img_file))
            if emb is not None:
                all_embeddings[person].append(emb)
                total_images += 1
            else:
                failed_images += 1

    # Compute per-person centroid (mean of all embeddings, then L2-normalised)
    for person, embs in all_embeddings.items():
        if not embs:
            continue
        stack    = np.stack(embs, axis=0)          # (N, 512)
        centroid = np.mean(stack, axis=0)           # (512,)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)  # L2 normalise
        centroids[person] = centroid
        log.info("  %-20s centroid from %d embeddings", person, len(embs))

    if not centroids:
        log.error("No centroids were built — check your dataset directory.")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(centroids, f)

    log.info("=" * 55)
    log.info("Centroids saved  : %s", out_path)
    log.info("People indexed   : %d", len(centroids))
    log.info("Images processed : %d  (failed: %d)", total_images, failed_images)
    log.info("=" * 55)
    log.info("Restart security_system.py — centroids will load automatically.")


# -------------------------
# Diagnostic: print distances
# -------------------------
def print_distances(out_path: str):
    """
    After building, print pairwise centroid distances so you can
    verify CENTROID_ACCEPT_THRESHOLD is set correctly in config.py.
    """
    with open(out_path, "rb") as f:
        centroids = pickle.load(f)

    names = list(centroids.keys())
    if len(names) < 2:
        return

    log.info("")
    log.info("Pairwise centroid distances (lower = more similar):")
    log.info("Use these to tune CENTROID_ACCEPT_THRESHOLD in config.py")
    log.info("-" * 55)

    from scipy.spatial.distance import cosine
    for i, a in enumerate(names):
        for b in names[i+1:]:
            d = cosine(centroids[a], centroids[b])
            log.info("  %-15s  <->  %-15s  dist = %.4f", a, b, d)

    log.info("")
    log.info("Recommended CENTROID_ACCEPT_THRESHOLD:")
    log.info("  Set it roughly halfway between the smallest inter-person")
    log.info("  distance and 0.90 (typical unknown distance).")
    log.info("  E.g. if closest pair is 0.40 and unknowns score ~0.70+,")
    log.info("  a threshold of 0.55 is a safe middle ground.")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FaceNet centroids from dataset")
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"Path to dataset folder (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Output centroids.pkl path (default: {DEFAULT_OUT})"
    )
    args = parser.parse_args()

    build_centroids(args.dataset, args.out)
    print_distances(args.out)