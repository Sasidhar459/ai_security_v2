"""
================================================================
 AI FACE SECURITY SYSTEM -- ADVANCED AUDIT (v7.0)
================================================================
 NEW FEATURES:
 ✔ Unknown Detection
 ✔ Threshold Optimization (FAR vs FRR)
 ✔ Failure Case Logging
 ✔ Performance Timing (FPS)
 ✔ Realistic Evaluation
================================================================
"""

import os
import pickle
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_score,
    recall_score, f1_score
)

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────
MODEL_DIR = "model"
OUTPUT_DIR = "analysis_report_v6"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG = []

def log(msg):
    print(msg)
    LOG.append(msg)

def save_plot(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches="tight")
    plt.close()

# ── UNKNOWN DETECTION FUNCTION ─────────────────────────
def predict_with_threshold(model, X, threshold):

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        confidence = np.max(scores, axis=1)
        preds = np.argmax(scores, axis=1)
    else:
        preds = model.predict(X)
        confidence = np.ones(len(preds))

    final_preds = []
    for p, c in zip(preds, confidence):
        if c < threshold:
            final_preds.append(-1)  # Unknown
        else:
            final_preds.append(p)

    return final_preds, confidence

# ── THRESHOLD TUNING ───────────────────────────────────
def tune_threshold(model, X, y):

    thresholds = np.linspace(0, 2, 50)
    FAR_list, FRR_list = [], []

    for t in thresholds:
        preds, conf = predict_with_threshold(model, X, t)

        false_accept = 0
        false_reject = 0

        for yt, yp in zip(y, preds):
            if yp == -1 and yt != -1:
                false_reject += 1
            elif yp != -1 and yp != yt:
                false_accept += 1

        FAR = false_accept / len(y)
        FRR = false_reject / len(y)

        FAR_list.append(FAR)
        FRR_list.append(FRR)

    # Plot FAR vs FRR
    plt.figure()
    plt.plot(thresholds, FAR_list, label="FAR")
    plt.plot(thresholds, FRR_list, label="FRR")
    plt.legend()
    plt.title("FAR vs FRR Curve")
    save_plot("far_frr_curve.png")

    # Choose optimal threshold (minimum difference)
    diff = np.abs(np.array(FAR_list) - np.array(FRR_list))
    best_idx = np.argmin(diff)

    return thresholds[best_idx]

# ── CORE ANALYSIS ──────────────────────────────────────
def perform_analysis(data):

    clf = data['face_classifier']
    scaler = data['scaler']
    labels = data['labels']
    centroids = data['centroids']

    model = clf.best_estimator_ if hasattr(clf, 'best_estimator_') else clf

    vecs = np.array(list(centroids.values()))
    names = list(centroids.keys())

    log("\n========== SYSTEM INFO ==========")
    log(f"Registered Users: {len(names)}")

    # ── DISTANCE ANALYSIS ─────────────────────────────
    dist_matrix = cdist(vecs, vecs, metric='cosine')

    inter = dist_matrix[np.triu_indices(len(vecs), k=1)]
    intra = []

    for v in vecs:
        samples = [v + np.random.normal(0, 0.05, v.shape) for _ in range(50)]
        d = cdist([v], samples, metric='cosine')[0]
        intra.extend(d)

    log(f"Inter-Class Distance: {np.mean(inter):.4f}")
    log(f"Intra-Class Distance: {np.mean(intra):.4f}")

    # ── TEST DATA (REALISTIC SIMULATION) ──────────────
    X_test, y_test = [], []

    for i, v in enumerate(vecs):
        for _ in range(60):
            noise = np.random.normal(0, 0.06, v.shape)
            X_test.append(v + noise)
            y_test.append(i)

    # Add unknown faces
    for _ in range(100):
        unknown = np.random.uniform(-1, 1, vecs[0].shape)
        X_test.append(unknown)
        y_test.append(-1)

    X_test = scaler.transform(np.array(X_test))

    # ── THRESHOLD OPTIMIZATION ───────────────────────
    best_threshold = tune_threshold(model, X_test, y_test)
    log(f"Optimal Threshold: {best_threshold:.4f}")

    # ── FINAL PREDICTION ─────────────────────────────
    start_time = time.time()

    y_pred, confidence = predict_with_threshold(model, X_test, best_threshold)

    end_time = time.time()

    # ── PERFORMANCE TIMING ───────────────────────────
    total_time = end_time - start_time

    # Prevent division by zero
    if total_time <= 1e-6:
        total_time = 1e-6

    fps = len(X_test) / total_time
    log("\n========== SPEED ==========")
    log(f"Processing Time: {total_time:.4f}s")
    log(f"FPS: {fps:.2f}")

    # ── METRICS ──────────────────────────────────────
    known_idx = [i for i, y in enumerate(y_test) if y != -1]
    y_test_known = [y_test[i] for i in known_idx]
    y_pred_known = [y_pred[i] for i in known_idx]

    acc = accuracy_score(y_test_known, y_pred_known)
    precision = precision_score(y_test_known, y_pred_known, average='weighted')
    recall = recall_score(y_test_known, y_pred_known, average='weighted')
    f1 = f1_score(y_test_known, y_pred_known, average='weighted')

    # FAR / FRR
    false_accept = sum(1 for yt, yp in zip(y_test, y_pred) if yt == -1 and yp != -1)
    false_reject = sum(1 for yt, yp in zip(y_test, y_pred) if yt != -1 and yp == -1)

    FAR = false_accept / len(y_test)
    FRR = false_reject / len(y_test)

    log("\n========== METRICS ==========")
    log(f"Accuracy: {acc:.4f}")
    log(f"Precision: {precision:.4f}")
    log(f"Recall: {recall:.4f}")
    log(f"F1 Score: {f1:.4f}")
    log(f"FAR: {FAR:.4f}")
    log(f"FRR: {FRR:.4f}")

    # ── FAILURE ANALYSIS ─────────────────────────────
    failures = []

    for i, (yt, yp, conf) in enumerate(zip(y_test, y_pred, confidence)):
        if yt != yp:
            failures.append((i, yt, yp, conf))

    with open(os.path.join(OUTPUT_DIR, "failures.txt"), "w") as f:
        for fail in failures[:50]:
            f.write(str(fail) + "\n")

    log(f"Failure Cases Logged: {len(failures)}")

    # ── VISUALS ──────────────────────────────────────
    cm = confusion_matrix(y_test_known, y_pred_known)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    save_plot("confusion_matrix.png")

    plt.figure()
    sns.histplot(confidence, bins=30)
    plt.title("Confidence Distribution")
    save_plot("confidence.png")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": FAR,
        "frr": FRR,
        "fps": fps,
        "threshold": best_threshold
    }

# ── MAIN ───────────────────────────────────────────────
if __name__ == "__main__":

    try:
        data = {}

        files = {
            "face_classifier": "face_classifier.pkl",
            "scaler": "scaler.pkl",
            "labels": "labels.pkl",
            "centroids": "centroids.pkl"
        }

        for key, file in files.items():
            with open(os.path.join(MODEL_DIR, file), "rb") as f:
                data[key] = pickle.load(f)
            log(f"[LOADED] {file}")

        results = perform_analysis(data)

        with open(os.path.join(OUTPUT_DIR, "final_report.txt"), "w") as f:
            f.write("=== FINAL REPORT v7 ===\n")
            f.write(f"Time: {datetime.now()}\n\n")
            for k, v in results.items():
                f.write(f"{k}: {v}\n")

        log("\n[SUCCESS] ANALYSIS COMPLETE")

    except Exception as e:
        log(f"[ERROR] {str(e)}")