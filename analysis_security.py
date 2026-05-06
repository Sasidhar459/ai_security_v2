"""
Analyze training and real-time security-system results.

Examples:
    python analysis_security.py --mode training
    python analysis_security.py --mode realtime
    python analysis_security.py --mode realtime --test-name "Sasidhar, Mohan"
    python analysis_security.py --mode realtime --true-people "Sasidhar, Mohan"
    python analysis_security.py --mode realtime --no-label-prompt
    python analysis_security.py --mode both
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean


DEFAULT_TRAINING_REPORT = Path("analysis_report/training_evaluation_report.txt")
DEFAULT_TRAINING_MATRIX = Path("analysis_report/training_confusion_matrix.csv")
DEFAULT_DETECTION_LOG = Path("logs/detection_log.csv")
DEFAULT_PERFORMANCE_LOG = Path("logs/performance_log.csv")
DEFAULT_FRAME_LOG = Path("logs/frame_log.csv")
DEFAULT_OUTPUT = Path("analysis_report/combined_analysis_report.txt")
DEFAULT_GRAPH_DIR = Path("analysis_report/graphs")
DEFAULT_TEST_RUNS_DIR = Path("logs/test_runs")
DEFAULT_TEST_REPORTS_DIR = Path("analysis_report/test_reports")

UNKNOWN_PREDICTIONS = {
    "",
    "?",
    "unknown",
    "checking",
    "checking...",
    "none",
}
UNAUTHORIZED_TRUE_LABELS = {
    "unauthorized",
    "unknown",
    "intruder",
}


def slugify_test_name(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned or cleaned == "?":
        cleaned = "unlabeled"
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unlabeled"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latest_test_run_dir(test_runs_dir: Path = DEFAULT_TEST_RUNS_DIR) -> Path | None:
    if not test_runs_dir.exists():
        return None
    dirs = [path for path in test_runs_dir.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.stat().st_mtime)


def find_test_run_dir(test_name: str,
                      test_runs_dir: Path = DEFAULT_TEST_RUNS_DIR) -> Path | None:
    if not test_name:
        return None

    direct = test_runs_dir / test_name
    if direct.is_dir():
        return direct

    slug = slugify_test_name(test_name)
    direct = test_runs_dir / slug
    if direct.is_dir():
        return direct

    if not test_runs_dir.exists():
        return None

    matches = [
        path for path in test_runs_dir.iterdir()
        if path.is_dir() and path.name.startswith(slug)
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def first_matching_file(directory: Path, pattern: str, fallback_name: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if matches:
        return matches[0]
    return directory / fallback_name


def realtime_paths_from_test_dir(test_dir: Path) -> tuple[Path, Path, Path]:
    return (
        first_matching_file(test_dir, "detection_log_*.csv", "detection_log.csv"),
        first_matching_file(test_dir, "performance_log_*.csv", "performance_log.csv"),
        first_matching_file(test_dir, "frame_log_*.csv", "frame_log.csv"),
    )


def default_test_output(mode: str, test_id: str) -> Path:
    report_name = "realtime_analysis_report" if mode == "realtime" else "combined_analysis_report"
    return DEFAULT_TEST_REPORTS_DIR / test_id / f"{report_name}_{test_id}.txt"


def resolve_realtime_artifacts(args):
    test_dir = args.test_dir
    if test_dir is None and args.test_name:
        test_dir = find_test_run_dir(args.test_name)
        if test_dir is None:
            print(f"[WARN] Test run not found for '{args.test_name}'. Falling back to explicit/default log paths.")

    explicit_logs = any([args.detection_log, args.performance_log, args.frame_log])
    if test_dir is None and not explicit_logs:
        test_dir = latest_test_run_dir()
        if test_dir:
            print(f"[INFO] Using latest test run logs: {test_dir}")

    if test_dir:
        detection_log, performance_log, frame_log = realtime_paths_from_test_dir(test_dir)
        test_id = test_dir.name
    else:
        detection_log = args.detection_log or DEFAULT_DETECTION_LOG
        performance_log = args.performance_log or DEFAULT_PERFORMANCE_LOG
        frame_log = args.frame_log or DEFAULT_FRAME_LOG
        if detection_log == DEFAULT_DETECTION_LOG:
            test_id = "legacy_logs"
        else:
            test_id = slugify_test_name(args.true_people or detection_log.parent.name or "default")

    output = args.output
    graph_dir = args.graph_dir
    if args.mode in {"realtime", "both"} and output is None:
        output = default_test_output(args.mode, test_id)
    if args.mode in {"realtime", "both"} and graph_dir is None:
        graph_dir = DEFAULT_TEST_REPORTS_DIR / test_id / "graphs"

    if output is None:
        output = DEFAULT_OUTPUT
    if graph_dir is None:
        graph_dir = DEFAULT_GRAPH_DIR

    return detection_log, performance_log, frame_log, output, graph_dir, test_id


def safe_float(value: str, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def parse_training_matrix(path: Path):
    if not path.exists():
        return None

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2 or len(rows[0]) < 2:
        return None

    labels = rows[0][1:]
    matrix = []
    for row in rows[1:]:
        matrix.append([int(value) for value in row[1:]])

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0]))))
    accuracy = correct / total if total else None
    return labels, matrix, accuracy, total, correct


def parse_training_metrics(report_path: Path) -> dict[str, float]:
    metrics = {}
    if not report_path.exists():
        return metrics

    metric_names = {
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-score",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1-score",
    }
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key in metric_names:
            parsed = safe_float(value)
            if parsed is not None:
                metrics[key] = parsed
    return metrics


def summarize_training(report_path: Path, matrix_path: Path) -> list[str]:
    lines = [
        "Training Data Analysis",
        "======================",
    ]

    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        lines.append(f"Source report: {report_path}")
        wanted_prefixes = (
            "Total embeddings",
            "Classes",
            "Train samples",
            "Validation samples",
            "Best params",
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1-score",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1-score",
        )
        for raw_line in report_text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith(wanted_prefixes):
                lines.append(stripped)
    else:
        lines.append(f"Training report not found: {report_path}")

    matrix_summary = parse_training_matrix(matrix_path)
    if matrix_summary:
        labels, _matrix, accuracy, total, correct = matrix_summary
        lines.append(f"Confusion matrix: {matrix_path}")
        lines.append(f"Matrix classes: {len(labels)}")
        lines.append(f"Matrix samples: {total}")
        lines.append(f"Matrix correct: {correct}")
        lines.append(f"Matrix accuracy: {format_number(accuracy)}")
    else:
        lines.append(f"Training confusion matrix not found or invalid: {matrix_path}")

    return lines


def is_authorized_prediction(label: str) -> bool:
    normalized = label.strip().lower()
    if normalized in UNKNOWN_PREDICTIONS:
        return False
    if normalized.startswith("intruder") or "intruder" in normalized:
        return False
    return True


def is_authorized_true_label(label: str) -> bool:
    normalized = label.strip().lower()
    if normalized in UNAUTHORIZED_TRUE_LABELS:
        return False
    return True


def normalize_people_label(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "?"
    if cleaned.lower() in {"none", "unknown", "intruder", "unauthorized"}:
        return "unauthorized"
    if "," in cleaned:
        return "authorized"
    return cleaned


def parse_people_names(value: str) -> list[str]:
    normalized = value.strip()
    if normalized.lower() in {"", "?", "authorized", "unauthorized", "unknown", "none", "intruder"}:
        return []
    return [name.strip() for name in normalized.split(",") if name.strip()]


def collect_ground_truth_people(detection_rows: list[dict[str, str]],
                                frame_rows: list[dict[str, str]]) -> list[str]:
    people = Counter()
    for row in detection_rows + frame_rows:
        for name in parse_people_names(row.get("true_people", "")):
            people[name] += 1
    return [name for name, _count in people.most_common()]


def is_labeled_row(row: dict[str, str]) -> bool:
    return row.get("true_label", "").strip() not in {"", "?"}


def is_known_test_row(row: dict[str, str]) -> bool:
    return is_labeled_row(row) and is_authorized_true_label(row.get("true_label", ""))


def is_unknown_test_row(row: dict[str, str]) -> bool:
    return is_labeled_row(row) and not is_authorized_true_label(row.get("true_label", ""))


def prediction_matches_expected_people(predicted_label: str,
                                       expected_people: list[str]) -> bool:
    normalized = predicted_label.strip().lower()
    return normalized in {name.lower() for name in expected_people}


def binary_metrics(labeled_rows: list[dict[str, str]]):
    tp = tn = fp = fn = 0
    for row in labeled_rows:
        true_authorized = is_authorized_true_label(row.get("true_label", ""))
        pred_authorized = is_authorized_prediction(row.get("predicted_label", ""))

        if true_authorized and pred_authorized:
            tp += 1
        elif not true_authorized and not pred_authorized:
            tn += 1
        elif not true_authorized and pred_authorized:
            fp += 1
        elif true_authorized and not pred_authorized:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "FAR": far,
        "FRR": frr,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def realtime_metric_values(detection_rows: list[dict[str, str]]) -> dict[str, float]:
    labeled_rows = [
        row for row in detection_rows
        if row.get("true_label", "").strip() not in {"", "?"}
    ]
    if not labeled_rows:
        return {}
    metrics = binary_metrics(labeled_rows)
    return {
        key: value for key, value in metrics.items()
        if key in {"Accuracy", "Precision", "Recall", "F1-score", "FAR", "FRR"}
        and value is not None
    }


def identity_metrics(labeled_rows: list[dict[str, str]]):
    checked = 0
    correct = 0
    for row in labeled_rows:
        true_label = row.get("true_label", "").strip()
        predicted = row.get("predicted_label", "").strip()
        if not true_label or true_label == "?" or "," in true_label:
            continue
        if true_label.lower() in UNAUTHORIZED_TRUE_LABELS or true_label.lower() == "authorized":
            continue
        checked += 1
        if predicted.lower() == true_label.lower():
            correct += 1
    return checked, correct, (correct / checked if checked else None)


def known_unknown_metric_values(detection_rows: list[dict[str, str]],
                                frame_rows: list[dict[str, str]]) -> dict[str, float]:
    expected_people = collect_ground_truth_people(detection_rows, frame_rows)
    known_rows = [row for row in detection_rows if is_known_test_row(row)]
    unknown_rows = [row for row in detection_rows if is_unknown_test_row(row)]
    metrics = {}

    if known_rows:
        known_accepted = [
            row for row in known_rows
            if is_authorized_prediction(row.get("predicted_label", ""))
        ]
        known_rejected = len(known_rows) - len(known_accepted)
        metrics["Known acceptance"] = len(known_accepted) / len(known_rows)
        metrics["FRR"] = known_rejected / len(known_rows)
        if expected_people:
            correct_identity = [
                row for row in known_accepted
                if prediction_matches_expected_people(
                    row.get("predicted_label", ""),
                    expected_people,
                )
            ]
            metrics["Known identity"] = len(correct_identity) / len(known_rows)

    if unknown_rows:
        unknown_rejected = [
            row for row in unknown_rows
            if not is_authorized_prediction(row.get("predicted_label", ""))
        ]
        false_accepts = len(unknown_rows) - len(unknown_rejected)
        metrics["Unknown rejection"] = len(unknown_rejected) / len(unknown_rows)
        metrics["FAR"] = false_accepts / len(unknown_rows)

    return metrics


def current_rescue_config() -> tuple[float, float] | None:
    try:
        from config import CENTROID_RESCUE_CONFIDENCE, CENTROID_RESCUE_THRESHOLD
    except Exception:
        return None
    return float(CENTROID_RESCUE_CONFIDENCE), float(CENTROID_RESCUE_THRESHOLD)


def projected_label_with_current_gate(row: dict[str, str]) -> str:
    predicted = row.get("predicted_label", "").strip()
    if is_authorized_prediction(predicted):
        return predicted

    rescue_config = current_rescue_config()
    if rescue_config is None:
        return predicted

    rescue_confidence, rescue_threshold = rescue_config
    top_prediction = row.get("top_prediction", "").strip()
    nearest_centroid = row.get("nearest_centroid", "").strip()
    confidence = safe_float(row.get("confidence"))
    distance = safe_float(row.get("centroid_distance"))

    if (
        top_prediction
        and top_prediction == nearest_centroid
        and confidence is not None
        and distance is not None
        and confidence >= rescue_confidence
        and distance <= rescue_threshold
    ):
        return top_prediction
    return predicted


def append_current_gate_projection(lines: list[str],
                                   known_rows: list[dict[str, str]],
                                   expected_people: list[str]):
    rescue_config = current_rescue_config()
    if not known_rows or rescue_config is None:
        return

    rescue_confidence, rescue_threshold = rescue_config
    projected_labels = [projected_label_with_current_gate(row) for row in known_rows]
    projected_accepted = [
        label for label in projected_labels
        if is_authorized_prediction(label)
    ]
    projected_acceptance = len(projected_accepted) / len(known_rows)

    lines.append("    Current gate tuning estimate for next run:")
    lines.append(
        "      Rescue gate: "
        f"prob >= {rescue_confidence:.2f}, centroid distance <= {rescue_threshold:.2f}, "
        "classifier == nearest centroid"
    )
    lines.append(
        "      Projected known acceptance accuracy: "
        f"{format_number(projected_acceptance)} ({len(projected_accepted)}/{len(known_rows)})"
    )

    if expected_people:
        correct_identity = [
            label for label in projected_accepted
            if prediction_matches_expected_people(label, expected_people)
        ]
        projected_identity = len(correct_identity) / len(known_rows)
        lines.append(
            "      Projected expected-name identity accuracy: "
            f"{format_number(projected_identity)} ({len(correct_identity)}/{len(known_rows)})"
        )


def append_known_unknown_summary(lines: list[str],
                                 detection_rows: list[dict[str, str]],
                                 frame_rows: list[dict[str, str]]):
    expected_people = collect_ground_truth_people(detection_rows, frame_rows)
    known_rows = [row for row in detection_rows if is_known_test_row(row)]
    unknown_rows = [row for row in detection_rows if is_unknown_test_row(row)]

    lines.append("Known vs Unknown Test Split:")
    if expected_people:
        lines.append(f"  Expected known people: {', '.join(expected_people)}")
    else:
        lines.append("  Expected known people: not available in logs")

    if known_rows:
        known_accepted = [
            row for row in known_rows
            if is_authorized_prediction(row.get("predicted_label", ""))
        ]
        known_rejected = [
            row for row in known_rows
            if not is_authorized_prediction(row.get("predicted_label", ""))
        ]
        acceptance_rate = len(known_accepted) / len(known_rows)
        frr = len(known_rejected) / len(known_rows)

        lines.append("  Known-person test:")
        lines.append(f"    Rows: {len(known_rows)}")
        lines.append(f"    Accepted as known: {len(known_accepted)}")
        lines.append(f"    Rejected as unknown/checking: {len(known_rejected)}")
        lines.append(f"    Known acceptance accuracy: {format_number(acceptance_rate)}")
        lines.append(f"    FRR: {format_number(frr)}")

        if expected_people:
            correct_identity = [
                row for row in known_accepted
                if prediction_matches_expected_people(
                    row.get("predicted_label", ""),
                    expected_people,
                )
            ]
            wrong_identity = [
                row for row in known_accepted
                if not prediction_matches_expected_people(
                    row.get("predicted_label", ""),
                    expected_people,
                )
            ]
            identity_total_rate = len(correct_identity) / len(known_rows)
            identity_accepted_rate = (
                len(correct_identity) / len(known_accepted)
                if known_accepted else None
            )
            lines.append(
                "    Expected-name identity accuracy: "
                f"{format_number(identity_total_rate)} ({len(correct_identity)}/{len(known_rows)})"
            )
            lines.append(
                "    Identity accuracy among accepted known rows: "
                f"{format_number(identity_accepted_rate)} ({len(correct_identity)}/{len(known_accepted)})"
            )
            if wrong_identity:
                wrong_counts = Counter(row.get("predicted_label", "") for row in wrong_identity)
                lines.append("    Wrong accepted identities:")
                for label, count in wrong_counts.most_common(5):
                    lines.append(f"      {label or 'blank'}: {count}")
        else:
            lines.append("    Identity accuracy skipped because true_people is missing.")

        if known_rejected:
            reason_counts = Counter(row.get("verify_reason", "") for row in known_rejected)
            top_counts = Counter(row.get("top_prediction", "") for row in known_rejected)
            nearest_counts = Counter(row.get("nearest_centroid", "") for row in known_rejected)
            lines.append("    Top rejection reasons:")
            for reason, count in reason_counts.most_common(5):
                lines.append(f"      {reason or 'blank'}: {count}")
            lines.append("    Top rejected classifier predictions:")
            for label, count in top_counts.most_common(5):
                lines.append(f"      {label or 'blank'}: {count}")
            lines.append("    Top rejected nearest centroids:")
            for label, count in nearest_counts.most_common(5):
                lines.append(f"      {label or 'blank'}: {count}")
        append_current_gate_projection(lines, known_rows, expected_people)
    else:
        lines.append("  Known-person test: no labeled known rows")

    if unknown_rows:
        unknown_rejected = [
            row for row in unknown_rows
            if not is_authorized_prediction(row.get("predicted_label", ""))
        ]
        false_accepts = [
            row for row in unknown_rows
            if is_authorized_prediction(row.get("predicted_label", ""))
        ]
        rejection_rate = len(unknown_rejected) / len(unknown_rows)
        far = len(false_accepts) / len(unknown_rows)
        lines.append("  Unknown/intruder test:")
        lines.append(f"    Rows: {len(unknown_rows)}")
        lines.append(f"    Correctly rejected: {len(unknown_rejected)}")
        lines.append(f"    False accepted as known: {len(false_accepts)}")
        lines.append(f"    Unknown rejection accuracy: {format_number(rejection_rate)}")
        lines.append(f"    FAR: {format_number(far)}")
        if false_accepts:
            false_accept_counts = Counter(row.get("predicted_label", "") for row in false_accepts)
            lines.append("    False accepted identities:")
            for label, count in false_accept_counts.most_common(5):
                lines.append(f"      {label or 'blank'}: {count}")
    else:
        lines.append("  Unknown/intruder test: no labeled unknown rows")


def apply_realtime_labels(detection_path: Path, frame_path: Path,
                          people: str | None = None,
                          keep_existing: bool = False):
    if people is None:
        people = input(
            "Who was present in front of the camera during this real-time test? "
            "Use names separated by comma, type unauthorized/none, or press Enter to skip: "
        ).strip()
    if not people or not people.strip():
        print("[INFO] No real-time ground-truth label entered; logs left unchanged.")
        return

    people = people.strip()
    true_label = normalize_people_label(people)
    true_people = people

    detection_rows = read_csv_rows(detection_path)
    if detection_rows:
        fields = list(detection_rows[0].keys())
        if "true_label" not in fields:
            fields.append("true_label")
        updated = 0
        for row in detection_rows:
            if not keep_existing or row.get("true_label", "").strip() in {"", "?"}:
                row["true_label"] = true_label
                updated += 1
        write_csv_rows(detection_path, detection_rows, fields)
        print(f"[OK] Updated {updated} detection labels in {detection_path}")

    frame_rows = read_csv_rows(frame_path)
    if frame_rows:
        fields = list(frame_rows[0].keys())
        if "true_label" not in fields:
            fields.append("true_label")
        if "true_people" not in fields:
            fields.append("true_people")
        updated = 0
        for row in frame_rows:
            if not keep_existing or row.get("true_label", "").strip() in {"", "?"}:
                row["true_label"] = true_label
                updated += 1
            if not keep_existing or row.get("true_people", "").strip() in {"", "?"}:
                row["true_people"] = true_people
        write_csv_rows(frame_path, frame_rows, fields)
        print(f"[OK] Updated {updated} frame labels in {frame_path}")


def summarize_realtime(detection_path: Path, performance_path: Path,
                       frame_path: Path) -> list[str]:
    detection_rows = read_csv_rows(detection_path)
    performance_rows = read_csv_rows(performance_path)
    frame_rows = read_csv_rows(frame_path)

    lines = [
        "Real-Time Data Analysis",
        "=======================",
        f"Detection log: {detection_path}",
        f"Performance log: {performance_path}",
        f"Frame log: {frame_path}",
        f"Detection rows: {len(detection_rows)}",
        f"Performance rows: {len(performance_rows)}",
        f"Frame rows: {len(frame_rows)}",
    ]

    if detection_rows:
        confidences = [
            value for value in
            (safe_float(row.get("confidence")) for row in detection_rows)
            if value is not None and value >= 0
        ]
        latencies = [
            value for value in
            (safe_float(row.get("processing_time")) for row in detection_rows)
            if value is not None
        ]
        predictions = Counter(row.get("predicted_label", "") for row in detection_rows)
        intruder_rows = [
            row for row in detection_rows
            if not is_authorized_prediction(row.get("predicted_label", ""))
        ]

        lines.append(f"Average confidence: {format_number(mean(confidences) if confidences else None)}")
        lines.append(f"Minimum confidence: {format_number(min(confidences) if confidences else None)}")
        lines.append(f"Maximum confidence: {format_number(max(confidences) if confidences else None)}")
        lines.append(f"Average latency ms: {format_number(mean(latencies) if latencies else None, 2)}")
        lines.append(f"Intruder/unknown decisions: {len(intruder_rows)}")
        lines.append("Top predictions:")
        for label, count in predictions.most_common(10):
            lines.append(f"  {label or 'blank'}: {count}")

        labeled_rows = [
            row for row in detection_rows
            if row.get("true_label", "").strip() not in {"", "?"}
        ]
        lines.append(f"Labeled detection rows: {len(labeled_rows)}")
        if labeled_rows:
            metrics = binary_metrics(labeled_rows)
            lines.append("Binary authorized/unauthorized metrics:")
            for key in ["Accuracy", "Precision", "Recall", "F1-score", "FAR", "FRR"]:
                lines.append(f"  {key}: {format_number(metrics[key])}")
            lines.append(
                f"  TP={metrics['TP']} TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']}"
            )
            checked, correct, identity_accuracy = identity_metrics(labeled_rows)
            if checked:
                lines.append(
                    "Single-person identity accuracy: "
                    f"{format_number(identity_accuracy)} ({correct}/{checked})"
                )
            else:
                lines.append("Single-person identity accuracy skipped for this multi-person/binary test.")
            append_known_unknown_summary(lines, labeled_rows, frame_rows)
        else:
            lines.append(
                "Accuracy/FAR/FRR skipped because true_label is missing or '?' in the real-time log."
            )

    if frame_rows:
        processed_frames = [
            row for row in frame_rows
            if row.get("processed", "").strip().lower() in {"1", "true", "yes"}
        ]
        frame_latencies = [
            value for value in
            (safe_float(row.get("processing_time")) for row in frame_rows)
            if value is not None
        ]
        face_counts = [
            int(value) for value in
            (row.get("face_count", "0") for row in frame_rows)
            if str(value).strip().isdigit()
        ]
        true_people = Counter(
            row.get("true_people", "").strip()
            for row in frame_rows
            if row.get("true_people", "").strip() not in {"", "?"}
        )

        lines.append("Frame-level summary:")
        lines.append(f"  Total frames logged: {len(frame_rows)}")
        lines.append(f"  Processed frames: {len(processed_frames)}")
        lines.append(f"  Skipped frames: {len(frame_rows) - len(processed_frames)}")
        lines.append(f"  Frames with faces: {sum(1 for count in face_counts if count > 0)}")
        lines.append(f"  Average faces/frame: {format_number(mean(face_counts) if face_counts else None, 2)}")
        lines.append(f"  Average frame log latency ms: {format_number(mean(frame_latencies) if frame_latencies else None, 2)}")
        if true_people:
            lines.append("Ground-truth people recorded:")
            for label, count in true_people.most_common(5):
                lines.append(f"  {label}: {count} frames")

    if performance_rows:
        fps_values = [
            value for value in
            (safe_float(row.get("fps")) for row in performance_rows)
            if value is not None
        ]
        latency_values = [
            value for value in
            (safe_float(row.get("avg_latency_ms")) for row in performance_rows)
            if value is not None
        ]
        lines.append(f"Average FPS: {format_number(mean(fps_values) if fps_values else None, 2)}")
        lines.append(f"Minimum FPS: {format_number(min(fps_values) if fps_values else None, 2)}")
        lines.append(f"Maximum FPS: {format_number(max(fps_values) if fps_values else None, 2)}")
        lines.append(
            f"Average frame latency ms: {format_number(mean(latency_values) if latency_values else None, 2)}"
        )

    return lines


def write_report(lines: list[str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved analysis report: {output_path}")


def load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"[WARN] Graph generation skipped: matplotlib is not available ({exc})")
        return None


def save_bar_chart(plt, labels: list[str], values: list[float], title: str,
                   ylabel: str, output_path: Path, ylim: tuple[float, float] | None = None):
    if not values:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#1f77b4")
    plt.title(title)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(*ylim)
    plt.xticks(rotation=30, ha="right")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_line_chart(plt, values: list[float], title: str, ylabel: str,
                    output_path: Path):
    if not values:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(values) + 1), values, linewidth=1.8)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_dual_line_chart(plt, first: list[float], second: list[float],
                         first_label: str, second_label: str, title: str,
                         ylabel: str, output_path: Path):
    if not first and not second:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    if first:
        plt.plot(range(1, len(first) + 1), first, label=first_label, linewidth=1.8)
    if second:
        plt.plot(range(1, len(second) + 1), second, label=second_label, linewidth=1.8)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_confusion_matrix_chart(plt, matrix_summary, output_path: Path):
    if not matrix_summary:
        return None
    labels, matrix, _accuracy, _total, _correct = matrix_summary
    if not matrix:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Training Confusion Matrix")
    plt.colorbar()
    if len(labels) <= 25:
        ticks = range(len(labels))
        plt.xticks(ticks, labels, rotation=90, fontsize=7)
        plt.yticks(ticks, labels, fontsize=7)
    else:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"Predicted label ({len(labels)} classes)")
        plt.ylabel(f"True label ({len(labels)} classes)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_training_graphs(report_path: Path, matrix_path: Path,
                             graph_dir: Path) -> list[Path]:
    plt = load_matplotlib()
    if plt is None:
        return []

    generated = []
    metrics = parse_training_metrics(report_path)
    if metrics:
        labels = list(metrics.keys())
        values = [metrics[label] for label in labels]
        path = save_bar_chart(
            plt,
            labels,
            values,
            "Training Evaluation Metrics",
            "Score",
            graph_dir / "training_metrics.png",
            ylim=(0, 1.05),
        )
        if path:
            generated.append(path)

    path = save_confusion_matrix_chart(
        plt,
        parse_training_matrix(matrix_path),
        graph_dir / "training_confusion_matrix.png",
    )
    if path:
        generated.append(path)
    return generated


def generate_realtime_graphs(detection_path: Path, performance_path: Path,
                             frame_path: Path, graph_dir: Path) -> list[Path]:
    plt = load_matplotlib()
    if plt is None:
        return []

    detection_rows = read_csv_rows(detection_path)
    performance_rows = read_csv_rows(performance_path)
    frame_rows = read_csv_rows(frame_path)
    generated = []

    realtime_metrics = realtime_metric_values(detection_rows)
    if realtime_metrics:
        labels = list(realtime_metrics.keys())
        values = [realtime_metrics[label] for label in labels]
        path = save_bar_chart(
            plt,
            labels,
            values,
            "Real-Time Accuracy, FAR, and FRR",
            "Score",
            graph_dir / "realtime_metrics.png",
            ylim=(0, 1.05),
        )
        if path:
            generated.append(path)

    split_metrics = known_unknown_metric_values(detection_rows, frame_rows)
    if split_metrics:
        labels = list(split_metrics.keys())
        values = [split_metrics[label] for label in labels]
        path = save_bar_chart(
            plt,
            labels,
            values,
            "Known vs Unknown Real-Time Metrics",
            "Score",
            graph_dir / "realtime_known_unknown_metrics.png",
            ylim=(0, 1.05),
        )
        if path:
            generated.append(path)

    confidences = [
        value for value in
        (safe_float(row.get("confidence")) for row in detection_rows)
        if value is not None and value >= 0
    ]
    path = save_line_chart(
        plt,
        confidences,
        "Real-Time Confidence by Detection",
        "Confidence",
        graph_dir / "realtime_confidence.png",
    )
    if path:
        generated.append(path)

    if confidences:
        graph_dir.mkdir(parents=True, exist_ok=True)
        output_path = graph_dir / "realtime_confidence_distribution.png"
        plt.figure(figsize=(10, 5))
        plt.hist(confidences, bins=20, color="#2ca02c", edgecolor="black")
        plt.title("Real-Time Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        generated.append(output_path)

    predictions = Counter(row.get("predicted_label", "") or "blank" for row in detection_rows)
    if predictions:
        top_predictions = predictions.most_common(10)
        path = save_bar_chart(
            plt,
            [item[0] for item in top_predictions],
            [float(item[1]) for item in top_predictions],
            "Top Real-Time Predictions",
            "Count",
            graph_dir / "realtime_predictions.png",
        )
        if path:
            generated.append(path)

    fps_values = [
        value for value in
        (safe_float(row.get("fps")) for row in performance_rows)
        if value is not None
    ]
    latency_values = [
        value for value in
        (safe_float(row.get("avg_latency_ms")) for row in performance_rows)
        if value is not None
    ]
    path = save_dual_line_chart(
        plt,
        fps_values,
        latency_values,
        "FPS",
        "Average latency ms",
        "Real-Time FPS and Latency",
        "Value",
        graph_dir / "realtime_fps_latency.png",
    )
    if path:
        generated.append(path)

    face_counts = [
        int(value) for value in
        (row.get("face_count", "0") for row in frame_rows)
        if str(value).strip().isdigit()
    ]
    path = save_line_chart(
        plt,
        face_counts,
        "Faces Detected per Frame",
        "Face count",
        graph_dir / "realtime_faces_per_frame.png",
    )
    if path:
        generated.append(path)

    return generated


def append_graphs_to_report(output_path: Path, generated: list[Path]):
    if not generated:
        return
    with output_path.open("a", encoding="utf-8") as f:
        f.write("\nGenerated Graphs\n")
        f.write("================\n")
        for path in generated:
            f.write(f"{path}\n")
    print("\nGenerated graphs:")
    for path in generated:
        print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze training and real-time security data.")
    parser.add_argument("--mode", choices=["training", "realtime", "both"], default="both")
    parser.add_argument("--training-report", type=Path, default=DEFAULT_TRAINING_REPORT)
    parser.add_argument("--training-matrix", type=Path, default=DEFAULT_TRAINING_MATRIX)
    parser.add_argument("--detection-log", type=Path, default=None)
    parser.add_argument("--performance-log", type=Path, default=None)
    parser.add_argument("--frame-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument(
        "--test-name",
        default=None,
        help="Analyze a named test run from logs/test_runs. Partial names use the newest matching run.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Analyze a specific test-run directory containing detection/performance/frame logs.",
    )
    parser.add_argument("--no-graphs", action="store_true", help="Skip PNG graph generation.")
    parser.add_argument(
        "--label-realtime",
        action="store_true",
        help="Ask who was present and fill missing real-time true_label fields before analysis. This is now automatic for realtime/both modes unless --no-label-prompt is used.",
    )
    parser.add_argument(
        "--true-people",
        default=None,
        help="Names present during the real-time recording. Avoids the prompt for realtime/both analysis.",
    )
    parser.add_argument(
        "--no-label-prompt",
        action="store_true",
        help="Do not ask who was present before realtime/both analysis.",
    )
    parser.add_argument(
        "--keep-existing-labels",
        action="store_true",
        help="Only fill blank/'?' realtime labels instead of replacing labels with the prompted ground truth.",
    )
    args = parser.parse_args()

    detection_log, performance_log, frame_log, output, graph_dir, test_id = resolve_realtime_artifacts(args)

    should_label_realtime = (
        args.mode in {"realtime", "both"}
        and (args.true_people is not None or not args.no_label_prompt)
    ) or args.label_realtime
    if should_label_realtime:
        apply_realtime_labels(
            detection_log,
            frame_log,
            args.true_people,
            keep_existing=args.keep_existing_labels,
        )

    report_lines = []
    if args.mode in {"training", "both"}:
        report_lines.extend(summarize_training(args.training_report, args.training_matrix))
    if args.mode == "both":
        report_lines.append("")
    if args.mode in {"realtime", "both"}:
        report_lines.extend(summarize_realtime(
            detection_log,
            performance_log,
            frame_log,
        ))

    write_report(report_lines, output)

    if not args.no_graphs:
        generated = []
        if args.mode in {"training", "both"}:
            generated.extend(generate_training_graphs(
                args.training_report,
                args.training_matrix,
                graph_dir,
            ))
        if args.mode in {"realtime", "both"}:
            generated.extend(generate_realtime_graphs(
                detection_log,
                performance_log,
                frame_log,
                graph_dir,
            ))
        append_graphs_to_report(output, generated)


if __name__ == "__main__":
    main()
