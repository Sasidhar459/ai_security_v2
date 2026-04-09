"""
Clear old runtime log history and/or analysis reports.

Usage:
    python clear_logs.py
    python clear_logs.py --dry-run
    python clear_logs.py --target logs
    python clear_logs.py --target reports
    python clear_logs.py --target both
    python clear_logs.py --include-labeled
"""

import argparse
import csv
from pathlib import Path


LOG_FILES = {
    Path("security.log"): None,
    Path("logs/retrain.log"): None,
    Path("logs/detection_log.csv"): [
        "timestamp",
        "frame_id",
        "face_id",
        "confidence",
        "true_label",
        "predicted_label",
        "processing_time",
        "top_prediction",
        "verify_reason",
        "centroid_distance",
        "nearest_centroid",
    ],
    Path("logs/performance_log.csv"): [
        "timestamp",
        "fps",
        "avg_latency_ms",
    ],
    Path("logs/frame_log.csv"): [
        "timestamp",
        "frame_id",
        "processed",
        "face_count",
        "predicted_labels",
        "true_label",
        "true_people",
        "processing_time",
    ],
}

OPTIONAL_LOG_FILES = {
    Path("logs/detection_log_labeled.csv"): [
        "timestamp",
        "frame_id",
        "face_id",
        "confidence",
        "true_label",
        "predicted_label",
        "processing_time",
        "top_prediction",
        "verify_reason",
        "centroid_distance",
        "nearest_centroid",
    ],
}

REPORT_PATHS = [
    Path("analysis_report/combined_analysis_report.txt"),
    Path("analysis_report/realtime_analysis_report.txt"),
    Path("analysis_report/training_evaluation_report.txt"),
    Path("analysis_report/training_confusion_matrix.csv"),
]

REPORT_DIRS = [
    Path("analysis_report/graphs"),
    Path("analysis_report/test_reports"),
]

LOG_DIRS = [
    Path("logs/test_runs"),
]


def clear_file(path: Path, header: list[str] | None, dry_run: bool):
    action = "Reset CSV" if header else "Clear log"
    if dry_run:
        print(f"[DRY RUN] {action}: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if header:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
    else:
        path.write_text("", encoding="utf-8")
    print(f"[OK] {action}: {path}")


def remove_path(path: Path, dry_run: bool):
    if not path.exists():
        return
    if dry_run:
        print(f"[DRY RUN] Remove report artifact: {path}")
        return

    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                remove_path(child, dry_run)
            else:
                child.unlink()
        path.rmdir()
    else:
        path.unlink()
    print(f"[OK] Removed report artifact: {path}")


def ask_target() -> str:
    print("What do you want to clean?")
    print("  1. Logs only")
    print("  2. Analysis reports only")
    print("  3. Both logs and reports")
    print("  4. Cancel")
    choice = input("Enter choice [1-4]: ").strip()
    return {
        "1": "logs",
        "2": "reports",
        "3": "both",
        "4": "cancel",
    }.get(choice, "cancel")


def clean_logs(include_labeled: bool, dry_run: bool):
    files = dict(LOG_FILES)
    if include_labeled:
        files.update(OPTIONAL_LOG_FILES)

    for path, header in files.items():
        clear_file(path, header, dry_run)
    for path in LOG_DIRS:
        remove_path(path, dry_run)


def clean_reports(dry_run: bool):
    for path in REPORT_PATHS:
        remove_path(path, dry_run)
    for path in REPORT_DIRS:
        remove_path(path, dry_run)


def main():
    parser = argparse.ArgumentParser(description="Clear old log history.")
    parser.add_argument(
        "--target",
        choices=["logs", "reports", "both"],
        default=None,
        help="What to clean. If omitted, the script asks interactively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be cleared without changing them.",
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="Also clear logs/detection_log_labeled.csv.",
    )
    args = parser.parse_args()

    target = args.target or ask_target()
    if target == "cancel":
        print("[CANCELLED] Nothing cleaned.")
        return

    if target in {"logs", "both"}:
        clean_logs(args.include_labeled, args.dry_run)
    if target in {"reports", "both"}:
        clean_reports(args.dry_run)


if __name__ == "__main__":
    main()
