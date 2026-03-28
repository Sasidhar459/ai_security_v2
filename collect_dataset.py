import cv2
import os
import time
from config import DATASET_DIR

# Updated angle phases and counts so each subject collects 1000 images total.
ANGLE_PHASES = [
    {
        "label": "Front - Neutral",
        "instruction": "Look straight at the camera with a neutral expression.",
        "count": 200,
    },
    {
        "label": "Front - Expressions",
        "instruction": "Look straight at the camera. Slowly change expressions: smile, frown, raise eyebrows.",
        "count": 200,
    },
    {
        "label": "Left Side",
        "instruction": "Slowly turn your head to the LEFT. Go from center to full left and back.",
        "count": 150,
    },
    {
        "label": "Right Side",
        "instruction": "Slowly turn your head to the RIGHT. Go from center to full right and back.",
        "count": 150,
    },
    {
        "label": "Up & Down Tilt",
        "instruction": "Tilt your head slowly UP then DOWN, like nodding gently.",
        "count": 150,
    },
    {
        "label": "Distance Variation",
        "instruction": "Move CLOSER to the camera, then slowly move FURTHER away. Repeat.",
        "count": 150,
    },
]

TOTAL_IMAGES = sum(p["count"] for p in ANGLE_PHASES)  # 1000


def draw_overlay(frame, phase_label, instruction, phase_idx, phase_total, count, phase_count, countdown=None):
    """Draw a clean HUD overlay on the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Phase label
    cv2.putText(frame, f"Phase {phase_idx + 1}/{phase_total}: {phase_label}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)

    # Image count
    cv2.putText(frame, f"Phase images: {count}/{phase_count}  |  Total: {sum(p['count'] for p in ANGLE_PHASES[:phase_idx]) + count}/{TOTAL_IMAGES}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Instruction box at the bottom
    box_h = 70
    cv2.rectangle(frame, (0, h - box_h), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "Instruction:", (10, h - box_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    # Word-wrap instruction across up to 2 lines
    words = instruction.split()
    lines, line = [], ""
    for word in words:
        if len(line + word) < 72:
            line += word + " "
        else:
            lines.append(line.strip())
            line = word + " "
    lines.append(line.strip())
    for i, l in enumerate(lines[:2]):
        cv2.putText(frame, l, (10, h - box_h + 44 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # Countdown overlay
    if countdown is not None:
        cv2.putText(frame, f"Starting in {countdown}...",
                    (w // 2 - 130, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 80, 255), 3)

    return frame


def countdown_screen(cap, phase_label, instruction, phase_idx, phase_total, phase_count, seconds=4):
    """Show a live countdown before each phase begins."""
    print(f"\n{'='*60}")
    print(f"  Phase {phase_idx + 1}/{phase_total}: {phase_label}")
    print(f"  Instruction: {instruction}")
    print(f"  Starting in {seconds} seconds... (Press 'q' to quit)")
    print(f"{'='*60}")

    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        remaining = max(0, seconds - int(time.time() - start))
        frame = draw_overlay(frame, phase_label, instruction,
                             phase_idx, phase_total, 0, phase_count, countdown=remaining)
        cv2.imshow("Face Data Collection", frame)
        if remaining == 0:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False  # Signal to quit
    return True


def collect_person(name):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("\n" + "="*60)
    print(f"  Face Dataset Collection for YOLOv8")
    print(f"  Subject : {name}")
    print(f"  Total   : {TOTAL_IMAGES} images across {len(ANGLE_PHASES)} angle phases")
    print("="*60)
    print("  Controls: 'q' = quit early | any key = skip countdown")
    print("="*60 + "\n")

    total_collected = 0

    for phase_idx, phase in enumerate(ANGLE_PHASES):
        label       = phase["label"]
        instruction = phase["instruction"]
        phase_count = phase["count"]

        # --- Countdown ---
        proceed = countdown_screen(cap, label, instruction,
                                   phase_idx, len(ANGLE_PHASES), phase_count)
        if not proceed:
            break

        # --- Capture loop ---
        count = 0
        while count < phase_count:
            ret, frame = cap.read()
            if not ret:
                break

            # Save image
            img_name = f"{name}_p{phase_idx}_{count:04d}.jpg"
            cv2.imwrite(os.path.join(person_dir, img_name), frame)
            count += 1
            total_collected += 1

            # Draw HUD
            display = draw_overlay(frame.copy(), label, instruction,
                                   phase_idx, len(ANGLE_PHASES),
                                   count, phase_count)
            cv2.imshow("Face Data Collection", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Collection interrupted by user.")
                cap.release()
                cv2.destroyAllWindows()
                print(f"\n[DONE] Collected {total_collected} images saved to: {person_dir}")
                return

        print(f"  [✓] Phase '{label}' complete — {count} images captured.")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print(f"  Collection complete!")
    print(f"  Subject : {name}")
    print(f"  Images  : {total_collected} saved to '{person_dir}'")
    print("="*60)
    print("\n  Next steps for YOLOv8 training:")
    print("  1. Annotate images with a tool like LabelImg or Roboflow")
    print("  2. Export labels in YOLO format (class x_center y_center w h)")
    print("  3. Split into train/val sets (80/20 recommended)")
    print("  4. Train: yolo detect train data=data.yaml model=yolov8n.pt epochs=50")


if __name__ == "__main__":
    name = input("Enter person's name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
    else:
        collect_person(name)
