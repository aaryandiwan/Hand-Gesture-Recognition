"""
data_collection.py — Collect hand gesture samples via webcam.

Usage
-----
    python data_collection.py

Controls
--------
    0–9 : Label the current hand pose as gesture class 0–9 and save a sample.
    q   : Quit the application.

Each sample is a 73-element feature vector (see utils.py) plus its label.
Samples are appended to  dataset/gesture_data.csv.
"""

import os
import csv
import cv2
import mediapipe as mp
from utils import extract_features, get_feature_names, GESTURE_NAMES

# ── Paths ────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
CSV_PATH = os.path.join(DATASET_DIR, "gesture_data.csv")

# ── MediaPipe Hands setup ────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def ensure_csv_header() -> None:
    """Create the CSV file with a header row if it doesn't already exist."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        header = get_feature_names() + ["label"]
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(header)


def count_samples_per_class() -> dict[str, int]:
    """Return a dict mapping each label to the number of collected samples."""
    counts: dict[str, int] = {}
    if not os.path.isfile(CSV_PATH):
        return counts
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                label = row[-1]
                counts[label] = counts.get(label, 0) + 1
    return counts


def save_sample(features, label: str) -> None:
    """Append one feature-vector + label row to the CSV."""
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(features) + [label])


def main() -> None:
    ensure_csv_header()
    counts = count_samples_per_class()

    # ── Open webcam ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Webcam opened. Press 0–9 to save a gesture sample. Press 'q' to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame.")
                break

            # Flip for a mirror-view experience
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR → RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True

                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # ── HUD: instructions & sample counts ────────────────────
            y_offset = 30
            cv2.putText(
                frame,
                "Press 0-9 to label gesture | 'q' to quit",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y_offset += 30

            if not hand_detected:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                y_offset += 30

            # Show per-class sample counts
            for label in sorted(counts.keys()):
                name = GESTURE_NAMES.get(label, label)
                text = f"{label} ({name}): {counts[label]} samples"
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                y_offset += 22

            # Set up fullscreen window
            window_name = "Hand Gesture Data Collection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, frame)

            # ── Keyboard input ───────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            # Quit on 'q' or ESC
            if key == ord("q") or key == 27:  # 27 = ESC key
                print("[INFO] Quitting data collection.")
                break

            # Save sample if a digit key 0–9 is pressed
            if ord("0") <= key <= ord("9") and hand_detected:
                label = chr(key)
                # Use the first detected hand
                features = extract_features(
                    results.multi_hand_landmarks[0].landmark
                )
                save_sample(features, label)
                counts[label] = counts.get(label, 0) + 1
                name = GESTURE_NAMES.get(label, label)
                print(f"[INFO] Saved sample for class {label} ({name})  "
                      f"(total: {counts[label]})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
