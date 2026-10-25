
from pathlib import Path
import csv
import json

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------- SETTINGS ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model" / "gesture_model.pkl"
TEST_FILE = BASE_DIR / "data" / "hand_gestures_test.csv"
METRICS_FILE = BASE_DIR / "model" / "evaluation_metrics.json"

TARGET_SAMPLES = 60
SAMPLE_EVERY_N_FRAMES = 5

GESTURES = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "peace",
    ord("4"): "thumbs_up",
    ord("5"): "pointing",
}

# ---------------- LOAD TRAINED MODEL ----------------

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

saved_data = joblib.load(MODEL_FILE)
model = saved_data["model"]
feature_names = saved_data.get("feature_names")

if not feature_names:
    feature_names = [
        f"{axis}{i}"
        for i in range(21)
        for axis in ("x", "y", "z")
    ]

BASE_DIR.joinpath("data").mkdir(exist_ok=True)
METRICS_FILE.parent.mkdir(exist_ok=True)

counts = {name: 0 for name in GESTURES.values()}

columns = ["label"] + list(feature_names)

print("\nIndependent webcam test")
print("This creates a NEW test CSV; it does not change your training CSV.")
print("Collect 60 samples for each gesture.")
print("1 Open Palm | 2 Fist | 3 Peace")
print("4 Thumbs Up | 5 Pointing")
print("S = Pause | Q = Finish and evaluate")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit(
        "Could not open webcam. Close other camera programs first."
    )

current_label = None
recording = False
frame_counter = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Each run starts a fresh independent test session.
with open(TEST_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(columns)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    wrist = hand.landmark[0]
                    features = []

                    for point in hand.landmark:
                        features.extend([
                            point.x - wrist.x,
                            point.y - wrist.y,
                            point.z - wrist.z,
                        ])

                    scale = max(abs(value) for value in features)

                    if (
                        recording
                        and current_label is not None
                        and counts[current_label] < TARGET_SAMPLES
                        and scale > 0
                    ):
                        frame_counter += 1

                        # Avoid recording every consecutive frame.
                        if frame_counter % SAMPLE_EVERY_N_FRAMES == 0:
                            features = [
                                value / scale for value in features
                            ]

                            writer.writerow([current_label] + features)
                            file.flush()
                            counts[current_label] += 1

                            if counts[current_label] >= TARGET_SAMPLES:
                                recording = False
                                print(
                                    current_label,
                                    "completed: 60 samples",
                                )

                key = cv2.waitKey(1) & 0xFF

                if key in GESTURES:
                    current_label = GESTURES[key]
                    frame_counter = 0

                    if counts[current_label] < TARGET_SAMPLES:
                        recording = True
                        print("Recording:", current_label)
                    else:
                        recording = False
                        print("This class already has 60 samples.")

                elif key == ord("s"):
                    recording = False

                elif key == ord("q"):
                    break

                status = "RECORDING" if recording else "PAUSED"
                selected = current_label or "Choose gesture (1-5)"

                cv2.putText(
                    frame,
                    f"Gesture: {selected}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Status: {status}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                y_pos = 95
                for label, count in counts.items():
                    cv2.putText(
                        frame,
                        f"{label}: {count}/{TARGET_SAMPLES}",
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    y_pos += 25

                cv2.imshow("Independent Model Evaluation", frame)

    finally:
        camera.release()
        cv2.destroyAllWindows()

print("\nCollected samples:", counts)

# Do not report a complete evaluation if classes are incomplete.
incomplete = {
    label: count
    for label, count in counts.items()
    if count < TARGET_SAMPLES
}

if incomplete:
    raise SystemExit(
        "Test incomplete. Run this script again and collect "
        "60 samples for all five gestures."
    )

# ---------------- EVALUATE SAVED MODEL ----------------

test_data = pd.read_csv(TEST_FILE)

missing_features = [
    name for name in feature_names
    if name not in test_data.columns
]

if missing_features:
    raise ValueError(
        f"Missing test features: {missing_features}"
    )

X_test = test_data[list(feature_names)]
y_test = test_data["label"]

predictions = model.predict(X_test)

classes = [str(value) for value in model.classes_]

accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    labels=list(model.classes_),
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=list(model.classes_),
)

per_class = {}

for label in classes:
    per_class[label] = {
        "precision": float(report[label]["precision"]),
        "recall": float(report[label]["recall"]),
        "f1-score": float(report[label]["f1-score"]),
        "support": int(report[label]["support"]),
    }

metrics = {
    "evaluation_type": (
        "Fresh webcam session, separate from the training CSV"
    ),
    "accuracy": float(accuracy),
    "test_samples": int(len(test_data)),
    "classes": classes,
    "per_class": per_class,
    "confusion_matrix": matrix.tolist(),
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nFresh-session accuracy:", f"{accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("\nConfusion matrix (actual rows, predicted columns):")
print(matrix)
print("\nEvaluation saved to:", METRICS_FILE)
