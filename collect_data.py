
import cv2
import mediapipe as mp
import csv
import os

# Five gesture categories
KEY_TO_GESTURE = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "peace",
    ord("4"): "thumbs_up",
    ord("5"): "pointing",
}

MAX_SAMPLES = 200
DATA_FOLDER = "data"
DATA_FILE = os.path.join(DATA_FOLDER, "hand_gestures.csv")

os.makedirs(DATA_FOLDER, exist_ok=True)

# Start a fresh dataset for this collection session
gesture_names = list(KEY_TO_GESTURE.values())
sample_counts = {name: 0 for name in gesture_names}

# CSV will contain a label and 63 landmark features
columns = ["label"]
for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

current_label = None
recording = False

print("Gesture dataset collection started.")
print("Press 1-5 to select a gesture.")
print("Press S to pause. Press Q to quit.")

with open(DATA_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(columns)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                # Use the wrist as the reference point
                wrist = hand.landmark[0]
                features = []

                for point in hand.landmark:
                    features.extend([
                        point.x - wrist.x,
                        point.y - wrist.y,
                        point.z - wrist.z
                    ])

                # Scale features to reduce size variation
                scale = max(abs(v) for v in features)

                if scale > 0:
                    features = [v / scale for v in features]

                if (
                    recording
                    and current_label is not None
                    and sample_counts[current_label] < MAX_SAMPLES
                ):
                    writer.writerow([current_label] + features)
                    file.flush()
                    sample_counts[current_label] += 1

                    if sample_counts[current_label] >= MAX_SAMPLES:
                        recording = False

            # Read keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key in KEY_TO_GESTURE:
                current_label = KEY_TO_GESTURE[key]

                if sample_counts[current_label] < MAX_SAMPLES:
                    recording = True
                else:
                    recording = False
                    print(current_label, "already has 200 samples.")

            elif key == ord("s"):
                recording = False

            elif key == ord("q"):
                break

            # Display instructions and progress
            label_text = current_label or "Select gesture: 1-5"
            status_text = "RECORDING" if recording else "PAUSED"

            cv2.putText(
                frame, f"Gesture: {label_text}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Status: {status_text}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 2
            )

            y_position = 95
            for name in gesture_names:
                text = f"{name}: {sample_counts[name]}/{MAX_SAMPLES}"
                cv2.putText(
                    frame, text, (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )
                y_position += 25

            cv2.imshow("Gesture Dataset Collection", frame)

camera.release()
cv2.destroyAllWindows()

print("\nDataset collection finished.")
print("Samples collected:", sample_counts)
print("Dataset saved to:", DATA_FILE)
