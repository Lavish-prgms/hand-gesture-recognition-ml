
import cv2
import mediapipe as mp
import csv
import os

DATA_FILE = "data/hand_gestures.csv"
TARGET_SAMPLES = 450

GESTURES = {
    ord("1"): "open_palm",
    ord("2"): "fist",
    ord("3"): "peace",
    ord("4"): "thumbs_up",
    ord("5"): "pointing",
}

# Check the existing dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "Dataset not found: data/hand_gestures.csv"
    )

# Count samples already collected for each gesture
counts = {name: 0 for name in GESTURES.values()}

with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    if not reader.fieldnames or "label" not in reader.fieldnames:
        raise ValueError("Dataset must contain a label column.")

    for row in reader:
        label = row.get("label")
        if label in counts:
            counts[label] += 1

print("Existing dataset samples:", counts)

if all(count >= TARGET_SAMPLES for count in counts.values()):
    print("All gestures already have 300 samples. No collection needed.")
    raise SystemExit

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Could not open webcam. Check camera access.")

current_label = None
recording = False

print("\nAdditional sample collection started.")
print("1 = Open Palm | 2 = Fist | 3 = Peace")
print("4 = Thumbs Up | 5 = Pointing")
print("S = Pause | Q = Quit")
print("Keep each gesture clearly visible.")

try:
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

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
                            point.z - wrist.z
                        ])

                    scale = max(abs(value) for value in features)

                    if (
                        recording
                        and current_label is not None
                        and counts[current_label] < TARGET_SAMPLES
                        and scale > 0
                    ):
                        features = [v / scale for v in features]

                        writer.writerow([current_label] + features)
                        file.flush()
                        counts[current_label] += 1

                        if counts[current_label] >= TARGET_SAMPLES:
                            recording = False
                            print(current_label, "completed!")

                key = cv2.waitKey(1) & 0xFF

                if key in GESTURES:
                    current_label = GESTURES[key]

                    if counts[current_label] < TARGET_SAMPLES:
                        recording = True
                        print("Collecting:", current_label)
                    else:
                        recording = False
                        print(current_label, "already has enough samples.")

                elif key == ord("s"):
                    recording = False

                elif key == ord("q"):
                    break

                status = "RECORDING" if recording else "PAUSED"
                selected = current_label or "Select a gesture"

                cv2.putText(
                    frame, f"Gesture: {selected}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

                cv2.putText(
                    frame, f"Status: {status}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

                y = 95
                for name, count in counts.items():
                    cv2.putText(
                        frame, f"{name}: {count}/{TARGET_SAMPLES}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1
                    )
                    y += 25

                cv2.imshow("Additional Gesture Collection", frame)

finally:
    camera.release()
    cv2.destroyAllWindows()

print("\nFinal sample counts:", counts)
print("Existing dataset preserved.")
