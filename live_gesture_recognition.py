
import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

# Load the trained model
MODEL_FILE = "model/gesture_model.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

saved_data = joblib.load(MODEL_FILE)
model = saved_data["model"]

print("Trained model loaded successfully!")
print("Recognized gestures:", list(model.classes_))
print("Press Q to close the webcam.")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Open the laptop webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

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

        # Mirror the webcam view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Draw detected hand landmarks
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            # Extract the same 63 features used during training
            wrist = hand.landmark[0]
            features = []

            for point in hand.landmark:
                features.extend([
                    point.x - wrist.x,
                    point.y - wrist.y,
                    point.z - wrist.z
                ])

            # Apply the same scaling used during data collection
            scale = max(abs(value) for value in features)

            if scale > 0:
                features = [value / scale for value in features]

                # Predict the gesture
                prediction = model.predict([features])[0]
                probabilities = model.predict_proba([features])[0]
                confidence = float(np.max(probabilities))

                # Display a cautious result
                if confidence >= 0.60:
                    result_text = str(prediction).replace(
                        "_", " "
                    ).upper()
                else:
                    result_text = "UNCERTAIN"

                cv2.putText(
                    frame,
                    result_text,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Model score: {confidence * 100:.1f}%",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

        else:
            cv2.putText(
                frame,
                "Show your hand",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("AI Gesture Recognition", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

print("Gesture recognition stopped.")
