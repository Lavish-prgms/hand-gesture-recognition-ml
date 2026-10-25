
import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
import subprocess
import win32com.client
import os

from collections import Counter, deque

# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_FILE = "model/gesture_model.pkl"
CONFIDENCE_LIMIT = 0.60
STABLE_FRAMES = 5

# ==================================================
# 2. LOAD TRAINED MACHINE LEARNING MODEL
# ==================================================

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

saved_data = joblib.load(MODEL_FILE)
model = saved_data["model"]
feature_names = saved_data.get("feature_names")

print("\nTrained ML model loaded successfully!")
print("Recognized gestures:", list(model.classes_))

# ==================================================
# 3. CONNECT TO POWERPOINT WHEN AVAILABLE
# ==================================================

powerpoint = None


def get_powerpoint_app():
    """Connect to desktop PowerPoint if it is running."""
    global powerpoint

    if powerpoint is None:
        try:
            powerpoint = win32com.client.GetActiveObject(
                "PowerPoint.Application"
            )
        except Exception:
            return None

    return powerpoint


def get_slideshow_view():
    """Return the active PowerPoint slideshow view, if available."""
    app = get_powerpoint_app()

    if app is None:
        return None

    try:
        if app.SlideShowWindows.Count > 0:
            return app.SlideShowWindows.Item(1).View
    except Exception:
        return None

    return None


if get_powerpoint_app() is not None:
    print("PowerPoint connected!")
else:
    print("PowerPoint is not open. Non-PowerPoint actions can still work.")

print("\nGESTURE CONTROLS")
print("------------------------------")
print("PEACE       -> Next slide")
print("FIST        -> Previous slide")
print("THUMBS UP   -> Exit slideshow and stop controller")
print("OPEN PALM   -> Open File Explorer")
print("POINTING    -> Open Notepad")
print("------------------------------")
print("Remove your hand between commands.")
print("Press Q in the camera window to quit.\n")

# ==================================================
# 4. INITIALIZE WEBCAM AND MEDIAPIPE
# ==================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    camera.release()
    raise SystemExit(
        "Could not open webcam. Close other camera applications and try again."
    )

history = deque(maxlen=6)

can_act = True
absent_frames = 0
controller_running = True

# ==================================================
# 5. REAL-TIME HAND GESTURE RECOGNITION
# ==================================================

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while controller_running:

            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror webcam image
            frame = cv2.flip(frame, 1)

            # Convert BGR image to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb_frame)

            stable_label = None
            confidence = 0.0
            display_text = "Show your hand"

            # ------------------------------------------
            # DETECT HAND AND EXTRACT 63 FEATURES
            # ------------------------------------------

            if results.multi_hand_landmarks:

                absent_frames = 0
                hand = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist = hand.landmark[0]
                features = []

                for point in hand.landmark:
                    features.extend([
                        point.x - wrist.x,
                        point.y - wrist.y,
                        point.z - wrist.z
                    ])

                # Apply the same feature scaling used in training
                scale = max(abs(value) for value in features)

                if scale > 0:

                    features = [
                        value / scale for value in features
                    ]

                    # Use the same feature names as the trained model
                    if feature_names is not None:
                        input_data = pd.DataFrame(
                            [features],
                            columns=feature_names
                        )
                    else:
                        input_data = np.array([features])

                    # Predict gesture using the trained ML model
                    prediction = str(
                        model.predict(input_data)[0]
                    )

                    probabilities = model.predict_proba(
                        input_data
                    )[0]

                    confidence = float(np.max(probabilities))

                    # Keep recent predictions for stable detection
                    history.append((prediction, confidence))

                    if len(history) == history.maxlen:

                        labels = [
                            item[0] for item in history
                        ]

                        common_label, count = Counter(
                            labels
                        ).most_common(1)[0]

                        matching_scores = [
                            score
                            for label, score in history
                            if label == common_label
                        ]

                        mean_confidence = float(
                            np.mean(matching_scores)
                        )

                        if (
                            count >= STABLE_FRAMES
                            and mean_confidence >= CONFIDENCE_LIMIT
                        ):
                            stable_label = common_label

                    if stable_label:
                        display_text = stable_label.upper().replace(
                            "_", " "
                        )
                    else:
                        display_text = f"Detecting: {prediction}"

                    # --------------------------------------
                    # 6. GESTURE-BASED COMPUTER CONTROL
                    # --------------------------------------

                    if stable_label is not None and can_act:

                        # PEACE: NEXT POWERPOINT SLIDE
                        if stable_label == "peace":

                            view = get_slideshow_view()

                            if view is not None:
                                try:
                                    view.Next()
                                    print("ACTION: NEXT SLIDE")
                                except Exception as error:
                                    print("Next slide failed:", error)
                            else:
                                print(
                                    "No active slideshow. Open PowerPoint "
                                    "and select Slide Show > From Beginning."
                                )

                            can_act = False

                        # FIST: PREVIOUS POWERPOINT SLIDE
                        elif stable_label == "fist":

                            view = get_slideshow_view()

                            if view is not None:
                                try:
                                    view.Previous()
                                    print("ACTION: PREVIOUS SLIDE")
                                except Exception as error:
                                    print("Previous slide failed:", error)
                            else:
                                print(
                                    "No active PowerPoint slideshow."
                                )

                            can_act = False

                        # THUMBS UP: EXIT SLIDESHOW AND STOP CONTROLLER
                        elif stable_label == "thumbs_up":

                            view = get_slideshow_view()

                            if view is not None:
                                try:
                                    view.Exit()
                                    print("ACTION: SLIDESHOW EXITED")
                                except Exception as error:
                                    print("Could not exit slideshow:", error)
                            else:
                                print(
                                    "No active slideshow to exit."
                                )

                            print("Stopping gesture controller...")
                            controller_running = False

                        # OPEN PALM: OPEN FILE EXPLORER
                        elif stable_label == "open_palm":

                            try:
                                subprocess.Popen(["explorer.exe"])
                                print("ACTION: FILE EXPLORER OPENED")
                            except Exception as error:
                                print("Could not open File Explorer:", error)

                            can_act = False

                        # POINTING: OPEN NOTEPAD
                        elif stable_label == "pointing":

                            try:
                                subprocess.Popen(["notepad.exe"])
                                print("ACTION: NOTEPAD OPENED")
                            except Exception as error:
                                print("Could not open Notepad:", error)

                            can_act = False

            else:
                # Reset gesture history when the hand disappears
                history.clear()
                absent_frames += 1

                # Require the hand to leave the camera briefly
                # before another command can be executed
                if absent_frames >= 5:
                    can_act = True

            # ------------------------------------------
            # 7. DISPLAY LIVE RESULTS
            # ------------------------------------------

            cv2.putText(
                frame,
                display_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Model score: {confidence * 100:.1f}%",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Remove hand between commands | Q = Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "AI Gesture Computer Control",
                frame
            )

            # Quit manually using Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Gesture controller stopped.")
