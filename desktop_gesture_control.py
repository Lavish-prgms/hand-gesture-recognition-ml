
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import subprocess

# ---------------- SETTINGS ----------------

WINDOW = "Gesture Mouse Control"

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

SCREEN_W, SCREEN_H = pyautogui.size()

SMOOTHING = 0.30
PINCH_START = 0.040
PINCH_RELEASE = 0.065
SCROLL_THRESHOLD = 0.020

# ---------------- WEBCAM ----------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit(
        "Webcam unavailable. Close other webcam programs first."
    )

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- STATE ----------------

cursor_x = None
cursor_y = None

pinch_active = False
previous_pinch = False

scroll_anchor = None
explorer_opened = False

print("Gesture mouse control started!")
print("OPEN PALM: Open File Explorer once")
print("INDEX FINGER: Move cursor")
print("THUMB + INDEX PINCH: Double-click")
print("INDEX + MIDDLE UP: Scroll")
print("Press Q in the camera window to quit.")

try:
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

            status = "Show your hand"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                # Finger extension checks
                index_up = lm[8].y < lm[6].y
                middle_up = lm[12].y < lm[10].y
                ring_up = lm[16].y < lm[14].y
                pinky_up = lm[20].y < lm[18].y

                # Distance between thumb tip and index tip
                pinch_distance = np.hypot(
                    lm[4].x - lm[8].x,
                    lm[4].y - lm[8].y
                )

                # Pinch hysteresis avoids repeated clicks
                if pinch_active:
                    if pinch_distance > PINCH_RELEASE:
                        pinch_active = False
                elif pinch_distance < PINCH_START:
                    pinch_active = True

                # Four fingers extended = open palm
                open_palm = (
                    index_up and middle_up
                    and ring_up and pinky_up
                    and pinch_distance > 0.10
                )

                # Index and middle raised; ring and pinky folded
                scroll_mode = (
                    index_up and middle_up
                    and not ring_up and not pinky_up
                    and not pinch_active
                )

                # -------- OPEN FILE EXPLORER --------

                if open_palm:
                    scroll_anchor = None
                    status = "OPEN PALM"

                    if not explorer_opened:
                        subprocess.Popen(["explorer.exe"])
                        explorer_opened = True
                        status = "FILE EXPLORER OPENED"
                        print(status)

                # -------- SCROLL UP / DOWN --------

                elif scroll_mode:
                    status = "SCROLL: Move hand up/down"

                    current_y = (
                        lm[8].y + lm[12].y
                    ) / 2

                    if scroll_anchor is not None:
                        delta = scroll_anchor - current_y

                        if delta >= SCROLL_THRESHOLD:
                            pyautogui.scroll(4)
                            scroll_anchor = current_y
                            status = "SCROLLING UP"

                        elif delta <= -SCROLL_THRESHOLD:
                            pyautogui.scroll(-4)
                            scroll_anchor = current_y
                            status = "SCROLLING DOWN"

                    else:
                        scroll_anchor = current_y

                # -------- MOVE CURSOR --------

                else:
                    scroll_anchor = None

                    cursor_mode = (
                        index_up
                        and not middle_up
                        and not ring_up
                        and not pinky_up
                    )

                    if cursor_mode:
                        tip = lm[8]

                        target_x = int(np.interp(
                            tip.x, [0.08, 0.92],
                            [20, SCREEN_W - 20]
                        ))

                        target_y = int(np.interp(
                            tip.y, [0.08, 0.92],
                            [20, SCREEN_H - 20]
                        ))

                        if cursor_x is None:
                            cursor_x = target_x
                            cursor_y = target_y
                        else:
                            cursor_x = (
                                SMOOTHING * target_x
                                + (1 - SMOOTHING) * cursor_x
                            )
                            cursor_y = (
                                SMOOTHING * target_y
                                + (1 - SMOOTHING) * cursor_y
                            )

                        pyautogui.moveTo(
                            int(cursor_x), int(cursor_y)
                        )
                        status = "MOVING CURSOR"

                    # -------- DOUBLE-CLICK --------
                    if pinch_active and not previous_pinch:
                        pyautogui.doubleClick(interval=0.15)
                        status = "DOUBLE CLICK"
                        print("Double-click performed")

                previous_pinch = pinch_active

            else:
                scroll_anchor = None
                pinch_active = False
                previous_pinch = False

            # -------- CAMERA STATUS --------

            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                "Index: Move | Pinch: Double-click",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (255, 255, 255), 1
            )

            cv2.putText(
                frame, "Two fingers: Scroll | Q: Quit",
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (255, 255, 255), 1
            )

            cv2.imshow(WINDOW, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except pyautogui.FailSafeException:
    print("Mouse safety stop triggered.")

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Gesture mouse control stopped.")
