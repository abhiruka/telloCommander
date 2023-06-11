import time

import cv2
import mediapipe as mp
from djitellopy import Tello

# Threshold for detecting a gesture
GESTURE_DETECTION_TIMEOUT = 10
# Threshold for detecting a gesture
GESTURE_DETECTION_THRESHOLD = 5

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Initialize Tello drone
tello = Tello()
tello.connect()
tello.streamon()

frame_read = tello.get_frame_read()

gesture_start_time = None
last_gesture = None
no_hands_start_time = None

while True:
    try:
        # Get the current frame from the Tello drone's video feed
        img = frame_read.frame

        # Convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks is not None:
            # If hands are detected, draw hand landmarks on the image
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if thumb is up or down
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_index_distance = ((thumb.x - index_finger.x) ** 2 + (thumb.y - index_finger.y) ** 2) ** 0.5

                if thumb_index_distance < 0.1:
                    gesture = "Fist"
                elif thumb.y > thumb_base.y:
                    gesture = "Thumb Down"
                else:
                    gesture = "Thumb Up"

                # Display the detected gesture on the image
                cv2.putText(img, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # If gesture has changed, reset the start time for the new gesture
                if gesture != last_gesture:
                    gesture_start_time = time.time()
                    last_gesture = gesture

                # If the same gesture has been held for more than 5 seconds, execute the command for that gesture
                elif time.time() - gesture_start_time > GESTURE_DETECTION_THRESHOLD:
                    if gesture == "Thumb Up":
                        tello.takeoff()  # Make the drone take off when thumbs up is detected
                    elif gesture == "Thumb Down":
                        tello.land()  # Make the drone land when thumbs down is detected

                    # Reset the start time to prevent the command from being executed repeatedly
                    gesture_start_time = time.time()

                # Reset the no hands start time since a hand is detected
                no_hands_start_time = None

        else:
            # No hands detected
            if no_hands_start_time is None:
                no_hands_start_time = time.time()
            elif time.time() - no_hands_start_time > GESTURE_DETECTION_TIMEOUT:
                # If no hands have been detected for more than 10 seconds, land the drone as a failsafe
                tello.land()

        # Show the image in a window
        cv2.imshow("Tello", img)

        # If the 'q' key is pressed, break the loop and close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        tello.land()
        break

# Clean up
tello.land()  # Make the drone land
cv2.destroyAllWindows()
