import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


#CONFIG
MODEL_PATH = "hand_landmarker.task"

screen_w, screen_h = pyautogui.size()
smoothening = 3
prev_x, prev_y = 0, 0
prev_scroll_y = 0

dragging = False
frameskip = 2
framecount = 0

#Margins
frame_margin_x = 70
frame_margin_y = 110

#MediaPipe HandLandmarker
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

opt = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    
)

landmarker = HandLandmarker.create_from_options(opt)

#Webcam
cap = cv2.VideoCapture(0)
timestamp = 0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


#FUNCTIONS
def calcdist(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.hypot(x2 - x1, y2 - y1)


#LOOP
while cap.isOpened():
    check, frame = cap.read()
    if not check:
        break

    framecount += 1

    if framecount % frameskip != 0:

        #Flip Camera
        frame = cv2.flip(frame, 1)

        #Camera Shape
        h, w, _ = frame.shape

        #Color Convert
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp += 1

        #Get Image Result
        result = landmarker.detect_for_video(mp_image, timestamp)

    
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Hand Width (Scale Reference)
            indexroot = hand[5]
            pinkyroot = hand[17]

            handsize = calcdist(indexroot, pinkyroot, w, h)

            #Middle Finger Base (Position Reference)
            midfingerbase = hand[9]
            x = int(midfingerbase.x * w)
            y = int(midfingerbase.y * h)

            screen_x = np.interp(
            x,
            (frame_margin_x, w - frame_margin_x),
            (0, screen_w)
            )

            screen_y = np.interp(
            y,
            (50, h - frame_margin_y),
            (0, screen_h)
            )

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            #Finger point references
            thumb = hand[4]
            indfingerbase = hand[6]
            indfingertip = hand[8]
            midfingertip = hand[12]

            #IndexBase-Thumb Distance (Click)
            distBT = calcdist(indfingerbase, thumb, w, h)

            if distBT < handsize * 0.45:
                pyautogui.click()
                pyautogui.sleep(0.25)

            #IndexTip-Thumb Distance (Drag)
            distTT = calcdist(indfingertip, thumb, w, h)

            if distTT < handsize * 0.45 and not dragging:
                pyautogui.mouseDown()
                dragging = True

            if distTT > handsize * 0.55 and dragging:
                pyautogui.mouseUp()
                dragging = False

            #IndexTip-MiddleTip (Scroll)
            distIM = calcdist(indfingertip, midfingertip, w, h)

            if distIM < handsize * 0.35:
                scroll_amount = int((prev_scroll_y - curr_y) * 2)
                pyautogui.scroll(scroll_amount)
                prev_scroll_y = curr_y

            #Draw Points
            for landmark in hand:
                if(landmark == hand[6] or
                landmark == hand[8] or
                landmark == hand[12] or
                landmark == hand[4]):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 127, 255), -1)

                if landmark == hand[9]:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (225, 225, 225), -1)

        cv2.rectangle(
        frame,
        (frame_margin_x, 50),
        (w - frame_margin_x, h - frame_margin_y),
        (0, 127, 255),
        4
        )
    
        cv2.imshow("Hand-Controlled Cursor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
