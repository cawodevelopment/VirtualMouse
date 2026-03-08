import cv2 as cv
import numpy as np
import autopy as ap
import HandTrackingModule as htm
import mediapipe as mp

camWidth, camHeight = 640, 480
screenWidth, screenHeight = ap.screen.size()
frameR = 100

cap = cv.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = htm.HandDetector(max_num_hands=1)
left_click_latch = False
right_click_latch = False

while True:
    ret, frame = cap.read()
    flipped_frame = cv.flip(frame, 1)

    flipped_frame = detector.findHands(flipped_frame)
    lmList = detector.findPosition(flipped_frame, draw=False)

    if len(lmList) > 12:
        x0, y0 = lmList[4][1], lmList[4][2]
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]

        cv.circle(flipped_frame, (x0, y0), 10, (255, 0, 255), cv.FILLED)
        cv.circle(flipped_frame, (x1, y1), 10, (255, 0, 255), cv.FILLED)
        cv.circle(flipped_frame, (x2, y2), 10, (255, 0, 255), cv.FILLED)

        fingers = detector.fingersUp()
        print(fingers)

        if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0:
            cv.rectangle(flipped_frame, (frameR, frameR), (camWidth - frameR, camHeight - frameR), (255, 0, 255), 2)

            x3 = np.interp(x1, (frameR, camWidth - frameR), (0, screenWidth))
            y3 = np.interp(y1, (frameR, camHeight - frameR), (0, screenHeight))

            ap.mouse.move(x3, y3)
        
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0:
            if not right_click_latch:
                ap.mouse.click(ap.mouse.Button.RIGHT)
                right_click_latch = True
            cv.circle(flipped_frame, (x1, y1), 15, (0, 255, 0), cv.FILLED)
        else:
            right_click_latch = False

        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:
            if not left_click_latch:
                ap.mouse.click()
                left_click_latch = True
            cv.circle(flipped_frame, (x1, y1), 15, (0, 255, 0), cv.FILLED)
        else:
            left_click_latch = False


    cv.imshow("Virtual Mouse", flipped_frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
