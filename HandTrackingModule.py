import cv2 as cv
import mediapipe as mp

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for idx, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([idx, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 10, (255, 0, 255), cv.FILLED)

        return lmList
    
    def fingersUp(self):
        fingers = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            
            threshold = 0.05

            if myHand.landmark[4].x < myHand.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if myHand.landmark[id * 4 + 4].y < myHand.landmark[id * 4 + 2].y - threshold:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers
    
    def findDistance(self, p1, p2, frame, draw=True):
        x1, y1 = self.results.multi_hand_landmarks[0].landmark[p1].x, self.results.multi_hand_landmarks[0].landmark[p1].y
        x2, y2 = self.results.multi_hand_landmarks[0].landmark[p2].x, self.results.multi_hand_landmarks[0].landmark[p2].y

        h, w, c = frame.shape
        cx1, cy1 = int(x1 * w), int(y1 * h)
        cx2, cy2 = int(x2 * w), int(y2 * h)

        if draw:
            cv.circle(frame, (cx1, cy1), 10, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (cx2, cy2), 10, (255, 0, 255), cv.FILLED)
            cv.line(frame, (cx1, cy1), (cx2, cy2), (255, 0, 255), 3)

        length = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

        return length, frame, [cx1, cy1, cx2, cy2]
    
    