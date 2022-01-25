import cv2
import mediapipe as mp
import time

class handDetector():
    # inititalising class variables
    def __init__(self, mode=False, maxHands=2, complexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        # static_img_mode
        self.maxHands = maxHands
        self.complexity = complexity

        # confidences
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # hands object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon,)

        # Drawing on image object used in findHands
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        # mediapipe uses on RGB image, convert
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # returns array of all landmark coordinates
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)

        # if landmarks are found
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        # landmark list
        lmList = []
        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # to find pixel position
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                # draw separately
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()