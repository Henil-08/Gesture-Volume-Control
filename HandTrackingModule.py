import cv2 as cv
import mediapipe as mp 
import numpy as np  
import time as t

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectConfi=0.5, trackConfi=0.5):
        # Initializig the Modes
        self.mode = mode
        self.maxHands = maxHands
        self.detectConfi = detectConfi
        self.trackConfi = trackConfi
        
        # Initializing the Mediapipe
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectConfi, self.trackConfi)

        # Initializing FPS Counter
        self.pTime = 0
        self.cTime = 0

    def findHands(self, frame, draw=True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Getting the Landmarks
        self.results = self.hands.process(frameRGB)
        # print(results.multi_hand_landmarks)

        # Drawing the Landmarks on Hands
        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLMs, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self, frame, handNo=0, boxDraw=False):
        lmList = []
        bBox = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # Giving Every point(landmarks) on the hand their IDs
            for id, lm in enumerate(myHand.landmark):
                # As the x and y values are in aspect ratio we convert them to pixel    
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            
            # Min and Maximum for cx and cy values
            _, xMin, yMin = np.min(lmList, axis=0)
            _, xMax, yMax = np.max(lmList, axis=0)

            bBox = [xMin, yMin, xMax, yMax]

            if boxDraw:
                cv.rectangle(frame, (bBox[0]-30, bBox[1]-30), (bBox[2]+30, bBox[3]+30), (0, 255, 0), 3)
        if boxDraw:
            return lmList, bBox
        else:
            return lmList

    def fingerCheck(self, lmList):
        tipIDs = [4, 8, 12, 16, 20]
        fCheck = []

        # For thumb
        if lmList[4][1] < lmList[3][1]:
            fCheck.append(True)
        else:
            fCheck.append(False)

        # For other fngers
        for id in range(1, 5):
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                fCheck.append(True)
            else:
                fCheck.append(False)

        return fCheck

    def addFPS(self, frame):
        # Calculating the FPS
        self.cTime = t.time()
        fps = 1/(self.cTime - self.pTime)
        self.pTime = self.cTime * 0.90 + self.pTime * 0.10
        cv.putText(frame, f"FPS: {int(fps)}", (10, 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

def main():
    cap = cv.VideoCapture(0)
    detect = HandDetector()

    # Initializing the Time Variables
    pTime = 0
    cTime = 0

    while(cap.isOpened()):
        isSuccess, frame = cap.read()

        if isSuccess:
            frame = detect.findHands(frame)
            lmList = detect.findPosition(frame, boxDraw=False)
    
            # Printing the list
            if len(lmList) != 0:
                print(lmList[4])
            
            detect.addFPS(frame)

            cv.imshow("Video", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()