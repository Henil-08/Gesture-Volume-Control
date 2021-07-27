from cv2 import cv2 as cv
import HandTrackingModule as htm
from math import sqrt
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# PyCaw Initializations
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Getting Volume Range
volRange = volume.GetVolumeRange() 
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

# Initializing Videocapture and HandDetectors
cap = cv.VideoCapture(0)
detect = htm.HandDetector(detectConfi=0.7, maxHands=1)


def volCondition(lmList):
    # Making a Condition that only when index and thumb are open, volcontrol will commence
    fingerCheck = detect.fingerCheck(lmList)  
    
    if fingerCheck[0] and not(fingerCheck[2] and fingerCheck[3] and fingerCheck[4]):
        volControl = True
    else: 
        volControl = False
    
    # Pinky Check
    if fingerCheck[4] and volControl:
        pinky = True
    else:
        pinky = False

    return volControl, pinky


def calcDistance(frame, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Calculating Midpoint
    mx, my = (x1+x2)/2, (y1+y2)/2 

    # For Min (0%) Volume
    if distance < 30:
        cv.circle(frame, (int(mx), int(my)), 7, (0, 0, 255), -1)

    # For Max (100%) Volume
    elif distance > 180:
        cv.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv.circle(frame, (int(mx), int(my)), 7, (255, 0, 0), -1)
        cv.circle(frame, (x1, y1), 7, (255, 0, 0), -1)
        cv.circle(frame, (x2, y2), 7, (255, 0, 0), -1)
    
    # For inBetween Values
    else:
        cv.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv.circle(frame, (int(mx), int(my)), 7, (255, 255, 0), -1)
        cv.circle(frame, (x1, y1), 7, (255, 255, 0), -1)
        cv.circle(frame, (x2, y2), 7, (255, 255, 0), -1)

    return distance


def volGraphics(frame, distance):
    # Mapping the Values like in Arduino
    volPer = np.interp(distance, [30, 180], [0, 100])
    volBar = np.interp(volPer, [0, 100], [400, 150])
    
    # Setting the volume at every 5 Levels 
    smoothness = 5
    volPer = smoothness * round(volPer/smoothness)
    
    if pinky:
        # Setting the Desired Volume
        volume.SetMasterVolumeLevelScalar(volPer/100, None)
    
    getVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv.putText(frame, f"Volume Set {getVol}%", (320, 20), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)

    # Drawing the Volume Bar 
    cv.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
    cv.rectangle(frame, (50, int(volBar)), (85, 400), (255, 0, 0), -1)
    cv.putText(frame, f"{int(volPer)}%", (40, 450), cv.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3)

    return frame


while cap.isOpened():
    isSuccess, frame = cap.read()

    if isSuccess:
        frame = detect.findHands(frame)
        lmList, bBox = detect.findPosition(frame, boxDraw=True)

        if len(lmList) != 0:
            # Condition Check
            volControl, pinky = volCondition(lmList)

            if volControl:
                tx, ty = lmList[4][1], lmList[4][2]
                ix, iy = lmList[8][1], lmList[8][2]

                distance = calcDistance(frame, [tx, ty], [ix, iy])
                frame = volGraphics(frame, distance)
                
                cv.putText(frame, f"Volume Set ON", (bBox[0]-30, bBox[1]-30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            else:
                cv.putText(frame, f"Volume Set OFF", (bBox[0]-30, bBox[1]-30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                                            
        # Showing the FPS
        detect.addFPS(frame)

        # Showing the Frames
        cv.imshow("Video", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv.destroyAllWindows()
