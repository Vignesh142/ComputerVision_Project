import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math
import VolumeControlModule as vcm

##################################
wCam, hCam = 640, 480
##################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime=0

detector = htm.HandDetector(detectionCon=0.7)
volControl = vcm.VolumeHandControl()

while True:
    success, img = cap.read()
    detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    img = volControl.disp_volBar(img)
    if len(lmList) != 0:
        img = volControl.control(img, lmList)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()