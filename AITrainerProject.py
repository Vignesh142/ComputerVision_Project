import cv2
import numpy as np
import time
import os
import PoseEstimationModule as pm

cap = cv2.VideoCapture('AITrainer/exercise2.mp4')
detector = pm.PoseDetector()
cap.set(3, 720)
cap.set(4, 720)
pTime = 0
count = 0
dir = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Video not found")
        break
    img = cv2.resize(img, (720, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPositions(img, draw=False)
    if len(lmList) != 0:
        angle_left = detector.findAngle(img, 11, 13, 15, right=False)
        angle_right = detector.findAngle(img, 12, 14, 16)

        per = np.interp(angle_left, (45, 180), (100, 0))
        bar = np.interp(angle_left, (45, 180), (100, 650))
        # print(angle_left, per, bar)

        # check for the dumbbell curls
        color= (255, 0, 255)
        if per >=80:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per <= 25:
            color = (0, 0, 255)
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)

        # draw bar
        cv2.rectangle(img, (640, 100), (690, 650), color, 3)
        cv2.rectangle(img, (640, int(bar)), (690, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (590, 75), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

        # draw counter
        cv2.rectangle(img, (0, 500), (230, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (40, 670), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 10)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "fps: "+str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Video", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()