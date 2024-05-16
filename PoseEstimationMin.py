import cv2
import mediapipe as mp
import time
import PoseEstimationModule as pem

detector = pem.PoseDetector()
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPositions(img)
    print(lmList[14])#elbow
    cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()