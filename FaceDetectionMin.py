import cv2
import mediapipe as mp
import time
import FaceDetectionModule as fdm

cap = cv2.VideoCapture(0)
pTime = 0
detector = fdm.FaceDetector()
while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img, draw=False)
    # print(bboxs)
    if bboxs:
        for bbox in bboxs:
            img = detector.facny_draw(img, bbox)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, "fps: "+str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()