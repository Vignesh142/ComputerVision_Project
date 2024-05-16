import cv2
import mediapipe as mp
import time
import FaceMeshModule as fmm

cap = cv2.VideoCapture(0)
pTime = 0
detector = fmm.FaceMeshDetector()

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)
    # print(len(faces), end="")

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, "fps:"+str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()