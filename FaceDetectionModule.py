import cv2
import mediapipe as mp
import time


class FaceDetector():

    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.min_detection_confidence, self.model_selection)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if draw:
                    self.mpDraw.draw_detection(img, detection)
                # print(id, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                    int(bboxC.width*iw), int(bboxC.height*ih)
                bboxs.append([id, bbox, detection.score])
        return img, bboxs

    def facny_draw(self, img, bboxs, l=30, t=10, rt=1):
        x, y, w, h = bboxs[1]
        x1, y1 = x+w, y+h
        score = bboxs[2][0]

        cv2.rectangle(img, bboxs[1], (255, 0, 255), rt)
        # Show accuracy
        cv2.putText(img, f'{int(score*100)}%', (x, y-20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        # Top Left x,y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right x1,y
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)
        # Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)

        return img
