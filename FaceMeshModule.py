import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
        refine_landmarks=False
    ):

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode,
            max_num_faces,
            refine_landmarks,
            min_detection_confidence,
            min_tracking_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpecs = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpecs, self.drawSpecs)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    face.append([id, x, y])
                faces.append(face)
        return img, faces