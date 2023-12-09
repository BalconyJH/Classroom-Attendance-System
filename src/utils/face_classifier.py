import asyncio

import cv2
import dlib
from numpy import ndarray

from src.config import config


class FaceClassifier:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(
            config.predictor_model_path / "shape_predictor_68_face_landmarks.dat"
        ))

    @staticmethod
    async def image_preprocessing(path: str) -> list[ndarray]:
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None,
            FaceClassifier._process_image,
            path
        )
        return result

    @staticmethod
    def _process_image(path: str) -> list[ndarray]:
        face_detector = dlib.get_frontal_face_detector()
        raw_image = cv2.imread(path)
        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector(gray_image, 1)
        cropped_faces = []

        for face_rect in detected_faces:
            x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
            crop = gray_image[y1:y2, x1:x2]
            cropped_faces.append(crop)

        return cropped_faces

    @staticmethod
    async def save_face_classifier():
        pass

    async def train_face_classifier(self):
        pass
