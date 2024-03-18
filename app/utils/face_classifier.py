from typing import Union, Optional

import cv2
import dlib
from loguru import logger
from sklearn.svm import OneClassSVM

from config import config
from app.utils import message_translator


class FaceClassifier:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat")
        self.face_recognition_model = dlib.face_recognition_model_v1(
            f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat"
        )

    async def face_descriptors(self, image_path: str) -> Union[dlib.vector, None]:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(message_translator("ERRORS.FILE.STATUS.NOT_FOUND"))
            return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = self.face_detector(rgb_image, 1)

        if len(faces) != 1:
            logger.warning(f"{len(faces)} faces detected. Expected exactly one face.")
            return None

        shape = self.shape_predictor(rgb_image, faces[0])
        face_descriptor = self.face_recognition_model.compute_face_descriptor(rgb_image, shape)
        return face_descriptor

    def preprocess_face(self, image_path):
        img = cv2.imread(image_path)
        # rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(img, 1)
        if len(faces) == 0:
            logger.warning("No face detected.")
            return None
        if len(faces) > 1:
            logger.warning("More than one face detected.")
            return None

        for face in faces:
            shape = self.shape_predictor(img, face)

            face_chip = dlib.get_face_chip(img, shape)
            cv2.imwrite(image_path, face_chip)

    @staticmethod
    async def train_face_classifier(
        face_descriptors: list[list[float]],
    ) -> Optional[OneClassSVM]:
        if len(face_descriptors) == 0:
            logger.warning("MODEL.TRAIN.PARAMETERS_MISSING")
            return None
        logger.debug(f"face_descriptors_length: {len(face_descriptors)}")
        logger.debug(f"sample_width: {len(face_descriptors[0])}")
        model = OneClassSVM(gamma="auto")
        model.fit(face_descriptors)

        return model

    # @staticmethod
    # async def save_face_classifier(data: Union[list[float], OneClassSVM], uid: str, path: str):
    #     file_path = os.path.join(path, f"{uid}.pkl")
    #     with open(file_path, "wb") as f:
    #         pickle.dump(data, f)
