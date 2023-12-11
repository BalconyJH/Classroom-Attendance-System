
import cv2
import dlib
from sklearn.svm import OneClassSVM

from src.config import config
from src.utils import logger, message_translator


class FaceClassifier:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(
            f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat"
        )
        self.face_recognition_model = dlib.face_recognition_model_v1(
            f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat"
        )

    async def face_descriptors(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            logger.error(message_translator("ERRORS.FILE.STATUS.NOT_FOUND"))
            return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detected_faces = self.face_detector(rgb_image)
        face_descriptors = []
        for face_rect in detected_faces:
            shape = self.shape_predictor(rgb_image, face_rect)
            face_descriptor = self.face_recognition_model.compute_face_descriptor(rgb_image, shape)
            face_descriptors.append(face_descriptor)
        return face_descriptors[0]

    def preprocess_face(self, image_path):
        img = cv2.imread(image_path)
        # rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(img, 1)
        if len(faces) == 0:
            logger.warning("No face detected.")
            return None
        if len(faces) > 1:
            logger.warning("More than one face detected.")

        for face in faces:
            shape = self.shape_predictor(img, face)

            face_chip = dlib.get_face_chip(img, shape)
            cv2.imwrite(image_path, face_chip)

    @staticmethod
    async def face_svm(face_descriptors):
        OneClassSVM(gamma="auto").fit(face_descriptors)

    @staticmethod
    async def save_face_classifier():
        pass

    async def train_face_classifier(self):
        pass
