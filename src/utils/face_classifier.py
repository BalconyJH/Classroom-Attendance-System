
import cv2
import dlib


class FaceClassifier:
    def __init__(self):
        pass

    @staticmethod
    async def image_preprocessing(path: str):
        face_detector = dlib.get_frontal_face_detector()
        raw_image = cv2.imread(path)
        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector(gray_image, 1)
        cropped_faces = []

        for i, face_rect in enumerate(detected_faces):
            x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
            crop = gray_image[y1:y2, x1:x2]
            cropped_faces.append(crop)

        return cropped_faces

    @staticmethod
    async def save_face_classifier():
        pass

    async def train_face_classifier(self):
        pass
