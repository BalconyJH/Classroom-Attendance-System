import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import dlib
import numpy as np
from config import config


def extract_and_resize_face(image: np.ndarray, face_rect: dlib.rectangle, scale_factor: int = 2) -> np.ndarray:
    """
    提取并放大图像中的人脸区域。

    :param image: 原始图像, 类型为 np.ndarray。
    :param face_rect: 人脸的矩形框, 类型为 dlib.rectangle。
    :param scale_factor: 放大倍数, 默认为 2。
    :return: 放大后的人脸图像区域, 类型为 np.ndarray。
    """
    face_height = face_rect.bottom() - face_rect.top()
    face_width = face_rect.right() - face_rect.left()
    half_height = int(face_height / 2)
    half_width = int(face_width / 2)

    new_height, new_width = face_height * scale_factor, face_width * scale_factor
    img_blank = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    top, left = face_rect.top() - half_height, face_rect.left() - half_width
    for row_index in range(new_height):
        for col_index in range(new_width):
            source_row = top + row_index
            source_col = left + col_index
            if 0 <= source_row < image.shape[0] and 0 <= source_col < image.shape[1]:
                img_blank[row_index, col_index] = image[source_row, source_col]

    return img_blank


def is_face_within_bounds(face_rect: dlib.rectangle, bounds: tuple[int, int]) -> bool:
    """
    判断人脸是否在指定的范围内。

    :param face_rect: 人脸的矩形框。
    :param bounds: 允许的最大范围。
    :return: 如果人脸在范围内, 返回True; 否则返回False。
    """
    face_height = face_rect.bottom() - face_rect.top()
    face_width = face_rect.right() - face_rect.left()
    half_height = int(face_height / 2)
    half_width = int(face_width / 2)

    return not (
        (face_rect.right() + half_width) > bounds[0]
        or (face_rect.bottom() + half_height > bounds[1])
        or (face_rect.left() - half_width < 0)
        or (face_rect.top() - half_height < 0)
    )


class FaceFeatureProcessor:
    def __init__(self, logger):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat")
        self.face_recognition_model = dlib.face_recognition_model_v1(
            f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat"
        )
        self.logger = logger

    def process_single_face_image(self, path: str) -> str:
        # 读取人脸图像
        img_rd = cv2.imread(path)
        # Dlib的人脸检测器
        faces: list[dlib.rectangle] = self.face_detector(img_rd, 0)

        if len(faces) != 1:
            self.logger.warning(f"{len(faces)} faces detected. Expected exactly one face.")
            return "false"

        face_rect = faces[0]
        if not is_face_within_bounds(face_rect, (640, 480)):
            self.logger.info(f"Face out of range, discarding image: {path}")
            return "big"

        # 提取并放大人脸
        img_blank = extract_and_resize_face(img_rd, face_rect)
        cv2.imwrite(path, img_blank)
        self.logger.info(f"Processed and saved image: {path}")
        return "right"

    def extract_face_features_128d(self, path_img: str):
        image = cv2.imread(path_img)
        faces = self.face_detector(image, 1)

        if faces:
            shape = self.shape_predictor(image, faces[0])
            return self.face_recognition_model.compute_face_descriptor(image, shape)
        else:
            self.logger.warning(f"No face detected in image: {path_img}")
            return None

    def calculate_average_face_features(self, path: str) -> np.ndarray:
        def process_image(photo_path):
            features_128d = self.extract_face_features_128d(photo_path)
            if features_128d is not None:
                return features_128d
            return None

        photo_paths = [os.path.join(path, photo_name) for photo_name in os.listdir(path)]
        face_feature_vectors = []

        with ProcessPoolExecutor() as executor:
            results = executor.map(process_image, photo_paths)

        for result in results:
            if result is not None:
                face_feature_vectors.append(result)

        if face_feature_vectors:
            return np.array(face_feature_vectors).mean(axis=0)
        return np.zeros(128, dtype=int, order="C")
