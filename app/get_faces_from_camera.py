import cv2
import dlib
import numpy as np

from . import app

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()


def extract_and_resize_face(image: np.ndarray, face_rect: dlib.rectangle, scale_factor: int = 2) -> np.ndarray:
    """
    提取并放大图像中的人脸区域。

    :param image: 原始图像，类型为 np.ndarray。
    :param face_rect: 人脸的矩形框，类型为 dlib.rectangle。
    :param scale_factor: 放大倍数，默认为 2。
    :return: 放大后的人脸图像区域，类型为 np.ndarray。
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
    :param image_shape: 图像的尺寸。
    :param bounds: 允许的最大范围。
    :return: 如果人脸在范围内，返回True；否则返回False。
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


class FaceRegister:
    def __init__(self, logger):
        self.logger = logger

    async def check_camera(self):
        stream = cv2.VideoCapture(self.video_stream)
        app.logger.debug(f"Stream Fps: {round(stream.get(cv2.CAP_PROP_FPS), 2)}")
        self.logger.debug(f"Stream Backend: {stream.getBackendName()}")
        self.logger.debug(
            f"Stream Size(W*H): {stream.get(cv2.CAP_PROP_FRAME_WIDTH)}*" f"{stream.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
        )
        self.logger.info("Camera Check Start, press ESC to exit")
        while stream.isOpened():
            ret, frame = stream.read()
            if not ret:
                self.logger.error("Camera is not working")
                break
            cv2.imshow("Camera Checking", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        stream.release()
        cv2.destroyAllWindows()

    def process_single_face_image(self, path: str) -> str:
        # 读取人脸图像
        img_rd = cv2.imread(path)
        # Dlib的人脸检测器
        faces: list[dlib.rectangle] = detector(img_rd, 0)

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
