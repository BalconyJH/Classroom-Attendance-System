import os
from typing import Union

import cv2
from src.config import config

from app.utils import logger

HAAR_CASCADE_PATH = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_alt2.xml")


class Camera:
    def __init__(self, video_stream: Union[int, str]):
        self.video_stream = video_stream

    async def check_camera(self):
        stream = cv2.VideoCapture(self.video_stream)
        logger.debug(f"Stream Fps: {round(stream.get(cv2.CAP_PROP_FPS), 2)}")
        logger.debug(f"Stream Backend: {stream.getBackendName()}")
        logger.debug(
            f"Stream Size(W*H): {stream.get(cv2.CAP_PROP_FRAME_WIDTH)}*" f"{stream.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
        )
        logger.info("Camera Check Start, press ESC to exit")
        while stream.isOpened():
            ret, frame = stream.read()
            if not ret:
                logger.error("Camera is not working")
                break
            cv2.imshow("Camera Checking", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        stream.release()
        cv2.destroyAllWindows()

    async def face_gather(self, uid):
        stream = cv2.VideoCapture(self.video_stream)
        face_classifier = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
        increment_num = 0
        dataset_num: int = 15

        path = f"{config.cache_path}/dataset/images/{uid}"
        if not os.path.exists(path):
            os.makedirs(path)

        while True:
            ret, img = stream.read()
            if not ret:
                break

            img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)

            # detect faces using haar cascade detector
            faces = face_classifier.detectMultiScale(img, 1.0485258, 6)
            for x, y, w, h in faces:
                increment_num += 1

                # 保存裁剪的脸部图像
                cv2.imwrite(f"{path}/{increment_num}.jpg", img)

                # 在原始图像上画框
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 更新进度条
                progress = (increment_num / dataset_num) * 100
                progress_bar = (
                    f"Progress:" f" [{'#' * int(progress / 10)}{'.' * (10 - int(progress / 10))}]" f" {int(progress)}%"
                )
                logger.info(progress_bar, end="\r")
            cv2.imshow("Capturing Face", img)
            if cv2.waitKey(1) & 0xFF == 27 or increment_num == dataset_num:
                break
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break
        stream.release()
        cv2.destroyAllWindows()

    @staticmethod
    async def load_face_classifier():
        pass

    async def face_recognition(self):
        logger.info("Face Recognition Start, press ESC to exit")
        stream = cv2.VideoCapture(self.video_stream)

        while True:
            _, img = stream.read()
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.imshow("Face Recognition", img)
            # todo: face recognition powered by dlib
            if cv2.waitKey(1) & 0xFF == 27:
                break
