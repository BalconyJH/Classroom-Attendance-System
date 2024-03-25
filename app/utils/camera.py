import os
import time
from typing import Union

import cv2
import dlib
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from sklearn.neighbors import NearestNeighbors

from app import config
from app.data_access.student_faces_repo import StudentFaces

HAAR_CASCADE_PATH = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_alt2.xml")


class Camera:
    def __init__(self, video_stream: Union[int, str] = 0):
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

            img = cv2.resize(
                img,
                (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)),
                interpolation=cv2.INTER_AREA,
            )

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


class VideoCamera:
    def __init__(self, video_stream: Union[int, str] = 0):
        # sklearn NearestNeighbors model initialization
        self.face_recognizer = NearestNeighbors(n_neighbors=1, algorithm="auto")
        self.match_threshold = 0.3

        # face changed flag
        self.faces_changed = None

        # dlib face recognition model initialization
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat")
        self.face_recognition_model = dlib.face_recognition_model_v1(
            f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat"
        )

        # font
        self.font = config.font_path

        # video stream
        self.video = cv2.VideoCapture(video_stream)

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_cnt = 0

        # known faces and names, from the Faces table in the database
        self.features_known_list = []
        self.name_known_list = []

        self.current_frame_name_list = ["Unknown"]

        # Face counter
        self.last_frame_faces_cnt = 0
        self.current_frame_face_cnt = 0

        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        # load database into memory when the camera is initialized
        if not self.get_face_database():
            logger.error("No face features found in database.")
            return
        else:
            self.train_face_recognizer()
            logger.info("Face features loaded from database.")

    def __del__(self):
        self.video.release()

    def train_face_recognizer(self) -> None:
        """
        训练人脸识别模型, 并初始化人脸识别器。
        :return: None
        """
        feature_database = np.array(self.features_known_list)
        self.face_recognizer.fit(feature_database)

    def get_face_database(self):
        from_db_all_features = StudentFaces.get_all_student_faces()
        if not from_db_all_features:
            logger.error("No face features found in database.")
            return False

        self.name_known_list = [face.s_id for face in from_db_all_features]
        self.features_known_list = [
            [float(feature) if feature else 0 for feature in str(face.feature).split(",")]
            for face in from_db_all_features
        ]
        return True

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # def init_camera(self) -> bool:
    #     """
    #     摄像头初始化检查
    #     :return: bool
    #     """
    #     try:
    #         if not self.video.isOpened():
    #             logger.error("摄像头未开启")
    #             return False
    #         elif self.video.isOpened():
    #             if

    def get_frame(self):
        # 检查是否能够获取人脸数据库
        # if self.get_face_database() is False:
        #     return

        while self.video.isOpened():
            self.frame_cnt += 1
            flag, img_rd = self.video.read()
            # 调整图像大小以加快处理速度
            img_rd = cv2.resize(
                img_rd,
                (int(img_rd.shape[1] * 0.5), int(img_rd.shape[0] * 0.5)),
                interpolation=cv2.INTER_AREA,
            )

            # 检测当前帧的人脸
            logger.info(f"img size: {img_rd.shape}")
            faces: dlib.rectangles = self.face_detector(img_rd, 0)
            logger.debug(f"当前帧人脸数: {len(faces)}")
            if len(faces) > 1:
                logger.warning("multiple faces detected")
                continue
            # self.update_face_counts(len(faces))

            match_name = "Unknown"
            if len(faces) > 0:
                face_box, match_name = self.recognize_faces(faces[0], img_rd)
                img_rd = self.draw_name(img_rd, match_name, face_box)

            self.draw_note(img_rd)
            self.update_fps()

            # 返回处理后的图像
            status, jpeg = cv2.imencode(".jpg", img_rd)
            return jpeg.tobytes(), match_name

    def update_face_counts(self, current_count):
        """更新当前帧和上一帧的人脸数量。"""
        logger.debug(f"当前帧人脸数: {current_count}")
        logger.debug(f"上一帧人脸数: {self.last_frame_faces_cnt}")
        logger.debug(f"当前帧人脸数: {self.current_frame_face_cnt}")
        self.last_frame_faces_cnt = self.current_frame_face_cnt
        self.current_frame_face_cnt = current_count
        self.faces_changed = self.last_frame_faces_cnt != self.current_frame_face_cnt

    def reset_face_lists(self):
        """重置与人脸识别相关的列表。"""
        self.current_frame_name_list = []

    def recognize_faces(
        self, faces: dlib.rectangle, img_rd: np.ndarray
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], str]:
        """
        识别人脸并返回人脸框和匹配的名字。
        :param faces: 人脸实例。
        :param img_rd: 原始图像。
        :return: 人脸框和匹配的名字。
        """
        x, y, w, h = faces.left(), faces.top(), faces.width(), faces.height()
        face_box = (x, y), (x + w, y + h)

        # 提取人脸特征
        shape = self.shape_predictor(img_rd, faces)
        face_descriptor = self.face_recognition_model.compute_face_descriptor(img_rd, shape)
        face_feature = np.array(face_descriptor)
        face_feature_reshaped = face_feature.reshape(1, -1)

        distances, indices = self.face_recognizer.kneighbors(face_feature_reshaped)

        if distances[0][0] < self.match_threshold:
            match_index = indices[0][0]
            match_name = self.name_known_list[match_index]
        else:
            match_name = "Unknown"

        cv2.rectangle(img_rd, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return face_box, match_name

    def process_faces(self, faces, img_rd) -> np.ndarray:
        """处理当前帧的人脸识别逻辑。"""
        logger.debug("开始处理人脸识别逻辑")
        if faces:
            face_box, match_name = self.recognize_faces(faces[0], img_rd)
            img_rd = self.draw_name(img_rd, match_name, face_box)
            return img_rd
        else:
            self.reclassify_interval_cnt = 0

    def update_and_process_faces(self, faces: dlib.rectangles, img_rd: np.ndarray) -> np.ndarray:
        """更新人脸数量并处理人脸识别。"""
        current_count = len(faces)
        self.update_face_counts(current_count)

        # 如果人脸数量发生变化或达到重分类间隔
        logger.debug(f"{self.faces_changed}, {self.reclassify_interval_cnt >= self.reclassify_interval}")
        if self.faces_changed or self.reclassify_interval_cnt >= self.reclassify_interval:
            # 重置相关列表和计数器
            self.reset_face_lists()
            self.reclassify_interval_cnt = 0  # 重置计数器

            # 然后处理人脸识别逻辑
            return self.process_faces(faces, img_rd)

        # 如果存在未知人脸, 增加重分类计数器
        if "Unknown" in self.current_frame_name_list:
            self.reclassify_interval_cnt += 1
        return img_rd

    def release(self):
        """释放视频捕获对象的资源。"""
        if self.video.isOpened():
            self.video.release()
            logger.info("VideoCapture资源已释放")

    def draw_name(
        self,
        img_rd: np.ndarray,
        text: str,
        box: tuple[tuple[int, int], tuple[int, int]],
    ) -> np.ndarray:
        """
        在图像上绘制人脸名字, 注意, 该方法使用的是PIL库。
        :param img_rd: 原始图像。
        :param text: 要绘制的文本。
        :param box: 人脸的位置。
        :return: 绘制了名字的图像。
        """
        font_size = 24

        img_pil = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        font = ImageFont.truetype(self.font, font_size)

        text_position = box[0]

        draw.text(text_position, text, font=font, fill=(0, 255, 0))

        img_rd = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return img_rd

    def draw_note(self, img_rd: np.ndarray):
        # note
        cv2.putText(
            img_rd,
            "One person at a time",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        # fps info
        cv2.putText(
            img_rd,
            "FPS: " + str(self.fps.__round__(2)),
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
