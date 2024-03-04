import os
import time
from typing import Union

import cv2
import dlib
import numpy as np
from PIL import ImageFont, Image, ImageDraw

from app import config

from loguru import logger

from app.data_access.student_faces_repo import StudentFaces

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


class VideoCamera:
    def __init__(self, video_stream: Union[int, str] = 0):
        self.faces_changed = None
        self.match_threshold = 0.5
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat")
        self.face_recognition_model = dlib.face_recognition_model_v1(
            f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat"
        )
        self.font = cv2.FONT_ITALIC
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(video_stream)

        # 统计 FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

        # 统计帧数
        self.frame_cnt = 0

        # 用来存储所有录入人脸特征的数组
        self.features_known_list = []
        # 用来存储录入人脸名字
        self.name_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标
        self.last_frame_centroid_list = []
        self.current_frame_centroid_list = []

        # 用来存储当前帧检测出目标的名字
        self.current_frame_name_list = []

        # 上一帧和当前帧中人脸数的计数器
        self.last_frame_faces_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征
        self.current_frame_face_feature_list = []

        # 控制再识别的后续帧数
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    def __del__(self):
        self.video.release()

    def get_face_database(self, cid):
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

    def get_frame(self, cid):
        # 检查是否能够获取人脸数据库
        if self.get_face_database(cid) is False:
            return

        stream = self.video
        while stream.isOpened():
            self.frame_cnt += 1
            flag, img_rd = stream.read()
            # 调整图像大小以加快处理速度
            img_rd = cv2.resize(
                img_rd, (int(img_rd.shape[1] * 0.5), int(img_rd.shape[0] * 0.5)), interpolation=cv2.INTER_AREA
            )

            # 检测当前帧的人脸
            faces = self.face_detector(img_rd, 0)
            logger.debug(f"当前帧人脸数: {len(faces)}")
            self.update_face_counts(len(faces))

            filename = "attendance.txt"
            with open(filename, "a", encoding="utf-8") as file:
                self.update_and_process_faces(faces, img_rd, file)

            img_rd = self.draw_name(img_rd)

            # 添加说明文字和更新FPS
            self.draw_note(img_rd)
            self.update_fps()

            # 返回处理后的图像
            status, jpeg = cv2.imencode(".jpg", img_rd)
            return jpeg.tobytes()

    def update_face_counts(self, current_count):
        """更新当前帧和上一帧的人脸数量。"""
        self.last_frame_faces_cnt = self.current_frame_face_cnt
        self.current_frame_face_cnt = current_count
        self.faces_changed = self.last_frame_faces_cnt != self.current_frame_face_cnt

    def reset_face_lists(self):
        """重置与人脸识别相关的列表。"""
        self.current_frame_face_position_list = []
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_feature_list = []
        self.current_frame_name_list = []

    def recognize_faces(self, faces, img_rd):
        # 遍历当前帧检测到的每个人脸
        for face in faces:
            # 获取人脸区域的坐标
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # 提取人脸特征
            shape = self.shape_predictor(img_rd, face)
            face_descriptor = self.face_recognition_model.compute_face_descriptor(img_rd, shape)
            face_feature = np.array(face_descriptor)

            # 初始化匹配结果
            match_name = "Unknown"
            min_distance = float("inf")

            # 进行特征匹配
            for i, known_feature in enumerate(self.features_known_list):
                distance = np.linalg.norm(face_feature - np.array(known_feature))
                if distance < min_distance and distance < self.match_threshold:
                    min_distance = distance
                    match_name = self.name_known_list[i]  # noqa

            # 在图像上绘制人脸框并标注识别结果
            cv2.rectangle(img_rd, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img_rd

    def faces_count_changed(self):
        """检查当前帧和上一帧的人脸数量是否发生变化。"""
        return self.current_frame_face_cnt != self.last_frame_faces_cnt

    def process_faces(self, faces, img_rd, file):
        """处理当前帧的人脸识别逻辑。"""
        if faces:
            self.recognize_faces(faces, img_rd)
        else:
            # 如果当前帧没有检测到人脸, 重置相关计数器
            self.reclassify_interval_cnt = 0

    def update_and_process_faces(self, faces, img_rd, file):
        """更新人脸数量并处理人脸识别。"""
        current_count = len(faces)
        self.update_face_counts(current_count)

        # 如果人脸数量发生变化或达到重分类间隔, 则重置相关列表和计数器
        if self.faces_changed or self.reclassify_interval_cnt >= self.reclassify_interval:
            self.reset_face_lists()
            self.reclassify_interval_cnt = 0  # 重置计数器

        # 如果存在未知人脸, 增加重分类计数器
        if "unknown" in self.current_frame_name_list:
            self.reclassify_interval_cnt += 1

        # 如果人脸数量发生变化, 或者达到了重分类间隔, 则处理人脸识别逻辑
        if self.faces_changed or self.reclassify_interval_cnt >= self.reclassify_interval:
            self.process_faces(faces, img_rd, file)

    def release(self):
        """释放视频捕获对象的资源。"""
        if self.video.isOpened():
            self.video.release()
            logger.info("VideoCapture资源已释放")

    def draw_label(self, img, text, pos, bg_color):
        """
        在图像上绘制带有背景颜色的文本标签。

        :param img: 原始图像。
        :param text: 要绘制的文本。
        :param pos: 文本的左上角坐标。
        :param bg_color: 背景颜色。
        """
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(text, self.font, font_scale, font_thickness)[0]
        text_x = pos[0]
        text_y = pos[1] - 5
        cv2.rectangle(img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), bg_color, cv2.FILLED)
        cv2.putText(img, text, (text_x, text_y), self.font, font_scale, (255, 255, 255), font_thickness)

    def draw_name(self, img_rd: np.ndarray) -> np.ndarray:
        if self.current_frame_face_position_list and self.current_frame_name_list:
            font = ImageFont.truetype("simsun.ttc", 30)
            img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            draw.text(
                xy=self.current_frame_face_position_list[0],
                text=self.current_frame_name_list[0],
                font=font,
            )
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd

    def draw_note(self, img_rd: np.ndarray):
        cv2.putText(
            img_rd,
            "FPS:   " + str(self.fps.__round__(2)),
            (20, 100),
            self.font,
            0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
