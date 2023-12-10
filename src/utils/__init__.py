import os
import bz2
import shutil
import asyncio

import numpy as np

from src.utils.log import SingletonLogger
from src.utils.translator import Translator

logger = SingletonLogger.get_instance().logger
message_translator = Translator().get_local_translation


def unzip_static(static_zip_path: str, static_unzip_path: str) -> bool:
    """
    解压静态资源文件。
    :param static_zip_path: 压缩文件路径
    :param static_unzip_path: 解压后文件路径
    :return: 解压成功返回 True，否则返回 False
    """
    try:
        with bz2.BZ2File(static_zip_path, "rb") as file:
            with open(static_unzip_path, "wb") as new_file:
                shutil.copyfileobj(file, new_file)
        os.remove(static_zip_path)
        return True
    except (FileNotFoundError, OSError, EOFError) as e:
        logger.error(e)
        return False


def return_euclidean_distance(feature_1: list[float], feature_2: list[float]) -> bool:
    """
    计算并返回两个特征向量之间的欧氏距离是否小于预设阈值。

    :param feature_1: 第一个特征向量，表示为浮点数列表。
    :param feature_2: 第二个特征向量，表示为浮点数列表。

    :return: 如果两个特征向量的欧氏距离小于或等于0.4，则返回 True，表示它们相似， 则返回 False，表示它们不相似。
    """
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("欧式距离: ", dist)

    if dist > 0.4:
        return False
    else:
        return True


async def init():
    async def database_init():
        """初始化数据库"""
        from src.database.db import create_db, test_db_connection
        if await test_db_connection() is True:
            logger.info(message_translator("ERRORS.DATABASE.CONNECTION.SUCCESS"))
        else:
            logger.error(message_translator("ERRORS.DATABASE.CONNECTION.FAILED"))
            await create_db()

    async def init_shape_predictor_model_file():
        """初始化人脸关键点检测模型文件"""
        import dlib

        from src.config import config
        try:
            _ = dlib.shape_predictor(
                f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat"
            )
            logger.info(message_translator("MODEL.MODEL_LOAD_SUCCESS"))
            _ = None
        except Exception as e:
            logger.error(message_translator("MODEL.MODEL_LOAD_FAILED"))
            logger.error(e)
            unzip_static(
                config.static_path / "shape_predictor_68_face_landmarks.dat.bz2",
                config.predictor_model_path / "shape_predictor_68_face_landmarks.dat"
            )

    async def init_face_recognition_model_file():
        """初始化人脸识别模型文件"""
        import dlib

        from src.config import config
        try:
            _ = dlib.face_recognition_model_v1(
                f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat"
            )
            logger.info(message_translator("MODEL.MODEL_LOAD_SUCCESS"))
            _ = None
        except Exception as e:
            logger.error(message_translator("MODEL.MODEL_LOAD_FAILED"))
            logger.error(e)
            unzip_static(
                config.static_path / "dlib_face_recognition_resnet_model_v1.dat.bz2",
                config.face_model_path / "dlib_face_recognition_resnet_model_v1.dat"
            )

    await asyncio.gather(
        database_init(),
        init_shape_predictor_model_file(),
        init_face_recognition_model_file()
    )

