import asyncio
import bz2
import os
import shutil

import numpy as np
from loguru import logger

from app.utils.translator import message_translator


def unzip_static(static_zip_path: str, static_unzip_path: str) -> bool:
    """
    解压静态资源文件。
    :param static_zip_path: 压缩文件路径
    :param static_unzip_path: 解压后文件路径
    :return: 解压成功返回 True, 否则返回 False
    """
    try:
        with bz2.BZ2File(static_zip_path, "rb") as file:
            with open(static_unzip_path, "wb") as new_file:
                shutil.copyfileobj(file, new_file)
        os.remove(static_zip_path)
        return True
    except (FileNotFoundError, OSError, EOFError) as e:
        logger.info(e)
        return False


def return_euclidean_distance(feature_1: list[float], feature_2: list[float]) -> float:
    """
    计算并返回两个特征向量之间的欧氏距离是否小于预设阈值。

    :param feature_1: 第一个特征向量, 表示为浮点数列表。
    :param feature_2: 第二个特征向量, 表示为浮点数列表。

    :return: 返回两个特征向量之间的欧氏距离。
    """
    feature_1_np = np.array(feature_1)
    feature_2_np = np.array(feature_2)
    return np.sqrt(np.sum(np.square(feature_1_np - feature_2_np)))


# async def database_init():
#     """初始化数据库"""
#
#     # if await test_db_connection() is True:
#     #     logger.info(("ERRORS.DATABASE.CONNECTION.SUCCESS"))
#     # else:
#     #     logger.info(("ERRORS.DATABASE.CONNECTION.FAILED"))
#     try:
#         import create_db
#     except SQLAlchemyError:
#         logger.info(message_translator("ERRORS.DATABASE.INIT.FAILED"))


async def init_shape_predictor_model_file():
    """初始化人脸关键点检测模型文件"""
    import dlib
    from app.config import config

    logger.info(message_translator("MODEL.LOAD.START") + "shape_predictor_model_file")
    try:
        _ = dlib.shape_predictor(f"{config.face_model_path}/shape_predictor_68_face_landmarks.dat")
        logger.info(message_translator("MODEL.LOAD.SUCCESS") + "shape_predictor_model_file")
        _ = None
    except Exception as e:
        logger.info(message_translator("MODEL.LOAD.SUCCESS") + "shape_predictor_model_file")
        logger.info(e)
        unzip_static(
            config.static_path / "shape_predictor_68_face_landmarks.dat.bz2",
            config.face_model_path / "shape_predictor_68_face_landmarks.dat",
        )


async def init_face_recognition_model_file():
    """初始化人脸识别模型文件"""
    import dlib
    from app.config import config

    logger.info(message_translator("MODEL.LOAD.START") + "face_recognition_model_file")
    try:
        _ = dlib.face_recognition_model_v1(f"{config.face_model_path}/dlib_face_recognition_resnet_model_v1.dat")
        logger.info(message_translator("MODEL.LOAD.SUCCESS") + "face_recognition_model_file")
        _ = None
    except Exception as e:
        logger.info(message_translator("MODEL.LOAD.FAILED") + "face_recognition_model_file")
        logger.info(e)
        unzip_static(
            config.static_path / "dlib_face_recognition_resnet_model_v1.dat.bz2",
            config.face_model_path / "dlib_face_recognition_resnet_model_v1.dat",
        )


async def init():
    """初始化"""
    await asyncio.gather(init_shape_predictor_model_file(), init_face_recognition_model_file())
