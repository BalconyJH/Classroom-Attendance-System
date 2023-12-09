# 解压静态资源
import os
import bz2
import shutil
import asyncio

from src.utils.log import SingletonLogger
from src.utils.translator import Translator

logger = SingletonLogger.get_instance()
message_translator = Translator().get_local_translation


def unzip_static(static_zip_path: str, static_unzip_path: str) -> bool:
    try:
        with bz2.BZ2File(static_zip_path, "rb") as file:
            with open(static_unzip_path, "wb") as new_file:
                shutil.copyfileobj(file, new_file)
        os.remove(static_zip_path)
        return True
    except (FileNotFoundError, OSError, EOFError) as e:
        logger.error(e)
        return False


async def init():
    async def database_init():
        from src.resources.database.db import create_db
        await create_db()

    async def release_model_file():
        from src.config import config
        unzip_static(
            config.static_path / "shape_predictor_68_face_landmarks.dat.bz2",
            config.predictor_model_path / "shape_predictor_68_face_landmarks.dat"
        )

    await asyncio.gather(database_init())
