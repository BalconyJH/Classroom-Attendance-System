import sys
import logging
from typing import TYPE_CHECKING

import loguru

from src.config import config

if TYPE_CHECKING:
    from loguru import Logger  # noqa F401


class SingletonLogger:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.logger = loguru.logger
        self._configure_logger()
        self._configure_sqlalchemy_logger()

    def _configure_logger(self):
        default_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        self.logger.remove()
        self.logger.add(
            sys.stdout,
            level=config.log_level,
            format=default_format,
        )

    @staticmethod
    def _configure_sqlalchemy_logger():
        sqlalchemy_logger = logging.getLogger("sqlalchemy.engine")
        sqlalchemy_logger.addHandler(LoguruHandler())
        sqlalchemy_logger.setLevel(config.log_level)


class LoguruHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        logger = SingletonLogger.get_instance().logger
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )
