import sys
import logging
from typing import TYPE_CHECKING

import loguru

from config import config

if TYPE_CHECKING:
    from loguru import Logger

logger: "Logger" = loguru.logger


class LoguruHandler(logging.Handler):

    def emit(self, record: logging.LogRecord):
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


default_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
logger.remove()
logger_id = logger.add(
    sys.stdout,
    level=config.log_level,
    format=default_format,
)
