from loguru import logger
import logging
from datetime import datetime
from app.config import config


def setup_logger():
    logging.getLogger("werkzeug").handlers.clear()

    class LoguruHandler(logging.Handler):
        def emit(self, record):
            level = logger.level(record.levelname).name
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    loguru_handler = LoguruHandler()
    loguru_handler.setLevel(config.log_level)
    root_logger = logging.getLogger()
    root_logger.addHandler(loguru_handler)
    root_logger.setLevel(config.log_level)

    config.log_path.mkdir(parents=True, exist_ok=True)

    log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    log_file_path = config.log_path / log_file_name
    logger.add(
        log_file_path,
        rotation="1 day",
        retention="30 days",
        level=config.log_level,
        encoding="utf-8",
    )
