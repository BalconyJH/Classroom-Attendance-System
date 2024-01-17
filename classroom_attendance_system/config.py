from pathlib import Path
from typing import Union

import pytz
from pydantic import Extra
from pytz.tzinfo import DstTzInfo
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    log_level: Union[int, str] = "INFO"
    timezone: DstTzInfo = pytz.timezone("Asia/Shanghai")
    database_url: str = "sqlite+aiosqlite:///resources/data/data.sqlite3"
    cache_path: Path = Path(__file__).parent / "resources" / "caches"
    static_path: Path = Path(__file__).parent / "resources" / "statics"
    face_model_path: Path = Path(__file__).parent / "resources" / "models"
    translation_path: Path = Path(__file__).parent / "resources" / "translations"
    language_type: str = "zh_CN"
    sentry_dsn: str = ""

    class Config:
        env_file = ".env"
        extra = Extra.allow


config = Config()
