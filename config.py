from pathlib import Path
from datetime import timedelta

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    log_level: str = "INFO"
    cache_path: Path = Path(__file__).parent / "resources" / "caches"
    static_path: Path = Path(__file__).parent / "resources" / "statics"
    face_model_path: Path = Path(__file__).parent / "resources" / "models"
    translation_path: Path = Path(__file__).parent / "resources" / "translations"
    language_type: str = "zh_CN"

    # Flask specific settings
    secret_key: SecretStr = "123456"  # 使用 SecretStr 以保护敏感信息
    jinja_env_auto_reload: bool = True
    send_file_max_age_default: timedelta = timedelta(seconds=1)

    # SQLAlchemy settings
    sqlalchemy_database_uri: SecretStr = (
        "mysql+pymysql://root:070499@localhost:3306/test?charset=utf8"
    )
    sqlalchemy_track_modifications: bool = True

    class Config:
        env_file = ".env"


config = Config()
