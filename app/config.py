from pathlib import Path
from datetime import timedelta

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    log_level: str = "INFO"
    cache_path: Path = Path(__file__).parent / "static" / "caches"
    static_path: Path = Path(__file__).parent / "static" / "compress_models"
    face_model_path: Path = Path(__file__).parent / "static" / "models"
    translation_path: Path = Path(__file__).parent / "static" / "translations"
    language_type: str = "zh_CN"

    # Flask specific settings
    secret_key: SecretStr
    jinja_env_auto_reload: bool = True
    send_file_max_age_default: timedelta = timedelta(seconds=1)

    # SQLAlchemy settings
    sqlalchemy_database_uri: SecretStr
    sqlalchemy_track_modifications: bool = True

    # sentry settings
    sentry_dsn: SecretStr
    enable_tracing: bool = False

    class Config:
        env_file = "../.env"
        extra = "allow"


config = Config()
