from datetime import timedelta
from pathlib import Path

from pydantic import SecretStr, model_validator
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
    sentry_environment: str = "production"

    class Config:
        env_file = "../.env"
        extra = "allow"

    @model_validator(mode="after")
    def set_sentry_env(self):
        if self.log_level == "DEBUG":
            self.sentry_environment = "development"
        else:
            self.sentry_environment = "production"


config = Config()
