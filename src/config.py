from pathlib import Path

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///files/data.sqlite3"
    static_path: Path = Path(__file__).parent / "resources" / "statics"
    predictor_model_path: Path = Path(__file__).parent / "resources" / "models"
    translation_path: Path = Path(__file__).parent / "resources" / "translations"
    language_type: str = "zh_CN"

    class Config:
        env_file = ".env"


config = Config()
