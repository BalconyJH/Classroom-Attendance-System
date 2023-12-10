from pathlib import Path

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///resources/data/data.sqlite3"
    cache_path: Path = Path(__file__).parent / "resources" / "caches"
    static_path: Path = Path(__file__).parent / "resources" / "statics"
    face_model_path: Path = Path(__file__).parent / "resources" / "models"
    translation_path: Path = Path(__file__).parent / "resources" / "translations"
    language_type: str = "zh_CN"

    class Config:
        env_file = ".env"


config = Config()
