from pydantic_settings import BaseSettings


class Config(BaseSettings):
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///files/data.db"

    class Config:
        env_file = ".env"


config = Config()
