from pydantic_settings import BaseSettings


class Config(BaseSettings):
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


config = Config()
