from pydantic import BaseSettings


class Config(BaseSettings):
    log_level: str = "INFO"

    class Config:
        env_file = '.env'


config = Config()
