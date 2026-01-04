import os
from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Central settings for the unified app."""

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
