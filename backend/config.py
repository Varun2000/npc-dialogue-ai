from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    mistral_api_key: str
    mistral_model: str = "mistral-small-latest"
    chroma_db_path: str = "./chroma_db"
    max_history_messages: int = 20
    rag_top_k: int = 3

    class Config:
        env_file = str(ENV_FILE)


@lru_cache
def get_settings() -> Settings:
    return Settings()
