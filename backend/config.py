from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    mistral_api_key: str
    mistral_model: str = "mistral-small-latest"
    chroma_db_path: str = "./chroma_db"
    max_history_messages: int = 20
    rag_top_k: int = 3

    class Config:
        env_file = ".env"


# Enter Model API Key in the param
@lru_cache
def get_settings() -> Settings:
    return Settings()
