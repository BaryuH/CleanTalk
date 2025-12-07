# backend/app/core/config.py
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    MODELS_STORE_DIR: Path = BASE_DIR / "models_store"
    TOXICITY_MODEL_PATH: Path = MODELS_STORE_DIR / "CleanTalk1" / "svm_model.pkl"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-distilroberta-v1"
    EMBEDDING_CACHE_DIR: Path = MODELS_STORE_DIR / "embeddings"
    PROJECT_NAME: str = "CleanTalk"
    API_V1_PREFIX: str = "/api/v1"
    class Config:
        env_file = ".env"
        case_sensitive = True
settings = Settings()
