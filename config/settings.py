from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise Agentic RAG Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database & Vector Store
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # LLM & Embeddings
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_BASE_URL: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    LLM_MODEL: str = "gpt-4o"
    
    # RAG Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    RRF_K: int = 60
    TOP_K: int = 5
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
