from typing import Optional
from decimal import Decimal
from pydantic import Field
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

    # Full-run evaluation uses local inference unless explicitly priced otherwise.
    EVAL_LLM_BASE_URL: str = "http://127.0.0.1:8091/v1"
    EVAL_LLM_MODEL: str = "evidencerag-local"
    EVAL_LLM_LOCAL: bool = True
    EVAL_LLM_API_KEY: str = ""
    EVAL_BUDGET_USD: Decimal = Field(default=Decimal("2.99"), gt=0, lt=3)
    EVAL_INPUT_USD_PER_MILLION: Decimal = Field(default=Decimal("0"), ge=0)
    EVAL_OUTPUT_USD_PER_MILLION: Decimal = Field(default=Decimal("0"), ge=0)
    EVAL_REQUEST_INTERVAL_SECONDS: float = Field(default=12, ge=0)
    EVAL_RATE_LIMIT_RETRIES: int = Field(default=6, ge=0)
    
    # RAG Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    RRF_K: int = 60
    TOP_K: int = 5
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
