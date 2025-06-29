from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "insurance_agent"
    redis_url: str = "redis://localhost:6379"
    redis_ttl: int = 3600
    
    # LLM Configuration
    use_local_llm: bool = True
    openai_api_key: Optional[str] = None
    local_llm_model: str = "microsoft/DialoGPT-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Application Configuration
    debug: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()