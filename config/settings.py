from typing import Any, Dict, List
from pathlib import Path
import yaml
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_app_config() -> Dict[str, Any]:
    """Load application configuration from YAML file."""
    config_path = Path(__file__).parent / "app_config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            result = yaml.safe_load(file)
            return dict(result) if result else {}
    except Exception as e:
        print(f"Warning: Failed to load config file: {e}. Using defaults.")
        return {}


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config files."""
    
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "insurance_agent"
    
    llm_primary_model: str = "ollama/gemma3:4b"
    llm_fallback_models: str = "gpt-4o-mini,huggingface/google/gemma-3-4b-it"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_timeout: int = 60
    llm_retry_attempts: int = 3

    # Application Configuration
    debug: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501
    default_timeout: int = 5
    long_timeout: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'
    )
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._app_config = load_app_config()

    @property
    def required_fields(self) -> List[str]:
        """Get required fields list."""
        return list(self._app_config.get("required_fields", []))
    
    @property
    def duplicate_detection_config(self) -> Dict[str, Any]:
        """Get duplicate detection configuration."""
        return dict(self._app_config.get("duplicate_detection", {}))
    
    @property
    def database_collections_config(self) -> Dict[str, str]:
        """Get database collections configuration."""
        return dict(self._app_config.get("database", {}).get("collections", {}))
    
    @property
    def llm_intents_config(self) -> Dict[str, str]:
        """Get LLM intents configuration."""
        return dict(self._app_config.get("llm_intents", {}))
    
    @property
    def conversation_status_config(self) -> Dict[str, str]:
        """Get conversation status configuration."""
        return dict(self._app_config.get("conversation_status", {}))
    
    @property
    def prompts(self) -> Dict[str, str]:
        """Get all prompts from app config."""
        return dict(self._app_config.get("prompts", {}))
    
    @property
    def response_templates(self) -> Dict[str, str]:
        """Get response templates from app config."""
        return dict(self._app_config.get("response_templates", {}))
    
    def get_prompt(self, prompt_name: str) -> str:
        """Get specific prompt by name."""
        return self.prompts.get(prompt_name, "")
    
    def get_response_template(self, template_name: str) -> str:
        """Get specific response template by name."""
        return self.response_templates.get(template_name, "")
    
    def get_config(self, config_name: str, default: Any = None) -> Any:
        """Get any configuration value by name."""
        return self._app_config.get(config_name, default)

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        fallback_models = []
        if self.llm_fallback_models:
            fallback_models = [model.strip() for model in self.llm_fallback_models.split(",")]
        
        return {
            "primary_model": self.llm_primary_model,
            "fallback_models": fallback_models,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "timeout": self.llm_timeout,
            "retry_attempts": self.llm_retry_attempts,
        }


settings = Settings()