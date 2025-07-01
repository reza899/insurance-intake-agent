from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_app_config() -> Dict[str, Any]:
    """Load application configuration from YAML file."""
    config_path = Path(__file__).parent / "app_config.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        result = yaml.safe_load(file)
        return dict(result) if result else {}


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config files."""
    
    # Database Configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "insurance_agent"
    
    # LLM Configuration
    use_hf_local: bool = False
    llm_primary_provider: str = "openai"
    llm_fallback_provider: str = "external_api"
    
    # External API Provider (OpenAI-compatible APIs)
    ext_provider_api_key: Optional[str] = None
    ext_provider_model: str = "gpt-4o-mini"
    ext_provider_base_url: str = "https://api.openai.com/v1"
    ext_provider_fallback_api_key: Optional[str] = None
    ext_provider_fallback_model: str = "gemma3:4b"
    ext_provider_fallback_base_url: str = "http://localhost:11434/v1"
    
    # Local Model Provider (HuggingFace)
    local_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    local_model_device: str = "cpu"
    
    # Application Configuration
    debug: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Load app config after initialization
        self._app_config = load_app_config()
    
    @property
    def app_config(self) -> Dict[str, Any]:
        """Get the full app configuration."""
        return self._app_config
    
    @property
    def system_prompt(self) -> str:
        """Get system prompt from app config."""
        return str(self._app_config.get("prompts", {}).get("system", ""))
    
    @property
    def llm_defaults(self) -> Dict[str, Any]:
        """Get LLM defaults from app config."""
        return dict(self._app_config.get("llm", {}))
    
    @property
    def conversation_config(self) -> Dict[str, Any]:
        """Get conversation configuration."""
        return dict(self._app_config.get("conversation", {}))
    
    @property
    def duplicate_detection_config(self) -> Dict[str, Any]:
        """Get duplicate detection configuration."""
        return dict(self._app_config.get("duplicate_detection", {}))
    
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


# Global settings instance
settings = Settings()