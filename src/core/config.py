from pathlib import Path
from typing import Any, Dict

import yaml

from config.settings import settings


def _load_app_config() -> Dict[str, Any]:
    """Load application configuration from YAML file."""
    config_path = Path("config/app_config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_config() -> Dict[str, Any]:
    """Get combined configuration from environment and YAML."""
    app_config = _load_app_config()

    # Add environment settings to config
    config = {
        "app": app_config,
        "env": {
            "mongodb_url": settings.mongodb_url,
            "mongodb_database": settings.mongodb_database,
            "redis_url": settings.redis_url,
            "redis_ttl": settings.redis_ttl,
            "use_local_llm": settings.use_local_llm,
            "openai_api_key": settings.openai_api_key,
            "local_llm_model": settings.local_llm_model,
            "local_embedding_model": settings.local_embedding_model,
            "debug": settings.debug,
            "log_level": settings.log_level,
            "host": settings.host,
            "api_port": settings.api_port,
            "ui_port": settings.ui_port,
        },
    }

    return config
