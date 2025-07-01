from typing import Dict, Type

from .providers.base import BaseLLMProvider
from .providers.huggingface import HuggingFaceProvider
from .providers.openai_compatible import OpenAICompatibleProvider


class LLMProviderFactory:
    """Factory for creating LLM providers with extensible registry."""

    # Provider registry for extensibility
    _providers: Dict[str, Type[BaseLLMProvider]] = {
        "huggingface": HuggingFaceProvider,
        "openai_compatible_llm": OpenAICompatibleProvider,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Register a new provider type."""
        cls._providers[name] = provider_class

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())

    @classmethod
    def create_provider(cls, provider_name: str) -> BaseLLMProvider:
        """Create an LLM provider instance."""
        if provider_name not in cls._providers:
            available = ", ".join(cls.get_available_providers())
            raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")

        provider_class = cls._providers[provider_name]
        return provider_class()
