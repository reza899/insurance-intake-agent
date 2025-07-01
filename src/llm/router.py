"""
LLM Router - Simple router delegating all provider creation to factory.
"""

from config.settings import settings
from src.models import LLMRequest, LLMResponse

from .factory import LLMProviderFactory


class LLMRouter:
    """Simple router using factory for all provider creation."""

    def __init__(self) -> None:
        """Initialize router with appropriate provider via factory."""
        provider_name = "huggingface" if settings.use_hf_local else "openai_compatible_llm"
        self.provider = LLMProviderFactory.create_provider(provider_name)
        self.provider_type = provider_name

    async def route_request(self, request: LLMRequest) -> LLMResponse:
        """Route request to provider with retry logic."""
        return await self.provider.generate_response_with_retry(request)

    async def health_check(self) -> dict:
        """Check health of router."""
        is_healthy = await self.provider.health_check()
        return {self.provider_type: is_healthy}
