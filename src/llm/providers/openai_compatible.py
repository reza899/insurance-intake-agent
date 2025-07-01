import time
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config.settings import settings
from src.models import LLMError, LLMRequest, LLMResponse

from .base import BaseLLMProvider


class OpenAICompatibleProvider(BaseLLMProvider):
    """Slim provider using LangChain for API models with fallback."""

    def __init__(self) -> None:
        """Initialize provider with generic OpenAI-compatible API clients."""
        super().__init__({"name": "openai_compatible_llm"})

        self.primary_model = self._create_primary_model()
        fallback_model = self._create_fallback_model() if settings.llm_fallback_provider else None

        self.chain = self.primary_model.with_fallbacks([fallback_model]) if fallback_model else self.primary_model

    @staticmethod
    def _create_primary_model() -> ChatOpenAI:
        """Create primary OpenAI-compatible model."""
        defaults = settings.llm_defaults

        if not settings.ext_provider_model:
            raise LLMError(
                provider="primary",
                message="Primary external provider model name is required but not configured",
                retryable=False,
            )

        return ChatOpenAI(
            model=settings.ext_provider_model,
            api_key=SecretStr(settings.ext_provider_api_key or "not-needed"),  # Some APIs don't need keys
            base_url=settings.ext_provider_base_url,
            temperature=defaults["default_temperature"],
            timeout=defaults["timeout"],
        )

    @staticmethod
    def _create_fallback_model() -> ChatOpenAI:
        """Create fallback OpenAI-compatible model."""
        defaults = settings.llm_defaults

        if not settings.ext_provider_fallback_model:
            raise LLMError(
                provider="fallback",
                message="Fallback external provider model name is required but not configured",
                retryable=False,
            )

        return ChatOpenAI(
            model=settings.ext_provider_fallback_model,
            api_key=SecretStr(
                settings.ext_provider_fallback_api_key or "not-needed"
            ),  # Support both API-key and Ollama
            base_url=settings.ext_provider_fallback_base_url,
            temperature=defaults["default_temperature"],
            timeout=defaults["timeout"],
        )

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using LangChain with automatic fallback."""
        start_time = time.time()

        try:
            # Build messages
            messages: List[BaseMessage] = []
            if request.context:
                messages.append(SystemMessage(content=request.context))
            messages.append(HumanMessage(content=request.prompt))

            # Apply runtime parameters if needed
            chain = self._bind_params(request) if request.temperature or request.max_tokens else self.chain

            response = await chain.ainvoke(messages)

            # Extract token usage properly from LangChain's response
            tokens_used = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = response.usage_metadata.get("total_tokens")
            elif hasattr(response, "response_metadata") and response.response_metadata:
                # Fallback to response metadata for some providers
                usage = response.response_metadata.get("token_usage", {})
                tokens_used = usage.get("total_tokens")

            return LLMResponse(
                content=str(response.content),
                provider_name="openai_compatible_llm",
                model=settings.ext_provider_model,
                tokens_used=tokens_used,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            raise LLMError(
                provider="openai_compatible_llm", message=f"Failed to generate response: {str(e)}", retryable=True
            )

    def _bind_params(self, request: LLMRequest):
        """Bind runtime parameters to model while preserving fallback."""
        params = {}
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        # Only bind if we have parameters
        if not params:
            return self.chain

        # Bind parameters to the chain to preserve fallback
        return self.chain.bind(**params)

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            await self.generate_response(LLMRequest(prompt="Hello", context=None, max_tokens=10, temperature=0.1))
            return True
        except Exception:
            return False
