import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.models import LLMError, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers with retry capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize provider with optional configuration."""
        self.config = config or {}
        self.name = self.config.get("name", "unknown")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)

    async def generate_response_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Generate response with exponential backoff retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.generate_response(request)
            except Exception as e:
                last_error = e

                if attempt == self.max_retries:
                    # Final attempt failed
                    logger.error(f"Provider {self.name} failed after {self.max_retries + 1} attempts: {e}")
                    if isinstance(e, LLMError) and not e.retryable:
                        raise e
                    raise LLMError(
                        provider=self.name,
                        message=f"Failed after {self.max_retries + 1} attempts: {str(e)}",
                        retryable=False,
                    )

                # Check if error is retryable
                if isinstance(e, LLMError) and not e.retryable:
                    raise e

                # Exponential backoff
                delay = self.retry_delay * (2**attempt)
                logger.warning(f"Provider {self.name} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        raise last_error or LLMError(provider=self.name, message="Unknown error", retryable=False)

    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for the given request."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and available."""
        pass

    def validate_config(self, required_fields: list[str]) -> None:
        """Validate that required configuration fields are present."""
        missing = [field for field in required_fields if not getattr(self, field, None)]
        if missing:
            raise LLMError(
                provider=self.name, message=f"Missing required configuration: {', '.join(missing)}", retryable=False
            )
