import asyncio
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import litellm
from litellm import acompletion

from config.settings import settings
from src.models.llm import LLMError, LLMRequest, LLMResponse
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Single global LLM provider instance
_llm_provider: Optional['LLMProvider'] = None


class LLMProvider:
    """Universal LLM provider with automatic fallback support."""

    def __init__(self) -> None:
        """Initialize LLM provider from configuration."""
        self.config = settings.llm_config
        self.models = [self.config["primary_model"]] + self.config["fallback_models"]

        litellm.num_retries = self.config["retry_attempts"]
        litellm.request_timeout = self.config["timeout"]
        litellm.success_callback = ["logger"]
        litellm.failure_callback = ["logger"]

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response with automatic model fallback and retry logic."""
        for attempt in range(self.config["retry_attempts"]):
            start_time = time.time()

            params = {
                "messages": self._format_messages(request),
                "temperature": request.temperature if request.temperature is not None else self.config["temperature"],
                "max_tokens": request.max_tokens if request.max_tokens is not None else self.config["max_tokens"],
                "timeout": self.config["timeout"],
                "num_retries": self.config["retry_attempts"],
            }

            for i, model in enumerate(self.models):
                try:
                    response = await acompletion(model=model, **params)

                    message = response.choices[0].message
                    content = message.content or ""

                    if not hasattr(message, "reasoning_content"):
                        content = self._clean_response(content)

                    return LLMResponse(
                        content=content,
                        provider_name="llm",
                        model=model,
                        tokens_used=response.usage.total_tokens if response.usage else None,
                        latency_ms=(time.time() - start_time) * 1000,
                        timestamp=datetime.now(),
                    )

                except Exception as e:
                    if i == len(self.models) - 1:
                        error = LLMError(
                            provider="llm",
                            message=f"All models failed. Last error: {str(e)}",
                            retryable=self._is_retryable_error(e),
                        )
                        if not error.retryable or attempt == self.config["retry_attempts"] - 1:
                            raise error
                        await asyncio.sleep(2**attempt)
                        break
                    continue

        raise LLMError(provider="llm", message="Max retries exceeded", retryable=False)

    @staticmethod
    def _format_messages(request: LLMRequest) -> List[Dict[str, str]]:
        """Format request into message list."""
        messages = []
        if request.context:
            messages.append({"role": "system", "content": request.context})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    @staticmethod
    def _clean_response(content: str) -> str:
        """Clean response by removing XML-like tags and extra whitespace."""
        # Remove any XML-like tags (more generic approach)
        content = re.sub(r"<[^>]+>.*?</[^>]+>", "", content, flags=re.DOTALL)

        # Clean up whitespace
        content = re.sub(r"\n\s*\n", "\n", content)
        return content.strip()

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Check if error should be retried."""
        error_str = str(error).lower()

        if any(
            pattern in error_str
            for pattern in [
                "invalid api key",
                "unauthorized",
                "not found",
                "400",
                "401",
                "403",
                "404",
                "content policy",
                "context window",
            ]
        ):
            return False

        if any(
            pattern in error_str
            for pattern in [
                "rate limit",
                "timeout",
                "connection",
                "server error",
                "internal error",
                "503",
                "502",
                "500",
                "429",
            ]
        ):
            return True

        return True


# Global provider singleton
def get_llm_provider() -> LLMProvider:
    """Get or create the global LLM provider instance."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LLMProvider()
    return _llm_provider
