from typing import Optional

from config.settings import settings
from src.llm.provider import get_llm_provider
from src.models import LLMRequest
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


async def create_llm_request_and_get_response(
    prompt: str,
    context: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """Create LLM request and get response with error handling."""
    try:
        # Use settings defaults if not provided
        request = LLMRequest(
            prompt=prompt,
            context=context,
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens
        )

        response = await get_llm_provider().generate_response(request)
        return response.content.strip() if response and response.content else None

    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return None


def create_standard_llm_request(
    prompt: str,
    context: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> LLMRequest:
    """Create a standard LLM request with settings defaults."""
    return LLMRequest(
        prompt=prompt,
        context=context,
        temperature=temperature or settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens
    )
