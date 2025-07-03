from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """Request model for LLM interactions."""

    prompt: str = Field(..., description="The input prompt for the LLM")
    context: Optional[str] = Field(None, description="Additional context for the prompt")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)


class LLMResponse(BaseModel):
    """Response model for LLM interactions."""

    content: str = Field(..., description="Generated response content")
    provider_name: str = Field(..., description="Provider that generated the response")
    model: str = Field(..., description="Specific model used")
    tokens_used: Optional[int] = Field(None, description="Number of tokens consumed")
    latency_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LLMError(Exception):
    """Exception for LLM-related errors."""

    def __init__(self, provider: str, message: str, retryable: bool = False):
        self.provider = provider
        self.message = message
        self.retryable = retryable
        super().__init__(f"{provider}: {message}")


class LLMServiceResponse(BaseModel):
    """Response format from LLM service."""

    content: str = Field(..., description="Processed response content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    success: bool = Field(..., description="Whether the request was successful")
    error_message: Optional[str] = Field(None, description="Error details if failed")
