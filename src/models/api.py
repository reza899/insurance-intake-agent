from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationHistoryItem(BaseModel):
    """Model for a single conversation history item."""

    role: str = Field(..., description="Role of the speaker (user/assistant)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User message")
    conversation_history: List[ConversationHistoryItem] = Field(
        default_factory=list, description="Previous conversation messages"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(..., description="Agent's response")
    extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Data extracted so far")
    missing_fields: List[str] = Field(default_factory=list, description="Fields still missing")
    status: str = Field(default="processing", description="Current conversation status")
    errors: Optional[List[str]] = Field(None, description="Validation errors if any")
    duplicates: Optional[List[Dict[str, Any]]] = Field(None, description="Potential duplicate registrations found")
    registration_id: Optional[str] = Field(None, description="Registration ID if successfully created")
    error: Optional[str] = Field(None, description="Error message if processing failed")
