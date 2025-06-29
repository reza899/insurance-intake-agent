from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Message type enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """Individual conversation message model."""

    type: MessageType
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ConversationSession(BaseModel):
    """Complete conversation session model."""

    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Data collection progress
    customer_data_complete: bool = False
    car_data_complete: bool = False
    registration_complete: bool = False

    # Extracted data (partial or complete)
    extracted_customer: Optional[Dict[str, Any]] = None
    extracted_car: Optional[Dict[str, Any]] = None

    # Session metadata
    user_ip: Optional[str] = None
    user_agent: Optional[str] = None

    def add_message(
        self,
        message_type: MessageType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to the conversation."""
        message = ConversationMessage(
            type=message_type, content=content, metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_conversation_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history in a format suitable for LLM."""
        messages = self.messages[-limit:] if limit else self.messages
        return [
            {
                "role": msg.type.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in messages
        ]

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
