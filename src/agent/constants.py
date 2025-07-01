class ConversationStatus:
    """Status constants for conversation flow."""

    COLLECTING_DATA = "collecting_data"
    VALIDATION_ERROR = "validation_error"
    DUPLICATE_FOUND = "duplicate_found"
    COMPLETED = "completed"
    ERROR = "error"
    DUPLICATE_REVIEW_REQUESTED = "duplicate_review_requested"
    CLARIFICATION_NEEDED = "clarification_needed"


class DatabaseCollections:
    """Database collection names."""

    REGISTRATIONS = "registrations"


class LLMProvider:
    """LLM provider constants."""

    HUGGINGFACE = "huggingface"
    OPENAI_COMPATIBLE_LLM = "openai_compatible_llm"

    # Individual provider types (for DSPy compatibility)
    OPENAI = "openai"

    # Model types
    CHAT_MODEL = "chat"

    # URL/Model prefixes for DSPy
    OPENAI_MODEL_PREFIX = "openai/"


class DuplicateKeywords:
    """Keywords for duplicate detection flow."""

    DUPLICATE_CONTEXT = ["duplicate", "similar registration", "proceed anyway", "check this existing"]

    PROCEED_KEYWORDS = ["yes", "proceed", "continue", "update", "go ahead"]

    CHECK_KEYWORDS = ["no", "check", "stop", "cancel", "review"]

    CONFIRMATION_WORDS = ["yes", "no", "proceed", "check", "update", "continue", "stop"]


class RequiredFields:
    """Required field names for registration."""

    ALL = ["car_type", "manufacturer", "year", "license_plate", "customer_name", "birth_date"]


class ErrorMessages:
    """Common error messages."""

    DSPY_INIT_FAILED = "Failed to initialize DSPy"
    DSPY_EXTRACTION_FAILED = "DSPy extraction failed"
    DSPY_NOT_AVAILABLE = "DSPy extraction not available, returning existing data"
    HF_NOT_RECOMMENDED = "HuggingFace local models are not recommended for extraction tasks"
    NO_LLM_PROVIDER = "No suitable LLM provider available for extraction"
    DUPLICATE_DETECTION_FAILED = "Duplicate detection failed"
    REGISTRATION_SAVE_FAILED = "Failed to save registration"
