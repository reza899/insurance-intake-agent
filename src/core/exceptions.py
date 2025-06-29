class InsuranceAgentError(Exception):
    """Base exception for insurance agent application."""

    pass


class ConfigurationError(InsuranceAgentError):
    """Raised when there's a configuration issue."""

    pass


class DatabaseError(InsuranceAgentError):
    """Raised when there's a database operation issue."""

    pass


class ValidationError(InsuranceAgentError):
    """Raised when data validation fails."""

    pass


class DuplicateDetectionError(InsuranceAgentError):
    """Raised when duplicate detection fails."""

    pass


class LLMError(InsuranceAgentError):
    """Raised when LLM operations fail."""

    pass


class RAGError(InsuranceAgentError):
    """Raised when RAG operations fail."""

    pass
