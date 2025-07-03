class InsuranceAgentError(Exception):
    """Base exception for insurance agent application."""

    pass


class DatabaseError(InsuranceAgentError):
    """Raised when there's a database operation issue."""

    pass


class DataExtractionError(InsuranceAgentError):
    """Raised when there's an issue with data extraction from conversations."""

    pass


class DuplicateDetectionError(InsuranceAgentError):
    """Raised when there's an issue with duplicate detection processing."""

    pass


class RegistrationError(InsuranceAgentError):
    """Base exception for registration-related issues."""

    pass


class RegistrationSaveError(RegistrationError):
    """Raised when there's an issue saving a registration."""

    pass


class RegistrationNotFoundError(RegistrationError):
    """Raised when a registration cannot be found."""

    pass


class ConversationError(InsuranceAgentError):
    """Raised when there's an issue with conversation flow."""

    pass


class ConfigurationError(InsuranceAgentError):
    """Raised when there's a configuration issue."""

    pass
