class InsuranceAgentError(Exception):
    """Base exception for insurance agent application."""

    pass


class DatabaseError(InsuranceAgentError):
    """Raised when there's a database operation issue."""

    pass
