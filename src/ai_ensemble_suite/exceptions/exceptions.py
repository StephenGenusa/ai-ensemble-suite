# src/ai_ensemble_suite/models/exceptions.py

"""Exception classes for ai-ensemble-suite."""


class AiEnsembleSuiteError(Exception):
    """Base exception for all ai-ensemble-suite errors."""

    def __init__(self, message: str = "An error occurred in ai-ensemble-suite") -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the issue.
        """
        self.message = message
        super().__init__(self.message)


class ConfigurationError(AiEnsembleSuiteError):
    """Error raised when configuration is invalid."""

    def __init__(self, message: str = "Invalid configuration") -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the configuration issue.
        """
        self.message = message
        super().__init__(self.message)


class ModelError(AiEnsembleSuiteError):
    """Error raised when model operations fail."""

    def __init__(self, message: str = "Model operation failed") -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the model issue.
        """
        self.message = message
        super().__init__(self.message)


class CollaborationError(AiEnsembleSuiteError):
    """Error raised when collaboration phase execution fails."""

    def __init__(self, message: str = "Collaboration phase execution failed") -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the collaboration issue.
        """
        self.message = message
        super().__init__(self.message)


class AggregationError(AiEnsembleSuiteError):
    """Error raised when aggregation fails."""

    def __init__(self, message: str = "Aggregation failed") -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the aggregation issue.
        """
        self.message = message
        super().__init__(self.message)


class ValidationError(AiEnsembleSuiteError):
    """Error raised when validation fails."""

    def __init__(self, message: str = "Validation failed") -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the validation issue.
        """
        self.message = message
        super().__init__(self.message)


class ResourceError(AiEnsembleSuiteError):
    """Error raised when required resources are unavailable."""

    def __init__(self, message: str = "Required resources are unavailable") -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the resource issue.
        """
        self.message = message
        super().__init__(self.message)


class TimeoutError(AiEnsembleSuiteError):
    """Error raised when an operation times out."""

    def __init__(self, message: str = "Operation timed out") -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the timeout issue.
        """
        self.message = message
        super().__init__(self.message)
