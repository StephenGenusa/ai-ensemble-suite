# src/ai_ensemble_suite/utils/logging.py

"""Logging utilities for ai-ensemble-suite."""

import logging
import sys
from typing import Dict, Any, Optional, Union, List, Callable


class EnsembleLogger:
    """Logging utility for ai-ensemble-suite.

    Provides structured logging with context information and configurable verbosity.
    """

    def __init__(
        self,
        name: str = "ai_ensemble_suite",
        level: Union[int, str] = logging.DEBUG,
        format_string: Optional[str] = None,
        handlers: Optional[List[logging.Handler]] = None,
    ) -> None:
        """Initialize the EnsembleLogger.

        Args:
            name: The logger name.
            level: The logging level.
            format_string: Custom format string for log messages.
            handlers: Custom log handlers.
        """
        self.name = name
        self.logger = logging.getLogger(name)

        # Set default format if not provided
        if format_string is None:
            format_string = (
                "%(asctime)s [%(levelname)s] %(name)s: "
                "%(message)s (%(filename)s:%(lineno)d)"
            )

        # Create formatter
        formatter = logging.Formatter(format_string)

        # Use provided handlers or create default handler
        if handlers is None:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            for handler in handlers:
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

        # Set level
        self.set_level(level)

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger.

        Returns:
            The configured logger instance.
        """
        return self.logger

    def set_level(self, level: Union[int, str]) -> None:
        """Set the logging level.

        Args:
            level: The new logging level.
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(level)
        self.logger.setLevel(logging.DEBUG)

    def _sanitize(self, msg: str, sanitize: bool = False) -> str:
        """Sanitize sensitive information from log messages.

        Args:
            msg: The message to sanitize.
            sanitize: Whether to sanitize the message.

        Returns:
            Sanitized message.
        """
        if not sanitize:
            return msg

        # Simple sanitization for now - could be expanded with regex patterns
        # for more sophisticated redaction
        if len(msg) > 100:
            return f"{msg[:50]}...{msg[-50:]}"
        return msg

    def _format_extra(self, extra: Dict[str, Any]) -> str:
        """Format extra context information.

        Args:
            extra: Extra context information.

        Returns:
            Formatted extra context.
        """
        if not extra:
            return ""
        return " | " + " | ".join(f"{k}={v}" for k, v in extra.items())

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message.

        Args:
            msg: The message to log.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. Special keys:
                extra: Dict[str, Any] - Extra context information
                sanitize: bool - Whether to sanitize sensitive information
        """
        extra = kwargs.pop("extra", {})
        sanitize = kwargs.pop("sanitize", False)

        formatted_msg = self._sanitize(msg, sanitize)
        if extra:
            formatted_msg += self._format_extra(extra)

        self.logger.debug(formatted_msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message.

        Args:
            msg: The message to log.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. Special keys:
                extra: Dict[str, Any] - Extra context information
                sanitize: bool - Whether to sanitize sensitive information
        """
        extra = kwargs.pop("extra", {})
        sanitize = kwargs.pop("sanitize", False)

        formatted_msg = self._sanitize(msg, sanitize)
        if extra:
            formatted_msg += self._format_extra(extra)

        self.logger.info(formatted_msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message.

        Args:
            msg: The message to log.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. Special keys:
                extra: Dict[str, Any] - Extra context information
                sanitize: bool - Whether to sanitize sensitive information
        """
        extra = kwargs.pop("extra", {})
        sanitize = kwargs.pop("sanitize", False)

        formatted_msg = self._sanitize(msg, sanitize)
        if extra:
            formatted_msg += self._format_extra(extra)

        self.logger.warning(formatted_msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message.

        Args:
            msg: The message to log.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. Special keys:
                extra: Dict[str, Any] - Extra context information
                sanitize: bool - Whether to sanitize sensitive information
        """
        extra = kwargs.pop("extra", {})
        sanitize = kwargs.pop("sanitize", False)

        formatted_msg = self._sanitize(msg, sanitize)
        if extra:
            formatted_msg += self._format_extra(extra)

        self.logger.error(formatted_msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message.

        Args:
            msg: The message to log.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. Special keys:
                extra: Dict[str, Any] - Extra context information
                sanitize: bool - Whether to sanitize sensitive information
        """
        extra = kwargs.pop("extra", {})
        sanitize = kwargs.pop("sanitize", False)

        formatted_msg = self._sanitize(msg, sanitize)
        if extra:
            formatted_msg += self._format_extra(extra)

        self.logger.critical(formatted_msg, *args, **kwargs)


# Create a global logger instance for convenience
logger = EnsembleLogger()
