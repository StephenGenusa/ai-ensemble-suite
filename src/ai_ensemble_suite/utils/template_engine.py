# ai_ensemble_suite/utils/template_engine.py

"""Template engine module for rendering templates with Jinja2."""

from jinja2 import Environment, BaseLoader
from typing import Dict, Any, Optional, Callable, Union
from ai_ensemble_suite.utils.logging import logger


class TemplateEngine:
    """Handles template rendering using Jinja2 with configuration for AI prompts.

    This class provides a configurable Jinja2 environment with custom delimiters
    and filters optimized for AI template rendering.
    """

    def __init__(self) -> None:
        """Initialize the Jinja2 environment with custom settings."""
        self.env = Environment(
            loader=BaseLoader(),
        )

        # Register custom filters
        self._register_filters()

    def _register_filters(self) -> None:
        """Register custom filters for template rendering."""
        self.env.filters['substring'] = self._substring_filter
        self.env.filters['truncate'] = self._truncate_filter

    def _substring_filter(self, s: Any, start: int, length: int) -> str:
        """Custom filter for substring operations.

        Args:
            s: The string to extract substring from.
            start: Starting index.
            length: Length of substring to extract.

        Returns:
            Extracted substring.
        """
        if s is None:
            return ""
        s_str = str(s)
        start_idx = int(start)
        length_val = int(length)
        return s_str[start_idx:start_idx + length_val]

    def _truncate_filter(self, s: Any, length: int) -> str:
        """Custom filter for truncating strings.

        Args:
            s: The string to truncate.
            length: Maximum length.

        Returns:
            Truncated string.
        """
        if s is None:
            return ""
        s_str = str(s)
        return s_str[:int(length)]

    def render(self, template_str: str, context: Dict[str, Any]) -> str:
        """Render a template string with provided context variables.

        Args:
            template_str: The template string to render.
            context: Dictionary of variables to use in rendering.

        Returns:
            The rendered string.

        Raises:
            ValueError: If template rendering fails.
        """
        try:
            template = self.env.from_string(template_str)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            raise ValueError(f"Failed to render template: {e}")
