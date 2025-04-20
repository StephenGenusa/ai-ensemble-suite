# src/ai_ensemble_suite/utils/template_manager.py

"""Template management module for AI ensemble."""

from typing import Dict, Any, Optional
from ai_ensemble_suite.utils.template_engine import TemplateEngine
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.exceptions import ConfigurationError


class TemplateManager:
    """Manages templates for the AI ensemble system.

    Provides an interface for retrieving and rendering templates using
    Jinja2, with integration to the configuration system.
    """

    def __init__(self, config_manager: "ConfigManager") -> None:
        """Initialize the template manager.

        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self.engine = TemplateEngine()
        logger.debug("TemplateManager initialized with Jinja2 engine")

    def get_raw_template(self, template_name: str) -> str:
        """Get a raw template string by name.

        Args:
            template_name: Name of the template to retrieve.

        Returns:
            The raw template string.

        Raises:
            ConfigurationError: If template is not found.
        """
        template = self.config_manager.get_template(template_name)
        if template is None:
            raise ConfigurationError(f"Template not found: {template_name}")

        return template

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template by name with the provided context.

        Args:
            template_name: Name of the template to render.
            context: Dictionary of variables for rendering.

        Returns:
            The rendered template string.

        Raises:
            ConfigurationError: If template is not found.
            ValueError: If template rendering fails.
        """
        template_str = self.get_raw_template(template_name)
        logger.debug(f"Rendering template '{template_name}' with context keys: {list(context.keys())}")
        return self.engine.render(template_str, context)

    def render_string(self, template_str: str, context: Dict[str, Any]) -> str:
        """Render an arbitrary template string with the provided context.

        Args:
            template_str: Template string to render.
            context: Dictionary of variables for rendering.

        Returns:
            The rendered string.

        Raises:
            ValueError: If template rendering fails.
        """
        return self.engine.render(template_str, context)
