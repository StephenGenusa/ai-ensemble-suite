# src/ai_ensemble_suite/config/__init__.py

"""Configuration management for ai-ensemble-suite."""

from ai_ensemble_suite.config.config_manager import ConfigManager
from ai_ensemble_suite.config.defaults import DEFAULT_CONFIG
from ai_ensemble_suite.config.templates import (
    ALL_TEMPLATES,
    BASIC_TEMPLATES,
    DEBATE_TEMPLATES,
    ROLE_TEMPLATES,
    HIERARCHICAL_TEMPLATES,
)

__all__ = [
    "ConfigManager",
    "DEFAULT_CONFIG",
    "ALL_TEMPLATES",
    "BASIC_TEMPLATES",
    "DEBATE_TEMPLATES",
    "ROLE_TEMPLATES",
    "HIERARCHICAL_TEMPLATES",
]
