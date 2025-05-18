# ai_ensemble_suite/__init__.py

"""
ai-ensemble-suite: A Python framework for multiple GGUF language models collaboration.

This library enables multiple smaller GGUF-format language models to collaborate
on tasks using structured communication patterns, aggregating their outputs into
coherent responses.
"""

__version__ = "0.5.1"

# Public API exports
from ai_ensemble_suite.ensemble import Ensemble

__all__ = ["Ensemble"]
