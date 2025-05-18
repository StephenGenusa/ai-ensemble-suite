# ai_ensemble_suite/models/__init__.py

"""Model management for ai-ensemble-suite."""

from ai_ensemble_suite.models.gguf_model import GGUFModel
from ai_ensemble_suite.models.model_manager import ModelManager
from ai_ensemble_suite.models.confidence import (
    calculate_token_confidence,
    get_model_self_evaluation,
    measure_consistency_confidence,
    calculate_similarity,
    get_confidence_score,
    get_combined_confidence,
)

__all__ = [
    "GGUFModel",
    "ModelManager",
    "calculate_token_confidence",
    "get_model_self_evaluation",
    "measure_consistency_confidence",
    "calculate_similarity",
    "get_confidence_score",
    "get_combined_confidence",
]
