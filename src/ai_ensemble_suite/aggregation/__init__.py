
"""Aggregation strategies for ai-ensemble-suite."""

from ai_ensemble_suite.aggregation.base import BaseAggregator
from ai_ensemble_suite.aggregation.weighted_voting import WeightedVoting
from ai_ensemble_suite.aggregation.sequential_refinement import SequentialRefinement
from ai_ensemble_suite.aggregation.confidence_based import ConfidenceBased
from ai_ensemble_suite.aggregation.multidimensional_voting import MultidimensionalVoting
from ai_ensemble_suite.aggregation.ensemble_fusion import EnsembleFusion
from ai_ensemble_suite.aggregation.adaptive_selection import AdaptiveSelection


__all__ = [
    "BaseAggregator",
    "WeightedVoting",
    "SequentialRefinement",
    "ConfidenceBased",
    "MultidimensionalVoting",
    "EnsembleFusion",
    "AdaptiveSelection",
]
