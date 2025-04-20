
"""Structured debate collaboration patterns."""

from ai_ensemble_suite.collaboration.structured_debate.base_debate import BaseDebate
from ai_ensemble_suite.collaboration.structured_debate.critique import StructuredCritique
from ai_ensemble_suite.collaboration.structured_debate.synthesis import SynthesisOriented
from ai_ensemble_suite.collaboration.structured_debate.role_based import RoleBasedDebate

__all__ = [
    "BaseDebate",
    "StructuredCritique",
    "SynthesisOriented",
    "RoleBasedDebate",
]
