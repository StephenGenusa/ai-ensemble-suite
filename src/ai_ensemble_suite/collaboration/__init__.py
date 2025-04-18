
"""Collaboration phases for ai-ensemble-suite."""

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.collaboration.async_thinking import AsyncThinking
from ai_ensemble_suite.collaboration.integration import Integration
from ai_ensemble_suite.collaboration.expert_committee import ExpertCommittee
from ai_ensemble_suite.collaboration.hierarchical_review import HierarchicalReview
from ai_ensemble_suite.collaboration.competitive_evaluation import CompetitiveEvaluation
from ai_ensemble_suite.collaboration.perspective_rotation import PerspectiveRotation
from ai_ensemble_suite.collaboration.chain_of_thought import ChainOfThoughtBranching
from ai_ensemble_suite.collaboration.adversarial_improvement import AdversarialImprovement
from ai_ensemble_suite.collaboration.role_based_workflow import RoleBasedWorkflow
from ai_ensemble_suite.collaboration.bagging import Bagging
from ai_ensemble_suite.collaboration.uncertaintybased import UncertaintyBasedCollaboration
from ai_ensemble_suite.collaboration.stackedgeneralization import StackedGeneralization

from ai_ensemble_suite.collaboration.structured_debate import (
    BaseDebate,
    StructuredCritique,
    SynthesisOriented,
    RoleBasedDebate,
)

__all__ = [
    "BaseCollaborationPhase",
    "AsyncThinking",
    "Integration",
    "ExpertCommittee",
    "HierarchicalReview",
    "CompetitiveEvaluation",
    "PerspectiveRotation",
    "ChainOfThoughtBranching",
    "AdversarialImprovement",
    "RoleBasedWorkflow",
    "BaseDebate",
    "StructuredCritique",
    "SynthesisOriented",
    "RoleBasedDebate",
    "Bagging",
    "UncertaintyBasedCollaboration",
    "StackedGeneralization",
]
