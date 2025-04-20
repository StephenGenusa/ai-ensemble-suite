# src/ai_ensemble_suite/utils/__init__.py

"""Utility functions for the ai-ensemble-suite package."""

from ai_ensemble_suite.utils.logging import EnsembleLogger, logger
from ai_ensemble_suite.utils.async_utils import (
    gather_with_concurrency,
    run_in_threadpool,
)
from ai_ensemble_suite.utils.prompt_utils import (
    format_prompt,
    truncate_text,
    create_system_message,
    create_user_message,
    create_assistant_message,
    create_chat_prompt,
)
from ai_ensemble_suite.utils.tracing import (
    ModelTrace,
    PhaseTrace,
    AggregationTrace,
    SessionTrace,
    TraceCollector,
)

__all__ = [
    "EnsembleLogger",
    "logger",
    "gather_with_concurrency",
    "run_in_threadpool",
    "format_prompt",
    "truncate_text",
    "create_system_message",
    "create_user_message",
    "create_assistant_message",
    "create_chat_prompt",
    "ModelTrace",
    "PhaseTrace",
    "AggregationTrace",
    "SessionTrace",
    "TraceCollector",
]
