"""Confidence-Based aggregation strategy."""

from typing import Dict, Any, Optional, List, Set, Tuple, TYPE_CHECKING
import time

from ai_ensemble_suite.aggregation.base import BaseAggregator
from ai_ensemble_suite.exceptions import AggregationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector
import copy # Import copy


# Type hint imports
if TYPE_CHECKING:
    from ai_ensemble_suite.config import ConfigManager
    from ai_ensemble_suite.models import ModelManager


class ConfidenceBased(BaseAggregator):
    """Confidence-Based aggregation strategy.

    Selects the output from the collaboration phase that has the highest
    associated confidence score, potentially applying a minimum threshold.
    """

    # Override __init__ to accept optional model_manager and override
    # Although this strategy doesn't use model_manager, maintain signature consistency
    def __init__(
        self,
        config_manager: "ConfigManager",
        strategy_name: str,
        model_manager: Optional["ModelManager"] = None,
        strategy_config_override: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the ConfidenceBased aggregator."""
        super().__init__(config_manager, strategy_name, model_manager, strategy_config_override)
        # No specific init needed


    async def aggregate(
        self,
        outputs: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Aggregate outputs by selecting the one with the highest confidence score.

        Args:
            outputs: Dictionary mapping phase names to their output data.
            context: Context information from collaboration phases.
            trace_collector: Optional trace collector.

        Returns:
            Dictionary containing:
                response: The text output from the highest confidence phase.
                confidence: The confidence score of the selected output.
                source_phase: The name of the phase providing the selected output.
                confidence_scores: Dictionary mapping all phase names to their scores.

        Raises:
            AggregationError: If no outputs with valid confidence scores are available.
        """
        start_time = time.time()
        logger.debug(f"Starting Confidence-Based aggregation for strategy '{self._strategy_name}'")

        if not outputs:
             raise AggregationError("No phase outputs provided for Confidence-Based aggregation.")

        try:
            # Get confidence threshold from strategy configuration
            threshold = self._config.get("threshold", 0.0) # Default is 0.0 (consider all)
            if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
                 logger.warning(f"Invalid confidence threshold '{threshold}', using 0.0.")
                 threshold = 0.0

            # --- Extract confidence scores and outputs ---
            scored_outputs: List[Tuple[str, str, float]] = [] # (phase_name, output_text, confidence_score)

            all_phase_confidences: Dict[str, float] = {} # Store all scores for tracing/result

            for phase_name, phase_output_data in outputs.items():
                output_text = self._extract_output(phase_output_data)
                # Skip phases where text extraction failed
                if not output_text:
                    logger.debug(f"Could not extract text from phase '{phase_name}', skipping.")
                    all_phase_confidences[phase_name] = -1.0 # Indicate missing text
                    continue

                confidence = -1.0 # Default to invalid score
                if isinstance(phase_output_data, dict) and "confidence" in phase_output_data:
                    conf_val = phase_output_data["confidence"]
                    if isinstance(conf_val, (int, float)):
                        confidence = float(conf_val)
                    elif isinstance(conf_val, dict) and "combined" in conf_val:
                        # Prioritize 'combined' score if available
                        combined_score = conf_val.get("combined")
                        if isinstance(combined_score, (int, float)):
                             confidence = float(combined_score)

                # Ensure confidence is within [0, 1] range
                if isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0:
                     all_phase_confidences[phase_name] = confidence
                     # Add to list if above the configured threshold
                     if confidence >= threshold:
                         scored_outputs.append((phase_name, output_text, confidence))
                         logger.debug(f"Phase '{phase_name}' added with confidence {confidence:.3f} (>= threshold {threshold:.3f})")
                     else:
                          logger.debug(f"Phase '{phase_name}' skipped, confidence {confidence:.3f} < threshold {threshold:.3f}")
                else:
                    # Log phases with missing or invalid confidence
                    logger.debug(f"Phase '{phase_name}' has missing or invalid confidence score.")
                    all_phase_confidences[phase_name] = -1.0 # Mark as invalid


            # --- Determine the best output ---
            if not scored_outputs:
                # No outputs met the threshold (or none had valid confidence)
                logger.warning(f"No outputs met the confidence threshold >= {threshold:.3f}. "
                               f"Falling back to highest available score (ignoring threshold) or default.")

                # Fallback 1: Find highest score among *all* phases with valid scores
                valid_scores = {name: score for name, score in all_phase_confidences.items() if score >= 0.0}
                if valid_scores:
                     best_fallback_phase = max(valid_scores, key=valid_scores.get)
                     best_fallback_score = valid_scores[best_fallback_phase]
                     best_fallback_text = self._extract_output(outputs[best_fallback_phase])
                     logger.info(f"Using fallback phase '{best_fallback_phase}' with confidence {best_fallback_score:.3f}.")
                     scored_outputs.append((best_fallback_phase, best_fallback_text, best_fallback_score))
                else:
                    # Fallback 2: If NO valid scores at all, error out
                    raise AggregationError("No outputs with valid confidence scores available for aggregation.")

            # Sort the valid (above threshold or best fallback) outputs by confidence (descending)
            scored_outputs.sort(key=lambda x: x[2], reverse=True)

            # Get the highest confidence output from the filtered/sorted list
            best_phase_name, best_output_text, best_confidence = scored_outputs[0]


            execution_time = time.time() - start_time
            logger.info(
                f"Confidence-Based aggregation completed in {execution_time:.2f}s. "
                f"Selected phase: '{best_phase_name}' (Confidence: {best_confidence:.3f})",
                 extra={ "threshold": threshold }
            )

            # Prepare final result dictionary
            aggregation_result = {
                "response": best_output_text,
                "confidence": best_confidence,
                "source_phase": best_phase_name,
                # Include map of all phase confidences (even invalid ones marked as -1)
                "confidence_scores": all_phase_confidences
            }

            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_aggregation_trace(
                    strategy_name=self._strategy_name,
                    inputs={
                        "phase_output_keys": list(outputs.keys()),
                        "context_keys": list(context.keys()),
                    },
                    output=aggregation_result, # Contains response, confidence, source, scores map
                    execution_time=execution_time,
                    parameters={ # Parameters used by this strategy
                        "threshold": threshold,
                        # Maybe trace the sorted list for debugging? Could be large.
                        # "sorted_scored_outputs": [(p, c) for p, _, c in scored_outputs]
                    }
                )

            return aggregation_result

        except AggregationError as e:
             logger.error(f"Confidence-based aggregation failed: {str(e)}")
             raise # Re-raise known aggregation errors
        except Exception as e:
            logger.error(f"Unexpected error during Confidence-Based aggregation: {str(e)}", exc_info=True)
            # Wrap unexpected errors
            raise AggregationError(f"Confidence-Based aggregation failed unexpectedly: {str(e)}")


    # Inherit _extract_output from BaseAggregator
    # def _extract_output(self, phase_output_data: Any) -> str: ...
