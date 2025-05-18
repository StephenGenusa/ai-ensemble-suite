# ai_ensemble_suite/aggregation/sequential_refinement.py

"""Sequential Refinement aggregation strategy."""

from typing import Dict, Any, Optional, List, Set, Tuple, TYPE_CHECKING
import time

from ai_ensemble_suite.aggregation.base import BaseAggregator
from ai_ensemble_suite.exceptions import AggregationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector
import copy

# Type hint imports
if TYPE_CHECKING:
    from ai_ensemble_suite.config import ConfigManager
    from ai_ensemble_suite.models import ModelManager


class SequentialRefinement(BaseAggregator):
    """Sequential Refinement aggregation strategy.

    Assumes phases run in a sequence where later phases refine earlier ones.
    This strategy simply selects the output of the designated 'final' phase.
    """

    # Override __init__ for consistency, though model_manager isn't used here
    def __init__(
        self,
        config_manager: "ConfigManager",
        strategy_name: str,
        model_manager: Optional["ModelManager"] = None,
        strategy_config_override: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the SequentialRefinement aggregator."""
        super().__init__(config_manager, strategy_name, model_manager, strategy_config_override)
        # No specific init needed


    async def aggregate(
        self,
        outputs: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Aggregate by selecting the output from the designated final phase.

        Args:
            outputs: Dictionary mapping phase names to their output data.
            context: Context information (used to infer final phase if not configured).
            trace_collector: Optional trace collector.

        Returns:
            Dictionary containing:
                response: The text output from the final phase.
                confidence: The confidence score associated with the final phase output.
                final_phase: The name of the phase selected as final.

        Raises:
            AggregationError: If no final phase can be determined or its output is missing.
        """
        start_time = time.time()
        logger.debug(f"Starting Sequential Refinement aggregation for strategy '{self._strategy_name}'")

        if not outputs:
             raise AggregationError("No phase outputs provided for Sequential Refinement aggregation.")

        try:
            final_phase_name: Optional[str] = None

            # --- Determine the Final Phase ---
            # 1. Check strategy configuration for 'final_phase'
            configured_final_phase = self.get_final_phase() # Method uses self._config
            if configured_final_phase and configured_final_phase in outputs:
                logger.debug(f"Using configured final_phase: '{configured_final_phase}'")
                final_phase_name = configured_final_phase

            # 2. Infer from 'sequence' key in strategy config if final_phase not set/found
            if not final_phase_name:
                sequence = self._config.get("sequence", [])
                if isinstance(sequence, list) and sequence:
                    potential_final = sequence[-1]
                    if potential_final in outputs:
                        logger.debug(f"Inferred final phase from config sequence: '{potential_final}'")
                        final_phase_name = potential_final

            # 3. Infer from 'phase_sequence' in the main context if still not found
            if not final_phase_name:
                context_sequence = context.get("phase_sequence", [])
                if isinstance(context_sequence, list) and context_sequence:
                    potential_final = context_sequence[-1]
                    if potential_final in outputs:
                        logger.debug(f"Inferred final phase from context sequence: '{potential_final}'")
                        final_phase_name = potential_final

            # 4. Fallback: Look for phases named like 'refinement', 'integration', 'final' etc.
            if not final_phase_name:
                 candidate_suffixes = ["refinement", "integration", "final", "synthesis", "summary", "committee"]
                 # Check in reverse order of phase execution if possible
                 phases_to_check = reversed(context.get("phase_sequence", list(outputs.keys())))
                 for phase_name_to_check in phases_to_check:
                      if phase_name_to_check in outputs:
                           lower_name = phase_name_to_check.lower()
                           if any(suffix in lower_name for suffix in candidate_suffixes):
                                logger.debug(f"Using phase '{phase_name_to_check}' as likely final phase based on name.")
                                final_phase_name = phase_name_to_check
                                break

            # 5. Last Resort: If still no final phase determined, use the last available output based on context sequence or dict order
            if not final_phase_name:
                phases_in_order = context.get("phase_sequence", list(outputs.keys()))
                # Find the last phase in the order that actually exists in outputs
                for potential_final in reversed(phases_in_order):
                     if potential_final in outputs:
                          logger.warning(f"Could not determine specific final phase, using last available phase: '{potential_final}'")
                          final_phase_name = potential_final
                          break
                # If even that fails (e.g., sequence empty and outputs empty, though caught earlier)
                if not final_phase_name:
                     raise AggregationError("Cannot determine final phase: No outputs available or sequence context missing.")


            # --- Extract Output and Confidence from Final Phase ---
            logger.info(f"Selected final phase for aggregation: '{final_phase_name}'")
            final_output_data = outputs.get(final_phase_name)

            if final_output_data is None:
                # Should not happen if logic above is correct, but safety check
                raise AggregationError(f"Selected final phase '{final_phase_name}' not found in outputs dictionary.")

            # Extract the response text using the utility method
            final_response_text = self._extract_output(final_output_data)
            # Note: _extract_output returns "" if extraction fails

            # Get confidence score from the final phase's output data
            confidence = 0.0 # Default if not found
            if isinstance(final_output_data, dict) and "confidence" in final_output_data:
                conf_val = final_output_data["confidence"]
                if isinstance(conf_val, (int, float)):
                    confidence = float(conf_val)
                elif isinstance(conf_val, dict) and "combined" in conf_val:
                    # Prioritize 'combined' score
                    combined_score = conf_val.get("combined")
                    if isinstance(combined_score, (int, float)):
                        confidence = float(combined_score)

            # Ensure confidence is valid [0, 1] range
            confidence = min(max(confidence, 0.0), 1.0)

            # If confidence is still very low/zero, maybe use the average from all phases?
            if confidence < 0.1: # Threshold check
                 average_confidence = self._calculate_confidence(outputs)
                 if average_confidence > confidence:
                      logger.debug(f"Final phase confidence ({confidence:.3f}) is low, using average confidence ({average_confidence:.3f}) instead.")
                      confidence = average_confidence


            execution_time = time.time() - start_time
            logger.info(
                f"Sequential Refinement aggregation completed in {execution_time:.2f}s. "
                f"Using output from phase: '{final_phase_name}'",
                 extra={ "final_confidence": confidence }
            )

            # Prepare final result dictionary
            aggregation_result = {
                "response": final_response_text,
                "confidence": confidence,
                "final_phase": final_phase_name
            }

            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_aggregation_trace(
                    strategy_name=self._strategy_name,
                    inputs={
                        "phase_output_keys": list(outputs.keys()),
                        "context_keys": list(context.keys()),
                        "determined_final_phase": final_phase_name
                    },
                    output=aggregation_result, # Response, confidence, final_phase name
                    execution_time=execution_time,
                    parameters={"final_phase_configured": configured_final_phase} # Specific params
                )

            return aggregation_result

        except AggregationError as e:
             logger.error(f"Sequential refinement aggregation failed: {str(e)}")
             raise # Re-raise known errors
        except Exception as e:
            logger.error(f"Unexpected error during Sequential Refinement aggregation: {str(e)}", exc_info=True)
            # Wrap unexpected errors
            raise AggregationError(f"Sequential Refinement aggregation failed unexpectedly: {str(e)}")

    # Inherit _extract_output from BaseAggregator
    # def _extract_output(self, phase_output_data: Any) -> str: ...

