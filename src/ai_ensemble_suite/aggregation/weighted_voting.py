"""Weighted Voting aggregation strategy."""

from typing import Dict, Any, Optional, List, Set, Tuple, TYPE_CHECKING
import time
import re
from collections import Counter # Use Counter for efficient voting

from ai_ensemble_suite.aggregation.base import BaseAggregator
from ai_ensemble_suite.exceptions import AggregationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector
import copy

# Type hint imports
if TYPE_CHECKING:
    from ai_ensemble_suite.config import ConfigManager
    from ai_ensemble_suite.models import ModelManager


class WeightedVoting(BaseAggregator):
    """Weighted Voting aggregation strategy.

    Aggregates by assigning weights to different phase outputs (based on config)
    and selecting the output text that receives the highest total weight.
    """

     # Override __init__ for consistency
    def __init__(
        self,
        config_manager: "ConfigManager",
        strategy_name: str,
        model_manager: Optional["ModelManager"] = None,
        strategy_config_override: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the WeightedVoting aggregator."""
        super().__init__(config_manager, strategy_name, model_manager, strategy_config_override)
        # No specific init needed


    async def aggregate(
        self,
        outputs: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Aggregate phase outputs using weighted voting based on configuration.

        Args:
            outputs: Dictionary mapping phase names to their output data.
            context: Context information from collaboration phases.
            trace_collector: Optional trace collector.

        Returns:
            Dictionary containing:
                response: The output text that received the highest total weight.
                confidence: Confidence score, potentially combining weights and source confidence.
                weights: Dictionary mapping output texts to their normalized weights.
                source_phase: Name(s) of the phase(s) that produced the winning output.

        Raises:
            AggregationError: If no valid outputs or weights are available.
        """
        start_time = time.time()
        logger.debug(f"Starting Weighted Voting aggregation for strategy '{self._strategy_name}'")

        if not outputs:
             raise AggregationError("No phase outputs provided for Weighted Voting aggregation.")

        try:
            # Get weights from strategy configuration: { "phase_name": weight }
            phase_weights = self._config.get("weights", {})
            if not isinstance(phase_weights, dict):
                 logger.warning("Invalid 'weights' config (must be dict), using default weights.")
                 phase_weights = {}

            default_weight = self._config.get("default_weight", 1.0)
            if not isinstance(default_weight, (int, float)):
                 logger.warning(f"Invalid 'default_weight' {default_weight}, using 1.0.")
                 default_weight = 1.0

            # --- Tally weighted votes for each unique output text ---
            output_votes: Counter[str] = Counter() # Use Counter for summing weights per text
            valid_phases_processed = 0

            for phase_name, phase_output_data in outputs.items():
                 # Get the configured weight for this phase, or use default
                 weight = phase_weights.get(phase_name, default_weight)

                 # Skip phases with non-positive weight
                 if weight <= 0:
                      logger.debug(f"Skipping phase '{phase_name}' due to weight <= 0.")
                      continue

                 # Extract the output text content
                 output_text = self._extract_output(phase_output_data)

                 # Skip phases where text extraction failed or resulted in empty string
                 if not output_text:
                      logger.debug(f"Could not extract valid text from phase '{phase_name}', skipping vote.")
                      continue

                 # Add the weight to the vote count for this specific output text
                 output_votes[output_text] += weight
                 valid_phases_processed += 1
                 logger.debug(f"Phase '{phase_name}' voted for output (hash:{hash(output_text)%1000:03d}) with weight {weight:.2f}")


            # Check if any valid votes were cast
            if not output_votes or valid_phases_processed == 0:
                # If no weights were configured AND a final phase exists, use fallback logic?
                # This duplicates logic from base/other strategies, maybe avoid here.
                # Let's just error if no votes.
                logger.error("No valid outputs or positive weights found for aggregation.")
                raise AggregationError("Weighted voting failed: No valid weighted outputs available.")

            # --- Determine the winning output ---
            # Find the output text with the highest total weight
            # most_common(1) returns list of tuples: [ (text, total_weight) ]
            winner_list = output_votes.most_common(1)
            if not winner_list: # Should not happen if output_votes is not empty
                 raise AggregationError("Internal error: Could not determine winner from votes.")

            best_output_text, highest_total_weight = winner_list[0]

            # --- Calculate Normalized Weights and Confidence ---
            # Calculate the sum of all weights cast
            total_weight_sum = sum(output_votes.values())

            # Normalize the weights for each unique output (for reporting/tracing)
            normalized_weights = {}
            if total_weight_sum > 0:
                 normalized_weights = {text: votes / total_weight_sum for text, votes in output_votes.items()}
                 # The weight of the winning output corresponds to its normalized score
                 best_output_normalized_weight = normalized_weights.get(best_output_text, 0.0)
            else: # Should not happen if weights > 0
                 logger.warning("Total weight sum is zero, cannot normalize.")
                 best_output_normalized_weight = 1.0 if len(output_votes) == 1 else 0.0 # Assign 1 if only one option

            # Calculate confidence: Blend the normalized weight of the winner
            # with the average confidence of the phases that produced the winning output.
            source_phases_confidences = []
            source_phase_names = []
            for phase_name, phase_output_data in outputs.items():
                 if self._extract_output(phase_output_data) == best_output_text:
                      source_phase_names.append(phase_name)
                      # Extract confidence from this source phase
                      if isinstance(phase_output_data, dict) and "confidence" in phase_output_data:
                           conf_val = phase_output_data["confidence"]
                           score = None
                           if isinstance(conf_val, (int, float)): score = float(conf_val)
                           elif isinstance(conf_val, dict): score = conf_val.get("combined")
                           if isinstance(score, (int, float)):
                                source_phases_confidences.append(min(max(float(score), 0.0), 1.0))

            avg_source_confidence = (sum(source_phases_confidences) / len(source_phases_confidences)) \
                                     if source_phases_confidences else 0.7 # Default if no source confidences

            # Blend: e.g., 60% weight score, 40% source confidence average
            final_confidence = (0.6 * best_output_normalized_weight) + (0.4 * avg_source_confidence)
            final_confidence = min(max(final_confidence, 0.0), 1.0) # Clamp to [0, 1]


            execution_time = time.time() - start_time
            logger.info(
                f"Weighted Voting aggregation completed in {execution_time:.2f}s. "
                f"Winner score: {best_output_normalized_weight:.3f} "
                f"(from phases: {source_phase_names})",
                 extra={"final_confidence": final_confidence}
            )

            # Prepare final result dictionary
            # Sort normalized weights for clearer reporting/tracing
            sorted_normalized_weights = dict(sorted(normalized_weights.items(), key=lambda item: item[1], reverse=True))

            aggregation_result = {
                "response": best_output_text,
                "confidence": final_confidence,
                 # Report normalized weights per unique text output
                "weights": sorted_normalized_weights,
                 # List phases that produced the winning output
                "source_phase": source_phase_names[0] if len(source_phase_names) == 1 else source_phase_names,
            }


            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_aggregation_trace(
                    strategy_name=self._strategy_name,
                    inputs={
                        "phase_output_keys": list(outputs.keys()),
                        "context_keys": list(context.keys()),
                        "configured_weights": phase_weights,
                        "default_weight": default_weight,
                    },
                    output=aggregation_result, # Contains response, conf, normalized weights map, source(s)
                    execution_time=execution_time,
                    parameters={ # Parameters influencing the vote
                        "configured_weights": phase_weights,
                        "default_weight": default_weight,
                         # Trace the raw votes before normalization for debugging
                        "raw_votes": dict(output_votes)
                    }
                )

            return aggregation_result

        except AggregationError as e:
             logger.error(f"Weighted voting aggregation failed: {str(e)}")
             raise # Re-raise known errors
        except Exception as e:
            logger.error(f"Unexpected error during Weighted Voting aggregation: {str(e)}", exc_info=True)
            # Wrap unexpected errors
            raise AggregationError(f"Weighted Voting aggregation failed unexpectedly: {str(e)}")


    # Inherit _extract_output from BaseAggregator
    # def _extract_output(self, phase_output_data: Any) -> str: ...


    # _get_output_source is slightly simplified in the return dict preparation now
    # Keeping it here in case it's needed for more complex scenarios later
    # def _get_output_source(self, output_text: str, outputs: Dict[str, Dict[str, Any]]) -> Union[str, List[str]]:
    #     """Find the phase(s) that produced the given output text."""
    #     source_phases = []
    #     for phase_name, phase_output_data in outputs.items():
    #          extracted = self._extract_output(phase_output_data)
    #          if extracted == output_text:
    #              source_phases.append(phase_name)
    #
    #     if not source_phases: return "unknown"
    #     if len(source_phases) == 1: return source_phases[0]
    #     return source_phases

