# ai_ensemble_suite/aggregation/base.py

"""Base class for aggregation strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Set, TYPE_CHECKING
import time
import copy

from ai_ensemble_suite.exceptions import AggregationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector

# Type hint imports
if TYPE_CHECKING:
    from ai_ensemble_suite.config import ConfigManager
    from ai_ensemble_suite.models import ModelManager


class BaseAggregator(ABC):
    """Abstract base class for aggregation strategies.

    Defines the interface for aggregators and provides common functionality
    like configuration handling and default confidence calculation.
    """

    def __init__(
        self,
        config_manager: "ConfigManager",
        strategy_name: str,
        model_manager: Optional["ModelManager"] = None, # Added model_manager
        strategy_config_override: Optional[Dict[str, Any]] = None # Added override
    ) -> None:
        """Initialize the aggregator.

        Args:
            config_manager: The main ConfigManager instance.
            strategy_name: The name of the strategy for configuration lookup.
            model_manager: Optional ModelManager instance, required by some strategies.
            strategy_config_override: Optional dictionary to directly use as the
                                      strategy's config, bypassing config_manager lookup.
                                      Used by AdaptiveSelection.

        Raises:
            ConfigurationError: If strategy configuration loading fails (and no override).
            TypeError: If config_manager is not provided.
        """
        if config_manager is None:
             raise TypeError("ConfigManager instance is required for BaseAggregator.")

        self._config_manager = config_manager
        self._strategy_name = strategy_name
        self._model_manager = model_manager # Store the model manager

        # Load strategy configuration
        if strategy_config_override is not None:
             logger.debug(f"Using provided config override for strategy '{strategy_name}'.")
             self._config = copy.deepcopy(strategy_config_override)
             # Ensure 'strategy' key exists in override for consistency? Maybe not needed.
             self._config.setdefault("strategy", strategy_name)
        else:
            try:
                # Fetch config using the strategy name
                self._config = self._config_manager.get_aggregation_config(strategy_name)
            except ConfigurationError as e:
                # Re-raise with more context
                raise ConfigurationError(
                    f"Failed to load configuration for aggregation strategy '{strategy_name}': {str(e)}"
                ) from e

        logger.debug(f"Initialized aggregation strategy '{strategy_name}'")

    @abstractmethod
    async def aggregate(
        self,
        outputs: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Aggregate the outputs from collaboration phases.

        Args:
            outputs: Dictionary mapping phase names to their full output dictionaries.
                     Each value typically contains 'output', 'confidence', etc.
            context: Dictionary containing context information from the ensemble run,
                     potentially including the original query, phase sequence, etc.
            trace_collector: Optional trace collector for gathering execution details.

        Returns:
            A dictionary containing at least:
                response: The final aggregated output as a string.
                confidence: A float score (0.0-1.0) representing the confidence
                            in the aggregated response.
            May contain additional strategy-specific information (e.g., 'best_phase').

        Raises:
            AggregationError: If the aggregation process fails significantly.
        """
        pass

    def _extract_output(self, phase_output_data: Any) -> str:
        """Extract the primary output text content from a phase's result data.

        Args:
            phase_output_data: The data structure returned by a collaboration phase.

        Returns:
            The extracted output text as a string, or an empty string if none found.
        """
        if isinstance(phase_output_data, str):
            return phase_output_data.strip()
        elif isinstance(phase_output_data, dict):
            # Try common keys in preferred order
            # Added 'result' based on potential model outputs
            for field in ["output", "response", "text", "result", "content", "final_output", "synthesis", "critique"]:
                content = phase_output_data.get(field)
                if isinstance(content, str):
                    return content.strip()
            # Fallback for nested results (like AsyncThinking might have 'outputs': {'model_id': {'text': ...}})
            outputs_dict = phase_output_data.get("outputs")
            if isinstance(outputs_dict, dict) and outputs_dict:
                 # Try to get the first output's text, somewhat arbitrary
                 first_output_key = next(iter(outputs_dict.keys()), None)
                 if first_output_key:
                      first_output_data = outputs_dict[first_output_key]
                      # Check if the nested output data itself is a dict with text
                      if isinstance(first_output_data, dict):
                           # Check common keys again within the nested structure
                           for field in ["text", "output", "response", "result"]:
                                content = first_output_data.get(field)
                                if isinstance(content, str):
                                     logger.debug(f"Extracted text from nested output under key '{first_output_key}'->'{field}'")
                                     return content.strip()
                      # Check if the nested output data is just the string itself
                      elif isinstance(first_output_data, str):
                          logger.debug(f"Extracted text directly from nested output under key '{first_output_key}'")
                          return first_output_data.strip()

        # If no common key found or input is not dict/str, try converting non-complex types
        try:
            # Avoid converting large dicts/lists/bytes directly, might indicate wrong field access or hide errors
            if isinstance(phase_output_data, (dict, list, bytes)):
                 # Check if it's a simple dict/list with a single string value (less common)
                 if len(phase_output_data) == 1:
                      single_value = None
                      if isinstance(phase_output_data, dict):
                           single_value = next(iter(phase_output_data.values()), None)
                      elif isinstance(phase_output_data, list):
                           single_value = phase_output_data[0]
                      if isinstance(single_value, str):
                           return single_value.strip()

                 # Otherwise, assume complex structure we can't reliably stringify
                 logger.debug(f"Cannot reliably extract text from complex data structure: {type(phase_output_data)}")
                 return ""
             # Check if it's a simple type that can be stringified safely (int, float, bool, NoneType)
            elif phase_output_data is None:
                return "" # Explicitly handle None
            elif isinstance(phase_output_data, (int, float, bool)): # Add other simple types as needed
                 return str(phase_output_data).strip()
            else: # Unknown potentially complex type
                 logger.warning(f"Cannot extract text from phase output of unexpected type {type(phase_output_data)}. Returning empty string.")
                 return ""
        except Exception as e:
            logger.debug(f"Could not convert phase output of type {type(phase_output_data)} to string: {e}")
            return "" # Return empty string if conversion fails or is inappropriate

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration specific to this aggregation strategy.

        Returns:
            A deep copy of the strategy's configuration dictionary.
        """
        return copy.deepcopy(self._config)

    def get_name(self) -> str:
        """Get the name of this aggregation strategy.

        Returns:
            The strategy name string.
        """
        return self._strategy_name

    # Kept this method, but note that model requirements might be needed for some strategies
    # Consider moving model requirement logic elsewhere or making it dynamic?
    def get_required_models(self) -> Set[str]:
        """Get the set of models specifically required BY THE AGGREGATOR itself.

        Note: This usually applies only to model-based aggregators like EnsembleFusion
              or evaluation models used in MultidimensionalVoting/AdaptiveSelection.

        Returns:
            Set of model IDs required directly by this aggregation strategy.
        """
        # Default: Most aggregators don't inherently require specific models themselves.
        # Strategies needing models should override this or get models from config.
        required = set()
        # Example for model-based aggregators (they should override this):
        if "fusion_model" in self._config:
             required.add(self._config["fusion_model"])
        if "evaluator_model" in self._config:
             required.add(self._config["evaluator_model"])
        if "selector_model" in self._config: # For AdaptiveSelection
             required.add(self._config["selector_model"])
        return required


    def get_final_phase(self) -> Optional[str]:
        """Get the name of the phase considered 'final' by this strategy's config.

        Used by strategies like SequentialRefinement or as a fallback.

        Returns:
            The name of the final phase string, or None if not specified.
        """
        # Get from the strategy-specific config
        return self._config.get("final_phase")

    def _calculate_confidence(
        self,
        phase_outputs: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate an overall confidence score based on input phase outputs.

        Provides a default implementation that averages confidence scores found
        in the phase outputs. Subclasses can override this for more specific logic.

        Args:
            phase_outputs: Dictionary mapping phase names to their outputs.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        confidence_scores = []

        for phase_name, output_data in phase_outputs.items():
            # Check if the output is a dict and contains a 'confidence' key
            if isinstance(output_data, dict) and "confidence" in output_data:
                confidence_value = output_data["confidence"]
                score = None
                if isinstance(confidence_value, (int, float)):
                    # Direct numeric confidence score
                    score = float(confidence_value)
                elif isinstance(confidence_value, dict):
                    # Nested dictionary, look for a 'combined' score first
                    score = confidence_value.get("combined")
                    # Fallback: average all numeric values in the confidence dict? Risky.
                    # Or maybe use a specific metric like 'token_prob'?
                    if score is None: score = confidence_value.get("token_prob") # Example fallback
                # Ensure score is a valid number
                if isinstance(score, (int, float)):
                     # Clamp score to valid range [0, 1]
                     clamped_score = min(max(float(score), 0.0), 1.0)
                     confidence_scores.append(clamped_score)
                # else:
                #     logger.debug(f"Non-numeric or missing 'combined' confidence found in phase '{phase_name}'.")

        # Calculate average confidence or return default if no scores available
        if confidence_scores:
            average_confidence = sum(confidence_scores) / len(confidence_scores)
            logger.debug(f"Calculated average confidence from {len(confidence_scores)} phases: {average_confidence:.3f}")
            return average_confidence
        else:
            logger.warning("No valid confidence scores found in any phase outputs, returning default confidence 0.7")
            return 0.7  # Default medium-high confidence
