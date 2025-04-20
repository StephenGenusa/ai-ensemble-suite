"""Adaptive Selection aggregation strategy."""

from typing import Dict, Any, Optional, List, Set, Tuple, Type,TYPE_CHECKING
import time
import re

from ai_ensemble_suite.aggregation.base import BaseAggregator
# Import necessary concrete aggregator types for instantiation
from ai_ensemble_suite.aggregation.sequential_refinement import SequentialRefinement
from ai_ensemble_suite.aggregation.confidence_based import ConfidenceBased
from ai_ensemble_suite.aggregation.weighted_voting import WeightedVoting
from ai_ensemble_suite.aggregation.multidimensional_voting import MultidimensionalVoting
from ai_ensemble_suite.aggregation.ensemble_fusion import EnsembleFusion
from ..exceptions import ModelError

from ai_ensemble_suite.exceptions import AggregationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector
import copy


# Type hint imports
if TYPE_CHECKING:
    from ai_ensemble_suite.config import ConfigManager
    from ai_ensemble_suite.models import ModelManager

# Map strategy names to their corresponding classes
# Keep this registry updated if new strategies are added
STRATEGY_CLASS_MAP: Dict[str, Type[BaseAggregator]] = {
    "sequential_refinement": SequentialRefinement,
    "confidence_based": ConfidenceBased,
    "weighted_voting": WeightedVoting,
    "multidimensional_voting": MultidimensionalVoting,
    "ensemble_fusion": EnsembleFusion,
    # AdaptiveSelection itself should maybe not be in the map it uses,
    # unless nesting is desired (which adds complexity).
    # "adaptive_selection": AdaptiveSelection, # Avoid self-reference usually
}
DEFAULT_STRATEGY_CLASS = SequentialRefinement
DEFAULT_STRATEGY_NAME = "sequential_refinement"


class AdaptiveSelection(BaseAggregator):
    """Adaptive Selection aggregation strategy.

    Dynamically selects and executes another aggregation strategy based on
    analysis of the query, context, phase outputs, or an explicit selector model.
    """

    # Override __init__ to accept model_manager
    def __init__(
        self,
        config_manager: "ConfigManager",
        strategy_name: str, # Should be 'adaptive_selection'
        model_manager: Optional["ModelManager"] = None, # Required for selector model
        strategy_config_override: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the AdaptiveSelection aggregator."""
        super().__init__(config_manager, strategy_name, model_manager, strategy_config_override)
        if strategy_name != "adaptive_selection":
             logger.warning(f"AdaptiveSelection initialized with unexpected strategy name '{strategy_name}'.")


    async def aggregate(
        self,
        outputs: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Select the most appropriate strategy and execute it.

        Args:
            outputs: Dictionary mapping phase names to their outputs.
            context: Context information from collaboration phases (should include 'query').
            trace_collector: Optional trace collector for gathering execution details.

        Returns:
            Dictionary containing the result from the executed sub-strategy, potentially
            augmented with information about which strategy was selected.

        Raises:
            AggregationError: If selection or execution of the sub-strategy fails.
        """
        start_time = time.time()
        logger.debug(f"Starting Adaptive Selection aggregation...")

        if not outputs:
             raise AggregationError("No phase outputs provided for Adaptive Selection.")

        try:
            # Get the configuration for available strategies FROM THIS strategy's config
            # The 'strategies' key should contain definitions like { "name": { "config_details" } }
            available_strategies_config = self._config.get("strategies", {})

            # Provide default strategies if none are defined in config
            if not available_strategies_config or not isinstance(available_strategies_config, dict):
                logger.warning("No valid 'strategies' defined in AdaptiveSelection config, using defaults.")
                available_strategies_config = {
                    "sequential_refinement": {
                        "description": "Uses the output of the final phase in a sequence.",
                        "conditions": ["sequential", "refinement", "conversation", "final"],
                        # Example config for this strategy if run directly
                        "final_phase": context.get("phase_sequence", [])[-1] if context.get("phase_sequence") else None,
                    },
                    "confidence_based": {
                        "description": "Selects the output with the highest confidence score.",
                        "conditions": ["factual", "objective", "specific", "high confidence"],
                        "threshold": 0.7, # Example config
                    },
                    "weighted_voting": {
                        "description": "Combines outputs based on pre-defined weights.",
                        "conditions": ["opinion", "subjective", "creative", "multiple options"],
                        "weights": {}, # Example config (should ideally be populated)
                    },
                     "multidimensional_voting": {
                        "description": "Evaluates outputs on multiple dimensions and selects highest weighted score.",
                        "conditions": ["complex query", "multi-faceted", "evaluation needed"],
                        "dimensions": ["accuracy", "clarity", "completeness"], # Example
                        "dimension_weights": {}, # Example
                    },
                    "ensemble_fusion": {
                         "description": "Uses a model to fuse multiple outputs into one.",
                         "conditions": ["integration", "synthesis", "combine perspectives"],
                         # Needs fusion_model specified here or via context
                    }
                }

            # Get contextual info like query type, if available
            query = context.get("query", "")
            query_type = context.get("query_type", "") # Often determined earlier

            # --- Select the best strategy ---
            selected_strategy_name = await self._select_strategy(
                outputs, context, available_strategies_config, query, query_type, trace_collector
            )

            logger.info(f"Adaptive Selection chose strategy: '{selected_strategy_name}'")

            # Get the specific configuration for the selected strategy
            # This config will be passed as an override to the selected aggregator instance
            strategy_config = available_strategies_config.get(selected_strategy_name, {})
            if not strategy_config:
                 logger.warning(f"No configuration found for selected strategy '{selected_strategy_name}', using empty config.")
                 strategy_config = {}

             # Ensure the strategy name is part of its config for consistency
            strategy_config['strategy'] = selected_strategy_name


            # --- Instantiate and Execute the Selected Strategy ---
            # Get the class corresponding to the selected strategy name
            aggregator_class = STRATEGY_CLASS_MAP.get(selected_strategy_name)

            if aggregator_class is None:
                logger.error(f"Cannot execute selected strategy '{selected_strategy_name}': Class not found in registry. Falling back to default.")
                aggregator_class = DEFAULT_STRATEGY_CLASS
                selected_strategy_name = DEFAULT_STRATEGY_NAME
                # Get default config? For now, use empty, BaseAggregator will load defaults.
                strategy_config = {"strategy": selected_strategy_name}


            # Create the aggregator instance, passing:
            # - The *original* config_manager (for template access etc.)
            # - The selected strategy name
            # - The model_manager (needed by some strategies)
            # - The *specific configuration* for the selected strategy as an override
            try:
                aggregator_instance = aggregator_class(
                    config_manager=self._config_manager, # Original manager
                    strategy_name=selected_strategy_name,
                    model_manager=self._model_manager,    # Pass model manager
                    strategy_config_override=strategy_config # Pass specific config
                )
            except Exception as e:
                 logger.error(f"Failed to instantiate aggregator for strategy '{selected_strategy_name}': {e}", exc_info=True)
                 raise AggregationError(f"Failed to create instance for strategy '{selected_strategy_name}': {e}")


            # Execute the chosen strategy's aggregate method
            result = await aggregator_instance.aggregate(outputs, context, trace_collector)

            # Add adaptive selection information to the final result
            result["selected_strategy"] = selected_strategy_name


            execution_time = time.time() - start_time
            logger.info(
                f"Adaptive Selection completed using '{selected_strategy_name}' in {execution_time:.2f}s."
            )

            # Add trace for AdaptiveSelection itself, including the sub-result
            if trace_collector:
                trace_collector.add_aggregation_trace(
                    # Name of THIS strategy
                    strategy_name=self._strategy_name, # "adaptive_selection"
                    inputs={
                        "phase_output_keys": list(outputs.keys()),
                        "context_keys": list(context.keys()),
                        "available_strategies": list(available_strategies_config.keys()),
                        "query": query[:100] + "..." if len(query) > 100 else query,
                        "query_type": query_type
                    },
                    # Output includes the result from the sub-strategy + selection info
                    output={
                        "_selected_strategy": selected_strategy_name,
                        **result # Unpack the sub-strategy's result here
                    },
                    execution_time=execution_time,
                    parameters={ # Parameters of THIS strategy
                        # Store the config structure used for selection
                        "strategy_definitions": available_strategies_config,
                        "selector_model": self._config.get("selector_model")
                    }
                )

            return result

        except AggregationError as e:
             logger.error(f"Adaptive Selection aggregation failed: {str(e)}")
             raise # Re-raise known aggregation errors
        except Exception as e:
            logger.error(f"Unexpected error during Adaptive Selection aggregation: {str(e)}", exc_info=True)
            # Wrap unexpected errors
            raise AggregationError(f"Adaptive Selection aggregation failed unexpectedly: {str(e)}")

    async def _select_strategy(
        self,
        outputs: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        available_strategies: Dict[str, Dict[str, Any]], # Configs for potential strategies
        query: str,
        query_type: str,
        trace_collector: Optional[TraceCollector] = None
    ) -> str:
        """Select the most appropriate aggregation strategy based on various factors.

        Args:
            outputs: Outputs from collaboration phases.
            context: Context dictionary.
            available_strategies: Dictionary mapping strategy names to their configurations/metadata.
            query: The original user query.
            query_type: Pre-determined type/category of the query (if available).
            trace_collector: Optional trace collector.

        Returns:
            The name (string) of the selected strategy. Falls back to a default if selection fails.
        """
        logger.debug("Selecting aggregation strategy...")

        strategies_to_consider = list(available_strategies.keys())
        if not strategies_to_consider:
             logger.warning("No available strategies defined for selection. Falling back to default.")
             return DEFAULT_STRATEGY_NAME

        # --- Selection Logic (Priority Order) ---

        # 1. Check for explicit strategy preference in context
        preferred_strategy = context.get("preferred_strategy")
        if preferred_strategy and isinstance(preferred_strategy, str) and preferred_strategy in strategies_to_consider:
            logger.debug(f"Using preferred strategy from context: '{preferred_strategy}'")
            return preferred_strategy

        # 2. Use dedicated strategy selector model if configured
        selector_model_id = self._config.get("selector_model")
        if selector_model_id:
            if self._model_manager is None:
                logger.warning("Selector model specified, but ModelManager not available. Cannot use selector model.")
            else:
                logger.debug(f"Using selector model '{selector_model_id}' to choose strategy.")
                try:
                      # Format strategy options for the selector prompt
                      strategy_options_prompt = ""
                      for name, config in available_strategies.items():
                           description = config.get("description", f"Strategy: {name}")
                           conditions = config.get("conditions", [])
                           conditions_str = ", ".join(conditions) if conditions else "General purpose"
                           strategy_options_prompt += f"- {name}: {description} (Best for: {conditions_str})\n"

                      # Create the prompt for the selector model
                      selector_prompt_template = self._config.get("selector_prompt_template", "adaptive_selector_default")
                      default_selector_prompt = f"""Given the user query and available aggregation strategies, select the *single best* strategy name.

USER QUERY:
{{query}}

AVAILABLE STRATEGIES:
{{strategy_options}}

Analyze the query and choose the strategy name from the list above that is most suitable.
Respond with ONLY the strategy name (e.g., 'confidence_based', 'ensemble_fusion')."""

                      selector_prompt = ""
                      try:
                           context = {"query": query, "strategy_options": strategy_options_prompt}
                           selector_prompt = self._config_manager.render_prompt(selector_prompt_template, context)
                      except (ConfigurationError, KeyError):
                            logger.warning(f"Selector template '{selector_prompt_template}' failed, using default.")
                            selector_prompt = default_selector_prompt.format(query=query, strategy_options=strategy_options_prompt)


                      # Run the selector model inference
                      selector_result_raw = await self._model_manager.run_inference(
                          model_id=selector_model_id,
                          prompt=selector_prompt,
                          temperature=0.1, # Low temp for deterministic selection
                          max_tokens=50    # Expect short response (name only)
                      )

                      # Add trace for the selector model call
                      if trace_collector:
                           trace_collector.add_model_trace(
                               model_id=selector_model_id,
                               input_prompt=selector_prompt,
                               output=selector_result_raw,
                               execution_time=selector_result_raw.get("total_time", 0),
                               parameters={"role": "strategy_selector"}
                           )

                      # Extract the selected strategy name from the model's response
                      selected_name_raw = selector_result_raw.get("text", "").strip().lower()
                      # Clean up potential extra text
                      selected_name_clean = re.split(r'[\s:,\.\-]+', selected_name_raw)[0]


                      # Validate if the selected name is one of the available strategies
                      if selected_name_clean in strategies_to_consider:
                          logger.debug(f"Selector model chose strategy: '{selected_name_clean}'")
                          return selected_name_clean
                      else:
                          logger.warning(f"Selector model returned invalid strategy name: '{selected_name_raw}'. Ignoring.")

                except ModelError as e:
                    logger.error(f"Strategy selector model '{selector_model_id}' failed: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error using strategy selector model: {str(e)}", exc_info=True)


        # 3. Use query type matching based on 'conditions' in strategy config
        if query_type:
            logger.debug(f"Trying strategy selection based on query_type: '{query_type}'")
            query_type_lower = query_type.lower()
            for name, config in available_strategies.items():
                 conditions = config.get("conditions", [])
                 if isinstance(conditions, list):
                      for cond in conditions:
                           if isinstance(cond, str) and cond.lower() in query_type_lower:
                                logger.debug(f"Matched query_type '{query_type}' to condition '{cond}' for strategy '{name}'")
                                return name

        # 4. Check phase outputs for keywords matching 'conditions' (Less reliable)
        # Combine text from all phase outputs
        # all_output_text = " ".join([self._extract_output(o_data) for o_data in outputs.values()])
        # if len(all_output_text) > 50: # Only if there's substantial text
        #     all_output_text_lower = all_output_text.lower()
        #     match_counts: Dict[str, int] = {}
        #     for name, config in available_strategies.items():
        #         conditions = config.get("conditions", [])
        #         count = 0
        #         if isinstance(conditions, list):
        #             for cond in conditions:
        #                 if isinstance(cond, str) and cond.lower() in all_output_text_lower:
        #                      count += 1
        #         if count > 0: match_counts[name] = count
        #
        #     if match_counts:
        #         # Select strategy with the most condition matches in the text
        #         best_match_strategy = max(match_counts, key=match_counts.get)
        #         logger.debug(f"Selected strategy '{best_match_strategy}' based on keyword matches in outputs "
        #                      f"(Matches: {match_counts[best_match_strategy]})")
        #         return best_match_strategy


        # 5. Fallback: Use weighted selection if weights are defined (less common for adaptive)
        # Or simply fallback to the default strategy if no specific selection criteria met.

        # --- Final Fallback ---
        logger.debug(f"No specific strategy selected based on criteria. Falling back to default: '{DEFAULT_STRATEGY_NAME}'")
        # Ensure the default is actually in the available list, otherwise pick the first available one
        if DEFAULT_STRATEGY_NAME in strategies_to_consider:
             return DEFAULT_STRATEGY_NAME
        elif strategies_to_consider:
             first_available = strategies_to_consider[0]
             logger.warning(f"Default strategy '{DEFAULT_STRATEGY_NAME}' not available, using first available: '{first_available}'")
             return first_available
        else:
             # This case should be caught earlier, but as a safety net:
             raise AggregationError("Cannot select strategy: No available strategies defined and no default applicable.")


    # Inherit _extract_output from BaseAggregator, no need to redefine unless specialized.
    # def _extract_output(self, phase_output: Any) -> str: ...
