# ai_ensemble_suite/ensemble.py

"""Main Ensemble class for ai-ensemble-suite."""

from typing import Dict, Any, Optional, Union, Type, List, TypeVar, Awaitable, Tuple, Set
from types import TracebackType
import time
import asyncio
import copy  # Import copy

# Configuration and Core Components
from ai_ensemble_suite.config import ConfigManager
from ai_ensemble_suite.models import ModelManager
from ai_ensemble_suite.exceptions import (
    AiEnsembleSuiteError, ConfigurationError, ModelError,
    CollaborationError, AggregationError, ValidationError
)
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector

# Collaboration Phase Imports
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
# Import Debate Subtypes for mapping
from ai_ensemble_suite.collaboration.structured_debate import (
    StructuredCritique, SynthesisOriented, RoleBasedDebate, BaseDebate
)
from ai_ensemble_suite.collaboration.bagging import Bagging
from ai_ensemble_suite.collaboration.uncertaintybased import UncertaintyBasedCollaboration
from ai_ensemble_suite.collaboration.stackedgeneralization import StackedGeneralization

# Aggregation Strategy Imports
from ai_ensemble_suite.aggregation.base import BaseAggregator
from ai_ensemble_suite.aggregation.weighted_voting import WeightedVoting
from ai_ensemble_suite.aggregation.sequential_refinement import SequentialRefinement
from ai_ensemble_suite.aggregation.confidence_based import ConfidenceBased
from ai_ensemble_suite.aggregation.multidimensional_voting import MultidimensionalVoting
from ai_ensemble_suite.aggregation.ensemble_fusion import EnsembleFusion
from ai_ensemble_suite.aggregation.adaptive_selection import AdaptiveSelection

# Template manager
from ai_ensemble_suite.config.template_manager import TemplateManager

# Type variable for self-referential type hints
T_Ensemble = TypeVar("T_Ensemble", bound="Ensemble")


class Ensemble:
    """Coordinates the collaboration of multiple AI models for complex tasks.

    Manages configuration, model loading, phase execution, and result aggregation
    based on defined collaboration modes and aggregation strategies.
    """

    # Registry for collaboration phase types (maps lowercase type name to class)
    # Registry for collaboration phase types (maps lowercase type name to class)
    _COLLABORATION_TYPES: Dict[str, Type[BaseCollaborationPhase]] = {
        # Simple Phases
        "async_thinking": AsyncThinking,
        "integration": Integration,
        "expert_committee": ExpertCommittee,
        "hierarchical_review": HierarchicalReview,

        # More Complex Phases
        "competitive_evaluation": CompetitiveEvaluation,
        "perspective_rotation": PerspectiveRotation,
        "chain_of_thought": ChainOfThoughtBranching,  # Correct class name confirmed
        "adversarial_improvement": AdversarialImprovement,
        "role_based_workflow": RoleBasedWorkflow,

        # Debate (needs special handling via subtype)
        "structured_debate": BaseDebate,  # Base class maps here, subtype determines actual class

        # Direct mapping for debate subtypes for clarity/flexibility in config
        "critique": StructuredCritique,  # Can be used directly or via structured_debate subtype
        "synthesis": SynthesisOriented,  # Can be used directly or via ...
        "role_based_debate": RoleBasedDebate,  # Can be used directly or via ...

        # Add this line to include Bagging
        "bagging": Bagging,
        "uncertainty_based": UncertaintyBasedCollaboration,
        "stacked_generalization": StackedGeneralization,
    }

    # Specific mapping for structured_debate subtypes (used by _get_phase_class)
    _DEBATE_SUBTYPES: Dict[str, Type[BaseDebate]] = {
        "critique": StructuredCritique,
        "synthesis": SynthesisOriented,
        "role_based_debate": RoleBasedDebate,
    }

    # Registry for aggregation strategy types (maps lowercase type name to class)
    _AGGREGATION_TYPES: Dict[str, Type[BaseAggregator]] = {
        "weighted_voting": WeightedVoting,
        "sequential_refinement": SequentialRefinement,
        "confidence_based": ConfidenceBased,
        "multidimensional_voting": MultidimensionalVoting,
        "ensemble_fusion": EnsembleFusion,
        "adaptive_selection": AdaptiveSelection,
    }

    def __init__(
            self,
            config_path: Optional[str] = None,
            config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the Ensemble orchestration layer.

        Args:
            config_path: Optional path to a YAML configuration file.
            config_dict: Optional dictionary containing configuration values.
                         If both are None, default configuration is used.
                         If both are provided, raises ConfigurationError.

        Raises:
            ConfigurationError: If configuration loading or validation fails,
                                or if both config_path and config_dict are given.
            AiEnsembleSuiteError: For other unexpected initialization errors.
        """
        logger.info("Initializing Ensemble...")
        self.config_manager: ConfigManager
        self.model_manager: ModelManager
        self._initialized: bool = False
        self._initialization_lock = asyncio.Lock()  # Lock for initialization process

        try:
            # Initialize the configuration manager
            # ConfigManager handles defaults and loading priority internally
            self.config_manager = ConfigManager(config_path, config_dict)
            logger.info("ConfigManager initialized.")

            # Initialize the template manager
            self.template_manager = TemplateManager(self.config_manager)
            logger.info("TemplateManager initialized.")

            # Initialize the model manager, passing the config manager and self-reference
            self.model_manager = ModelManager(self.config_manager, ensemble=self)
            logger.info("ModelManager initialized.")

        except (ConfigurationError, ValidationError) as e:
            logger.error(f"Ensemble initialization failed due to configuration error: {e}", exc_info=True)
            raise ConfigurationError(f"Configuration failed: {e}") from e  # Re-raise specific config errors
        except Exception as e:
            logger.error(f"Unexpected error during Ensemble setup: {e}", exc_info=True)
            # Wrap unexpected errors
            raise AiEnsembleSuiteError(f"Failed to set up Ensemble components: {e}") from e

        logger.info("Ensemble instance created. Call initialize() or use async context manager before 'ask'.")

    async def initialize(self) -> None:
        """Load models and prepare the ensemble for processing queries.

        This method is idempotent; it only performs initialization once.
        Must be called before `ask` if not using the async context manager (`async with`).

        Raises:
            ModelError: If model loading fails during ModelManager initialization.
            ConfigurationError: If configuration issues prevent model loading.
            AiEnsembleSuiteError: For unexpected errors during initialization.
        """
        async with self._initialization_lock:
            if self._initialized:
                logger.debug("Ensemble already initialized. Skipping.")
                return

            logger.info("Initializing ensemble resources (loading models)...")
            init_start_time = time.time()

            try:
                # Initialize the model manager (this loads the models)
                await self.model_manager.initialize()

                self._initialized = True
                init_duration = time.time() - init_start_time
                logger.info(f"Ensemble initialization complete in {init_duration:.2f}s. Ready for queries.")

            except (ModelError, ConfigurationError) as e:
                # ModelManager.initialize handles its own logging and potential shutdown on failure
                logger.error(f"Ensemble initialization failed: {e}")
                self._initialized = False  # Ensure state reflects failure
                raise  # Re-raise caught known errors
            except Exception as e:
                logger.error(f"Unexpected error during ensemble initialization: {e}", exc_info=True)
                # Attempt graceful shutdown if possible, though model_manager might have already tried
                try:
                    if self.model_manager: await self.model_manager.shutdown()
                except Exception as shutdown_e:
                    logger.error(f"Error during shutdown attempt after initialization failure: {shutdown_e}")
                self._initialized = False
                raise AiEnsembleSuiteError(f"Unexpected error during ensemble initialization: {e}") from e

    async def shutdown(self) -> None:
        """Release resources used by the ensemble (unload models, shutdown threads).

        This method is idempotent. Should be called when done with the ensemble
        if not using the async context manager.
        """
        # Use the lock to prevent race conditions with initialization or concurrent shutdowns
        async with self._initialization_lock:
            # Check if already effectively shut down or never initialized properly
            if not self._initialized and (not hasattr(self,
                                                      'model_manager') or self.model_manager is None or not self.model_manager.initialized):
                logger.info("Ensemble already shut down or was not successfully initialized.")
                return

            logger.info("Shutting down ensemble resources...")
            shutdown_start_time = time.time()

            try:
                # Shutdown the model manager (this unloads models and stops executor)
                if hasattr(self, 'model_manager') and self.model_manager:
                    await self.model_manager.shutdown()

            except Exception as e:
                # Log error but proceed to reset state
                logger.error(f"Error during ensemble shutdown: {e}", exc_info=True)

            finally:
                # Reset state regardless of shutdown errors
                self._initialized = False
                # Setting managers to None might prevent reuse, decide based on desired behavior.
                # Clearing might be safer if re-initialization with new config is possible.
                # self.model_manager = None
                # self.config_manager = None
                shutdown_duration = time.time() - shutdown_start_time
                logger.info(f"Ensemble shutdown completed in {shutdown_duration:.2f}s.")

    async def ask(
            self,
            query: str,
            **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Process a query through the configured collaboration and aggregation pipeline.

        Args:
            query: The user query or task description string.
            **kwargs: Optional overrides and parameters for the execution:
                trace (bool): If True, return a detailed trace dictionary along
                              with the response. Defaults to False.
                collaboration_mode (str): Override the collaboration mode from config.
                aggregation_strategy (str): Override the aggregation strategy from config.
                # Context can also be passed via kwargs to provide initial state
                initial_context (Dict[str, Any]): A dictionary to merge into the starting context.

        Returns:
            - If `trace=False` (default): The final aggregated response string.
            - If `trace=True`: A dictionary containing:
                - 'response': The final aggregated response string.
                - 'trace': A dictionary containing detailed execution trace data.
                - 'execution_time': Total time taken for the query processing.
                - 'confidence': Aggregated confidence score (if available).

        Raises:
            ModelError: If model inference fails.
            CollaborationError: If a collaboration phase encounters an error.
            AggregationError: If the aggregation strategy fails.
            ConfigurationError: If configuration is invalid or required elements missing.
            AiEnsembleSuiteError: For other generic ensemble errors.
        """
        if not self._initialized:
            logger.info("Ensemble not initialized, attempting auto-initialization for 'ask'...")
            await self.initialize()  # Auto-initialize if needed (will raise if it fails)
            if not self._initialized:  # Double check after attempt
                raise AiEnsembleSuiteError("Ensemble initialization failed. Cannot process query.")

        # Start timing the overall request processing
        start_time = time.time()

        # Determine if tracing is requested for this call
        include_trace = kwargs.get("trace", False)

        # Instantiate TraceCollector ONLY if tracing is enabled, passing config and logger
        trace_collector: Optional[TraceCollector] = None  # Initialize as None first
        if include_trace:
            # Pass the configuration and the imported logger instance
            trace_collector = TraceCollector(
                logger_instance=logger  # Pass the imported logger
            )
            # Now start session if successfully created
            trace_collector.start_session()
            logger.debug("Tracing enabled for this request.")

        try:
            logger.info(f"Processing query (first 100 chars): '{query[:100]}{'...' if len(query) > 100 else ''}'")

            # --- Set up Initial Context ---
            # Start with the query, allow merging external context if provided
            initial_context = {"query": query}
            provided_context = kwargs.get("initial_context")
            if isinstance(provided_context, dict):
                initial_context.update(provided_context)
                logger.debug(f"Merging provided initial context keys: {list(provided_context.keys())}")

            # --- Execute Collaboration Phases ---
            # Pass query, initial context, potential overrides (**kwargs), and tracer
            phase_outputs, final_context = await self._execute_collaboration_phases(
                query, initial_context, trace_collector, **kwargs
            )

            # --- Aggregate Results ---
            # Pass phase outputs, the final context from phases, tracer, and overrides
            aggregation_result = await self._aggregate_results(
                phase_outputs, final_context, trace_collector, **kwargs
            )

            # Extract final response text from aggregation result
            response_text = aggregation_result.get("response", "")
            if not isinstance(response_text, str):
                logger.warning(
                    f"Aggregation result 'response' is not a string (type: {type(response_text)}). Converting.")
                response_text = str(response_text)

            if not response_text:
                logger.warning("Aggregation resulted in an empty response.")

            # Calculate total execution time
            execution_time = time.time() - start_time
            logger.info(f"Query processed successfully in {execution_time:.2f} seconds.")

            # Finalize trace session if tracing was enabled
            if trace_collector:
                # Create a snapshot of relevant configuration (avoiding secrets)
                # Use the actual mode/strategy used, considering overrides
                collab_mode_used = kwargs.get("collaboration_mode", self.config_manager.get_collaboration_mode())
                agg_strategy_used = kwargs.get("aggregation_strategy", self.config_manager.get_aggregation_strategy())

                config_snapshot = {
                    "collaboration_mode_used": collab_mode_used,
                    "aggregation_strategy_used": agg_strategy_used,
                    "model_ids_configured": self.config_manager.get_model_ids(),
                    "phase_sequence_executed": final_context.get("phase_sequence", [])
                    # Add other relevant high-level config keys if needed
                }
                trace_collector.add_session_trace(
                    query=query,
                    final_response=response_text,
                    total_execution_time=execution_time,
                    configuration=config_snapshot  # Use sanitized snapshot
                )
                trace_collector.end_session()
                logger.debug("Trace session finalized.")

            # Return the final result (either text or dict with trace)
            if include_trace and trace_collector:
                return {
                    "response": response_text,
                    # Use get_trace_data() which includes stats
                    "trace": trace_collector.get_trace_data(),
                    "execution_time": execution_time,
                    # Include top-level confidence from aggregation if available
                    "confidence": aggregation_result.get("confidence")
                }
            else:
                # Default: return only the response string
                return response_text

        except (ModelError, CollaborationError, AggregationError, ConfigurationError, ValidationError) as e:
            # Log specific known errors and re-raise them
            logger.error(f"Error processing query: {type(e).__name__}: {str(e)}", exc_info=True)
            if trace_collector: trace_collector.end_session()  # Ensure trace session ends on error
            raise  # Re-raise the original error type
        except Exception as e:
            # Catch unexpected errors
            logger.error(f"Unexpected error processing query: {e}", exc_info=True)
            if trace_collector: trace_collector.end_session()
            # Wrap in a generic ensemble error
            raise AiEnsembleSuiteError(f"Failed to process query due to unexpected error: {e}") from e

    async def _execute_collaboration_phases(
            self,
            query: str,
            initial_context: Dict[str, Any],
            trace_collector: Optional[TraceCollector] = None,
            **kwargs: Any
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Execute the sequence of collaboration phases defined in the configuration.

        Args:
            query: The user query.
            initial_context: Starting context dictionary, including the query.
            trace_collector: Optional trace collector for gathering execution details.
            **kwargs: Optional configuration overrides (e.g., collaboration_mode).

        Returns:
            A tuple containing:
            - Dictionary mapping phase names to their full output dictionaries.
            - The final context dictionary after all phases have executed.

        Raises:
            CollaborationError: If any phase fails execution.
            ConfigurationError: If phase configuration is invalid or missing.
        """
        # Determine collaboration mode (allow override via kwargs)
        collaboration_mode = kwargs.get(
            "collaboration_mode",
            self.config_manager.get_collaboration_mode()  # Get from config manager
        )
        logger.info(f"Executing collaboration phases using mode: '{collaboration_mode}'")

        # Get phases configuration list
        try:
            # Get the full collaboration section config
            collaboration_config = self.config_manager.get_collaboration_config()  # Gets entire section
            phases_config_list = collaboration_config.get("phases", [])

            if not isinstance(phases_config_list, list):
                raise ConfigurationError(f"Config error: 'phases' must be a list in collaboration section.")
            if not phases_config_list:
                logger.warning(
                    f"No phases defined in collaboration configuration for mode '{collaboration_mode}'. Collaboration stage will be skipped.")
                # Return empty outputs and the initial context if no phases
                return {}, initial_context

        except ConfigurationError as e:
            logger.error(f"Failed to retrieve collaboration phase configuration: {e}")
            raise  # Re-raise config error

        # Check for circular dependencies between phases
        dependency_graph = {}
        for phase_config in phases_config_list:
            if not isinstance(phase_config, dict):
                continue

            phase_name = phase_config.get("name")
            if not phase_name:
                continue

            input_from = phase_config.get("input_from", [])
            if isinstance(input_from, str):
                input_from = [input_from]
            elif not isinstance(input_from, list):
                input_from = []

            dependency_graph[phase_name] = input_from

        # Detect circular dependencies
        def detect_cycle(node: str, visited: Set[str], path: List[str]) -> bool:
            """Recursive helper to detect cycles in the dependency graph."""
            if node in path:
                cycle_path = path[path.index(node):] + [node]
                raise CollaborationError(f"Circular dependency detected in phases: {' -> '.join(cycle_path)}")

            if node in visited:
                return False

            visited.add(node)
            path.append(node)

            for dep in dependency_graph.get(node, []):
                if dep in dependency_graph:  # Only check dependencies that are actual phases
                    detect_cycle(dep, visited, path)

            path.pop()
            return False

        # Check each phase for cycles in its dependencies
        visited: Set[str] = set()
        for phase_name in dependency_graph:
            if phase_name not in visited:
                detect_cycle(phase_name, visited, [])

        # Initialize execution variables
        current_context: Dict[str, Any] = copy.deepcopy(initial_context)
        phase_outputs: Dict[str, Dict[str, Any]] = {}  # Store results keyed by phase name
        phase_sequence: List[str] = []  # Track execution order

        # Execute each defined phase sequentially
        for phase_config in phases_config_list:
            if not isinstance(phase_config, dict):
                logger.warning(f"Skipping invalid phase config item (not a dict): {phase_config}")
                continue

            # Validate essential phase config keys
            phase_name = phase_config.get("name")
            phase_type = phase_config.get("type")
            if not phase_name or not isinstance(phase_name, str):
                raise ConfigurationError(
                    f"Phase configuration item missing required string field: 'name'. Config: {phase_config}")
            if not phase_type or not isinstance(phase_type, str):
                raise ConfigurationError(f"Phase '{phase_name}' missing required string field: 'type'")
            if phase_name in phase_outputs:
                raise ConfigurationError(f"Duplicate phase name detected: '{phase_name}'. Phase names must be unique.")

            logger.info(f"-- Executing Phase: '{phase_name}' (Type: '{phase_type}') --")

            # Get the appropriate phase class based on type and potentially subtype
            try:
                phase_class = self._get_phase_class(phase_type, phase_config)
            except ConfigurationError as e:
                logger.error(f"Cannot execute phase '{phase_name}' due to configuration issue: {e}")
                # Wrap in CollaborationError to indicate phase execution failure point
                raise CollaborationError(f"Configuration error for phase '{phase_name}': {e}") from e

            # Instantiate and execute the phase
            phase_instance: BaseCollaborationPhase
            try:
                # Phase constructor expects model_manager, config_manager, and phase_name
                phase_instance = phase_class(
                    model_manager=self.model_manager,
                    config_manager=self.config_manager,
                    phase_name=phase_name  # Pass name for config lookup within phase
                )
            except (ConfigurationError, ValidationError) as e:
                logger.error(f"Failed to initialize phase '{phase_name}' instance: {e}", exc_info=True)
                raise CollaborationError(f"Initialization failed for phase '{phase_name}': {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error initializing phase '{phase_name}' instance: {e}", exc_info=True)
                raise CollaborationError(f"Unexpected initialization failure for phase '{phase_name}': {e}") from e

            try:
                # Execute the phase, passing the current context
                # Phase returns its result dictionary
                phase_start_time = time.time()
                phase_result = await phase_instance.execute(query, current_context, trace_collector)
                phase_duration = time.time() - phase_start_time
                logger.debug(f"Phase '{phase_name}' executed in {phase_duration:.2f}s")

                # --- Update context and store results ---
                if not isinstance(phase_result, dict):
                    logger.warning(
                        f"Phase '{phase_name}' did not return a dictionary (returned {type(phase_result)}). Wrapping result.")
                    # Store something basic to avoid KeyError later
                    phase_outputs[phase_name] = {"output": str(phase_result), "confidence": 0.0,
                                                 "error": "Invalid return type"}
                else:
                    # Ensure core keys exist, provide defaults if missing
                    phase_result.setdefault("output", "")  # Expecting 'output' as primary text
                    phase_result.setdefault("confidence", 0.5)  # Default confidence
                    phase_outputs[phase_name] = phase_result

                # Update the main context with the *full results dict* of this phase, keyed by phase name
                # This makes the output available to subsequent phases via _get_inputs_from_context
                current_context[phase_name] = phase_outputs[phase_name]

                # Add phase name to the execution sequence tracker
                phase_sequence.append(phase_name)

            except CollaborationError as e:
                # Log phase-specific collaboration errors and re-raise
                logger.error(f"Execution failed for phase '{phase_name}': {e}", exc_info=True)
                # Re-raise to halt execution
                raise CollaborationError(f"Failed during execution of phase '{phase_name}': {e}") from e
            except Exception as e:
                # Catch unexpected errors during phase execution
                logger.error(f"Unexpected error during phase '{phase_name}' execution: {e}", exc_info=True)
                raise CollaborationError(f"Unexpected failure in phase '{phase_name}': {e}") from e

        # Add the final phase sequence to the context for reference (e.g., by aggregation)
        current_context["phase_sequence"] = phase_sequence

        logger.info("Completed execution of all collaboration phases.")
        return phase_outputs, current_context

    def _get_phase_class(
            self,
            phase_type: str,
            phase_config: Dict[str, Any]  # Pass config for subtype lookup
    ) -> Type[BaseCollaborationPhase]:
        """Retrieve the collaboration phase class based on its type name.

        Handles special cases like 'structured_debate' which uses a subtype.

        Args:
            phase_type: The type name of the phase (string).
            phase_config: The configuration dictionary for the phase (used for subtype).

        Returns:
            The corresponding BaseCollaborationPhase subclass.

        Raises:
            ConfigurationError: If the phase type or subtype is unknown.
        """
        phase_type_lower = phase_type.lower()

        # Handle structured_debate with subtypes
        if phase_type_lower == "structured_debate":
            subtype = phase_config.get("subtype")
            if not subtype or not isinstance(subtype, str):
                # If subtype is missing or invalid, default to critique
                logger.warning(
                    f"Phase '{phase_config.get('name', 'unknown')}' is type 'structured_debate' but missing valid 'subtype'. Defaulting to 'critique'.")
                subtype = "critique"

            subtype_lower = subtype.lower()
            phase_class = self._DEBATE_SUBTYPES.get(subtype_lower)
            if phase_class:
                logger.debug(
                    f"Resolved 'structured_debate' with subtype '{subtype_lower}' to class {phase_class.__name__}")
                return phase_class
            else:
                known_subtypes = list(self._DEBATE_SUBTYPES.keys())
                raise ConfigurationError(
                    f"Unknown structured_debate subtype: '{subtype}' for phase '{phase_config.get('name', 'unknown')}'. "
                    f"Known subtypes: {known_subtypes}"
                )

        # Handle direct mapping from type name for other phases
        phase_class = self._COLLABORATION_TYPES.get(phase_type_lower)
        if phase_class:
            # If the type happens to be a base debate class but used directly, maybe guide user?
            if phase_class == BaseDebate:
                logger.warning(f"Phase '{phase_config.get('name', 'unknown')}' uses 'structured_debate' directly. "
                               f"Consider using a specific subtype (critique, synthesis, role_based) for clearer behavior.")
                # Default to critique if BaseDebate used directly?
                return StructuredCritique

            logger.debug(f"Resolved phase type '{phase_type_lower}' to class {phase_class.__name__}")
            return phase_class
        else:
            # Type not found in any mapping
            known_types = list(self._COLLABORATION_TYPES.keys())
            raise ConfigurationError(
                f"Unknown collaboration phase type: '{phase_type}' for phase '{phase_config.get('name', 'unknown')}'. "
                f"Known types: {known_types}"
            )

    async def _aggregate_results(
            self,
            outputs: Dict[str, Dict[str, Any]],
            context: Dict[str, Any],
            trace_collector: Optional[TraceCollector] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """Aggregate the outputs from the collaboration phases using the configured strategy.

        Args:
            outputs: Dictionary mapping phase names to their full output dictionaries.
            context: Dictionary containing context information from the ensemble run.
            trace_collector: Optional trace collector for gathering execution details.
            **kwargs: Optional overrides (e.g., aggregation_strategy).

        Returns:
            Dictionary containing the aggregated response and metadata.

        Raises:
            AggregationError: If the aggregation process fails.
            ConfigurationError: If the aggregation strategy configuration is invalid.
        """
        # Determine aggregation strategy (allow override via kwargs)
        aggregation_strategy_name = kwargs.get(
            "aggregation_strategy",
            self.config_manager.get_aggregation_strategy()
        ).lower()

        logger.info(f"Aggregating phase results using strategy: '{aggregation_strategy_name}'")

        # Get the aggregator class from the registry with improved error handling
        aggregator_class = self._AGGREGATION_TYPES.get(aggregation_strategy_name)

        if aggregator_class is None:
            known_strategies = list(self._AGGREGATION_TYPES.keys())
            error_msg = f"Unknown aggregation strategy: '{aggregation_strategy_name}'. Known strategies: {known_strategies}"
            logger.error(error_msg)

            # Check for fallback strategy in configuration
            try:
                aggregation_config = self.config_manager.get_aggregation_config()
                fallback_config = aggregation_config.get("fallback", {})
                if isinstance(fallback_config, dict) and "strategy" in fallback_config:
                    fallback_strategy = fallback_config["strategy"].lower()
                    fallback_class = self._AGGREGATION_TYPES.get(fallback_strategy)

                    if fallback_class:
                        logger.warning(
                            f"Using fallback strategy '{fallback_strategy}' instead of unknown '{aggregation_strategy_name}'")
                        aggregator_class = fallback_class
                        aggregation_strategy_name = fallback_strategy
            except Exception as fallback_e:
                logger.error(f"Error attempting to use fallback strategy: {fallback_e}")

            # Last resort fallback to sequential_refinement
            if aggregator_class is None:
                if "sequential_refinement" in self._AGGREGATION_TYPES:
                    logger.warning(f"Using 'sequential_refinement' as last-resort fallback")
                    aggregator_class = self._AGGREGATION_TYPES["sequential_refinement"]
                    aggregation_strategy_name = "sequential_refinement"
                else:
                    raise AggregationError(error_msg)

        # Instantiate the aggregator
        aggregator_instance: BaseAggregator
        try:
            # Pass ConfigManager, strategy name, and ModelManager
            # ModelManager is needed by some strategies (e.g., evaluation, fusion, adaptive)
            # strategy_config_override is handled internally by AdaptiveSelection if used
            aggregator_instance = aggregator_class(
                config_manager=self.config_manager,
                strategy_name=aggregation_strategy_name,
                model_manager=self.model_manager  # Pass manager instance
            )
        except (ConfigurationError, TypeError, ValidationError) as e:
            logger.error(f"Configuration or Type error instantiating aggregator '{aggregation_strategy_name}': {e}",
                         exc_info=True)
            raise ConfigurationError(f"Failed to create aggregator '{aggregation_strategy_name}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error instantiating aggregator '{aggregation_strategy_name}': {e}", exc_info=True)
            raise AggregationError(
                f"Unexpected failure creating aggregator instance for '{aggregation_strategy_name}': {e}") from e

        # Execute the aggregation process
        try:
            # Pass phase outputs and the FINAL context from the collaboration stage
            aggregation_result = await aggregator_instance.aggregate(
                outputs, context, trace_collector
            )

            # Validate the result structure
            if not isinstance(aggregation_result, dict) or "response" not in aggregation_result:
                logger.error(
                    f"Aggregation strategy '{aggregation_strategy_name}' did not return a valid dictionary with a 'response' key. Result: {aggregation_result}")
                # Attempt a fallback based on the last phase output
                fallback_response = "Aggregation failed: No valid response generated."
                fallback_confidence = 0.0
                if context.get("phase_sequence") and outputs:
                    last_phase = context["phase_sequence"][-1]
                    if last_phase in outputs:
                        response = outputs[last_phase].get("output", fallback_response)
                        fallback_response = str(response)  # Ensure string
                        fallback_confidence = outputs[last_phase].get("confidence", fallback_confidence)

                return {
                    "response": fallback_response,
                    "confidence": fallback_confidence,
                    "error": f"Invalid result structure from aggregator '{aggregation_strategy_name}'"
                }

            # Ensure confidence key exists, provide default if missing
            aggregation_result.setdefault("confidence", 0.5)  # Default confidence if strategy forgets

            logger.info(f"Aggregation completed using strategy '{aggregation_strategy_name}'.")
            return aggregation_result

        except AggregationError as e:
            # Log and re-raise known aggregation errors
            logger.error(f"Aggregation failed using strategy '{aggregation_strategy_name}': {e}", exc_info=True)
            raise  # Re-raise the original error
        except Exception as e:
            # Catch unexpected errors during aggregation execution
            logger.error(f"Unexpected error during aggregation with strategy '{aggregation_strategy_name}': {e}",
                         exc_info=True)
            raise AggregationError(
                f"Aggregation strategy '{aggregation_strategy_name}' failed unexpectedly: {e}") from e

    def configure(
            self,
            config_dict: Dict[str, Any]
    ) -> None:
        """Update the ensemble's configuration dynamically.

        Applies the provided dictionary on top of the existing configuration.
        Re-validates the configuration after update. If validation fails,
        the configuration change is reverted.

        Note: This does NOT automatically reload models if model paths or parameters change.
              A subsequent manual re-initialization (`await ensemble.shutdown(); await ensemble.initialize()`)
              is typically required for model changes to take effect.

        Args:
            config_dict: Dictionary containing configuration values to update.

        Raises:
            ConfigurationError: If the update dict is not a dict or if the resulting
                                configuration is invalid after merge.
            ValidationError: If validation fails (subclass of ConfigurationError).
        """
        if self._initialized:
            logger.warning("Configuring Ensemble after initialization. This does NOT automatically reload models. "
                           "Call shutdown() and initialize() again for model config changes to apply.")

        logger.info("Attempting to update ensemble configuration...")
        try:
            # Use ConfigManager's update method which includes validation and rollback on failure
            self.config_manager.update(config_dict)
            # If successful, ModelManager implicitly uses the updated config on next access requiring config
            logger.info("Ensemble configuration updated successfully.")
        except (ConfigurationError, ValidationError) as e:
            # ConfigManager.update handles logging and rollback
            # Re-raise the error
            raise ConfigurationError(f"Configuration update failed validation: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during configuration update: {e}", exc_info=True)
            raise ConfigurationError(f"Unexpected error applying configuration update: {e}") from e

    # --- Async Context Manager Support ---

    async def __aenter__(self: T_Ensemble) -> T_Ensemble:
        """Async context manager entry: Initialize the ensemble."""
        await self.initialize()
        return self

    async def __aexit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType]
    ) -> None:
        """Async context manager exit: Shutdown the ensemble."""
        await self.shutdown()
