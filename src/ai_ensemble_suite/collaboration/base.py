# src/ai_ensemble_suite/collaboration/base.py

"""Base class for collaboration phases."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Set
import time

from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector
from ai_ensemble_suite.utils.prompt_utils import format_prompt


class BaseCollaborationPhase(ABC):
    """Abstract base class for collaboration phases.
    
    Defines the interface for collaboration phases and provides common functionality.
    """

    def __init__(
            self,
            model_manager: "ModelManager",
            config_manager: "ConfigManager",
            phase_name: str
    ) -> None:
        """Initialize the collaboration phase.

        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.

        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        self._model_manager = model_manager
        self._config_manager = config_manager
        self._phase_name = phase_name

        # Get reference to template manager through model_manager.ensemble
        self._template_manager = None
        if hasattr(model_manager, 'ensemble') and model_manager.ensemble and hasattr(model_manager.ensemble,
                                                                                     'template_manager'):
            self._template_manager = model_manager.ensemble.template_manager

        # Load phase configuration (existing code)
        try:
            self._config = self._config_manager.get_collaboration_config(phase_name)
        except ConfigurationError as e:
            raise ConfigurationError(f"Failed to load configuration for phase '{phase_name}': {str(e)}")

        # Get phase type
        self._phase_type = self._config.get("type")
        if not self._phase_type:
            raise ConfigurationError(f"Phase '{phase_name}' is missing required field: type")
            
        # Get model IDs for this phase
        self._model_ids = self._config.get("models", [])
        
        # Get input sources (previous phases)
        self._input_from = self._config.get("input_from", [])
        if isinstance(self._input_from, str):
            self._input_from = [self._input_from]
            
        # Get prompt template name
        self._prompt_template = self._config.get("prompt_template")
        
        logger.debug(
            f"Initialized collaboration phase '{phase_name}' of type '{self._phase_type}'",
            extra={"models": self._model_ids}
        )
    
    @abstractmethod
    async def execute(
        self, 
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the collaboration phase.
        
        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.
            
        Returns:
            Dictionary containing:
                output: The phase output.
                context: Updated context for the next phase.
            
        Raises:
            CollaborationError: If phase execution fails.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the phase configuration.
        
        Returns:
            Dictionary containing the phase configuration.
        """
        return self._config.copy()
    
    def get_name(self) -> str:
        """Get the phase name.
        
        Returns:
            The phase name.
        """
        return self._phase_name
    
    def get_type(self) -> str:
        """Get the phase type.
        
        Returns:
            The phase type.
        """
        return self._phase_type
    
    def get_required_models(self) -> Set[str]:
        """Get the models required by this phase.
        
        Returns:
            Set of model IDs required by this phase.
        """
        return set(self._model_ids)
    
    def get_input_phases(self) -> List[str]:
        """Get the input phases this phase depends on.
        
        Returns:
            List of phase names this phase takes input from.
        """
        return list(self._input_from)

    def render_template(
            self,
            template_name: str,
            context: Dict[str, Any]
    ) -> str:
        """Render a template using Jinja2 templating.

        Args:
            template_name: The name of the template to render.
            context: Dictionary of context variables for rendering.

        Returns:
            The rendered template string.

        Raises:
            ConfigurationError: If template is not found.
            ValueError: If template rendering fails.
        """
        if self._template_manager:
            # Use the template manager if available
            return self._template_manager.render_template(template_name, context)

    # def format_prompt(
    #         self,
    #         template_name: str,
    #         **kwargs: Any
    # ) -> str:
    #     """Format a prompt template with the provided values using simple substitution.
    #
    #     Note:
    #         For more advanced templating features, use render_template() instead.
    #
    #     Args:
    #         template_name: The name of the template to format.
    #         **kwargs: Values to format the template with.
    #
    #     Returns:
    #         The formatted prompt.
    #
    #     Raises:
    #         ConfigurationError: If the template does not exist.
    #     """
    #     template = self._config_manager.get_template(template_name)
    #     if template is None:
    #         raise ConfigurationError(f"Template not found: {template_name}")
    #
    #     return format_prompt(template, **kwargs)

    async def _run_models(
        self, 
        prompt: str,
        model_ids: Optional[List[str]] = None,
        trace_collector: Optional[TraceCollector] = None,
        **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        """Run inference on specific models or all phase models.
        
        Args:
            prompt: The prompt to send to the models.
            model_ids: Optional list of model IDs to use. If None, uses all phase models.
            trace_collector: Optional trace collector for gathering execution details.
            **kwargs: Additional parameters for model inference.
            
        Returns:
            Dictionary mapping model IDs to their outputs.
            
        Raises:
            CollaborationError: If model inference fails.
        """
        ids_to_use = model_ids if model_ids is not None else self._model_ids
        if not ids_to_use:
            raise CollaborationError(f"No models specified for phase '{self._phase_name}'")
            
        try:
            start_time = time.time()
            results = await self._model_manager.run_all_models(prompt, ids_to_use, **kwargs)
            
            # Add traces if collector is provided
            if trace_collector:
                for model_id, result in results.items():
                    trace_collector.add_model_trace(
                        model_id=model_id,
                        input_prompt=prompt,
                        output=result,
                        execution_time=result.get("generation_time", 0),
                        parameters=kwargs
                    )
                    
            return results
            
        except Exception as e:
            raise CollaborationError(
                f"Failed to run models for phase '{self._phase_name}': {str(e)}"
            )
    
    def _get_inputs_from_context(
        self, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract inputs from context based on input_from configuration.
        
        Args:
            context: The context dictionary containing previous phase outputs.
            
        Returns:
            Dictionary containing inputs for this phase.
            
        Raises:
            CollaborationError: If a required input is missing.
        """
        inputs = {}
        
        # If no input_from is specified, return empty inputs
        if not self._input_from:
            return inputs
            
        # Extract inputs from context
        for phase_name in self._input_from:
            if phase_name not in context:
                raise CollaborationError(
                    f"Phase '{self._phase_name}' requires input from phase '{phase_name}', "
                    "but it is not available in the context"
                )
                
            inputs[phase_name] = context[phase_name]
            
        return inputs
