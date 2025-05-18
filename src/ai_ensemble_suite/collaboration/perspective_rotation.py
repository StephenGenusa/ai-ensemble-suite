# ai_ensemble_suite/collaboration/perspective_rotation.py

"""Perspective Rotation collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set
import time
import asyncio

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class PerspectiveRotation(BaseCollaborationPhase):
    """Perspective Rotation collaboration phase.
    
    Models iterate on a problem by assuming different perspectives
    or stakeholder viewpoints in sequence.
    """
    
    def __init__(
        self, 
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the perspective rotation phase.
        
        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
            
        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)
        
        # Get perspectives to rotate through
        self._perspectives = self._config.get("perspectives", [])
        
        # If no perspectives defined, use default set
        if not self._perspectives:
            self._perspectives = [
                "technical", "ethical", "practical", "creative", "critical"
            ]
            logger.debug(
                f"No perspectives specified for PerspectiveRotation phase '{phase_name}', "
                f"using defaults: {self._perspectives}"
            )
            
        # Get template name
        self._perspective_template = self._config.get("perspective_template", "perspective")
        
        # Get synthesis template
        self._synthesis_template = self._config.get("synthesis_template", "synthesis")
        
        # Get synthesis model (default to first model)
        self._synthesis_model = self._config.get("synthesis_model")
        if not self._synthesis_model and self._model_ids:
            self._synthesis_model = self._model_ids[0]
            
        logger.debug(
            f"Initialized PerspectiveRotation phase '{phase_name}' with "
            f"{len(self._perspectives)} perspectives"
        )
    
    async def execute(
        self, 
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Perspective Rotation phase.
        
        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.
            
        Returns:
            Dictionary containing:
                output: The synthesized response.
                context: Updated context with perspectives and synthesis.
            
        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()
        
        try:
            # Get inputs from previous phases
            inputs = self._get_inputs_from_context(context)
            
            # Step 1: Generate responses from each perspective
            perspective_responses = await self._generate_perspective_responses(
                query, context, trace_collector
            )
            
            # Step 2: Synthesize the perspectives
            synthesis = await self._synthesize_perspectives(
                query, perspective_responses, context, trace_collector
            )
            
            execution_time = time.time() - start_time
            
            # Log completion
            logger.info(
                f"PerspectiveRotation phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={"perspectives": len(self._perspectives)}
            )
            
            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "context": context},
                    output_data={
                        "perspective_responses": perspective_responses,
                        "synthesis": synthesis
                    },
                    execution_time=execution_time,
                    phase_parameters=self._config
                )
                
            # Calculate confidence score
            confidence = synthesis.get("confidence", 0.7)
                
            # Return results
            return {
                "output": synthesis.get("text", ""),
                "perspective_responses": perspective_responses,
                "synthesis": synthesis,
                "perspectives": self._perspectives,
                "confidence": confidence
            }
            
        except Exception as e:
            raise CollaborationError(
                f"PerspectiveRotation phase '{self._phase_name}' failed: {str(e)}"
            )
    
    async def _generate_perspective_responses(
        self,
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, str]:
        """Generate responses from each perspective.
        
        Args:
            query: The user query.
            context: Context information from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary mapping perspective names to their responses.
            
        Raises:
            CollaborationError: If generation fails.
        """
        logger.debug(f"Generating responses from {len(self._perspectives)} perspectives")
        
        # Get inputs from previous phases
        inputs = self._get_inputs_from_context(context)
        
        # Distribute models across perspectives
        if len(self._model_ids) >= len(self._perspectives):
            # If we have enough models, assign one per perspective
            model_assignments = {
                perspective: self._model_ids[i % len(self._model_ids)]
                for i, perspective in enumerate(self._perspectives)
            }
        else:
            # Otherwise, use available models in rotation
            model_assignments = {
                perspective: self._model_ids[i % len(self._model_ids)]
                for i, perspective in enumerate(self._perspectives)
            }
            
        # Generate response for each perspective
        perspective_responses = {}
        
        for perspective in self._perspectives:
            # Get template (try perspective-specific first, then default)
            template_name = f"{perspective}_perspective"
            if not self._config_manager.get_template(template_name):
                template_name = self._perspective_template
                
            # Format prompt
            try:
                context = {
                    "query": query,
                    "perspective": perspective,
                    **inputs
                }
                perspective_prompt = self.render_template(template_name, context)
            except (ConfigurationError, KeyError) as e:
                logger.warning(f"Failed to format prompt for perspective '{perspective}': {str(e)}")
                continue
                
            # Get model for this perspective
            model_id = model_assignments.get(perspective)
            if not model_id:
                logger.warning(f"No model assigned for perspective '{perspective}'")
                continue
                
            # Run model
            try:
                model_result = await self._model_manager.run_inference(
                    model_id=model_id,
                    prompt=perspective_prompt
                )
                
                # Extract response
                perspective_responses[perspective] = model_result.get("text", "")
                
                # Add trace if collector is provided
                if trace_collector:
                    trace_collector.add_model_trace(
                        model_id=model_id,
                        input_prompt=perspective_prompt,
                        output=model_result,
                        execution_time=model_result.get("generation_time", 0),
                        parameters={"perspective": perspective}
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to generate response for perspective '{perspective}': {str(e)}")
                
        return perspective_responses
    
    async def _synthesize_perspectives(
        self,
        query: str,
        perspective_responses: Dict[str, str],
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Synthesize the perspectives into a final response.
        
        Args:
            query: The user query.
            perspective_responses: Responses from different perspectives.
            context: Context information from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary containing synthesis results.
            
        Raises:
            CollaborationError: If synthesis fails.
        """
        logger.debug("Synthesizing perspectives")
        
        # Check if synthesis model is available
        if not self._synthesis_model:
            raise CollaborationError("No synthesis model specified")
            
        # Format perspectives for synthesis
        formatted_perspectives = ""
        for perspective, response in perspective_responses.items():
            formatted_perspectives += f"\n\n## {perspective.capitalize()} Perspective\n\n{response}"
            
        # Format synthesis prompt
        try:
            context = {
                "query": query,
                "perspectives": formatted_perspectives
            }
            synthesis_prompt = self.render_template(self._synthesis_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format synthesis prompt: {str(e)}")
            
        # Run synthesis model
        try:
            synthesis_result = await self._model_manager.run_inference(
                model_id=self._synthesis_model,
                prompt=synthesis_prompt
            )
            
            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_model_trace(
                    model_id=self._synthesis_model,
                    input_prompt=synthesis_prompt,
                    output=synthesis_result,
                    execution_time=synthesis_result.get("generation_time", 0),
                    parameters={"role": "synthesizer"}
                )
                
            return synthesis_result
            
        except Exception as e:
            raise CollaborationError(f"Failed to synthesize perspectives: {str(e)}")
