
"""Integration/Refinement collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set
import time

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class Integration(BaseCollaborationPhase):
    """Integration/Refinement collaboration phase.
    
    Models refine responses based on feedback and insights from previous phases,
    integrating multiple perspectives into a coherent response.
    """
    
    async def execute(
        self, 
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Integration phase.
        
        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.
            
        Returns:
            Dictionary containing:
                output: The integrated/refined response.
                context: Updated context for the next phase.
            
        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()
        
        try:
            # Get inputs from previous phases
            inputs = self._get_inputs_from_context(context)
            
            # Validate inputs
            if not inputs and self._input_from:
                raise CollaborationError(
                    f"Integration phase '{self._phase_name}' requires inputs from "
                    f"previous phases: {', '.join(self._input_from)}"
                )
            
            # Get prompt template or use default integration prompt
            if not self._prompt_template:
                logger.warning(
                    f"No prompt template specified for Integration phase '{self._phase_name}', "
                    "using default integration prompt"
                )
                self._prompt_template = "refinement"
            
            # Format prompt with query and inputs
            try:
                context = {"query": query, **inputs}
                integration_prompt = self.render_template(self._prompt_template, context)
            except (ConfigurationError, KeyError) as e:
                raise CollaborationError(f"Failed to format integration prompt: {str(e)}")
            
            # Run models
            model_results = await self._run_models(
                prompt=integration_prompt,
                trace_collector=trace_collector
            )
            
            # Process outputs
            integrated_outputs = {}
            for model_id, result in model_results.items():
                integrated_outputs[model_id] = result.get("text", "")
            
            # Determine the primary integrated output
            primary_output = ""
            if len(integrated_outputs) == 1:
                # Single model case
                primary_output = list(integrated_outputs.values())[0]
            elif self._model_ids and len(self._model_ids) > 0:
                # Use the first model as primary
                primary_model = self._model_ids[0]
                if primary_model in integrated_outputs:
                    primary_output = integrated_outputs[primary_model]
                else:
                    # Fallback to first result
                    primary_output = next(iter(integrated_outputs.values()), "")
            
            execution_time = time.time() - start_time
            
            # Log completion
            logger.info(
                f"Integration phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={
                    "model_count": len(model_results),
                    "phase": self._phase_name
                }
            )
            
            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "prompt": integration_prompt, "inputs": inputs},
                    output_data={"outputs": integrated_outputs, "primary_output": primary_output},
                    execution_time=execution_time,
                    phase_parameters=self._config
                )
            
            # Calculate confidence score (average from all models)
            confidence_values = []
            for result in model_results.values():
                if "confidence" in result:
                    if isinstance(result["confidence"], dict) and "combined" in result["confidence"]:
                        confidence_values.append(result["confidence"]["combined"])
                    elif isinstance(result["confidence"], (float, int)):
                        confidence_values.append(result["confidence"])
            
            confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.7
            
            # Return results
            return {
                "output": primary_output,
                "integrated_outputs": integrated_outputs,
                "raw_results": model_results,
                "confidence": confidence
            }
            
        except Exception as e:
            raise CollaborationError(
                f"Integration phase '{self._phase_name}' failed: {str(e)}"
            )
