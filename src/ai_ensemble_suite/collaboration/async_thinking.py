# ai_ensemble_suite/collaboration/async_thinking.py

"""Asynchronous Thinking collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set
import time
import asyncio

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class AsyncThinking(BaseCollaborationPhase):
    """Asynchronous Thinking collaboration phase.
    
    Models work independently on a problem before their outputs are collected.
    This is the simplest form of collaboration where multiple models process
    the same prompt concurrently.
    """
    
    async def execute(
        self, 
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Asynchronous Thinking phase.
        
        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.
            
        Returns:
            Dictionary containing:
                output: Dictionary mapping model IDs to their responses.
                context: Updated context for the next phase.
            
        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()
        
        try:
            # Get prompt template or use the query directly
            prompt = query
            if self._prompt_template:
                try:
                    # Format prompt with query and any inputs from previous phases
                    inputs = self._get_inputs_from_context(context)
                    context_vars = {"query": query, **inputs}
                    prompt = self.render_template(self._prompt_template, context_vars)
                except (ConfigurationError, KeyError) as e:
                    raise CollaborationError(f"Failed to format prompt: {str(e)}")
            
            # Run models concurrently
            model_results = await self._run_models(
                prompt=prompt,
                trace_collector=trace_collector
            )
            
            # Process outputs for simpler consumption by next phases
            processed_outputs: Dict[str, str] = {}
            
            for model_id, result in model_results.items():
                processed_outputs[model_id] = result.get("text", "")
            
            # Select the primary output for single model case
            primary_output = ""
            if len(processed_outputs) == 1:
                primary_output = list(processed_outputs.values())[0]
            elif self._model_ids and len(self._model_ids) > 0:
                # Use the first model in the model_ids list as primary if available
                primary_model = self._model_ids[0]
                if primary_model in processed_outputs:
                    primary_output = processed_outputs[primary_model]
            
            execution_time = time.time() - start_time
            
            # Log completion
            logger.info(
                f"AsyncThinking phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={
                    "model_count": len(model_results),
                    "phase": self._phase_name
                }
            )
            
            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "prompt": prompt},
                    output_data={"outputs": processed_outputs},
                    execution_time=execution_time,
                    phase_parameters=self._config
                )
            
            # Calculate a combined confidence score (average of all models)
            confidence_values = []
            for result in model_results.values():
                if "confidence" in result:
                    if isinstance(result["confidence"], dict) and "combined" in result["confidence"]:
                        confidence_values.append(result["confidence"]["combined"])
                    elif isinstance(result["confidence"], (float, int)):
                        confidence_values.append(result["confidence"])
            
            confidence_score = sum(confidence_values) / len(confidence_values) if confidence_values else 0.7
            
            # Return results
            return {
                "output": primary_output,  
                "outputs": processed_outputs,
                "confidence": confidence_score,
                "raw_results": model_results  # Include full model results for advanced use
            }
            
        except Exception as e:
            raise CollaborationError(
                f"AsyncThinking phase '{self._phase_name}' failed: {str(e)}"
            )
