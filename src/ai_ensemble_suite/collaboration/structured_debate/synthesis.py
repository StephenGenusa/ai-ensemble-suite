# ai_ensemble_suite/collaboration/structured_debate/synthesis.py

"""Synthesis-oriented debate pattern implementation."""

from typing import Dict, Any, Optional, List, Set
import time

from ai_ensemble_suite.collaboration.structured_debate.base_debate import BaseDebate
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class SynthesisOriented(BaseDebate):
    """Synthesis-Oriented debate pattern.
    
    Models focus on finding common ground and integrating perspectives
    rather than critique. This pattern is more collaborative than adversarial.
    """
    
    async def _execute_debate_pattern(
        self,
        query: str,
        initial_response: str,
        inputs: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the synthesis-oriented debate pattern.
        
        Args:
            query: The user query to process.
            initial_response: The initial response to synthesize with.
            inputs: Inputs from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary containing debate results.
            
        Raises:
            CollaborationError: If execution fails.
        """
        # Get prompt template
        if not self._prompt_template:
            raise CollaborationError(
                f"Synthesis phase '{self._phase_name}' missing prompt template"
            )
        
        # Format prompt
        try:
            context = {
                "query": query,
                "response": initial_response,
                **inputs
            }
            synthesis_prompt = self.render_template(self._prompt_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format synthesis prompt: {str(e)}")
            
        # Run synthesis models
        model_results = await self._run_models(
            prompt=synthesis_prompt,
            trace_collector=trace_collector
        )
        
        # Process outputs
        synthesis_outputs = {}
        for model_id, result in model_results.items():
            synthesis_outputs[model_id] = result.get("text", "")
            
        # Combine syntheses if multiple models
        if len(synthesis_outputs) > 1:
            # Determine the primary model (first in model_ids list)
            primary_model = self._model_ids[0] if self._model_ids else None
            
            # Use the primary model's synthesis if available
            if primary_model and primary_model in synthesis_outputs:
                synthesis_text = synthesis_outputs[primary_model]
            else:
                # Or combine all syntheses
                combined_synthesis = "# Integrated Perspectives\n\n"
                for model_id, synthesis in synthesis_outputs.items():
                    combined_synthesis += f"## Perspective from {model_id}\n\n{synthesis}\n\n"
                synthesis_text = combined_synthesis
        elif synthesis_outputs:
            # Just use the single synthesis
            synthesis_text = list(synthesis_outputs.values())[0]
        else:
            synthesis_text = ""
            
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
            "output": synthesis_text,
            "synthesis": synthesis_text,
            "initial_response": initial_response,
            "individual_syntheses": synthesis_outputs,
            "raw_results": model_results,
            "confidence": confidence
        }
