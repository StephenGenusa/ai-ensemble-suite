
"""Structured critique debate pattern implementation."""

from typing import Dict, Any, Optional, List, Set
import time

from ai_ensemble_suite.collaboration.structured_debate.base_debate import BaseDebate
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class StructuredCritique(BaseDebate):
    """Structured Critique debate pattern.
    
    Models evaluate others' responses using structured formats with
    specific critique dimensions like accuracy, clarity, and completeness.
    """
    
    async def _execute_debate_pattern(
        self,
        query: str,
        initial_response: str,
        inputs: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the structured critique debate pattern.
        
        Args:
            query: The user query to process.
            initial_response: The initial response to critique.
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
                f"Critique phase '{self._phase_name}' missing prompt template"
            )
        
        # Format prompt
        try:
            context = {
                "query": query,
                "response": initial_response,
                **inputs
            }
            critique_prompt = self.render_template(self._prompt_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format critique prompt: {str(e)}")
            
        # Run critique models
        model_results = await self._run_models(
            prompt=critique_prompt,
            trace_collector=trace_collector
        )
        
        # Process outputs
        critiques = {}
        for model_id, result in model_results.items():
            critiques[model_id] = result.get("text", "")
            
        # Combine critiques if multiple critics
        if len(critiques) > 1:
            combined_critique = "# Combined Critiques\n\n"
            for model_id, critique in critiques.items():
                combined_critique += f"## Critique from {model_id}\n\n{critique}\n\n"
            critique_text = combined_critique
        elif critiques:
            # Just use the single critique
            critique_text = list(critiques.values())[0]
        else:
            critique_text = ""
            
        # Extract key points from the critique
        key_points = self._extract_key_points(critique_text)
        
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
            "output": critique_text,
            "critique": critique_text,
            "initial_response": initial_response,
            "key_points": key_points,
            "model_critiques": critiques,
            "raw_results": model_results,
            "confidence": confidence
        }
