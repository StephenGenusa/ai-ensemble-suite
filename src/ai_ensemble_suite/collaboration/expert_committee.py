# ai_ensemble_suite/collaboration/expert_committee.py

"""Expert Committee collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set, Tuple
import time
import re

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class ExpertCommittee(BaseCollaborationPhase):
    """Expert Committee collaboration phase.
    
    Final processing/structuring of model outputs before aggregation.
    Organizes information and evaluates the quality of responses.
    """
    
    def __init__(
        self, 
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the expert committee phase.
        
        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
            
        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)
        
        # Get committee-specific configuration
        self._committee_type = self._config.get("committee_type", "evaluative")
        self._evaluation_criteria = self._config.get("evaluation_criteria", [
            "accuracy", "completeness", "clarity", "reasoning"
        ])
        self._format_output = self._config.get("format_output", True)
        
        # Log configuration
        logger.debug(
            f"Initialized Expert Committee phase '{phase_name}' with "
            f"type '{self._committee_type}' and {len(self._evaluation_criteria)} criteria"
        )
    
    async def execute(
        self, 
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Expert Committee phase.
        
        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.
            
        Returns:
            Dictionary containing:
                output: The committee's final output.
                context: Updated context including evaluations and structured output.
            
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
                    f"Expert Committee phase '{self._phase_name}' requires inputs from "
                    f"previous phases: {', '.join(self._input_from)}"
                )
            
            # Find the content to evaluate based on inputs and committee type
            processing_results = await self._process_inputs(query, inputs, trace_collector)
            
            # If evaluative committee, perform evaluation
            if self._committee_type == "evaluative":
                evaluation_results = await self._evaluate_inputs(query, inputs, processing_results, trace_collector)
                processing_results.update(evaluation_results)
                
            # If formatting is enabled, format the final output
            if self._format_output:
                formatted_output = await self._format_final_output(
                    query, inputs, processing_results, trace_collector
                )
                processing_results["output"] = formatted_output
                
            execution_time = time.time() - start_time
            
            # Log completion
            logger.info(
                f"Expert Committee phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={
                    "committee_type": self._committee_type,
                    "phase": self._phase_name
                }
            )
            
            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "inputs": inputs},
                    output_data=processing_results,
                    execution_time=execution_time,
                    phase_parameters=self._config
                )
                
            return processing_results
            
        except Exception as e:
            raise CollaborationError(
                f"Expert Committee phase '{self._phase_name}' failed: {str(e)}"
            )
    
    async def _process_inputs(
        self,
        query: str,
        inputs: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Process and organize inputs from previous phases.
        
        Args:
            query: The user query.
            inputs: Inputs from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary containing processing results.
            
        Raises:
            CollaborationError: If processing fails.
        """
        # Get or create processing template
        template_name = self._prompt_template or "committee_processing"
        
        # Prepare inputs for processing
        organized_inputs = {}
        
        # Convert inputs to a format suitable for the template
        for phase_name, phase_output in inputs.items():
            if isinstance(phase_output, str):
                # Direct string output
                organized_inputs[phase_name] = phase_output
            elif isinstance(phase_output, dict):
                # Dictionary output - extract the most relevant field
                if "output" in phase_output:
                    organized_inputs[phase_name] = phase_output["output"]
                elif "text" in phase_output:
                    organized_inputs[phase_name] = phase_output["text"]
                elif "response" in phase_output:
                    organized_inputs[phase_name] = phase_output["response"]
                else:
                    # Try to find any string field
                    for key, value in phase_output.items():
                        if isinstance(value, str):
                            organized_inputs[phase_name] = value
                            break
            
            # If no suitable output found, log warning
            if phase_name not in organized_inputs:
                logger.warning(f"Could not extract output from phase '{phase_name}'")

        # Format prompt with organized inputs
        try:
            # Create a single combined input if there are multiple organized inputs
            if len(organized_inputs) > 1:
                combined_input = "# Inputs from Previous Phases\n\n"
                for phase_name, content in organized_inputs.items():
                    combined_input += f"## From {phase_name}\n\n{content}\n\n"

                context = {"query": query, "inputs": combined_input}

                # Also add individual inputs
                for phase_name, content in organized_inputs.items():
                    context[phase_name] = content
            else:
                # Single input case
                phase_name = next(iter(organized_inputs.keys()), "")
                content = next(iter(organized_inputs.values()), "")
                context = {"query": query, "input": content, phase_name: content}

            processing_prompt = self.render_template(template_name, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format processing prompt: {str(e)}")

        # Run processing models
        model_results = await self._run_models(
            prompt=processing_prompt,
            trace_collector=trace_collector
        )
        
        # Process outputs
        processed_outputs = {}
        for model_id, result in model_results.items():
            processed_outputs[model_id] = result.get("text", "")
            
        # Determine the primary processed output
        primary_output = ""
        if len(processed_outputs) == 1:
            # Single model case
            primary_output = list(processed_outputs.values())[0]
        elif self._model_ids and len(self._model_ids) > 0:
            # Use the first model as primary
            primary_model = self._model_ids[0]
            if primary_model in processed_outputs:
                primary_output = processed_outputs[primary_model]
            else:
                # Fallback to first result
                primary_output = next(iter(processed_outputs.values()), "")
                
        # Calculate confidence score (average from all models)
        confidence_values = []
        for result in model_results.values():
            if "confidence" in result:
                if isinstance(result["confidence"], dict) and "combined" in result["confidence"]:
                    confidence_values.append(result["confidence"]["combined"])
                elif isinstance(result["confidence"], (float, int)):
                    confidence_values.append(result["confidence"])
        
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.7
        
        # Return processing results
        return {
            "output": primary_output,
            "processed_outputs": processed_outputs,
            "raw_results": model_results,
            "organized_inputs": organized_inputs,
            "confidence": confidence
        }
    
    async def _evaluate_inputs(
        self,
        query: str,
        inputs: Dict[str, Any],
        processing_results: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Evaluate inputs based on specified criteria.
        
        Args:
            query: The user query.
            inputs: Inputs from previous phases.
            processing_results: Results from processing step.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary containing evaluation results.
            
        Raises:
            CollaborationError: If evaluation fails.
        """
        # Use a specific evaluation template if available
        eval_template_name = self._config.get("evaluation_template", "committee_evaluation")
        
        # Get content to evaluate (priority: processed output > inputs)
        content_to_evaluate = processing_results.get("output", "")
        if not content_to_evaluate:
            # Try to get from organized inputs
            organized_inputs = processing_results.get("organized_inputs", {})
            if organized_inputs:
                # Combine all inputs
                content_to_evaluate = "\n\n".join(organized_inputs.values())
                
        # If still no content, log warning and return empty results
        if not content_to_evaluate:
            logger.warning(f"No content to evaluate in committee phase '{self._phase_name}'")
            return {
                "evaluations": {},
                "evaluation_summary": "",
                "overall_score": 0.0
            }
            
        # Format evaluation prompt
        try:
            # Create criteria string
            criteria_str = ", ".join(self._evaluation_criteria)

            context = {
                "query": query,
                "content": content_to_evaluate,
                "criteria": criteria_str
            }
            eval_prompt = self.render_template(eval_template_name, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format evaluation prompt: {str(e)}")
            
        # Run evaluation models
        model_results = await self._run_models(
            prompt=eval_prompt,
            trace_collector=trace_collector
        )
        
        # Process outputs
        eval_outputs = {}
        for model_id, result in model_results.items():
            eval_outputs[model_id] = result.get("text", "")
            
        # Determine the primary evaluation output
        eval_summary = ""
        if len(eval_outputs) == 1:
            # Single model case
            eval_summary = list(eval_outputs.values())[0]
        elif self._model_ids and len(self._model_ids) > 0:
            # Use the first model as primary
            primary_model = self._model_ids[0]
            if primary_model in eval_outputs:
                eval_summary = eval_outputs[primary_model]
            else:
                # Fallback to first result
                eval_summary = next(iter(eval_outputs.values()), "")
                
        # Parse scores from evaluation text
        evaluations = {}
        overall_score = 0.0
        
        # Try to extract scores for each criterion
        for criterion in self._evaluation_criteria:
            score_pattern = rf'(?i){criterion}[^\d]*?(\d+(?:\.\d+)?)/10'
            match = re.search(score_pattern, eval_summary)
            if match:
                score = float(match.group(1)) / 10.0  # Normalize to 0-1 range
                evaluations[criterion] = score
                
        # Calculate overall score as average of individual scores
        if evaluations:
            overall_score = sum(evaluations.values()) / len(evaluations)
            
        # Return evaluation results
        return {
            "evaluations": evaluations,
            "evaluation_summary": eval_summary,
            "overall_score": overall_score,
            "evaluation_outputs": eval_outputs
        }
    
    async def _format_final_output(
        self,
        query: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> str:
        """Format the final output based on processing and evaluation.
        
        Args:
            query: The user query.
            inputs: Inputs from previous phases.
            results: Results from processing and evaluation.
            trace_collector: Optional trace collector.
            
        Returns:
            Formatted final output.
            
        Raises:
            CollaborationError: If formatting fails.
        """
        # Use a specific formatting template if available
        format_template_name = self._config.get("formatting_template", "committee_formatting")
        
        # Get processed output and evaluation summary
        processed_output = results.get("output", "")
        evaluation_summary = results.get("evaluation_summary", "")
        
        # If no processed output, return empty string
        if not processed_output:
            logger.warning(f"No processed output to format in committee phase '{self._phase_name}'")
            return ""
            
        # If no formatting template, just return the processed output
        if not self._config_manager.get_template(format_template_name):
            return processed_output
            
        # Format the output
        try:
            context = {
                "query": query,
                "content": processed_output,
                "evaluation": evaluation_summary
            }
            formatted_prompt = self.render_template(format_template_name, context)
        except (ConfigurationError, KeyError) as e:
            # Log warning and return unformatted output
            logger.warning(f"Failed to format output: {str(e)}")
            return processed_output
            
        # Run formatting models
        model_results = await self._run_models(
            prompt=formatted_prompt,
            trace_collector=trace_collector
        )
        
        # Get the first result
        if model_results:
            first_model_id = next(iter(model_results.keys()))
            formatted_output = model_results[first_model_id].get("text", "")
            return formatted_output
            
        # Fallback to unformatted output
        return processed_output
