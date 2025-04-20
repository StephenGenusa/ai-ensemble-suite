"""Stacked Generalization collaboration phase implementation."""

from typing import Dict, Any, Optional, List
import time

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class StackedGeneralization(BaseCollaborationPhase):
    """Stacked Generalization collaboration phase.

    Base models first process the input, then a meta-model combines their outputs
    to produce a more accurate result. This learns the strengths and weaknesses
    of each base model to produce optimal combinations.
    """

    def __init__(
            self,
            model_manager: "ModelManager",
            config_manager: "ConfigManager",
            phase_name: str
    ) -> None:
        """Initialize the Stacked Generalization collaboration phase.

        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
        """
        super().__init__(model_manager, config_manager, phase_name)

        # Load stacking-specific configuration
        self._base_models = self._config.get("base_models", [])
        self._meta_model = self._config.get("meta_model", "")
        self._combination_strategy = self._config.get("combination_strategy", "weighted")
        self._max_rounds = self._config.get("max_rounds", 1)
        self._use_feedback_loop = self._config.get("use_feedback_loop", False)

        # Validate configuration
        if not self._base_models:
            raise ConfigurationError(f"Stacked Generalization phase '{phase_name}' requires at least one base model")

        if not self._meta_model:
            raise ConfigurationError(f"Stacked Generalization phase '{phase_name}' requires a meta model")

        # Ensure meta model is in the list of model_ids
        if self._meta_model not in self._model_ids:
            raise ConfigurationError(f"Meta model '{self._meta_model}' not found in available models")

        # Ensure all base models are in the list of model_ids
        for model in self._base_models:
            if model not in self._model_ids:
                raise ConfigurationError(f"Base model '{model}' not found in available models")

        logger.debug(
            f"Configured Stacked Generalization phase with {len(self._base_models)} base models and '{self._meta_model}' as the meta model",
            extra={
                "combination_strategy": self._combination_strategy,
                "max_rounds": self._max_rounds,
                "use_feedback_loop": self._use_feedback_loop
            }
        )

    async def _run_base_models(
            self,
            prompt: str,
            trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run all base models on the prompt.

        Args:
            prompt: The input prompt.
            trace_collector: Optional trace collector.

        Returns:
            Dictionary mapping model IDs to their outputs.
        """
        # Run each base model on the prompt
        result = await self._run_models(
            prompt=prompt,
            model_ids=self._base_models,
            trace_collector=trace_collector
        )

        # Add trace if collector is provided
        if trace_collector:
            for model_id in self._base_models:
                trace_collector.add_model_trace(
                    model_id=f"{model_id}_base",
                    input_prompt=prompt,
                    output=result[model_id],
                    execution_time=result[model_id].get("generation_time", 0),
                    parameters={"role": "base_model"}
                )

        return result

    async def _run_meta_model(
            self,
            original_prompt: str,
            base_outputs: Dict[str, Dict[str, Any]],
            round_num: int = 1,
            trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Run the meta model to combine base model outputs.

        Args:
            original_prompt: The original input prompt.
            base_outputs: Dictionary mapping model IDs to their outputs.
            round_num: Current round number for multi-round stacking.
            trace_collector: Optional trace collector.

        Returns:
            Meta model's output.
        """
        # Extract text outputs from each model
        base_text_outputs = {model_id: output.get("text", "")
                             for model_id, output in base_outputs.items()}

        # Format the meta-prompt that includes base model outputs
        meta_prompt_template = self._config.get("meta_prompt_template", "")
        if not meta_prompt_template:
            # Default meta prompt if none is specified
            meta_prompt = f"""As a meta-model, your task is to combine and synthesize the outputs from multiple AI models 
into a single, coherent, and accurate response. Here is the original query and the outputs from each model:

ORIGINAL QUERY:
{original_prompt}

MODEL OUTPUTS:
"""
            for model_id, output in base_text_outputs.items():
                meta_prompt += f"\n--- {model_id} ---\n{output}\n"

            meta_prompt += """
Please analyze these outputs and create a comprehensive response that:
1. Integrates the strengths of each model's response
2. Resolves any contradictions between models
3. Provides a coherent, well-structured answer to the original query
4. Improves upon any weaknesses or gaps in the individual responses

Your synthesized response:"""
        else:
            # Format custom meta prompt template
            context = {
                "original_prompt": original_prompt,
                "base_outputs": base_text_outputs,
                "round": round_num,
            }
            meta_prompt = self.render_template(meta_prompt_template, context)

        # Run the meta model
        result = await self._run_models(
            prompt=meta_prompt,
            model_ids=[self._meta_model],
            trace_collector=trace_collector
        )

        # Add trace if collector is provided
        if trace_collector:
            trace_collector.add_model_trace(
                model_id=f"{self._meta_model}_meta_round{round_num}",
                input_prompt=meta_prompt,
                output=result[self._meta_model],
                execution_time=result[self._meta_model].get("generation_time", 0),
                parameters={"role": "meta_model", "round": round_num}
            )

        return result[self._meta_model]

    async def execute(
            self,
            query: str,
            context: Dict[str, Any],
            trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Stacked Generalization phase.

        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.

        Returns:
            Dictionary containing:
                output: The final meta-model output.
                base_outputs: Dictionary mapping base model IDs to their outputs.
                meta_output: The meta model's output.
                rounds: Number of rounds performed.
                context: Updated context for the next phase.

        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()

        try:
            # Prepare the base prompt
            base_prompt = query
            if self._prompt_template:
                try:
                    inputs = self._get_inputs_from_context(context)
                    context = {"query": query, **inputs}
                    base_prompt = self.render_template(self._prompt_template, context)
                except (ConfigurationError, KeyError) as e:
                    raise CollaborationError(f"Failed to format prompt: {str(e)}")

            # Run the base models
            base_results = await self._run_base_models(base_prompt, trace_collector)

            # Extract text outputs for easier processing
            base_text_outputs = {model_id: result.get("text", "")
                                 for model_id, result in base_results.items()}

            # Initialize meta output and round counter
            meta_result = None
            meta_output = ""
            rounds = 0
            all_round_outputs = {}
            all_round_results = {}

            # Iterative refinement through multiple rounds if configured
            current_base_results = base_results
            for round_num in range(1, self._max_rounds + 1):
                rounds = round_num

                # Run the meta model
                meta_result = await self._run_meta_model(
                    original_prompt=base_prompt,
                    base_outputs=current_base_results,
                    round_num=round_num,
                    trace_collector=trace_collector
                )

                meta_output = meta_result.get("text", "")
                all_round_outputs[f"round_{round_num}"] = meta_output
                all_round_results[f"round_{round_num}"] = meta_result

                # If feedback loop is enabled and we're not at the last round,
                # use the meta output to inform the next round
                if self._use_feedback_loop and round_num < self._max_rounds:
                    # For feedback loop, we provide the meta model's output back to base models
                    feedback_prompt = f"{base_prompt}\n\nPrevious synthesis: {meta_output}"

                    # Re-run base models with feedback from meta model
                    current_base_results = await self._run_base_models(feedback_prompt, trace_collector)
                else:
                    # We're done with all rounds
                    break

            execution_time = time.time() - start_time

            # Calculate confidence metrics
            confidence = 0.0

            # Confidence could be based on agreement between base models
            if len(base_text_outputs) > 1:
                # Simple agreement metric - count words that appear in multiple outputs
                word_sets = [set(output.lower().split()) for output in base_text_outputs.values()]
                union_words = set().union(*word_sets)
                if union_words:
                    # Count how many outputs contain each word
                    word_counts = {}
                    for word in union_words:
                        word_counts[word] = sum(1 for words in word_sets if word in words)

                    # Average agreement across all words
                    avg_agreement = sum(word_counts.values()) / (len(word_counts) * len(base_text_outputs))
                    confidence = avg_agreement
            else:
                confidence = 1.0

            # Log completion
            logger.info(
                f"Stacked Generalization phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={
                    "base_model_count": len(self._base_models),
                    "rounds": rounds,
                    "phase": self._phase_name,
                    "confidence": confidence
                }
            )

            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "base_prompt": base_prompt},
                    output_data={
                        "base_outputs": base_text_outputs,
                        "meta_output": meta_output,
                        "all_round_outputs": all_round_outputs
                    },
                    execution_time=execution_time,
                    phase_parameters={
                        "base_models": self._base_models,
                        "meta_model": self._meta_model,
                        "combination_strategy": self._combination_strategy,
                        "max_rounds": self._max_rounds,
                        "rounds_completed": rounds,
                        "use_feedback_loop": self._use_feedback_loop
                    }
                )

            # Return results
            return {
                "output": meta_output,  # The final output
                "base_outputs": base_text_outputs,  # All base model outputs
                "base_results": base_results,  # Raw base model results
                "meta_output": meta_output,  # Meta model output
                "meta_result": meta_result,  # Raw meta model result
                "all_round_outputs": all_round_outputs,  # Outputs from each round
                "all_round_results": all_round_results,  # Raw results from each round
                "rounds": rounds,  # Number of rounds performed
                "confidence": confidence,  # Agreement-based confidence
                "execution_time": execution_time  # Execution time
            }

        except Exception as e:
            raise CollaborationError(
                f"Stacked Generalization phase '{self._phase_name}' failed: {str(e)}"
            )
