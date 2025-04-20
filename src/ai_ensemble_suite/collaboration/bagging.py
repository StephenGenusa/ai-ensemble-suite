"""Bagging collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set
import time
import random
from collections import Counter

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class Bagging(BaseCollaborationPhase):
    """Bagging (Bootstrap Aggregating) collaboration phase.

    Models process different variations of the same input, then their outputs
    are aggregated to produce a more robust result. This reduces variance and
    increases stability in the ensemble's output.
    """

    def __init__(
        self,
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the Bagging collaboration phase.

        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
        """
        super().__init__(model_manager, config_manager, phase_name)

        # Load bagging-specific configuration
        self._sample_ratio = self._config.get("sample_ratio", 0.8)
        self._variation_strategy = self._config.get("variation_strategy", "token_sampling")
        self._aggregation_method = self._config.get("aggregation_method", "voting")
        self._num_variations = self._config.get("num_variations", len(self._model_ids))

        if self._num_variations < 1:
            raise ConfigurationError(f"Bagging phase '{phase_name}' requires at least 1 variation")

        logger.debug(
            f"Configured Bagging phase with {self._num_variations} variations using '{self._variation_strategy}' strategy",
            extra={"aggregation": self._aggregation_method, "sample_ratio": self._sample_ratio}
        )

    def _generate_variations(self, text: str, n: int) -> List[str]:
        """Generate input variations using the configured strategy.

        Args:
            text: The original input text.
            n: Number of variations to generate.

        Returns:
            List of text variations.
        """
        variations = []

        if self._variation_strategy == "token_sampling":
            # Sample tokens from the original text
            words = text.split()
            for _ in range(n):
                # Sample ~sample_ratio proportion of words
                sample_size = max(1, int(len(words) * self._sample_ratio))
                sampled_words = random.sample(words, sample_size)
                variations.append(" ".join(sampled_words))

        elif self._variation_strategy == "segment_focus":
            # Focus on different segments of the text
            if len(text) < 50:  # For short texts, use token sampling instead
                return self._generate_variations(text, n)

            segments = max(3, n)  # At least 3 segments
            segment_length = len(text) // segments

            for i in range(n):
                # Focus on a primary segment but include some context
                primary_segment = i % segments
                start = max(0, primary_segment * segment_length - segment_length//2)
                end = min(len(text), (primary_segment + 1) * segment_length + segment_length//2)
                variations.append(text[start:end])

        elif self._variation_strategy == "instruction_variation":
            # Create variations with different instructions/framing
            instruction_variations = [
                f"Please answer this question: {text}",
                f"Analyze the following: {text}",
                f"Consider this query: {text}",
                f"Respond to this input: {text}",
                f"Evaluate this statement: {text}"
            ]

            for i in range(n):
                variations.append(instruction_variations[i % len(instruction_variations)])

        else:  # Default or "none" - just use the original text repeated
            variations = [text] * n

        return variations

    def _aggregate_outputs(self, outputs: List[str]) -> str:
        """Aggregate multiple outputs into a single result.

        Args:
            outputs: List of model outputs to aggregate.

        Returns:
            Aggregated output string.
        """
        if self._aggregation_method == "voting":
            # Simple majority voting - works best for classification or short responses
            if all(len(output.strip()) < 100 for output in outputs):
                counter = Counter(outputs)
                return counter.most_common(1)[0][0]
            else:
                # For longer texts, voting might not make sense
                # Fall back to the first output
                logger.warning("Voting aggregation not suitable for long texts, using primary output instead")
                return outputs[0]

        elif self._aggregation_method == "concatenation":
            # Join all outputs with a delimiter
            return "\n\n===== NEXT RESPONSE =====\n\n".join(outputs)

        elif self._aggregation_method == "summarization":
            # This would require an additional model call to summarize
            # For now, we'll use a simple combination approach
            if len(outputs) <= 2:
                return "\n\n".join(outputs)

            # Take the first sentence from each output for a quick summary
            summary_parts = []
            for output in outputs:
                first_sentence = output.split(".")[0] + "." if "." in output else output
                summary_parts.append(first_sentence)

            return " ".join(summary_parts)

        else:  # Default to first output
            return outputs[0] if outputs else ""

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Bagging phase.

        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.

        Returns:
            Dictionary containing:
                output: The aggregated output from all variations.
                outputs: Dictionary mapping variation IDs to individual responses.
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
                    context_vars = {"query": query, **inputs}
                    base_prompt = self.render_template(self._prompt_template, context_vars)
                except (ConfigurationError, KeyError) as e:
                    raise CollaborationError(f"Failed to format prompt: {str(e)}")

            # Generate variations of the prompt
            variations = self._generate_variations(base_prompt, self._num_variations)

            # Execute each variation with a model
            all_outputs = {}
            raw_results = {}

            for i, variant_prompt in enumerate(variations):
                # Use modulo in case we have more variations than models
                model_idx = i % len(self._model_ids)
                model_id = self._model_ids[model_idx]

                # Run individual model on this variation
                result = await self._run_models(
                    prompt=variant_prompt,
                    model_ids=[model_id],
                    trace_collector=trace_collector
                )

                # Store the result
                variation_id = f"variation_{i+1}"
                all_outputs[variation_id] = result[model_id].get("text", "")
                raw_results[variation_id] = result

                # Add trace for this variation if collector is provided
                if trace_collector:
                    trace_collector.add_model_trace(
                        model_id=f"{model_id}_bagging_{i+1}",
                        input_prompt=variant_prompt,
                        output=result[model_id],
                        execution_time=result[model_id].get("generation_time", 0),
                        parameters={"variation_id": variation_id}
                    )

            # Aggregate the outputs into a single result
            aggregated_output = self._aggregate_outputs(list(all_outputs.values()))

            execution_time = time.time() - start_time

            # Calculate agreement score as a confidence measure
            if len(all_outputs) > 1:
                # Count matching words across outputs
                word_sets = [set(output.lower().split()) for output in all_outputs.values()]
                union_words = set().union(*word_sets)
                if not union_words:
                    agreement_score = 1.0
                else:
                    # Count how many outputs contain each word
                    word_counts = {}
                    for word in union_words:
                        word_counts[word] = sum(1 for words in word_sets if word in words)

                    # Average agreement across all words
                    avg_agreement = sum(word_counts.values()) / (len(word_counts) * len(all_outputs))
                    agreement_score = avg_agreement
            else:
                agreement_score = 1.0

            # Log completion
            logger.info(
                f"Bagging phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={
                    "variation_count": len(variations),
                    "phase": self._phase_name,
                    "agreement_score": agreement_score
                }
            )

            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "base_prompt": base_prompt},
                    output_data={"aggregated_output": aggregated_output, "variation_outputs": all_outputs},
                    execution_time=execution_time,
                    phase_parameters={
                        "variation_strategy": self._variation_strategy,
                        "sample_ratio": self._sample_ratio,
                        "aggregation_method": self._aggregation_method,
                        "num_variations": self._num_variations
                    }
                )

            # Return results
            return {
                "output": aggregated_output,
                "outputs": all_outputs,
                "confidence": agreement_score,
                "raw_results": raw_results,
                "variation_count": len(variations)
            }

        except Exception as e:
            raise CollaborationError(
                f"Bagging phase '{self._phase_name}' failed: {str(e)}"
            )
