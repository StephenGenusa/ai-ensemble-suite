"""Uncertainty-Based Collaboration phase implementation."""

from typing import Dict, Any, List, Optional, Set
import time
import numpy as np
from collections import Counter

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class UncertaintyBasedCollaboration(BaseCollaborationPhase):
    """Uncertainty-Based Collaboration phase.

    This collaborative phase uses uncertainty measurements to guide model interactions.
    Initial model outputs are analyzed for uncertainty, then the collaboration adapts
    by using more confident models to refine uncertain outputs or by selecting the
    most reliable outputs from multiple models.
    """

    def __init__(
            self,
            model_manager: "ModelManager",
            config_manager: "ConfigManager",
            phase_name: str
    ) -> None:
        """Initialize the Uncertainty-Based Collaboration phase."""
        super().__init__(model_manager, config_manager, phase_name)

        # Load uncertainty-specific configuration
        self._uncertainty_metric = self._config.get("uncertainty_metric", "disagreement")
        self._selection_method = self._config.get("selection_method", "least_uncertain")
        self._confidence_threshold = self._config.get("confidence_threshold", 0.6)
        self._fallback_model = self._config.get("fallback_model", None)
        self._refinement_iterations = self._config.get("refinement_iterations", 1)
        self._adaptive_selection = self._config.get("adaptive_selection", True)

        logger.debug(
            f"Configured UncertaintyBasedCollaboration phase with metric: {self._uncertainty_metric}",
            extra={
                "selection_method": self._selection_method,
                "confidence_threshold": self._confidence_threshold,
                "refinement_iterations": self._refinement_iterations
            }
        )

    def _calculate_uncertainty_metrics(self, outputs: List[str]) -> Dict[str, float]:
        """Calculate uncertainty metrics between multiple outputs."""
        metrics = {
            "disagreement": 0.0,
            "entropy": 0.0,
            "variance": 0.0,
            "confidence": 1.0  # Start with high confidence
        }

        if not outputs or len(outputs) < 2:
            # Can't calculate uncertainty with fewer than 2 outputs
            return metrics

        # Calculate word-level disagreement
        word_sets = []
        for output in outputs:
            # Create sets of significant words for comparison
            words = set(output.lower().split())
            word_sets.append(words)

        # Calculate average Jaccard distance between all pairs
        jaccard_distances = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                # Jaccard distance = 1 - (intersection / union)
                intersection = len(word_sets[i].intersection(word_sets[j]))
                union = len(word_sets[i].union(word_sets[j]))
                if union == 0:
                    distance = 0
                else:
                    distance = 1.0 - (intersection / union)
                jaccard_distances.append(distance)

        if jaccard_distances:
            metrics["disagreement"] = sum(jaccard_distances) / len(jaccard_distances)

        # Calculate word distribution entropy
        all_words = []
        for output in outputs:
            all_words.extend(output.lower().split())

        word_counts = Counter(all_words)
        total_words = len(all_words)

        if total_words > 0:
            # Calculate normalized entropy
            probs = [count / total_words for count in word_counts.values()]
            try:
                entropy = -sum(p * np.log2(p) for p in probs)
                # Normalize by maximum possible entropy
                max_entropy = np.log2(len(word_counts)) if word_counts else 0
                if max_entropy > 0:
                    metrics["entropy"] = min(1.0, entropy / max_entropy)
            except (ValueError, TypeError):
                # Handle numerical issues
                metrics["entropy"] = 0.5

        # Calculate output length variance (normalized)
        if len(outputs) > 1:
            lengths = [len(output) for output in outputs]
            mean_len = sum(lengths) / len(lengths)
            if mean_len > 0:
                variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
                # Normalize variance by mean squared to get scale-invariant measure
                metrics["variance"] = min(1.0, variance / (mean_len ** 2)) if mean_len > 0 else 0

        # Calculate overall confidence score (inverse of uncertainty)
        # Weight the metrics based on importance
        weights = {
            "disagreement": 0.6,  # Disagreement has highest weight
            "entropy": 0.3,  # Entropy next
            "variance": 0.1  # Length variance least important
        }

        uncertainty_score = sum(metrics[metric] * weight for metric, weight in weights.items())
        metrics["confidence"] = 1.0 - min(1.0, uncertainty_score)

        return metrics

    def _calculate_output_disagreement(self, target_output: str, other_outputs: List[str]) -> float:
        """Calculate how much a specific output disagrees with others."""
        if not other_outputs:
            return 0.0

        target_words = set(target_output.lower().split())
        all_disagreements = []

        for other in other_outputs:
            other_words = set(other.lower().split())
            union = len(target_words.union(other_words))

            if union == 0:  # Both empty
                all_disagreements.append(0.0)
            else:
                intersection = len(target_words.intersection(other_words))
                disagreement = 1.0 - (intersection / union)  # Jaccard distance
                all_disagreements.append(disagreement)

        # Return average disagreement with all other outputs
        return sum(all_disagreements) / len(all_disagreements) if all_disagreements else 0.0

    def _get_output_text(self, output_value: Any) -> str:
        """Safely extract text from an output value, handling different formats."""
        if isinstance(output_value, dict) and "text" in output_value:
            return output_value["text"]
        elif isinstance(output_value, str):
            return output_value
        else:
            logger.warning(f"Unexpected output format: {type(output_value)}")
            return str(output_value)

    def _normalize_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize outputs to a consistent format with 'text' key."""
        normalized = {}
        for key, value in outputs.items():
            if isinstance(value, dict) and "text" in value:
                normalized[key] = value
            elif isinstance(value, str):
                normalized[key] = {"text": value, "confidence": 0.5}
            else:
                logger.warning(f"Unexpected output format for key {key}: {type(value)}")
                normalized[key] = {"text": str(value), "confidence": 0.3}
        return normalized

    def _select_model_subset(
            self,
            model_outputs: Dict[str, Any],
            uncertainty_metrics: Dict[str, float]
    ) -> List[str]:
        """Select a subset of models for refinement based on uncertainty metrics."""
        normalized_outputs = self._normalize_outputs(model_outputs)

        if len(normalized_outputs) <= 1:
            return list(normalized_outputs.keys())

        # Calculate per-model disagreement
        model_scores = {}
        for model_id, output_data in normalized_outputs.items():
            output_text = output_data["text"]
            other_outputs = [
                normalized_outputs[m_id]["text"] for m_id in normalized_outputs
                if m_id != model_id
            ]
            disagreement = self._calculate_output_disagreement(output_text, other_outputs)
            agreement = 1.0 - disagreement
            model_scores[model_id] = agreement

        # If overall confidence is high, use all models
        if uncertainty_metrics["confidence"] >= self._confidence_threshold:
            return list(normalized_outputs.keys())

        # Otherwise, select top models based on agreement
        sorted_models = sorted(model_scores.keys(), key=lambda m: model_scores[m], reverse=True)

        # Take top half of models (at least 1)
        num_to_select = max(1, len(sorted_models) // 2)
        return sorted_models[:num_to_select]

    def _create_refinement_prompt(
            self,
            base_prompt: str,
            model_outputs: Dict[str, Any],
            uncertainty_metrics: Dict[str, float]
    ) -> str:
        """Create a prompt for the refinement models, incorporating uncertainty information."""
        normalized_outputs = self._normalize_outputs(model_outputs)

        refinement_prompt = (
            f"The following question needs a refined answer:\n\n"
            f"{base_prompt}\n\n"
            f"Initial responses have been generated with a confidence of {uncertainty_metrics['confidence']:.2f}.\n\n"
        )

        # Add information about areas of disagreement
        if uncertainty_metrics["disagreement"] > 0.3:
            refinement_prompt += "There is significant disagreement between initial responses. "

        if uncertainty_metrics["entropy"] > 0.5:
            refinement_prompt += "The responses show high variability in their content. "

        refinement_prompt += "Here are the previous responses:\n\n"

        # Add model outputs
        for i, (model_id, output_data) in enumerate(normalized_outputs.items()):
            output_text = output_data["text"]
            # Limit text length to avoid excessively long prompts
            if len(output_text) > 500:
                output_text = output_text[:497] + "..."
            refinement_prompt += f"Response {i+1}:\n{output_text}\n\n"

        refinement_prompt += (
            "Please provide a refined, accurate response that addresses any inconsistencies "
            "or uncertainties in the previous responses. Focus on generating a comprehensive "
            "and confident answer."
        )

        return refinement_prompt

    def _select_best_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best output from a set of outputs based on confidence or other metrics.

        Args:
            outputs: Dictionary of outputs to select from

        Returns:
            Dictionary containing the selected output information including model_id and text
        """
        if not outputs:
            return {"model_id": None, "text": "No outputs available."}

        normalized = self._normalize_outputs(outputs)

        # If only one output, return it
        if len(normalized) == 1:
            model_id = next(iter(normalized))
            return {
                "model_id": model_id,
                "text": normalized[model_id]["text"],
                "confidence": normalized[model_id].get("confidence", 0.5)
            }

        # Otherwise, select based on confidence
        best_model_id = None
        best_confidence = -1

        for model_id, output_data in normalized.items():
            confidence = output_data.get("confidence", 0.5)
            if confidence > best_confidence:
                best_confidence = confidence
                best_model_id = model_id

        # Return best output, or first one if all confidences are equal
        if best_model_id is None:
            best_model_id = next(iter(normalized))

        return {
            "model_id": best_model_id,
            "text": normalized[best_model_id]["text"],
            "confidence": normalized[best_model_id].get("confidence", 0.5)
        }

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Uncertainty-Based Collaboration phase."""
        start_time = time.time()

        try:
            # Prepare the base prompt
            base_prompt = query
            if self._prompt_template:
                try:
                    inputs = self._get_inputs_from_context(context)
                    context_dict = {"query": query, **inputs}
                    # import json
                    # print("DEBUG CONTEXT STRUCTURE:", json.dumps(context_dict, default=str, indent=2))
                    base_prompt = self.render_template(self._prompt_template, context_dict)
                except (ConfigurationError, KeyError) as e:
                    raise CollaborationError(f"Failed to format prompt: {str(e)}")

            # Check if we have inputs from previous phases to use
            initial_outputs = {}
            if 'input_from' in self._config and context:
                for phase_name in self._config['input_from']:
                    if phase_name in context:
                        phase_data = context[phase_name]
                        if 'outputs' in phase_data:
                            # Use outputs from previous phase as initial outputs
                            initial_outputs = phase_data['outputs']
                            logger.debug(f"Using {len(initial_outputs)} outputs from phase '{phase_name}'")
                        elif 'output' in phase_data:
                            # If only a single output is available
                            initial_outputs = {'previous_phase': phase_data['output']}
                            logger.debug(f"Using single output from phase '{phase_name}'")

            # If no previous outputs, get initial outputs from all models
            if not initial_outputs:
                initial_results = await self._run_models(
                    prompt=base_prompt,
                    model_ids=self._model_ids,
                    trace_collector=trace_collector
                )

                # Extract text outputs
                for model_id, result in initial_results.items():
                    if "text" in result:
                        initial_outputs[model_id] = {
                            "text": result["text"],
                            "confidence": result.get("confidence", 0.5),
                            "generation_time": result.get("generation_time", 0)
                        }

            # If still no outputs, raise an error
            if not initial_outputs:
                raise CollaborationError("No outputs available from models or previous phases")

            # Calculate uncertainty metrics for initial outputs - handle both dict and string outputs
            all_outputs = []
            for key, value in initial_outputs.items():
                all_outputs.append(self._get_output_text(value))

            uncertainty_metrics = self._calculate_uncertainty_metrics(all_outputs)

            # Store iterations for tracing
            iterations = [
                {
                    "iteration": 0,
                    "outputs": initial_outputs,
                    "uncertainty_metrics": uncertainty_metrics
                }
            ]

            # Normalize outputs for consistent processing
            current_outputs = self._normalize_outputs(initial_outputs)
            current_metrics = uncertainty_metrics

            # Determine if refinement is needed based on confidence
            needs_refinement = current_metrics["confidence"] < self._confidence_threshold
            refinement_count = 0
            refinement_success = False

            # Perform iterative refinement if needed
            while (needs_refinement and
                   refinement_count < self._refinement_iterations):

                refinement_count += 1
                logger.info(
                    f"Starting refinement iteration {refinement_count} due to low confidence "
                    f"({current_metrics['confidence']:.2f})",
                    extra={"threshold": self._confidence_threshold}
                )

                try:
                    # Select models for refinement
                    if self._adaptive_selection:
                        # Get model IDs that correspond to actual loaded models
                        available_models = []
                        for model_id in self._model_ids:
                            if model_id in current_outputs or model_id in self._model_manager.get_loaded_models():
                                available_models.append(model_id)

                        if not available_models:
                            logger.warning("No models available for refinement, using all configured models")
                            available_models = self._model_ids

                        refinement_models = self._select_model_subset(current_outputs, current_metrics)
                        # Ensure selected models are available
                        refinement_models = [m for m in refinement_models if m in available_models]
                    else:
                        # Use all models
                        refinement_models = self._model_ids

                    # Skip refinement if no models available
                    if not refinement_models:
                        logger.warning("No models available for refinement, skipping")
                        break

                    # Create refinement prompt with uncertainty information
                    refinement_prompt = self._create_refinement_prompt(
                        base_prompt, current_outputs, current_metrics
                    )

                    # Run selected models with refinement prompt
                    refinement_results = await self._run_models(
                        prompt=refinement_prompt,
                        model_ids=refinement_models,
                        trace_collector=trace_collector
                    )

                    # Extract refined outputs - check if we got any results
                    refined_outputs = {}
                    for model_id, result in refinement_results.items():
                        if "text" in result:
                            refined_outputs[model_id] = {
                                "text": result["text"],
                                "confidence": result.get("confidence", 0.5),
                                "generation_time": result.get("generation_time", 0),
                                "is_refinement": True
                            }

                    # If we got no refined outputs, stop refinement
                    if not refined_outputs:
                        logger.warning("No refined outputs produced in this iteration, stopping refinement")
                        break

                    # Calculate uncertainty metrics for refined outputs
                    refined_texts = [output["text"] for output in refined_outputs.values()]
                    refined_metrics = self._calculate_uncertainty_metrics(refined_texts)

                    # Store this iteration
                    iterations.append({
                        "iteration": refinement_count,
                        "outputs": refined_outputs,
                        "uncertainty_metrics": refined_metrics,
                        "refinement_models": refinement_models
                    })

                    # Update current state
                    current_outputs = refined_outputs
                    current_metrics = refined_metrics
                    refinement_success = True

                    # Check if further refinement is needed
                    needs_refinement = (
                        current_metrics["confidence"] < self._confidence_threshold and
                        refinement_count < self._refinement_iterations
                    )

                except Exception as e:
                    logger.error(f"Error during refinement iteration {refinement_count}: {e}")
                    # Continue with the next iteration if available, or exit the loop
                    if refinement_count >= self._refinement_iterations:
                        break

            # Select final output based on uncertainty metrics across all iterations
            best_iteration = 0
            best_confidence = iterations[0]["uncertainty_metrics"]["confidence"]

            for i, iteration_data in enumerate(iterations):
                confidence = iteration_data["uncertainty_metrics"]["confidence"]
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_iteration = i

            # Get outputs from best iteration
            best_iteration_data = iterations[best_iteration]
            best_outputs = best_iteration_data["outputs"]

            # Select best output
            best_output_info = self._select_best_output(best_outputs)

            # Get final output information
            best_model_id = best_output_info["model_id"]
            best_output = best_output_info["text"]
            best_metrics = best_iteration_data["uncertainty_metrics"]

            execution_time = time.time() - start_time

            # Log completion
            logger.info(
                f"Uncertainty-based collaboration phase completed in {execution_time:.2f}s "
                f"with {refinement_count} refinement iterations",
                extra={
                    "confidence": best_metrics["confidence"],
                    "best_iteration": best_iteration,
                    "best_model": best_model_id,
                    "refinement_success": refinement_success
                }
            )

            # Prepare result with all outputs and metrics
            final_result = {
                "output": best_output,
                "confidence": best_metrics["confidence"],
                "best_model": best_model_id,
                "best_iteration": best_iteration,
                "refinement_count": refinement_count,
                "refinement_success": refinement_success,
                "uncertainty_metrics": best_metrics,
                "iterations": iterations
            }

            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "base_prompt": base_prompt},
                    output_data=final_result,
                    execution_time=execution_time,
                    phase_parameters={
                        "uncertainty_metric": self._uncertainty_metric,
                        "selection_method": self._selection_method,
                        "confidence_threshold": self._confidence_threshold,
                        "refinement_iterations": self._refinement_iterations
                    }
                )

            return final_result

        except Exception as e:
            logger.error(f"Error in uncertainty-based collaboration: {e}", exc_info=True)
            raise CollaborationError(f"Uncertainty-based collaboration failed: {str(e)}")
