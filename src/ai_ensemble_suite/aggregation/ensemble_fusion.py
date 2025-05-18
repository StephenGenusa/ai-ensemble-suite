# ai_ensemble_suite/aggregation/ensemble_fusion.py

"""Ensemble Fusion aggregation strategy."""

from typing import Dict, Any, Optional, List, Set, Tuple, TYPE_CHECKING
import time
import re
import math

from ai_ensemble_suite.aggregation.base import BaseAggregator
from ai_ensemble_suite.exceptions import AggregationError, ConfigurationError, ModelError, ValidationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector
import copy

# Type hint imports
if TYPE_CHECKING:
    from ai_ensemble_suite.config import ConfigManager
    from ai_ensemble_suite.models import ModelManager


class EnsembleFusion(BaseAggregator):
    """Ensemble Fusion aggregation strategy.

    Uses a designated 'fusion' model to synthesize multiple phase outputs into
    a single, coherent response, aiming to integrate the best aspects of each input.
    """

    def __init__(
        self,
        config_manager: "ConfigManager",
        strategy_name: str,
        model_manager: Optional["ModelManager"] = None,
        strategy_config_override: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the EnsembleFusion aggregator."""
        super().__init__(config_manager, strategy_name, model_manager, strategy_config_override)
        if self._model_manager is None:
             logger.warning(f"EnsembleFusion strategy '{self._strategy_name}' initialized without a ModelManager. Aggregation will likely fail.")

    async def aggregate(
        self,
        outputs: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Aggregate phase outputs by using a specified model to fuse them."""
        start_time = time.time()
        logger.debug(f"Starting Ensemble Fusion aggregation using strategy '{self._strategy_name}'")

        if self._model_manager is None:
            raise AggregationError("ModelManager is required for Ensemble Fusion but not available.")

        if not outputs:
             raise AggregationError("No phase outputs provided for Ensemble Fusion.")

        try:
            # Determine the Fusion Model
            fusion_model_id = self._config.get("fusion_model") # Key checked by schema
            fusion_model_id = context.get("fusion_model", fusion_model_id) # Allow context override

            if not fusion_model_id:
                 # Schema validation should require fusion_model, so this is unlikely
                 logger.error("Configuration Error: 'fusion_model' is required but missing.")
                 raise AggregationError("Missing required 'fusion_model' configuration for Ensemble Fusion.")

            # Ensure the selected fusion model actually exists
            try:
                 fusion_model_instance = self._model_manager.get_model(fusion_model_id)
                 logger.info(f"Using fusion model: {fusion_model_id}")
            except ModelError:
                 logger.error(f"Specified fusion model '{fusion_model_id}' not found in ModelManager.")
                 raise AggregationError(f"Fusion model '{fusion_model_id}' not available.")


            # Prepare Inputs for Fusion
            outputs_to_fuse: Dict[str, str] = {}
            for phase_name, phase_output_data in outputs.items():
                extracted_text = self._extract_output(phase_output_data)
                if extracted_text:
                    outputs_to_fuse[phase_name] = extracted_text
                else:
                    logger.warning(f"Could not extract usable text from phase '{phase_name}' for fusion input.")

            if not outputs_to_fuse:
                 # Handle case where only one valid input exists - fusion might still be useful
                 if len(outputs) == 1:
                      logger.warning("Only one valid output found, using it directly instead of fusion.")
                      single_phase_name = list(outputs.keys())[0]
                      single_output_data = outputs[single_phase_name]
                      single_text = self._extract_output(single_output_data)
                      single_confidence = self._calculate_confidence({single_phase_name: single_output_data})
                      return {
                           "response": single_text,
                           "confidence": single_confidence,
                           "fusion_model": None, # Indicate no fusion occurred
                           "source_outputs": {single_phase_name: single_text}
                      }
                 else:
                      raise AggregationError("No valid outputs found to provide as input for fusion.")

            # Format the collected outputs into a single string
            fusion_input_str = self._format_fusion_input(outputs_to_fuse, context)

            # Prepare and Run Fusion Prompt
            fusion_template_name = self._config.get("fusion_template", "ensemble_fusion") # Checked by schema

            # Robust default template (in case formatting fails)
            default_fusion_template = f"""Synthesize the following outputs into a single, high-quality response to the original query. Focus on integrating the best aspects, ensuring accuracy, clarity, and coherence. Avoid simply listing the inputs; create a unified final answer.

ORIGINAL QUERY:
{{ query }}

INPUTS TO FUSE:
{{ fusion_input }}

Based on the query and the provided inputs, generate a comprehensive and well-structured final response:"""

            query_context = context.get("query", "N/A")
            if query_context == "N/A":
                 logger.warning("Query not found in context for fusion prompt.")

            fusion_prompt = ""
            try:
                context_dict = {"query": query_context, "fusion_input": fusion_input_str}
                fusion_prompt = self.render_template(fusion_template_name, context_dict)
            except (ConfigurationError, ValidationError) as e:
                 logger.warning(f"Error formatting fusion template '{fusion_template_name}': {e}. Using default.")
                 try:
                      fusion_prompt = default_fusion_template.format(query=query_context, fusion_input=fusion_input_str)
                 except KeyError: # Catch issues with the default template itself
                       logger.error("Failed to format even the default fusion template.", exc_info=True)
                       raise AggregationError("Failed to create fusion prompt.") from e
            except Exception as e:
                 logger.error(f"Unexpected error formatting fusion prompt: {e}", exc_info=True)
                 # Fallback to default on unexpected errors
                 try:
                      fusion_prompt = default_fusion_template.format(query=query_context, fusion_input=fusion_input_str)
                 except KeyError:
                       logger.error("Failed to format even the default fusion template on unexpected error.", exc_info=True)
                       raise AggregationError("Failed to create fusion prompt.") from e


            # Run the fusion model
            logger.debug(f"Sending fusion prompt to model '{fusion_model_id}'. Prompt length: {len(fusion_prompt)}")
            fusion_params = {
                 "temperature": self._config.get("fusion_temperature", 0.6),
                 "max_tokens": self._config.get("fusion_max_tokens", 2048),
                 "top_p": self._config.get("fusion_top_p"),
                 "top_k": self._config.get("fusion_top_k"),
                 "repeat_penalty": self._config.get("fusion_repeat_penalty")
            }
            fusion_params_filtered = {k: v for k, v in fusion_params.items() if v is not None}
            logger.debug(f"Fusion inference parameters: {fusion_params_filtered}")

            fusion_result_raw = await self._model_manager.run_inference(
                model_id=fusion_model_id,
                prompt=fusion_prompt,
                compute_confidence=True,
                **fusion_params_filtered
            )
            logger.debug("Received fusion result from model.")


            # Process Fusion Result
            fused_output_text = fusion_result_raw.get("text", "")
            if not fused_output_text:
                 logger.warning(f"Fusion model '{fusion_model_id}' returned empty text.")
                 # Consider fallback - maybe return best input based on confidence?
                 # For now, raise error if fusion genuinely failed.
                 raise AggregationError(f"Fusion model '{fusion_model_id}' failed to generate output.")

            # Determine confidence score
            confidence = 0.0
            fusion_model_confidence_data = fusion_result_raw.get("confidence")

            if isinstance(fusion_model_confidence_data, dict):
                 combined_conf = fusion_model_confidence_data.get("combined")
                 if isinstance(combined_conf, (int, float)) and not math.isnan(combined_conf):
                      confidence = float(combined_conf)
                 else:
                      numeric_scores = [v for v in fusion_model_confidence_data.values() if isinstance(v, (int, float)) and not math.isnan(v)]
                      confidence = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
            elif isinstance(fusion_model_confidence_data, (int, float)) and not math.isnan(fusion_model_confidence_data):
                 confidence = float(fusion_model_confidence_data)

            confidence = max(0.0, min(1.0, confidence)) # Clamp [0, 1]

            # Fallback confidence calculation if model confidence is low/zero
            if confidence <= 0.01:
                 logger.debug("Fusion model did not provide significant confidence, calculating average from input phases.")
                 # Use helper to calculate avg confidence of phases that contributed to the input
                 input_phase_names = list(outputs_to_fuse.keys())
                 # Filter original outputs dict to only include those used for fusion
                 fusion_source_outputs = {k: v for k, v in outputs.items() if k in input_phase_names}
                 average_input_confidence = self._calculate_confidence(fusion_source_outputs)
                 confidence = average_input_confidence
                 logger.info(f"Using average input confidence as fallback: {confidence:.3f}")

            execution_time = time.time() - start_time
            logger.info(
                f"Ensemble Fusion aggregation completed in {execution_time:.2f}s using model '{fusion_model_id}'. "
                f"Input phases fused: {len(outputs_to_fuse)}.",
                extra={"final_confidence": confidence}
            )

            # Prepare final result
            aggregation_result = {
                "response": fused_output_text,
                "confidence": confidence,
                "fusion_model": fusion_model_id,
                "source_outputs": outputs_to_fuse
            }

            # Add trace
            if trace_collector:
                 model_exec_time = fusion_result_raw.get("total_time", fusion_result_raw.get("generation_time", 0))
                 # Add trace for the fusion model call itself
                 trace_collector.add_model_trace(
                      model_id=fusion_model_id,
                      input_prompt=fusion_prompt,
                      output=fusion_result_raw,
                      execution_time=model_exec_time,
                      parameters={"role": "fusion_aggregator", **fusion_params_filtered}
                 )
                 # Add trace for the aggregation step
                 trace_collector.add_aggregation_trace(
                    strategy_name=self._strategy_name,
                    inputs={
                        "phase_output_keys": list(outputs.keys()),
                        "context_keys": list(context.keys()),
                    },
                    output=aggregation_result,
                    execution_time=execution_time,
                    parameters={
                        "fusion_model_used": fusion_model_id,
                        "fusion_template": fusion_template_name,
                        **fusion_params_filtered
                    }
                 )

            return aggregation_result

        except AggregationError as e:
             logger.error(f"Ensemble Fusion aggregation failed: {str(e)}")
             raise
        except ModelError as e:
             logger.error(f"Model error during Ensemble Fusion: {str(e)}")
             raise AggregationError(f"Ensemble Fusion failed due to model error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Ensemble Fusion aggregation: {str(e)}", exc_info=True)
            raise AggregationError(f"Ensemble Fusion aggregation failed unexpectedly: {str(e)}")


    def _format_fusion_input(
        self,
        outputs_to_fuse: Dict[str, str],
        context: Dict[str, Any]
    ) -> str:
        """Formats the extracted phase outputs into a structured string for the fusion prompt."""
        formatted_input_parts = []
        # Determine order: Use context sequence if available, else dict order
        phase_sequence = context.get("phase_sequence", list(outputs_to_fuse.keys()))
        ordered_phases = [p for p in phase_sequence if p in outputs_to_fuse]
        remaining_phases = [p for p in outputs_to_fuse if p not in ordered_phases]
        final_phase_order = ordered_phases + remaining_phases

        for i, phase_name in enumerate(final_phase_order):
             output_text = outputs_to_fuse[phase_name]
             # Use clear separators and headers
             header = f"--- Input {i+1} (from Phase: {phase_name}) ---"
             formatted_input_parts.append(f"{header}\n{output_text}")

        # Join with double newline for separation
        return "\n\n".join(formatted_input_parts)

