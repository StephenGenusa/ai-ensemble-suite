# src/ai_ensemble_suite/collaboration/hierarchical_review.py

"""Hierarchical Review collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set, Tuple, TYPE_CHECKING
import time
import asyncio
import math # Import math for confidence calculation

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError, ModelError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector

# Type hint imports
if TYPE_CHECKING:
    from ai_ensemble_suite.models import ModelManager
    from ai_ensemble_suite.config import ConfigManager


class HierarchicalReview(BaseCollaborationPhase):
    """Hierarchical Review collaboration phase.

    Content is progressively reviewed and refined by models in a hierarchical
    structure, with specialists focusing on different aspects of the content.
    """

    def __init__(
        self,
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the hierarchical review phase.

        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.

        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)

        # Get review levels and reviewers
        self._review_levels = self._config.get("review_levels", [])
        if not self._review_levels or not isinstance(self._review_levels, list):
            logger.warning(
                f"No valid 'review_levels' list specified for HierarchicalReview phase '{phase_name}', "
                "using default configuration based on available models."
            )
            # Set default review levels based on available models
            # Ensure _model_ids is available from super().__init__
            available_models = self._model_ids if hasattr(self, '_model_ids') else []
            num_models = len(available_models)

            # Generate default levels only if models are available
            self._review_levels = []
            if num_models >= 1:
                self._review_levels.append({
                    "name": "technical_review",
                     # Use first model
                    "models": available_models[:1],
                    "template": "hierarchical_technical_review" # Default template name
                })
            if num_models >= 2:
                 self._review_levels.append({
                    "name": "clarity_review",
                     # Use second model
                    "models": available_models[1:2],
                    "template": "hierarchical_clarity_review" # Default template name
                 })
            # Require at least 3 models for a separate final refinement step
            # Otherwise, the previous step is effectively the final one
            if num_models >= 3:
                 refinement_model_index = -1 # Use the last model
                 self._review_levels.append({
                    "name": "final_refinement",
                     # Use last model
                    "models": [available_models[refinement_model_index]],
                    "template": "hierarchical_final_refinement" # Default template name
                 })

            if not self._review_levels:
                 logger.warning(f"Could not create default review levels for phase '{phase_name}' as no models are assigned.")


        # Further validation of review levels structure
        if not self._review_levels:
            # Allow phase to potentially run with no review levels if draft is generated,
            # but log a warning. The execute method should handle this.
            logger.warning(
                f"HierarchicalReview phase '{phase_name}' has no review levels configured or generated. It might only produce a draft."
            )

        # Validate each level's structure
        all_level_models = set()
        phase_models_set = set(self._model_ids) # Models configured for the overall phase
        model_manager_models = self._model_manager.get_model_ids() if self._model_manager else set()

        for i, level in enumerate(self._review_levels):
             if not isinstance(level, dict):
                 raise ConfigurationError(f"Review level at index {i} for phase '{phase_name}' must be a dictionary.")

             # Ensure 'name' exists and is a string
             level_name = level.get("name")
             if not level_name or not isinstance(level_name, str):
                  default_name = f"level_{i}"
                  logger.warning(f"Review level {i} missing valid 'name', using default '{default_name}'.")
                  level["name"] = default_name # Assign default name back into the config dict for this instance

             # Ensure 'models' exists, is a list, and models are valid
             level_models = level.get("models")
             if not isinstance(level_models, list):
                  raise ConfigurationError(f"Review level '{level['name']}' must have 'models' as a list.")
             if not level_models:
                  logger.warning(f"Review level '{level['name']}' has an empty 'models' list. It will be skipped.")

             # Check if models in level exist in ModelManager
             for model_id in level_models:
                  if model_id not in model_manager_models:
                      raise ConfigurationError(f"Model '{model_id}' specified in review level '{level['name']}' not found in ModelManager.")
                  all_level_models.add(model_id)

             # Ensure 'template' exists and is a string
             level_template = level.get("template")
             if not level_template or not isinstance(level_template, str):
                  logger.warning(f"Review level '{level['name']}' missing 'template' string. Will attempt fallback to phase default or named template.")
                  # No need to assign here, execute logic handles fallback

        # Check if models used in levels are actually available in *this phase's* config list
        missing_in_phase_config = all_level_models - phase_models_set
        if missing_in_phase_config:
             logger.warning(
                 f"Models {missing_in_phase_config} used in review_levels are not listed "
                 f"in the main 'models' list ({list(phase_models_set)}) for phase '{phase_name}'. "
                 "Ensure these models are loaded by the ModelManager."
             )
             # This might be intentional (using models not explicitly listed for the phase), or an error.

        logger.debug(
            f"Initialized HierarchicalReview phase '{phase_name}' with "
            f"{len(self._review_levels)} review levels."
        )

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Hierarchical Review phase.

        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.

        Returns:
            Dictionary containing:
                output: The final reviewed and refined content.
                reviews: Dictionary mapping level names to their outputs/status.
                initial_content: The content before the review process started.
                review_progression: List tracking content changes after each level.
                confidence: Estimated confidence score for the final output.

        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()

        try:
            # Get inputs from previous phases this phase depends on
            inputs = self._get_inputs_from_context(context)

            # Step 1: Get initial content to review (either from inputs or generate a draft)
            initial_content = await self._get_initial_content(query, inputs, trace_collector) # Pass tracer
            if not initial_content and inputs: # If draft failed but inputs exist, maybe use first input?
                 logger.warning(f"No draft generated for phase '{self._phase_name}', trying to use primary input.")
                 # Try to extract from the first input specified in 'input_from'
                 primary_input_phase = self._input_from[0] if self._input_from else None
                 if primary_input_phase and primary_input_phase in inputs:
                      input_val = inputs[primary_input_phase]
                      extracted = self._extract_input_content(input_val) # Use helper
                      if extracted:
                           initial_content = extracted
                           logger.info(f"Using content from input phase '{primary_input_phase}' as initial content.")

            # Create review process context, starting with the initial content
            review_context = {
                "query": query,
                "initial_content": initial_content,
                "current_content": initial_content, # This will be updated
                "reviews": {}, # Store output of each review step keyed by level name
                **inputs # Include inputs from previous phases for template formatting
            }
            # Track content changes (start with initial content)
            review_progression = [initial_content]


            # Check if there are any review levels to execute
            if not self._review_levels:
                 logger.warning(f"Phase '{self._phase_name}' has no review levels. Output will be the initial content.")
                 last_valid_output = initial_content
            else:
                 # Process each review level defined in the configuration
                 last_valid_output = initial_content # Keep track of the last successful output
                 for level_idx, level_config in enumerate(self._review_levels):
                    level_name = level_config.get("name") # Should exist due to __init__ validation
                    level_models = level_config.get("models", [])
                    level_template = level_config.get("template") # Template name for this level

                    logger.debug(f"Executing review level '{level_name}' (Level {level_idx + 1}/{len(self._review_levels)})")

                    # If no models specified for this level, skip it
                    if not level_models:
                        logger.warning(f"No models specified for review level '{level_name}', skipping.")
                        review_context["reviews"][level_name] = {"status": "skipped", "reason": "no models"}
                        review_progression.append(last_valid_output) # Content doesn't change
                        continue

                    # Determine the template name to use
                    # Priority: level-specific template > phase default template > constructed fallback
                    template_to_use = level_template # From level config
                    if not template_to_use:
                        template_to_use = self._prompt_template # From phase config (via Base class)
                    if not template_to_use:
                         # Construct a fallback template name if none specified anywhere
                         template_to_use = f"hierarchical_{level_name}" # e.g., hierarchical_technical_review
                         logger.warning(f"No template specified for level '{level_name}', trying fallback name '{template_to_use}'.")


                    # Format prompt for this review level, ensuring current content is passed
                    try:
                        context = {**review_context}
                        # Ensure 'current_content' key exists even if empty
                        context.setdefault("current_content", "")
                        # Add a specific key for the content being reviewed *in this step* for clarity in templates
                        context["content_to_review"] = review_context.get("current_content", "")

                        # Format using BaseCollaborationPhase method
                        review_prompt = self.render_template(template_to_use, context)
                    except (ConfigurationError, KeyError, CollaborationError) as e: # Catch formatting or template errors
                        logger.error(f"Failed to format prompt for review level '{level_name}' using template '{template_to_use}': {str(e)}. Skipping level.", exc_info=True)
                        review_context["reviews"][level_name] = {"status": "error", "reason": f"prompt formatting error: {e}"}
                        review_progression.append(last_valid_output) # Content doesn't change
                        continue
                    except Exception as e: # Catch other unexpected formatting errors
                         logger.error(f"Unexpected error formatting prompt for level '{level_name}': {str(e)}.", exc_info=True)
                         review_context["reviews"][level_name] = {"status": "error", "reason": f"unexpected prompt error: {e}"}
                         review_progression.append(last_valid_output)
                         continue


                    # Run the specified models for this review level
                    try:
                        # Use _run_models which handles multiple models and tracing internally
                        # Pass role info for tracing
                        level_results = await self._run_models(
                            prompt=review_prompt,
                            model_ids=level_models,
                            trace_collector=trace_collector,
                            role=f"reviewer_{level_name}" # Add role context for tracing
                        )
                    except (CollaborationError, ModelError) as e:
                         logger.error(f"Failed to run models for review level '{level_name}': {str(e)}. Skipping level.")
                         review_context["reviews"][level_name] = {"status": "error", "reason": f"model execution error: {e}"}
                         review_progression.append(last_valid_output) # Content doesn't change
                         continue
                    except Exception as e:
                         logger.error(f"Unexpected error running models for level '{level_name}': {str(e)}.", exc_info=True)
                         review_context["reviews"][level_name] = {"status": "error", "reason": f"unexpected model error: {e}"}
                         review_progression.append(last_valid_output)
                         continue


                    # --- Process outputs from the level's models ---
                    # Extract text outputs from model results, handling potential errors
                    level_outputs_text: Dict[str, str] = {}
                    failed_models = []
                    for model_id, result in level_results.items():
                        if isinstance(result, dict):
                            if "error" in result:
                                 logger.warning(f"Model '{model_id}' failed during review level '{level_name}': {result['error']}")
                                 failed_models.append(model_id)
                            else:
                                 level_outputs_text[model_id] = result.get("text", "").strip()
                        else:
                             logger.warning(f"Unexpected result type from model '{model_id}' for level '{level_name}': {type(result)}")
                             failed_models.append(model_id)


                    # Combine outputs if multiple models contributed *successfully* to this level
                    successful_outputs = {mid: txt for mid, txt in level_outputs_text.items() if mid not in failed_models and txt}

                    level_output_final = "" # Initialize final output for the level
                    if len(successful_outputs) > 1:
                        logger.debug(f"Combining {len(successful_outputs)} outputs for level '{level_name}'")
                        # Combine outputs with headers - simple concatenation for now
                        combined_output = f"# Combined Review/Refinement from Level '{level_name}'\n\n"
                        for model_id, output_text in successful_outputs.items():
                            combined_output += f"## Contribution from {model_id}\n\n{output_text}\n\n---\n\n"
                        level_output_final = combined_output.strip() # Use combined text
                    elif len(successful_outputs) == 1:
                        # Just use the single successful model's output text
                        level_output_final = next(iter(successful_outputs.values()))
                    else:
                        # No valid successful outputs from this level
                        logger.warning(f"No valid text output received from models for review level '{level_name}'. Keeping previous content.")
                        level_output_final = last_valid_output # Keep the content from before this level
                        review_context["reviews"][level_name] = {"status": "warning", "reason": "no text output from models"}


                    # Update 'current_content' and store results *only if* new content was generated
                    if level_output_final != last_valid_output:
                         review_context["reviews"][level_name] = {
                             "status": "completed",
                             "output": level_output_final,
                             "raw_model_results": level_results # Store raw results too
                         }
                         # Add role-specific key in context for easier access in subsequent steps/templates
                         review_context[level_name] = level_output_final

                         review_context["current_content"] = level_output_final
                         last_valid_output = level_output_final # Update tracker
                         review_progression.append(level_output_final) # Add to progression
                    else:
                         # If content didn't change (e.g., no output), just record the status
                         if level_name not in review_context["reviews"]: # Avoid overwriting previous status
                              review_context["reviews"][level_name] = {
                                    "status": "no_change",
                                    "reason": "Output matched previous content or was empty.",
                                    "raw_model_results": level_results
                              }
                         review_progression.append(last_valid_output) # Content didn't change



            # --- Post-loop ---
            execution_time = time.time() - start_time

            logger.info(
                f"HierarchicalReview phase '{self._phase_name}' completed {len(self._review_levels)} levels in {execution_time:.2f}s"
            )

            # Calculate final confidence score (e.g., average confidence from the *last active* level)
            confidence = 0.7 # Default confidence
            last_active_level_name = None
            # Find the last level that actually completed or resulted in no change
            for level_config in reversed(self._review_levels):
                 lname = level_config["name"]
                 if lname in review_context["reviews"] and review_context["reviews"][lname].get("status") in ["completed", "no_change"]:
                      last_active_level_name = lname
                      break

            if last_active_level_name:
                 final_review_data = review_context["reviews"][last_active_level_name]
                 final_level_raw_results = final_review_data.get("raw_model_results", {})
                 confidence_values = []
                 for result in final_level_raw_results.values():
                       # Extract confidence from the result dict (using 'combined' if available)
                       model_conf = result.get("confidence") # Should be dict from run_inference
                       if isinstance(model_conf, dict):
                           confidence_values.append(model_conf.get("combined", 0.0)) # Use combined or 0
                       elif isinstance(model_conf, (float, int)):
                            # Less likely, but handle direct score
                           confidence_values.append(max(0.0, min(1.0, float(model_conf))))
                 if confidence_values:
                     confidence = sum(confidence_values) / len(confidence_values)


            # Add phase trace if collector is provided
            if trace_collector:
                # Create output data, potentially trimming large fields
                output_trace_data = {
                     "output": last_valid_output, # Final content
                     "reviews": review_context.get("reviews", {}), # Status/outputs of levels
                     "initial_content": initial_content,
                     "confidence": confidence,
                     # Consider not tracing full progression if very long
                     # "review_progression": review_progression
                }
                # Clean potentially large raw results from trace output
                if "reviews" in output_trace_data:
                     for level_name, level_data in output_trace_data["reviews"].items():
                          if "raw_model_results" in level_data:
                               level_data["raw_model_results"] = "..." # Placeholder

                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "inputs": list(inputs.keys())}, # Initial inputs keys
                    output_data=output_trace_data,
                    execution_time=execution_time,
                    phase_parameters=self._config # Include phase config (review levels etc.)
                )


            # Return the final results
            return {
                "output": last_valid_output, # The content after the last successful level
                "reviews": review_context.get("reviews", {}), # Details of each level
                "initial_content": initial_content,
                "review_progression": review_progression, # History of content state
                "confidence": confidence
            }

        except CollaborationError as e:
             # Log and re-raise known collaboration errors
             logger.error(f"Collaboration error in HierarchicalReview phase '{self._phase_name}': {str(e)}", exc_info=True)
             raise
        except Exception as e:
            # Catch-all for unexpected errors during execution
            logger.error(f"Unexpected error in HierarchicalReview phase '{self._phase_name}': {str(e)}", exc_info=True)
            # Wrap in CollaborationError for consistent handling upstream
            raise CollaborationError(f"HierarchicalReview phase '{self._phase_name}' failed unexpectedly: {str(e)}")


    async def _get_initial_content(
        self,
        query: str,
        inputs: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None # Added tracer
    ) -> str:
        """Get or generate the initial content to be reviewed by the hierarchy.

        Args:
            query: The user query.
            inputs: Inputs from previous phases (if specified by 'input_from').
            trace_collector: Optional trace collector.

        Returns:
            String containing the initial content, or empty string if none found/generated.
        """
        # --- Priority 1: Check if specified inputs already provide content ---
        # Use input_from defined in the phase config
        if self._input_from:
            logger.debug(f"Checking inputs {self._input_from} for initial content...")
            for source_phase_name in self._input_from:
                if source_phase_name in inputs:
                    source_data = inputs[source_phase_name]
                    content = self._extract_input_content(source_data) # Use helper
                    if content:
                         logger.info(f"Using initial content from previous phase: '{source_phase_name}'")
                         return content

        logger.debug("No suitable initial content found in configured inputs.")

        # --- Priority 2: Generate a draft if configured ---
        draft_template = self._config.get("draft_template")
        if draft_template:
            logger.debug(f"Attempting to generate initial draft using template: '{draft_template}'")
            try:
                # Combine query and any available inputs for the draft prompt
                context = {"query": query, **inputs}
                # Use render_template from Base Collaboration class
                draft_prompt = self.render_template(draft_template, context)

                # Determine which model(s) to use for drafting
                draft_model_id = self._config.get("draft_model") # Explicit draft model?
                draft_models = []
                if draft_model_id and isinstance(draft_model_id, str):
                    # Ensure specified draft model is managed
                    if draft_model_id in self._model_manager.get_model_ids():
                         draft_models = [draft_model_id]
                    else:
                         logger.warning(f"Specified draft_model '{draft_model_id}' not found in ModelManager. Cannot use for drafting.")
                elif self._model_ids: # Use first model assigned to the phase?
                    draft_models = self._model_ids[:1]

                if not draft_models:
                    logger.warning(f"No models specified or available for drafting in phase '{self._phase_name}'. Cannot generate draft.")
                    return "" # Cannot generate draft

                logger.info(f"Generating draft using model(s): {draft_models}")

                # Run drafting model(s)
                # *** FIX: Use await directly, remove asyncio.run() ***
                # Use _run_models from BaseCollaborationPhase
                draft_results = await self._run_models(
                    prompt=draft_prompt,
                    model_ids=draft_models,
                    trace_collector=trace_collector, # Pass tracer
                    role="draft_generator" # Add role for tracing
                )

                # Extract the first successful result's text
                if draft_results:
                    for model_id, result in draft_results.items(): # Iterate to find first success
                         if isinstance(result, dict) and "error" not in result:
                              draft_content = result.get("text", "").strip()
                              if draft_content:
                                   logger.info(f"Generated initial draft using model '{model_id}'.")
                                   return draft_content
                              else:
                                   logger.warning(f"Draft generation by '{model_id}' resulted in empty content.")
                         elif isinstance(result, dict) and "error" in result:
                              logger.warning(f"Draft generation failed for model '{model_id}': {result['error']}")

                    # If loop finishes without finding good content
                    logger.warning("Draft generation ran but produced no valid content.")
                else:
                     logger.warning("Draft generation failed or returned no results.")

            except (ConfigurationError, KeyError, CollaborationError) as e:
                logger.error(f"Failed to format or run draft generation using template '{draft_template}': {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error generating draft content: {str(e)}", exc_info=True)

        # --- Fallback: No content found or generated ---
        logger.warning(f"No initial content could be found or generated for HierarchicalReview phase '{self._phase_name}'. Proceeding with empty content.")
        return ""

    def _extract_input_content(self, source_data: Any) -> str:
        """Helper to extract string content from various input types."""
        content = ""
        if isinstance(source_data, str):
            content = source_data
        elif isinstance(source_data, dict):
             # Try common keys for output text
            content = source_data.get("output") or \
                      source_data.get("text") or \
                      source_data.get("response") or \
                      source_data.get("final_output") # Add other likely keys
        # Add handling for other potential types if necessary
        # else: logger.debug(...)

        return content.strip() if content and isinstance(content, str) else ""

