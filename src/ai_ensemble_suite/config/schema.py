# src/ai_ensemble_suite/config/schema.py

"""Schema definitions for validating configuration."""

from typing import Dict, Any, List, Optional, Union, Set, Callable, Type # Added Type
import os
import math # Need math for isnan/isinf checks eventually, maybe isclose

# Re-import ConfigurationError and ValidationError from exceptions
from ai_ensemble_suite.exceptions import ConfigurationError, ValidationError
from ai_ensemble_suite.utils.logging import logger # Import logger

# --- Path Validation ---
# NOTE: Path validation during initial schema check might be tricky due to relative paths.
# The example scripts handle path resolution and existence checks later.
# We'll keep basic path validation here but rely on later checks for full confirmation.
def _validate_path_exists(path: str, required: bool = True) -> bool:
    """Basic check if path string is non-empty (required) or looks plausible.
       Full existence check happens later after path resolution.
    """
    if not isinstance(path, str): return False
    if required:
        return bool(path) # Must be a non-empty string if required
    return True # Optional paths are valid even if empty string

# --- Range Validation ---
def _validate_float_range(value: float, min_val: float, max_val: float) -> bool:
    """Validate that a float is within a range [min_val, max_val]."""
    # Add checks for NaN or infinity which are usually invalid config values
    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
         return False
    return min_val <= float(value) <= max_val

def _validate_int_range(value: int, min_val: int, max_val: int) -> bool:
    """Validate that an integer is within a range [min_val, max_val]."""
    if not isinstance(value, int) or isinstance(value, bool): # bool is subclass of int
        return False
    return min_val <= value <= max_val

class ConfigSchema:
    """Schema definitions and validation methods for configuration."""

    @staticmethod
    def validate_model_config(model_id: str, config: Dict[str, Any]) -> None:
        """Validate model configuration.

        Args:
            model_id: The model ID
            config: Model configuration dictionary

        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
             raise ValidationError(f"Model '{model_id}' config must be a dictionary.")

        # Validate required fields
        required_fields = ["path"] # Role might also be considered required depending on usage
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Model '{model_id}' missing required field: '{field}'")

        # Basic path format check
        if not _validate_path_exists(config["path"], required=True):
            # More specific message if it's empty vs malformed later?
            raise ValidationError(f"Model '{model_id}' requires a non-empty 'path' string.")
            # Actual path existence checked later in example scripts

        # Validate parameters if provided
        if "parameters" in config:
            params = config["parameters"]
            if not isinstance(params, dict):
                 raise ValidationError(f"Model '{model_id}' parameters must be a dictionary.")

            # Validate specific parameter types and ranges
            param_validations = {
                "temperature": {"type": float, "min": 0.0, "max": 2.0},
                "top_p": {"type": float, "min": 0.0, "max": 1.0},
                "top_k": {"type": int, "min": 1}, # Max often model dependent, maybe remove upper bound
                "max_tokens": {"type": int, "min": 1, "max": 32768}, # Adjust max as needed
                "repeat_penalty": {"type": float, "min": 0.0}, # Often >= 1.0, but allow flexibility
                "n_ctx": {"type": int, "min": 1},
                "n_gpu_layers": {"type": int}, # Can be -1 (auto), 0 (CPU), or positive
                # Add other common parameters (e.g., stop sequences, mirostat)
            }

            for param_key, rules in param_validations.items():
                if param_key in params:
                    value = params[param_key]
                    expected_type = rules["type"]

                    # Check type (allow int for float params)
                    if expected_type is float and not isinstance(value, (int, float)):
                        raise ValidationError(f"Model '{model_id}' parameter '{param_key}' must be a number.")
                    elif expected_type is int and not isinstance(value, int):
                         # Special check for n_gpu_layers = -1 is okay
                         if param_key == "n_gpu_layers" and value == -1:
                             continue
                         # Also need to allow bools for integer checks? No, treat separately
                         if isinstance(value, bool):
                              raise ValidationError(f"Model '{model_id}' parameter '{param_key}' must be an integer, not a boolean.")
                         raise ValidationError(f"Model '{model_id}' parameter '{param_key}' must be an integer.")
                    # Add checks for bool, str, list[str] if needed
                    if expected_type is bool and not isinstance(value, bool):
                         raise ValidationError(f"Model '{model_id}' parameter '{param_key}' must be a boolean (true/false).")
                    if expected_type is str and not isinstance(value, str):
                         raise ValidationError(f"Model '{model_id}' parameter '{param_key}' must be a string.")
                    # Add list[str] check if needed


                    # Check range
                    min_val = rules.get("min")
                    max_val = rules.get("max")
                    if min_val is not None and max_val is not None:
                        if expected_type is float and not _validate_float_range(float(value), min_val, max_val):
                             raise ValidationError(f"Model '{model_id}' parameter '{param_key}' ({value}) must be between {min_val} and {max_val}.")
                        elif expected_type is int and not _validate_int_range(int(value), min_val, max_val):
                             raise ValidationError(f"Model '{model_id}' parameter '{param_key}' ({value}) must be between {min_val} and {max_val}.")
                    elif min_val is not None: # Check only min
                         if (expected_type is float and float(value) < min_val) or \
                            (expected_type is int and int(value) < min_val):
                              raise ValidationError(f"Model '{model_id}' parameter '{param_key}' ({value}) must be >= {min_val}.")
                    elif max_val is not None: # Check only max
                         if (expected_type is float and float(value) > max_val) or \
                            (expected_type is int and int(value) > max_val):
                              raise ValidationError(f"Model '{model_id}' parameter '{param_key}' ({value}) must be <= {max_val}.")

        # Validate other optional fields like 'role'
        if "role" in config and not isinstance(config["role"], str):
             raise ValidationError(f"Model '{model_id}' parameter 'role' must be a string.")


    @staticmethod
    def validate_collaboration_config(config: Dict[str, Any], model_ids: Set[str]) -> None:
        """Validate collaboration configuration.

        Args:
            config: Collaboration configuration dictionary
            model_ids: Set of available model IDs

        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
             raise ValidationError("Collaboration config must be a dictionary.")

        # Validate required fields
        required_fields = ["mode", "phases"]
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Collaboration configuration missing required field: '{field}'")

        # Validate mode is string
        if not isinstance(config["mode"], str) or not config["mode"]:
             raise ValidationError("Collaboration 'mode' must be a non-empty string.")

        # Validate phases
        if not isinstance(config["phases"], list) or not config["phases"]: # Must be non-empty list
            raise ValidationError("Collaboration 'phases' must be a non-empty list.")

        phase_names = set()
        for i, phase in enumerate(config["phases"]):
             if not isinstance(phase, dict):
                  raise ValidationError(f"Phase at index {i} must be a dictionary.")

             # Validate required phase fields
             if "name" not in phase or not isinstance(phase["name"], str) or not phase["name"]:
                 raise ValidationError(f"Phase at index {i} is missing required non-empty string field: 'name'")
             phase_name = phase["name"]

             if "type" not in phase or not isinstance(phase["type"], str) or not phase["type"]:
                 raise ValidationError(f"Phase '{phase_name}' is missing required non-empty string field: 'type'")

             # Check for duplicate phase names
             if phase_name in phase_names:
                 raise ValidationError(f"Duplicate phase name found: '{phase_name}'")
             phase_names.add(phase_name)

             # Validate model references (optional per phase, but must be valid if present)
             if "models" in phase:
                 if not isinstance(phase["models"], list):
                     raise ValidationError(f"Phase '{phase_name}': 'models' must be a list of strings.")
                 if not all(isinstance(model_id, str) and model_id for model_id in phase["models"]):
                     raise ValidationError(f"Phase '{phase_name}': 'models' list must contain only non-empty strings.")

                 for model_id in phase["models"]:
                     if model_id not in model_ids:
                         raise ValidationError(
                             f"Phase '{phase_name}' references unknown model: '{model_id}'. "
                             f"Available models: {model_ids or 'None'}"
                         )

             # Validate template references (optional per phase, checked later for existence)
             template_keys = ["prompt_template", "initial_template", "branch_template", # Add all known template keys
                              "evaluation_template", "critique_template", "improvement_template",
                              "draft_template", "competitor_template", "perspective_template",
                              "synthesis_template"]
             for key in template_keys:
                  if key in phase and not isinstance(phase[key], str):
                       raise ValidationError(f"Phase '{phase_name}': template key '{key}' must be a string.")

             # Validate structure of specific phase types if needed (e.g., workflow steps)
             if "workflow_steps" in phase:
                 if not isinstance(phase["workflow_steps"], list):
                      raise ValidationError(f"Phase '{phase_name}': 'workflow_steps' must be a list.")
                 for step_idx, step in enumerate(phase["workflow_steps"]):
                      if not isinstance(step, dict):
                           raise ValidationError(f"Phase '{phase_name}': Step {step_idx} in workflow_steps must be a dictionary.")
                      if "template" in step and not isinstance(step["template"], str):
                           raise ValidationError(f"Phase '{phase_name}': Step {step_idx} 'template' must be a string.")
                      # Add more step validation if needed (e.g., required role/task keys)

             if "review_levels" in phase: # Validation for Hierarchical Review
                  if not isinstance(phase["review_levels"], list):
                       raise ValidationError(f"Phase '{phase_name}': 'review_levels' must be a list.")
                  for level_idx, level in enumerate(phase["review_levels"]):
                      if not isinstance(level, dict):
                           raise ValidationError(f"Phase '{phase_name}': Level {level_idx} in review_levels must be a dictionary.")
                      if "name" not in level or not isinstance(level["name"], str) or not level["name"]:
                          raise ValidationError(f"Phase '{phase_name}': Level {level_idx} requires a non-empty 'name' string.")
                      if "models" not in level or not isinstance(level["models"], list) or not all(isinstance(m, str) and m for m in level["models"]):
                          raise ValidationError(f"Phase '{phase_name}': Level '{level['name']}' requires a 'models' list of non-empty strings.")
                      if "template" not in level or not isinstance(level["template"], str) or not level["template"]:
                           raise ValidationError(f"Phase '{phase_name}': Level '{level['name']}' requires a non-empty 'template' string.")
                      # Add validation against available models if needed:
                      # for model_id in level["models"]:
                      #      if model_id not in model_ids: ...


             # Validate input_from references (referencing previously defined phases)
             # Only phases after the first one can have input_from
             if i > 0 and "input_from" in phase:
                 input_from = phase["input_from"]
                 refs_to_check = []
                 if isinstance(input_from, str):
                     refs_to_check = [input_from]
                 elif isinstance(input_from, list):
                      if not all(isinstance(item, str) and item for item in input_from):
                           raise ValidationError(f"Phase '{phase_name}': 'input_from' list must contain only non-empty strings.")
                      refs_to_check = input_from
                 else:
                      raise ValidationError(f"Phase '{phase_name}': 'input_from' must be a string or list of non-empty strings.")

                 for ref_phase in refs_to_check:
                     # Check against names of *previously* defined phases
                     previously_defined = {p["name"] for p_idx, p in enumerate(config["phases"]) if p_idx < i and isinstance(p, dict) and "name" in p}
                     if ref_phase not in previously_defined:
                         raise ValidationError(
                             f"Phase '{phase_name}' 'input_from' references unknown or later-defined phase: '{ref_phase}'. "
                             f"Known previous phases: {previously_defined or 'None'}"
                         )


    @staticmethod
    def validate_aggregation_config(config: Dict[str, Any], phase_names: Set[str], model_ids: Set[str]) -> None:
        """Validate aggregation configuration based on the specified strategy.

        Args:
            config: Aggregation configuration dictionary.
            phase_names: Set of available phase names defined in collaboration.
            model_ids: Set of available model IDs defined globally.

        Raises:
            ValidationError: If configuration is invalid for the chosen strategy.
        """
        # 1. Validate existence and type of 'strategy' key
        if "strategy" not in config:
            raise ValidationError("Aggregation configuration missing required field: 'strategy'")
        if not isinstance(config["strategy"], str) or not config["strategy"]:
            raise ValidationError("Aggregation 'strategy' must be a non-empty string.")

        strategy = config["strategy"]
        context_msg = f"for strategy '{strategy}'" # Helper for error messages

        # --- Helper Functions ---
        def _validate_phase_ref(cfg: Dict[str, Any], key: str, context_msg: str, required: bool = True):
            """Validates if a phase reference exists and points to a known phase."""
            if key not in cfg:
                if required:
                    raise ValidationError(f"Aggregation config {context_msg} requires '{key}'.")
                else:
                    return  # Optional key not present, validation passes
            phase_ref = cfg[key]
            if not isinstance(phase_ref, str):
                raise ValidationError(f"Aggregation '{key}' {context_msg} must be a string (phase name).")
            if phase_ref not in phase_names:
                raise ValidationError(
                    f"Aggregation '{key}' ('{phase_ref}') {context_msg} references an unknown phase. "
                    f"Available phases: {phase_names or 'None'}"
                )

        # Check phase references in main aggregation config
        strategy = config["strategy"]
        context_msg = f"for strategy '{strategy}'"

        # Cross-validate phase references (final_phase, source_phase)
        if "final_phase" in config:
            _validate_phase_ref(config, "final_phase", context_msg, required=False)

        if "source_phase" in config:
            _validate_phase_ref(config, "source_phase", context_msg, required=False)

        # Validate phase references in sequence if present
        if "sequence" in config and isinstance(config["sequence"], list):
            for phase in config["sequence"]:
                if isinstance(phase, str) and phase not in phase_names:
                    raise ValidationError(
                        f"Phase '{phase}' in aggregation 'sequence' not found in available phases. "
                        f"Available phases: {phase_names or 'None'}"
                    )

        def _check_model_ref(cfg: Dict[str, Any], key: str, context_msg: str, required: bool = True):
            """Validates if a model reference exists and points to a known model."""
            if key not in cfg:
                 if required:
                      raise ValidationError(f"Aggregation config {context_msg} requires '{key}'.")
                 else:
                      return # Optional key not present
            model_ref = cfg[key]
            if not isinstance(model_ref, str):
                 raise ValidationError(f"Aggregation '{key}' {context_msg} must be a string (model ID).")
            if model_ref not in model_ids:
                 raise ValidationError(
                      f"Aggregation '{key}' ('{model_ref}') {context_msg} references an unknown model. "
                      f"Available models: {model_ids or 'None'}"
                 )

        def _check_template_ref(cfg: Dict[str, Any], key: str, context_msg: str, required: bool = True):
             """Checks if a template key exists and is a string. Actual template validation is elsewhere."""
             if key not in cfg:
                  if required:
                       raise ValidationError(f"Aggregation config {context_msg} requires template key '{key}'.")
                  else:
                       return # Optional key not present
             template_ref = cfg[key]
             if not isinstance(template_ref, str):
                  raise ValidationError(f"Aggregation template key '{key}' {context_msg} must be a string (template name).")
             # The actual check if the template *exists* happens in ConfigSchema.validate_templates

        def _validate_numeric_param(cfg: Dict[str, Any], key: str, context_msg: str, param_type: type = float, min_val=None, max_val=None):
             """Validates optional numeric parameters (int or float)."""
             if key in cfg:
                 value = cfg[key]
                 if not isinstance(value, (int, float)) or isinstance(value, bool) or \
                    (isinstance(value, float) and (math.isnan(value) or math.isinf(value))): # Added bool/nan/inf check
                      raise ValidationError(f"Aggregation parameter '{key}' {context_msg} must be a valid number (int or float). Found type {type(value)}.")
                 # Convert to target type for range check if needed
                 try:
                    value_typed = param_type(value)
                 except ValueError:
                     raise ValidationError(f"Aggregation parameter '{key}' {context_msg} cannot be converted to {param_type.__name__}.")

                 valid_range = True
                 err_msg = ""
                 if min_val is not None and max_val is not None:
                    if param_type is float: valid_range = _validate_float_range(value_typed, min_val, max_val)
                    elif param_type is int: valid_range = _validate_int_range(int(value_typed), min_val, max_val) # Use int() for int range check
                    err_msg = f"between {min_val} and {max_val}"
                 elif min_val is not None:
                    if (param_type is float and value_typed < min_val) or (param_type is int and int(value_typed) < min_val): valid_range = False
                    err_msg = f">= {min_val}"
                 elif max_val is not None:
                     if (param_type is float and value_typed > max_val) or (param_type is int and int(value_typed) > max_val): valid_range = False
                     err_msg = f"<= {max_val}"

                 if not valid_range:
                      raise ValidationError(f"Aggregation parameter '{key}' ({value}) {context_msg} must be {err_msg}.")


        # --- 2. Strategy-Specific Validation ---

        if strategy == "sequential_refinement":
            # final_phase is preferred but runtime has fallbacks. Validate if present.
            _validate_phase_ref(config, "final_phase", context_msg, required=False)
            if "sequence" in config: # Optional 'sequence' list
                if not isinstance(config["sequence"], list) or not all(isinstance(p, str) for p in config["sequence"]):
                     raise ValidationError(f"Aggregation 'sequence' {context_msg} must be a list of strings (phase names).")

        elif strategy == "confidence_based":
            # Needs source_phase implicitly to know which outputs' confidence to compare
            # *** FIX: Make source_phase optional as the implementation iterates all outputs anyway ***
            _validate_phase_ref(config, "source_phase", context_msg, required=False)
            _validate_numeric_param(config, "threshold", context_msg, float, min_val=0.0, max_val=1.0)

        elif strategy == "ensemble_fusion":
            # source_phase is not strictly needed as it fuses multiple inputs, but can be used for context
            _validate_phase_ref(config, "source_phase", context_msg, required=False)
            _check_model_ref(config, "fusion_model", context_msg, required=True)
            _check_template_ref(config, "fusion_template", context_msg, required=True)
            # Optional inference parameters
            _validate_numeric_param(config, "fusion_temperature", context_msg, float, min_val=0.0, max_val=2.0)
            _validate_numeric_param(config, "fusion_max_tokens", context_msg, int, min_val=1, max_val=32768)
            _validate_numeric_param(config, "fusion_top_p", context_msg, float, min_val=0.0, max_val=1.0)
            _validate_numeric_param(config, "fusion_top_k", context_msg, int, min_val=1)
            _validate_numeric_param(config, "fusion_repeat_penalty", context_msg, float, min_val=0.0)

        elif strategy == "weighted_voting":
            # Doesn't require source_phase for input selection, but good for context
            _validate_phase_ref(config, "source_phase", context_msg, required=False)
            if "weights" in config: # Optional weights dict
                 if not isinstance(config["weights"], dict):
                      raise ValidationError(f"Aggregation 'weights' {context_msg} must be a dictionary.")
                 for k, v in config["weights"].items():
                      if not isinstance(k, str):
                           raise ValidationError(f"Aggregation 'weights' keys {context_msg} must be phase names (strings). Found: {k}")
                      if not isinstance(v, (int, float)) or v < 0: # Weights should be non-negative
                           raise ValidationError(f"Aggregation 'weights' value for '{k}' {context_msg} must be a non-negative number. Found: {v}")
            # Optional default weight
            _validate_numeric_param(config, "default_weight", context_msg, float, min_val=0.0)

        elif strategy == "multidimensional_voting":
            # source_phase is not strictly needed but good for context if parsing/evaluating single phase
            _validate_phase_ref(config, "source_phase", context_msg, required=False)
            if "dimensions" in config: # Optional but usually needed
                if not isinstance(config["dimensions"], list) or not all(isinstance(d, str) and d for d in config["dimensions"]): # Check non-empty strings
                    raise ValidationError(f"Aggregation 'dimensions' {context_msg} must be a list of non-empty strings.")
            if "dimension_weights" in config: # Optional weights dict
                if not isinstance(config["dimension_weights"], dict):
                    raise ValidationError(f"Aggregation 'dimension_weights' {context_msg} must be a dictionary.")
                for k, v in config["dimension_weights"].items():
                    if not isinstance(k, str):
                         raise ValidationError(f"Aggregation 'dimension_weights' keys {context_msg} must be dimension names (strings). Found: {k}")
                    if not isinstance(v, (int, float)) or v < 0:
                         raise ValidationError(f"Aggregation 'dimension_weights' value for '{k}' {context_msg} must be a non-negative number. Found: {v}")
            # Optional evaluation model/template
            _check_model_ref(config, "evaluator_model", context_msg, required=False)
            _check_template_ref(config, "evaluation_template", context_msg, required=False)
            has_eval_model = "evaluator_model" in config
            has_eval_template = "evaluation_template" in config
            if has_eval_model != has_eval_template: # Require both or neither
                 raise ValidationError(f"Aggregation {context_msg} requires both 'evaluator_model' and 'evaluation_template' if either is specified.")
            # Optional evaluation inference parameters
            _validate_numeric_param(config, "evaluation_temperature", context_msg, float, min_val=0.0, max_val=2.0)
            _validate_numeric_param(config, "evaluation_max_tokens", context_msg, int, min_val=1, max_val=32768)


        elif strategy == "adaptive_selection":
            # Needs 'strategies' dictionary
            if "strategies" not in config or not isinstance(config["strategies"], dict):
                 raise ValidationError(f"Aggregation config {context_msg} requires a 'strategies' dictionary.")
            if not config["strategies"]:
                 raise ValidationError(f"Aggregation 'strategies' dictionary {context_msg} cannot be empty.")
            # selector_model is optional: selection logic can be rule-based
            _check_model_ref(config, "selector_model", context_msg, required=False)
            # If selector_model is present, template is required
            if "selector_model" in config:
                  _check_template_ref(config, "selector_prompt_template", context_msg, required=True)
            # Else template is optional (if selection is rule-based)


            # --- Recursive Validation for Sub-Strategies ---
            for sub_name, sub_config in config["strategies"].items():
                 if not isinstance(sub_config, dict):
                      raise ValidationError(f"Sub-strategy '{sub_name}' within '{strategy}' must be a dictionary.")

                 # If sub-config doesn't explicitly state strategy, assume key is the name
                 sub_config_to_validate = {**sub_config}
                 sub_strategy_name = sub_name # Assume key is name initially
                 if "strategy" in sub_config_to_validate:
                      if isinstance(sub_config_to_validate["strategy"], str) and sub_config_to_validate["strategy"]:
                           sub_strategy_name = sub_config_to_validate["strategy"]
                      else:
                           raise ValidationError(f"Sub-strategy '{sub_name}' within '{strategy}' has invalid 'strategy' key: {sub_config_to_validate['strategy']}. Must be non-empty string.")
                 else:
                      # Add the key as the strategy name if it's missing
                      sub_config_to_validate["strategy"] = sub_name
                      logger.debug(f"Assuming strategy name '{sub_name}' for validation of sub-strategy within '{strategy}'.")

                 # Prevent infinite recursion if adaptive selects itself directly
                 if sub_strategy_name == "adaptive_selection":
                      raise ValidationError("AdaptiveSelection cannot directly contain 'adaptive_selection' as a sub-strategy.")

                 try:
                      # ---> RECURSIVE CALL <---
                      ConfigSchema.validate_aggregation_config(sub_config_to_validate, phase_names, model_ids)
                 except ValidationError as e:
                      # Re-raise with context about the sub-strategy
                      raise ValidationError(f"Invalid configuration for sub-strategy '{sub_name}' "
                                           f"(type: '{sub_strategy_name}') "
                                           f"within '{strategy}': {str(e)}")


        # --- Handle the 'no_strategy' case or basic pass-through ---
        elif strategy in ["no_strategy", "pass_through", "first_valid"]: # Allow aliases
            # Assume these need a specific phase input
            # Allow source_phase OR final_phase
            has_source = "source_phase" in config
            has_final = "final_phase" in config
            if not has_source and not has_final:
                 raise ValidationError(f"Aggregation config for strategy '{strategy}' requires either 'source_phase' or 'final_phase'.")
            if has_source:
                 _validate_phase_ref(config, "source_phase", context_msg, required=False)
            if has_final:
                 _validate_phase_ref(config, "final_phase", context_msg, required=False)

        # --- Handle strategies not explicitly listed (e.g. user-defined/future) ---
        else:
            logger.warning(f"Validation rules not specifically defined in schema for aggregation strategy: '{strategy}'. Performing basic checks only.")
            # Minimal checks for unknown strategies
            if "source_phase" in config: _validate_phase_ref(config, "source_phase", context_msg, required=False)
            if "final_phase" in config: _validate_phase_ref(config, "final_phase", context_msg, required=False)
            if "evaluator_model" in config: _check_model_ref(config, "evaluator_model", context_msg, required=False)


        # --- 3. Common Validation (e.g., fallback) ---
        # This runs regardless of the primary strategy
        if "fallback" in config:
            if not isinstance(config["fallback"], dict):
                raise ValidationError("Aggregation fallback must be a dictionary.")

            # Validate the fallback configuration AS IF IT WERE a main strategy
            try:
                 temp_fallback_config = {**config["fallback"]} # Copy the fallback dict
                 if "strategy" not in temp_fallback_config:
                      raise ValidationError("Aggregation fallback must define its own 'strategy' key.")
                 fallback_strategy_name = temp_fallback_config["strategy"]
                 if not isinstance(fallback_strategy_name, str) or not fallback_strategy_name:
                      raise ValidationError("Aggregation fallback 'strategy' must be a non-empty string.")

                 # Prevent infinite recursion if fallback uses adaptive
                 if fallback_strategy_name == "adaptive_selection":
                     raise ValidationError("Aggregation fallback cannot use 'adaptive_selection'.")

                 # ---> RECURSIVE CALL FOR FALLBACK <---
                 ConfigSchema.validate_aggregation_config(temp_fallback_config, phase_names, model_ids)

                 # Check fallback-specific threshold if present
                 _validate_numeric_param(config["fallback"], "threshold", "for fallback", float, min_val=0.0, max_val=1.0)

            except ValidationError as e:
                 # Re-raise with context about the fallback
                 raise ValidationError(f"Invalid configuration for 'fallback' strategy "
                                      f"(type: '{config['fallback'].get('strategy', 'N/A')}'): {str(e)}")


    @staticmethod
    def validate_confidence_config(config: Dict[str, Any]) -> None:
        """Validate confidence configuration.

        Args:
            config: Confidence configuration dictionary

        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError("Confidence config must be a dictionary.")

        valid_methods = {"token_prob", "self_eval", "consistency", "combined"}

        if "default_method" in config:
             method = config["default_method"]
             is_valid = False
             if isinstance(method, str) and method in valid_methods:
                 is_valid = True
             elif isinstance(method, list): # Allow list of methods for combined
                 is_valid = all(isinstance(m, str) and m in valid_methods for m in method)

             if not is_valid:
                 raise ValidationError(
                     f"Invalid confidence method: '{config['default_method']}'. " +
                     f"Must be one or a list of: {', '.join(valid_methods)}"
                 )

        # Validate weights (for combined method mostly)
        weight_keys = ["token_prob_weight", "self_eval_weight", "consistency_weight"] # Add consistency weight
        for weight_name in weight_keys:
            if weight_name in config:
                if not isinstance(config[weight_name], (int, float)):
                    raise ValidationError(f"Confidence '{weight_name}' must be a number.")
                if not _validate_float_range(config[weight_name], 0.0, 1.0):
                    raise ValidationError(f"Confidence '{weight_name}' must be between 0.0 and 1.0.")

        # --- Validate that weights (if specified for combined) sum close to 1.0 ---
        # Method check logic needs refinement: if combined is *part* of a list, weights still relevant
        is_combined = False
        method_conf = config.get("default_method")
        if isinstance(method_conf, str) and method_conf == "combined":
             is_combined = True
        elif isinstance(method_conf, list) and "combined" in method_conf:
             is_combined = True

        if is_combined:
             total_weight = 0.0
             specified_weights = 0
             for weight_name in weight_keys:
                  if weight_name in config:
                       total_weight += float(config[weight_name])
                       specified_weights += 1
             # Only warn if multiple weights are specified and they don't sum correctly
             if specified_weights > 1 and not math.isclose(total_weight, 1.0):
                  logger.warning(f"Confidence weights ({weight_keys} present) sum to {total_weight}, not 1.0. Normalization might be applied at runtime.")


        # Validate consistency samples
        if "consistency_samples" in config:
            if not isinstance(config["consistency_samples"], int):
                raise ValidationError("Confidence 'consistency_samples' must be an integer.")
            if not _validate_int_range(config["consistency_samples"], 2, 10):
                raise ValidationError("Confidence 'consistency_samples' must be between 2 and 10.")

        # Validate self_eval_template is string if present
        if "self_eval_template" in config and not isinstance(config["self_eval_template"], str):
             raise ValidationError("Confidence 'self_eval_template' must be a string.")


    @staticmethod
    def validate_logging_config(config: Dict[str, Any]) -> None:
        """Validate logging configuration.

        Args:
            config: Logging configuration dictionary

        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValidationError("Logging config must be a dictionary.")

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

        if "level" in config and isinstance(config["level"], str):
            if config["level"].upper() not in valid_levels:
                raise ValidationError(
                    f"Invalid logging level: '{config['level']}'. " +
                    f"Must be one of: {', '.join(valid_levels)}"
                )

        if "sanitize_prompts" in config and not isinstance(config["sanitize_prompts"], bool):
            raise ValidationError("Logging 'sanitize_prompts' must be a boolean (true/false).")

        if "log_file" in config and not isinstance(config["log_file"], (str, type(None))): # Allow null/None
             raise ValidationError("Logging 'log_file' must be a string (path) or null.")


    @staticmethod
    def validate_templates(templates: Dict[str, str], required_templates: Set[str]) -> None:
        """Validate prompt templates dictionary.

        Checks if required templates exist and all template values are strings.

        Args:
            templates: Dictionary of template names to template strings.
            required_templates: Set of template names that must be present.

        Raises:
            ValidationError: If templates are invalid or required ones are missing.
        """
        if not isinstance(templates, dict):
             raise ValidationError("Configuration 'templates' section must be a dictionary.")

        # Check required templates
        missing_required = required_templates - set(templates.keys())
        if missing_required:
            # Provide helpful context about where requirements come from if possible
            # (This might require passing more context to _get_required_templates later)
            raise ValidationError(f"Missing required template(s): {', '.join(sorted(list(missing_required)))}."
                                  " These templates are referenced in collaboration phases or aggregation config.")

        # Check all template values are strings
        for name, template in templates.items():
            if not isinstance(template, str):
                raise ValidationError(f"Template '{name}' must be a string, but found type {type(template).__name__}.")
            # Optional: Check for basic placeholders? e.g., if template is empty string?
            if not template:
                logger.warning(f"Template '{name}' is an empty string.")

