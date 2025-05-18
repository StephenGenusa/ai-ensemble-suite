# ai_ensemble_suite/config/config_manager.py

"""Configuration management for ai-ensemble-suite."""

from typing import Dict, Any, Optional, List, Union, Set
import yaml
import os
import copy
from pathlib import Path

from ai_ensemble_suite.exceptions import ConfigurationError, ValidationError
from ai_ensemble_suite.config.defaults import DEFAULT_CONFIG
from ai_ensemble_suite.config.schema import ConfigSchema
from ai_ensemble_suite.config.templates import ALL_TEMPLATES
from ai_ensemble_suite.utils.logging import logger
# Correctly import the utility function
# from ai_ensemble_suite.utils.prompt_utils import format_prompt as _format_prompt_util

class ConfigManager:
    """Manages configuration for the ai-ensemble-suite.

    Handles loading, validation, and access to configuration values.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the ConfigManager.

        Args:
            config_path: Path to a YAML configuration file.
            config_dict: Dictionary containing configuration values.

        Raises:
            ConfigurationError: If both config_path and config_dict are provided
              or if configuration is invalid.
        """
        self.config: Dict[str, Any] = {}

        # Start with default configuration
        self._load_defaults()

        # Load configuration if provided
        try:
            if config_path or config_dict:
                self.load(config_path, config_dict)
        except (ConfigurationError, ValidationError) as e:
             # Catch validation errors during initial load and re-raise
             logger.error(f"Initial configuration loading failed: {e}", exc_info=True)
             raise ConfigurationError(f"Initial configuration loading failed: {e}") from e


    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.config = copy.deepcopy(DEFAULT_CONFIG)

        # Add all template variations from the templates module
        # Ensure ALL_TEMPLATES is a dict {name: string}
        if isinstance(ALL_TEMPLATES, dict):
            self.config.setdefault("templates", {}).update(ALL_TEMPLATES)
        else:
             logger.warning("ALL_TEMPLATES from templates module is not a dictionary, cannot load default templates.")


    def load(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load configuration from a file or dictionary, merging with defaults.

        Args:
            config_path: Path to a YAML configuration file.
            config_dict: Dictionary containing configuration values.

        Raises:
            ConfigurationError: If both config_path and config_dict are provided,
              or if file loading/parsing fails.
            ValidationError: If the merged configuration is invalid.
        """
        if config_path and config_dict:
            raise ConfigurationError(
                "Cannot specify both config_path and config_dict"
            )

        user_config: Optional[Dict[str, Any]] = None

        # Load from file if specified
        if config_path:
            try:
                # Ensure path is absolute or resolved correctly relative to execution context
                resolved_path = Path(config_path).resolve()
                logger.debug(f"Attempting to load configuration from: {resolved_path}")
                if not resolved_path.is_file():
                    raise FileNotFoundError(f"Configuration file not found or is not a file at: {resolved_path}")
                with open(resolved_path, "r", encoding="utf-8") as file:
                    loaded_content = yaml.safe_load(file)
                    if not isinstance(loaded_content, dict):
                        # Allow empty file (None) but not non-dict types
                        if loaded_content is not None:
                             raise ConfigurationError("Configuration file content must be a YAML dictionary (mapping).")
                        user_config = {} # Treat empty file as empty config
                    else:
                        user_config = loaded_content
            except (yaml.YAMLError, FileNotFoundError, OSError) as e:
                raise ConfigurationError(f"Failed to load or parse configuration file '{config_path}': {str(e)}")
        # Use provided dictionary if specified
        elif config_dict:
            if not isinstance(config_dict, dict):
                raise ConfigurationError("config_dict must be a dictionary")
            user_config = config_dict

        # Merge user config into existing (default) config if loaded
        if user_config is not None:
            try:
                # Deep update modifies self.config in place
                self._update_config_recursive(self.config, user_config)
                logger.debug("User configuration merged with defaults.")
            except Exception as e:
                 # Should not happen with basic dict merge, but safety catch
                 raise ConfigurationError(f"Internal error merging configuration: {str(e)}")

        # Always validate the final merged configuration
        try:
            self.validate()
            logger.info("Configuration loaded and validated successfully.")
        except ValidationError as e:
            # If validation fails after loading/merging, it's a fatal config error.
            # We don't revert here; the constructor or update method handles reversion.
            logger.error(f"Configuration validation failed: {e}")
            raise ValidationError(f"Invalid configuration after loading/merging: {str(e)}") from e


    def _update_config_recursive(self, config: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update configuration dictionary 'config' with 'updates'.

        Args:
            config: Existing configuration dictionary to update (modified in place).
            updates: New values to apply to the configuration.
        """
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                # Recursively update dictionaries
                self._update_config_recursive(config[key], value)
            # Allow replacing lists entirely, don't merge them element-wise by default
            # elif key in config and isinstance(config[key], list) and isinstance(value, list):
                 # List merging logic could go here if needed, but replacement is simpler
                 # config[key] = copy.deepcopy(value)
            elif isinstance(value, (dict, list)):
                # Use deepcopy for new/overwritten nested structures to avoid shared references
                config[key] = copy.deepcopy(value)
            else:
                # Replace or add the value (basic types are copied by assignment/immutable)
                config[key] = value

    def validate(self) -> None:
        """Validate the current configuration against the schema.

        Raises:
            ValidationError: If the configuration is invalid.
        """
        logger.debug("Validating configuration...")

        # Check for required top-level sections and their types
        required_sections = {
            "models": dict, # Models section is crucial even if empty
            "collaboration": dict,
            "aggregation": dict,
            "templates": dict
        }
        for section, expected_type in required_sections.items():
            if section not in self.config:
                raise ValidationError(f"Missing required configuration section: '{section}'")
            if not isinstance(self.config[section], expected_type):
                 raise ValidationError(f"Configuration section '{section}' must be a {expected_type.__name__}.")

        # --- Get IDs and Names for Cross-Validation ---
        # Get all model IDs
        model_ids = set(self.get_model_ids()) # Uses self.config['models']

        # Get all phase names
        phase_names = set()
        # Ensure collaboration and phases exist and are the correct type first
        if "collaboration" in self.config and isinstance(self.config["collaboration"], dict) and \
           "phases" in self.config["collaboration"] and isinstance(self.config["collaboration"]["phases"], list):
            phases_list = self.config["collaboration"]["phases"]
            for phase in phases_list:
                # Ensure phase is dict and has a string name
                if isinstance(phase, dict) and "name" in phase and isinstance(phase["name"], str):
                    phase_names.add(phase["name"])
                # else: Handled by collaboration config validation below

        # --- Run Schema Validations ---
        # Validate models configuration
        # Models section existence/type already checked, safe to access
        for model_id, model_config in self.config["models"].items():
            try:
                ConfigSchema.validate_model_config(model_id, model_config)
            except ValidationError as e:
                raise ValidationError(f"Invalid model configuration for '{model_id}': {str(e)}")

        # Validate collaboration configuration
        # Section existence/type checked, safe to access
        try:
            ConfigSchema.validate_collaboration_config(self.config["collaboration"], model_ids)
            # Re-extract phase names AFTER validation, in case validation modified format (unlikely)
            # Or rely on the initial extraction based on checked types. Using initial extraction is fine.
        except ValidationError as e:
            raise ValidationError(f"Invalid collaboration configuration: {str(e)}")

        try:
            # Pass model_ids and phase_names for cross-reference validation
            ConfigSchema.validate_aggregation_config(self.config["aggregation"], phase_names, model_ids)
        except ValidationError as e:
            raise ValidationError(f"Invalid aggregation configuration: {str(e)}")


        # Validate confidence configuration if provided
        if "confidence" in self.config:
             if not isinstance(self.config["confidence"], dict):
                  raise ValidationError("Configuration section 'confidence' must be a dictionary.")
             try:
                 ConfigSchema.validate_confidence_config(self.config["confidence"])
             except ValidationError as e:
                 raise ValidationError(f"Invalid confidence configuration: {str(e)}")

        # Validate logging configuration if provided
        if "logging" in self.config:
             if not isinstance(self.config["logging"], dict):
                  raise ValidationError("Configuration section 'logging' must be a dictionary.")
             try:
                 ConfigSchema.validate_logging_config(self.config["logging"])
             except ValidationError as e:
                 raise ValidationError(f"Invalid logging configuration: {str(e)}")

        # Validate templates: check if required templates exist
        # Section existence/type checked, safe to access
        try:
            required_templates = self._get_required_templates()
            ConfigSchema.validate_templates(self.config["templates"], required_templates)
        except ValidationError as e:
            raise ValidationError(f"Invalid templates configuration: {str(e)}")

        logger.debug("Configuration validation successful.")


    def _get_required_templates(self) -> Set[str]:
        """Determine the set of template names required by the current configuration.

        Scans collaboration phases and aggregation settings for template references.

        Returns:
            Set of required template names (strings).
        """
        required_templates: Set[str] = set()

        # --- Templates from Collaboration Phases ---
        if "collaboration" in self.config and isinstance(self.config["collaboration"], dict) and \
           "phases" in self.config["collaboration"] and isinstance(self.config["collaboration"]["phases"], list):
            phases_list = self.config["collaboration"]["phases"]
            for phase in phases_list:
                if isinstance(phase, dict):
                    # Add all known template keys that might exist in a phase config
                    template_keys_in_phase = [
                        "prompt_template", "initial_template", "branch_template",
                        "evaluation_template", "critique_template", "improvement_template",
                        "draft_template", "competitor_template", "perspective_template",
                        "synthesis_template"
                    ]
                    for key in template_keys_in_phase:
                         template_name = phase.get(key)
                         if isinstance(template_name, str) and template_name:
                              required_templates.add(template_name)

                    # Check within nested structures like workflow_steps or review_levels
                    if "workflow_steps" in phase and isinstance(phase["workflow_steps"], list):
                        for step in phase["workflow_steps"]:
                            if isinstance(step, dict):
                                 template_name = step.get("template")
                                 if isinstance(template_name, str) and template_name:
                                     required_templates.add(template_name)
                    if "review_levels" in phase and isinstance(phase["review_levels"], list):
                        for level in phase["review_levels"]:
                            if isinstance(level, dict):
                                 template_name = level.get("template")
                                 if isinstance(template_name, str) and template_name:
                                     required_templates.add(level["template"])


        # --- Templates from Aggregation Strategy ---
        def collect_aggregation_templates(agg_conf: Dict[str, Any], req_set: Set[str]):
            """Helper to collect template refs from an aggregation config dict."""
            if not isinstance(agg_conf, dict): return

            template_keys_in_agg = [
                "fusion_template", "evaluation_template", "selector_prompt_template"
            ]
            for key in template_keys_in_agg:
                 template_name = agg_conf.get(key)
                 if isinstance(template_name, str) and template_name:
                      req_set.add(template_name)

            # Recursively check fallback
            fallback_config = agg_conf.get("fallback")
            if isinstance(fallback_config, dict):
                collect_aggregation_templates(fallback_config, req_set)

            # Recursively check adaptive_selection sub-strategies
            if agg_conf.get("strategy") == "adaptive_selection":
                sub_strategies = agg_conf.get("strategies")
                if isinstance(sub_strategies, dict):
                     for sub_conf in sub_strategies.values():
                         collect_aggregation_templates(sub_conf, req_set)

        if "aggregation" in self.config and isinstance(self.config["aggregation"], dict):
             collect_aggregation_templates(self.config["aggregation"], required_templates)


        # --- Templates from Confidence Estimation ---
        if "confidence" in self.config and isinstance(self.config["confidence"], dict):
            conf_method = self.config["confidence"].get("default_method")
            needs_self_eval = False
            if isinstance(conf_method, str) and conf_method in ["self_eval", "combined"]:
                needs_self_eval = True
            elif isinstance(conf_method, list) and "self_eval" in conf_method:
                 needs_self_eval = True

            if needs_self_eval:
                 # Default name, or get from config if specified? Assume default for now.
                 self_eval_template_name = self.config["confidence"].get("self_eval_template", "self_evaluation")
                 if isinstance(self_eval_template_name, str) and self_eval_template_name:
                      required_templates.add(self_eval_template_name)


        logger.debug(f"Required templates identified: {required_templates or 'None'}")
        return required_templates


    # --- Getters and Setters ---

    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Get a configuration value, using dot notation for nested keys.

        Args:
            key: The configuration key (e.g., 'aggregation.strategy', 'models.model1.path').
            default: Default value if the key is not found.

        Returns:
            A deep copy of the configuration value if it's a dict or list,
            otherwise the value itself. Returns default if key not found.
        """
        try:
            if "." in key:
                parts = key.split(".")
                current = self.config
                for part in parts:
                    # Check if current level is a dict and part exists
                    if not isinstance(current, dict) or part not in current:
                        return default
                    current = current[part]
                # Return a deep copy for mutable types to prevent accidental modification
                return copy.deepcopy(current) if isinstance(current, (dict, list)) else current
            else:
                 # Top-level key
                 value = self.config.get(key, default)
                 # Deep copy if mutable
                 return copy.deepcopy(value) if isinstance(value, (dict, list)) else value
        except Exception as e:
             # Should not happen with .get and type checks, but safety first
             logger.warning(f"Error accessing config key '{key}': {e}. Returning default.")
             return default

    def update(
        self,
        config_dict: Dict[str, Any]
    ) -> None:
        """Update the current configuration with values from config_dict.

        Validates the configuration after merging. Reverts on validation failure.

        Args:
            config_dict: Dictionary containing configuration values to update.

        Raises:
            ConfigurationError: If config_dict is not a dictionary.
            ValidationError: If validation fails after merging.
        """
        if not isinstance(config_dict, dict):
             raise ConfigurationError("Update value must be a dictionary.")

        # Store the current config in case validation fails and we need to revert
        current_config_backup = copy.deepcopy(self.config)
        logger.debug("Attempting configuration update...")

        try:
            # Update the configuration recursively (modifies self.config)
            self._update_config_recursive(self.config, config_dict)
            # Validate the updated configuration
            self.validate()
            logger.info("Configuration updated and validated successfully.")
        except (ValidationError, ConfigurationError) as e: # Catch expected validation/merge errors
            # Restore the previous configuration if validation fails
            self.config = current_config_backup
            logger.error(f"Configuration update failed due to validation/merge error: {e}. Configuration reverted.")
            # Re-raise the error to signal failure
            raise ValidationError(f"Invalid configuration update: {str(e)}") from e
        except Exception as e:
             # Catch unexpected errors during update/validation
             self.config = current_config_backup
             logger.error(f"Unexpected error during configuration update: {e}. Configuration reverted.", exc_info=True)
             raise ConfigurationError(f"Unexpected error applying configuration update: {str(e)}")


    def get_model_config(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """Get the fully resolved configuration for a specific model, including defaults.

        Args:
            model_id: The ID of the model.

        Returns:
            Dictionary containing the model's merged configuration.

        Raises:
            ConfigurationError: If the model ID does not exist.
        """
        # Ensure models section exists and model_id is present
        if "models" not in self.config or not isinstance(self.config["models"], dict) or \
           model_id not in self.config["models"]:
            raise ConfigurationError(f"Model not found in configuration: '{model_id}'")

        # Get the model's specific config - deep copy first
        model_specific_config = copy.deepcopy(self.config["models"][model_id])
        if not isinstance(model_specific_config, dict):
             # Should be caught by validation, but safety check
             raise ConfigurationError(f"Configuration for model '{model_id}' is not a dictionary.")

        # Apply default parameters ('defaults.model_parameters')
        default_params = self.get_default_model_parameters() # Gets a deep copy
        if default_params:
             # Ensure 'parameters' key exists in model_specific_config
             if "parameters" not in model_specific_config or not isinstance(model_specific_config["parameters"], dict):
                 model_specific_config["parameters"] = {}
             # Apply defaults only if the parameter is NOT already set in the specific model's config
             for param, value in default_params.items():
                 model_specific_config["parameters"].setdefault(param, value)

        # Apply global model parameters ('defaults.global_model_parameters')
        # These override defaults but can be overridden by specific model params
        global_params = self.get("defaults.global_model_parameters", {})
        if isinstance(global_params, dict):
            if "parameters" not in model_specific_config or not isinstance(model_specific_config["parameters"], dict):
                model_specific_config["parameters"] = {}
            # Update defaults with global, then update with specific
            merged_params = {**default_params, **global_params, **model_specific_config["parameters"]}
            model_specific_config["parameters"] = merged_params


        return model_specific_config


    def get_collaboration_config(
        self,
        phase_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get config for a specific collaboration phase or the entire collaboration section.

        Args:
            phase_name: Name of the phase, or None for the whole section.

        Returns:
            Deep copy of the requested configuration dictionary.

        Raises:
            ConfigurationError: If collaboration section is missing or phase not found.
        """
        # Section existence/type checked during validation, access should be safe
        collab_section = self.config["collaboration"]

        if phase_name is None:
            # Return a deep copy of the entire collaboration configuration
            return copy.deepcopy(collab_section)

        # Find the phase with the specified name
        phases_list = collab_section.get("phases", []) # Should be a list due to validation
        if isinstance(phases_list, list):
             for phase in phases_list:
                 # Should be dict with str name due to validation
                 if isinstance(phase, dict) and phase.get("name") == phase_name:
                      # Return a deep copy of the phase config
                     return copy.deepcopy(phase)

        # Phase not found
        raise ConfigurationError(f"Phase not found in collaboration configuration: '{phase_name}'")


    def get_aggregation_config(
        self,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get config for a specific aggregation strategy or the entire aggregation section.

        Handles lookup for active strategy, fallback, or adaptive sub-strategies.

        Args:
            strategy_name: Name of the strategy, or None for the whole section.

        Returns:
            Deep copy of the requested configuration dictionary.

        Raises:
            ConfigurationError: If aggregation section missing or strategy details not found.
        """
        # Section existence/type checked during validation, access should be safe
        agg_section = self.config["aggregation"]

        if strategy_name is None:
            # Return a deep copy of the entire aggregation configuration
            return copy.deepcopy(agg_section)

        # Normalize the requested name
        strategy_name_lower = strategy_name.lower()
        active_strategy = agg_section.get("strategy") # Should be str due to validation

        # 1. Check if it matches the active strategy name
        if isinstance(active_strategy, str) and strategy_name_lower == active_strategy.lower():
            # Return the main config section, as it contains details for the active strategy
            return copy.deepcopy(agg_section)

        # 2. Check if it matches the fallback strategy name
        fallback_config = agg_section.get("fallback")
        if isinstance(fallback_config, dict):
            fallback_strategy_name = fallback_config.get("strategy") # Should be str due to validation
            if isinstance(fallback_strategy_name, str) and strategy_name_lower == fallback_strategy_name.lower():
                # Return the fallback's specific config dictionary
                return copy.deepcopy(fallback_config)

        # 3. Check if it's a sub-strategy under adaptive_selection
        if active_strategy == "adaptive_selection":
            adaptive_strategies_dict = agg_section.get("strategies") # Should be dict due to validation
            if isinstance(adaptive_strategies_dict, dict):
                 for sub_name, sub_config in adaptive_strategies_dict.items():
                      if isinstance(sub_name, str) and strategy_name_lower == sub_name.lower():
                           # Sub-config should be a dict due to validation
                           config_copy = copy.deepcopy(sub_config) if isinstance(sub_config, dict) else {}
                           # Ensure the strategy name is present in the returned dict for consistency
                           config_copy.setdefault("strategy", sub_name) # Use the key as name if missing
                           return config_copy


        # If not found in active, fallback, or adaptive sub-strategies
        raise ConfigurationError(
            f"Configuration details not found for aggregation strategy: '{strategy_name}'. "
            f"Checked active ('{active_strategy}'), fallback, and adaptive sub-strategies (if applicable)."
        )


    def get_model_ids(self) -> List[str]:
        """Get all model IDs defined in the configuration 'models' section.

        Returns:
            List of model ID strings. Returns empty list if 'models' is missing/invalid.
        """
        models_section = self.config.get("models")
        if not isinstance(models_section, dict):
            return []
        return list(models_section.keys())


    def get_all_models_config(self) -> Dict[str, Dict[str, Any]]:
        """Get fully resolved configuration for all models, applying defaults.

        Returns:
            Dictionary mapping model IDs to their merged configurations.
        """
        all_configs = {}
        model_ids = self.get_model_ids() # Get list of valid model IDs

        for model_id in model_ids:
            try:
                # get_model_config already handles defaults and returns a deep copy
                all_configs[model_id] = self.get_model_config(model_id)
            except ConfigurationError as e:
                 # Should not happen if get_model_ids is consistent, but log if it does
                 logger.warning(f"Error retrieving config for model ID '{model_id}' listed in get_model_ids(): {e}")
                 continue # Skip this model

        return all_configs

    # --- Simple Getters for Top-Level Info ---

    def get_collaboration_mode(self) -> str:
        """Get the active collaboration mode string."""
        # Validation ensures this path exists and is a string
        return self.config["collaboration"]["mode"]

    def get_aggregation_strategy(self) -> str:
        """Get the name of the active aggregation strategy string."""
        # Validation ensures this path exists and is a string
        return self.config["aggregation"]["strategy"]

    def get_default_model_parameters(self) -> Dict[str, Any]:
        """Get default model parameters from 'defaults.model_parameters'."""
        # Use self.get for safe access, returns {} if path doesn't exist or not dict
        params = self.get("defaults.model_parameters", {})
        # Ensure it's a dict before returning deep copy
        return copy.deepcopy(params) if isinstance(params, dict) else {}

    def get_template(self, template_name: str) -> Optional[str]:
        """Get a specific prompt template string by name.

        Args:
            template_name: The name of the template.

        Returns:
            The template string, or None if not found or not a string.
        """
        # Validation ensures 'templates' exists and is a dict
        template_value = self.config["templates"].get(template_name)

        # Return only if it's a string
        return template_value if isinstance(template_value, str) else None


    def get_all_templates(self) -> Dict[str, str]:
        """Get a dictionary of all configured templates."""
        # Validation ensures 'templates' exists and is a dict
        # Return a deep copy to prevent modification
        return copy.deepcopy(self.config["templates"])


    def get_confidence_config(self) -> Dict[str, Any]:
        """Get the confidence estimation configuration section."""
        # Use self.get for safe access, returns {} if not found/not dict
        conf_config = self.get("confidence", {})
        return copy.deepcopy(conf_config) if isinstance(conf_config, dict) else {}

    def get_logging_config(self) -> Dict[str, Any]:
        """Get the logging configuration section."""
        log_config = self.get("logging", {})
        return copy.deepcopy(log_config) if isinstance(log_config, dict) else {}


    # --- Formatting Utility ---

    # def format_prompt(self, template_name: str, **kwargs: Any) -> str:
    #     """Format a prompt template using its name and provided values.
    #
    #     Args:
    #         template_name: The name of the template to retrieve and format.
    #         **kwargs: Values to substitute into the template placeholders.
    #
    #     Returns:
    #         The formatted prompt string.
    #
    #     Raises:
    #         ConfigurationError: If the template name does not exist.
    #         ValidationError: If a required placeholder is missing in kwargs.
    #     """
    #     template_string = self.get_template(template_name)
    #     if template_string is None:
    #         # Check if it exists at all vs having wrong type (get_template returns None for both)
    #         if template_name not in self.config.get("templates", {}):
    #              raise ConfigurationError(f"Prompt template '{template_name}' not found in configuration.")
    #         else:
    #              raise ConfigurationError(f"Template '{template_name}' is not a string in configuration.")
    #
    #
    #     try:
    #         # Use the utility function, applying strict validation by default
    #         # Pass strict=True to ensure errors are raised for missing keys
    #         formatted = _format_prompt_util(template_string, strict=True, **kwargs)
    #         return formatted
    #     except ValidationError as e:
    #         # Re-raise validation errors related to missing keys/format issues
    #         raise ValidationError(f"Error formatting template '{template_name}': {str(e)}")
    #     except Exception as e:
    #          # Catch other potential formatting errors (less likely with strict=True)
    #          logger.error(f"Unexpected error formatting template '{template_name}': {e}", exc_info=True)
    #          raise ConfigurationError(f"Unexpected error formatting template '{template_name}': {str(e)}")

    def get_full_config(self) -> Dict[str, Any]:
        """Returns a deep copy of the entire current configuration dictionary."""
        # Assumes self.config exists and is a dictionary (which validation ensures)
        # Return a deep copy to prevent external modification of the internal state.
        return copy.deepcopy(self.config)
