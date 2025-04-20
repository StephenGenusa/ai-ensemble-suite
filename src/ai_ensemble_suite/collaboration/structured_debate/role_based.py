
"""Role-based debate pattern implementation."""

from typing import Dict, Any, Optional, List, Set, Tuple
import time
import asyncio

from ai_ensemble_suite.collaboration.structured_debate.base_debate import BaseDebate
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class RoleBasedDebate(BaseDebate):
    """Role-Based Debate pattern.
    
    Models interact according to assigned specialized roles such as
    Critic, Advocate, Synthesizer, and Fact-Checker.
    """
    
    def __init__(
        self, 
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the role-based debate phase.
        
        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
            
        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)
        
        # Get role assignments
        self._role_assignments = self._config.get("role_assignments", {})
        
        # If no explicit role assignments, try to use model roles
        if not self._role_assignments:
            self._use_model_roles = True
            logger.debug(
                f"No explicit role assignments for phase '{phase_name}', "
                "will use model roles"
            )
        else:
            self._use_model_roles = False
            logger.debug(
                f"Using explicit role assignments for phase '{phase_name}': "
                f"{self._role_assignments}"
            )
            
        # Get debate turn order (optional)
        self._turn_order = self._config.get("turn_order", [])
        
        # Get standard roles if not explicitly configured
        if not self._turn_order:
            self._turn_order = ["critic", "advocate", "synthesizer", "fact_checker"]
            logger.debug(f"Using default turn order for phase '{phase_name}'")
            
        # Get role-specific prompt templates
        self._role_templates = self._config.get("role_templates", {})
    
    async def _execute_debate_pattern(
        self,
        query: str,
        initial_response: str,
        inputs: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the role-based debate pattern.
        
        Args:
            query: The user query to process.
            initial_response: The initial response to debate.
            inputs: Inputs from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary containing debate results.
            
        Raises:
            CollaborationError: If execution fails.
        """
        # Map models to roles
        role_to_models = await self._map_roles_to_models()
        
        # If no roles mapped to models, raise an error
        if not role_to_models:
            raise CollaborationError(
                f"Role-based debate '{self._phase_name}' has no valid role-model mappings"
            )
            
        # Start with the initial context
        debate_context = {
            "query": query,
            "initial_response": initial_response,
            "exchanges": []
        }
        
        # Add inputs from previous phases
        debate_context.update(inputs)
        
        # Execute debate rounds
        for round_num in range(self._rounds):
            logger.debug(f"Starting debate round {round_num + 1}/{self._rounds}")
            
            # Process each role in turn order
            for role in self._turn_order:
                # Skip if no models assigned to this role
                if role not in role_to_models or not role_to_models[role]:
                    logger.debug(f"Skipping role '{role}' (no models assigned)")
                    continue
                    
                # Get the template for this role
                template_name = self._get_template_for_role(role)
                if not template_name:
                    logger.warning(
                        f"No template found for role '{role}', using default template"
                    )
                    template_name = self._prompt_template
                    
                # If still no template, skip this role
                if not template_name:
                    logger.warning(f"Skipping role '{role}' (no template available)")
                    continue
                
                # Format prompt for this role
                try:
                    role_prompt = self.render_template(template_name, debate_context)
                except (ConfigurationError, KeyError) as e:
                    logger.warning(f"Failed to format prompt for role '{role}': {str(e)}")
                    continue
                    
                # Run models for this role
                models_for_role = role_to_models[role]
                role_results = await self._run_models(
                    prompt=role_prompt,
                    model_ids=models_for_role,
                    trace_collector=trace_collector
                )
                
                # Process outputs
                responses = {}
                for model_id, result in role_results.items():
                    responses[model_id] = result.get("text", "")
                    
                # Combine responses if multiple models for this role
                if len(responses) > 1:
                    combined_response = f"# Combined {role.capitalize()} Perspective\n\n"
                    for model_id, response in responses.items():
                        combined_response += f"## From {model_id}\n\n{response}\n\n"
                    role_response = combined_response
                elif responses:
                    # Just use the single response
                    role_response = list(responses.values())[0]
                else:
                    role_response = ""
                    
                # Add to exchanges
                if role_response:
                    debate_context["exchanges"].append({
                        "role": role,
                        "content": role_response,
                        "round": round_num + 1
                    })
                    
                    # Also add role-specific key in context
                    debate_context[f"{role}_perspective"] = role_response
                    
                    # Update for the next role
                    debate_context["latest_exchange"] = role_response
        
        # Determine the final output
        final_perspective = ""
        
        # Prioritize the synthesizer role if present
        if "synthesizer_perspective" in debate_context:
            final_perspective = debate_context["synthesizer_perspective"]
        # Otherwise use the last exchange
        elif debate_context["exchanges"]:
            final_perspective = debate_context["exchanges"][-1]["content"]
            
        # Calculate confidence (average from all model results across all roles)
        all_confidences = []
        for role, role_models in role_to_models.items():
            for model_id in role_models:
                # Find all results for this model
                for exchange in debate_context.get("exchanges", []):
                    if exchange["role"] == role:
                        # Look for confidence in the raw results
                        for round_results in exchange.get("raw_results", {}).values():
                            if isinstance(round_results, dict) and "confidence" in round_results:
                                if isinstance(round_results["confidence"], dict) and "combined" in round_results["confidence"]:
                                    all_confidences.append(round_results["confidence"]["combined"])
                                elif isinstance(round_results["confidence"], (float, int)):
                                    all_confidences.append(round_results["confidence"])
        
        confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.7
        
        # Return results
        return {
            "output": final_perspective,
            "final_perspective": final_perspective,
            "exchanges": debate_context["exchanges"],
            "initial_response": initial_response,
            "role_perspectives": {
                role: debate_context.get(f"{role}_perspective", "")
                for role in role_to_models.keys()
            },
            "confidence": confidence
        }
    
    async def _map_roles_to_models(self) -> Dict[str, List[str]]:
        """Map roles to model IDs based on configuration.
        
        Returns:
            Dictionary mapping role names to lists of model IDs.
            
        Raises:
            CollaborationError: If role mapping fails.
        """
        role_to_models: Dict[str, List[str]] = {}
        
        # Check if using explicit role assignments
        if not self._use_model_roles:
            # Use explicit assignments from configuration
            for role, model_ids in self._role_assignments.items():
                if isinstance(model_ids, str):
                    # Single model ID
                    role_to_models[role] = [model_ids]
                elif isinstance(model_ids, list):
                    # List of model IDs
                    role_to_models[role] = model_ids
                    
            return role_to_models
            
        # Otherwise use model roles
        for model_id in self._model_ids:
            try:
                model = self._model_manager.get_model(model_id)
                role = model.get_role()
                
                if role:
                    if role not in role_to_models:
                        role_to_models[role] = []
                    role_to_models[role].append(model_id)
            except Exception as e:
                logger.warning(f"Failed to get role for model '{model_id}': {str(e)}")
                
        return role_to_models
    
    def _get_template_for_role(self, role: str) -> Optional[str]:
        """Get the template name for a specific role.
        
        Args:
            role: The role to get template for.
            
        Returns:
            Template name or None if not found.
        """
        # First check role-specific templates in this phase's config
        if self._role_templates and role in self._role_templates:
            return self._role_templates[role]
            
        # Then check for a template with the role name
        role_template_name = f"{role}_template"
        if role_template_name in self._config:
            return self._config[role_template_name]
            
        # Then look for a standard template with the role name pattern
        standard_template = f"role_{role}"
        template = self._config_manager.get_template(standard_template)
        if template is not None:
            return standard_template
            
        # Fall back to the default prompt template
        return self._prompt_template
