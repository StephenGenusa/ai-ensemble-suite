
"""Role-Based Workflow collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set, Tuple
import time
import asyncio

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class RoleBasedWorkflow(BaseCollaborationPhase):
    """Role-Based Workflow collaboration phase.
    
    Models function in specialized roles like researcher, analyst,
    and writer to create a structured workflow for complex tasks.
    """
    
    def __init__(
        self, 
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the role-based workflow phase.
        
        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
            
        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)
        
        # Get workflow steps
        self._workflow_steps = self._config.get("workflow_steps", [])
        
        # If no workflow steps defined, use default workflow
        if not self._workflow_steps:
            self._workflow_steps = [
                {
                    "role": "researcher",
                    "task": "research",
                    "description": "Gather comprehensive information on the topic",
                    "template": "role_researcher"
                },
                {
                    "role": "analyst",
                    "task": "analyze",
                    "description": "Analyze the research and identify key insights",
                    "template": "role_analyst"
                },
                {
                    "role": "writer",
                    "task": "write",
                    "description": "Create a well-structured response based on the analysis",
                    "template": "role_writer"
                },
                {
                    "role": "editor",
                    "task": "edit",
                    "description": "Review and refine the written content for clarity and accuracy",
                    "template": "role_editor"
                }
            ]
            logger.debug(
                f"No workflow steps defined for phase '{phase_name}', using default workflow"
            )
            
        # Validate workflow steps
        for i, step in enumerate(self._workflow_steps):
            if "role" not in step:
                step["role"] = f"role_{i+1}"
                logger.warning(f"Step {i+1} missing role, using '{step['role']}'")
                
            if "task" not in step:
                step["task"] = f"task_{i+1}"
                logger.warning(f"Step {i+1} missing task, using '{step['task']}'")
                
            if "template" not in step:
                step["template"] = f"role_{step['role']}"
                logger.warning(f"Step {i+1} missing template, using '{step['template']}'")
            
        logger.debug(
            f"Initialized RoleBasedWorkflow phase '{phase_name}' with "
            f"{len(self._workflow_steps)} workflow steps"
        )

    async def execute(
            self,
            query: str,
            context: Dict[str, Any],
            trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Role-Based Workflow phase."""
        start_time = time.time()

        try:
            # Get inputs from previous phases
            inputs = self._get_inputs_from_context(context)

            # Map models to roles
            role_assignments = self._assign_models_to_roles()

            # Initialize workflow context
            workflow_context = {
                "query": query,
                "steps": {},
                "final_output": "",
                **inputs
            }

            # Execute workflow steps in sequence
            for step_idx, step in enumerate(self._workflow_steps):
                role = step["role"]
                task = step["task"]
                template_name = step["template"]

                logger.debug(f"Executing workflow step {step_idx + 1}: {role} - {task}")

                # Get model for this role
                model_id = role_assignments.get(role)
                if not model_id:
                    logger.warning(f"No model assigned for role '{role}', skipping step")
                    continue

                # Format prompt for this step - IMPROVED VARIABLE HANDLING
                try:
                    # Start with a clean set of variables for this step
                    prompt_vars = {}

                    # Always include the query
                    prompt_vars["query"] = query

                    # Include inputs from previous phases
                    prompt_vars.update(inputs)

                    # Handle input_from parameter for this step
                    input_from_list = step.get("input_from", [])
                    if isinstance(input_from_list, str):
                        input_from_list = [input_from_list]

                    # If no input_from specified but not first step, use all previous steps
                    if not input_from_list and step_idx > 0:
                        input_from_list = [prev_step["task"] for prev_step in self._workflow_steps[:step_idx]]

                    # Add the specified inputs from workflow context AS DIRECT VARIABLES
                    for input_task in input_from_list:
                        if input_task in workflow_context["steps"]:
                            prompt_vars[input_task] = workflow_context["steps"][input_task]
                        else:
                            logger.warning(f"Step '{task}' requires input from '{input_task}', but it is not available")

                    # Format the prompt with the prepared variables
                    step_prompt = self.render_template(template_name, prompt_vars)

                except (ConfigurationError, KeyError) as e:
                    logger.warning(f"Failed to format prompt for step '{task}': {str(e)}")
                    continue

                # Run model with the formatted prompt
                try:
                    step_result = await self._model_manager.run_inference(
                        model_id=model_id,
                        prompt=step_prompt
                    )

                    # Add trace if collector is provided
                    if trace_collector:
                        trace_collector.add_model_trace(
                            model_id=model_id,
                            input_prompt=step_prompt,
                            output=step_result,
                            execution_time=step_result.get("generation_time", 0),
                            parameters={"role": role, "task": task}
                        )

                    # Extract output
                    step_output = step_result.get("text", "")

                    # Store step results in workflow context by task name
                    workflow_context["steps"][task] = step_output

                    # Also store by role name for easier reference
                    workflow_context[role] = step_output

                    # Update final output (always use the latest step)
                    workflow_context["final_output"] = step_output

                except Exception as e:
                    logger.warning(f"Failed to execute step '{task}': {str(e)}")

            execution_time = time.time() - start_time
            
            # Log completion
            logger.info(
                f"RoleBasedWorkflow phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={"steps": len(self._workflow_steps)}
            )
            
            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "inputs": inputs},
                    output_data=workflow_context,
                    execution_time=execution_time,
                    phase_parameters=self._config
                )
                
            # Get final output
            final_output = workflow_context.get("final_output", "")
            if not final_output and workflow_context["steps"]:
                # Use the last step's output
                final_step = list(workflow_context["steps"].keys())[-1]
                final_output = workflow_context["steps"][final_step]
                
            # Calculate confidence (average from step results that have confidence)
            confidence_values = []
            for step_name, step_result in workflow_context.get("steps", {}).items():
                if isinstance(step_result, dict) and "confidence" in step_result:
                    if isinstance(step_result["confidence"], dict) and "combined" in step_result["confidence"]:
                        confidence_values.append(step_result["confidence"]["combined"])
                    elif isinstance(step_result["confidence"], (float, int)):
                        confidence_values.append(step_result["confidence"])
                        
            confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.7
                
            # Return results
            return {
                "output": final_output,
                "workflow_steps": workflow_context["steps"],
                "confidence": confidence
            }
            
        except Exception as e:
            raise CollaborationError(
                f"RoleBasedWorkflow phase '{self._phase_name}' failed: {str(e)}"
            )
    
    def _assign_models_to_roles(self) -> Dict[str, str]:
        """Assign models to roles in the workflow.
        
        Returns:
            Dictionary mapping roles to model IDs.
        """
        # Check for explicit role assignments in configuration
        role_assignments = self._config.get("role_assignments", {})
        
        # If explicit assignments provided, use them
        if role_assignments:
            return role_assignments
            
        # Otherwise, try to use model roles
        role_to_model: Dict[str, str] = {}
        
        # First try to match roles based on model metadata
        for model_id in self._model_ids:
            try:
                model = self._model_manager.get_model(model_id)
                model_role = model.get_role()
                
                if model_role:
                    # Check if this role is used in the workflow
                    for step in self._workflow_steps:
                        if step["role"] == model_role and model_role not in role_to_model:
                            role_to_model[model_role] = model_id
                            break
            except Exception:
                pass
                
        # For any unassigned roles, distribute remaining models
        available_models = [m for m in self._model_ids if m not in role_to_model.values()]
        model_idx = 0
        
        for step in self._workflow_steps:
            role = step["role"]
            
            if role not in role_to_model and available_models:
                role_to_model[role] = available_models[model_idx % len(available_models)]
                model_idx += 1
                
        # Log assignments
        logger.debug(f"Role assignments: {role_to_model}")
        
        return role_to_model
