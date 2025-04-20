# src/ai_ensemble_suite/collaboration/chain_of_thought.py

"""Chain of Thought Branching collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set, Tuple
import time
import asyncio

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class ChainOfThoughtBranching(BaseCollaborationPhase):
    """Chain of Thought Branching collaboration phase.

    Models trace through multiple reasoning paths, then evaluate
    and select the most promising path.
    """

    def __init__(
            self,
            model_manager: "ModelManager",
            config_manager: "ConfigManager",
            phase_name: str
    ) -> None:
        """Initialize the chain of thought branching phase.

        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.

        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)

        # Get branch count
        self._branch_count = self._config.get("branch_count", 3)
        if self._branch_count < 1:
            logger.warning(
                f"Invalid branch_count ({self._branch_count}) for phase '{phase_name}', "
                "using default: 3"
            )
            self._branch_count = 3

        # Get branch depth
        self._branch_depth = self._config.get("branch_depth", 2)
        if self._branch_depth < 1:
            logger.warning(
                f"Invalid branch_depth ({self._branch_depth}) for phase '{phase_name}', "
                "using default: 2"
            )
            self._branch_depth = 2

        # Get template names
        self._initial_template = self._config.get("initial_template", "cot_initial")
        self._branch_template = self._config.get("branch_template", "cot_branch")
        self._evaluation_template = self._config.get("evaluation_template", "cot_evaluation")

        # Get evaluation model (default to first model)
        self._evaluation_model = self._config.get("evaluation_model")
        if not self._evaluation_model and self._model_ids:
            self._evaluation_model = self._model_ids[0]

        logger.debug(
            f"Initialized ChainOfThoughtBranching phase '{phase_name}' with "
            f"{self._branch_count} branches of depth {self._branch_depth}"
        )

    async def execute(
            self,
            query: str,
            context: Dict[str, Any],
            trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Chain of Thought Branching phase.

        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.

        Returns:
            Dictionary containing:
                output: The final response after evaluating reasoning branches.
                context: Updated context with branch information.

        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()

        try:
            # Get inputs from previous phases
            inputs = self._get_inputs_from_context(context)

            # Step 1: Generate initial thoughts
            initial_thoughts = await self._generate_initial_thoughts(
                query, context, trace_collector
            )

            # Step 2: Develop reasoning branches
            branches = await self._develop_branches(
                query, initial_thoughts, context, trace_collector
            )

            # Step 3: Evaluate branches
            evaluation_results = await self._evaluate_branches(
                query, branches, context, trace_collector
            )

            execution_time = time.time() - start_time

            # Log completion
            logger.info(
                f"ChainOfThoughtBranching phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={"branches": len(branches), "depth": self._branch_depth}
            )

            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "context": context},
                    output_data={
                        "initial_thoughts": initial_thoughts,
                        "branches": branches,
                        "evaluation_results": evaluation_results
                    },
                    execution_time=execution_time,
                    phase_parameters=self._config
                )

            # Get final output
            final_output = evaluation_results.get("best_branch_conclusion", "")
            if not final_output:
                final_output = evaluation_results.get("evaluation_text", "")

            # Calculate confidence
            confidence = evaluation_results.get("confidence", 0.7)

            # Return results
            return {
                "output": final_output,
                "initial_thoughts": initial_thoughts,
                "branches": branches,
                "evaluation_results": evaluation_results,
                "confidence": confidence
            }

        except Exception as e:
            raise CollaborationError(
                f"ChainOfThoughtBranching phase '{self._phase_name}' failed: {str(e)}"
            )

    async def _generate_initial_thoughts(
            self,
            query: str,
            context: Dict[str, Any],
            trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Generate initial thoughts on the query.

        Args:
            query: The user query.
            context: Context information from previous phases.
            trace_collector: Optional trace collector.

        Returns:
            Dictionary with initial thoughts.

        Raises:
            CollaborationError: If generation fails.
        """
        logger.debug("Generating initial thoughts")

        # Get inputs from previous phases
        inputs = self._get_inputs_from_context(context)

        # Format initial prompt
        try:
            context = {"query": query, **inputs}
            initial_prompt = self.render_template(self._initial_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format initial thoughts prompt: {str(e)}")

        # Select model for initial thoughts (first model)
        initial_model = self._model_ids[0] if self._model_ids else None
        if not initial_model:
            raise CollaborationError("No models available for generating initial thoughts")

        # Run model
        try:
            initial_result = await self._model_manager.run_inference(
                model_id=initial_model,
                prompt=initial_prompt
            )

            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_model_trace(
                    model_id=initial_model,
                    input_prompt=initial_prompt,
                    output=initial_result,
                    execution_time=initial_result.get("generation_time", 0),
                    parameters={}
                )

            return {
                "initial_thoughts": initial_result.get("text", ""),
                "model_id": initial_model,
                "raw_result": initial_result
            }

        except Exception as e:
            raise CollaborationError(f"Failed to generate initial thoughts: {str(e)}")

    async def _develop_branches(
            self,
            query: str,
            initial_thoughts: Dict[str, Any],
            context: Dict[str, Any],
            trace_collector: Optional[TraceCollector] = None
    ) -> List[Dict[str, Any]]:
        """Develop multiple reasoning branches from initial thoughts.

        Args:
            query: The user query.
            initial_thoughts: Results from initial thoughts step.
            context: Context information from previous phases.
            trace_collector: Optional trace collector.

        Returns:
            List of branch dictionaries.

        Raises:
            CollaborationError: If branch development fails.
        """
        logger.debug(f"Developing {self._branch_count} reasoning branches")

        initial_text = initial_thoughts.get("initial_thoughts", "")
        branches = []

        # Distribute models across branches
        model_assignments = []
        for i in range(self._branch_count):
            model_idx = i % len(self._model_ids)
            model_assignments.append(self._model_ids[model_idx])

        # Create initial branches
        for branch_idx in range(self._branch_count):
            branch = {
                "branch_id": f"branch_{branch_idx + 1}",
                "initial_thoughts": initial_text,
                "steps": [],
                "model_id": model_assignments[branch_idx]
            }
            branches.append(branch)

        # Develop each branch step by step
        for step_idx in range(self._branch_depth):
            logger.debug(f"Developing step {step_idx + 1}/{self._branch_depth}")

            for branch in branches:
                # Format branch prompt
                try:
                    # Gather previous steps for context
                    previous_steps = ""
                    for prev_step in branch["steps"]:
                        previous_steps += f"\nStep {prev_step['step_number']}: {prev_step['content']}\n"

                    context = {
                        "query": query,
                        "initial_thoughts": branch["initial_thoughts"],
                        "previous_steps": previous_steps,
                        "step_number": step_idx + 1
                    }
                    branch_prompt = self.render_template(self._branch_template, context)
                except (ConfigurationError, KeyError) as e:
                    logger.warning(
                        f"Failed to format prompt for branch {branch['branch_id']}, "
                        f"step {step_idx + 1}: {str(e)}"
                    )
                    continue

                # Run model for this branch step
                try:
                    model_id = branch["model_id"]
                    step_result = await self._model_manager.run_inference(
                        model_id=model_id,
                        prompt=branch_prompt
                    )

                    # Add step to branch
                    branch["steps"].append({
                        "step_number": step_idx + 1,
                        "content": step_result.get("text", ""),
                        "model_id": model_id
                    })

                    # Add trace if collector is provided
                    if trace_collector:
                        trace_collector.add_model_trace(
                            model_id=model_id,
                            input_prompt=branch_prompt,
                            output=step_result,
                            execution_time=step_result.get("generation_time", 0),
                            parameters={
                                "branch_id": branch["branch_id"],
                                "step_number": step_idx + 1
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to develop step {step_idx + 1} for branch {branch['branch_id']}: {str(e)}"
                    )

        # Extract conclusions from final steps
        for branch in branches:
            if branch["steps"]:
                final_step = branch["steps"][-1]
                branch["conclusion"] = final_step["content"]
            else:
                branch["conclusion"] = "No conclusion reached."

        return branches

    async def _evaluate_branches(
            self,
            query: str,
            branches: List[Dict[str, Any]],
            context: Dict[str, Any],
            trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Evaluate reasoning branches and select the best one.

        Args:
            query: The user query.
            branches: List of developed reasoning branches.
            context: Context information from previous phases.
            trace_collector: Optional trace collector.

        Returns:
            Dictionary with evaluation results.

        Raises:
            CollaborationError: If evaluation fails.
        """
        logger.debug("Evaluating reasoning branches")

        # Check if evaluation model is available
        if not self._evaluation_model:
            raise CollaborationError("No evaluation model specified")

        # Format branches for evaluation
        formatted_branches = ""
        for branch in branches:
            formatted_branches += f"\n\n## Branch {branch['branch_id']}\n\n"
            formatted_branches += f"Initial thoughts:\n{branch['initial_thoughts']}\n\n"

            for step in branch["steps"]:
                formatted_branches += f"Step {step['step_number']}: {step['content']}\n\n"

            formatted_branches += f"Conclusion: {branch.get('conclusion', '')}\n"

        # Format evaluation prompt
        try:
            context = {
                "query": query,
                "branches": formatted_branches
            }
            evaluation_prompt = self.render_template(self._evaluation_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format evaluation prompt: {str(e)}")

        # Run evaluation model
        try:
            evaluation_result = await self._model_manager.run_inference(
                model_id=self._evaluation_model,
                prompt=evaluation_prompt
            )

            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_model_trace(
                    model_id=self._evaluation_model,
                    input_prompt=evaluation_prompt,
                    output=evaluation_result,
                    execution_time=evaluation_result.get("generation_time", 0),
                    parameters={"role": "evaluator"}
                )

            # Default values in case evaluation_result is None
            evaluation_text = ""
            if evaluation_result is not None:
                evaluation_text = evaluation_result.get("text", "")

            # Try to identify best branch
            best_branch_id = None
            for branch in branches:
                branch_id = branch["branch_id"]
                if (evaluation_text and
                        (f"best branch is {branch_id}" in evaluation_text.lower() or
                         f"select {branch_id}" in evaluation_text.lower() or
                         f"choose {branch_id}" in evaluation_text.lower())):
                    best_branch_id = branch_id
                    break

            # If explicit selection not found, look for branch numbers
            if not best_branch_id and evaluation_text:
                for branch in branches:
                    branch_num = branch["branch_id"].split("_")[-1]
                    if (f"branch {branch_num} provides" in evaluation_text.lower() or
                            f"branch {branch_num} is" in evaluation_text.lower()):
                        best_branch_id = branch["branch_id"]
                        break

            # Get best branch conclusion
            best_branch_conclusion = ""
            if best_branch_id:
                for branch in branches:
                    if branch["branch_id"] == best_branch_id:
                        best_branch_conclusion = branch.get("conclusion", "")
                        break
            else:
                # If no best branch identified, use the evaluation text itself or first branch
                if evaluation_text:
                    best_branch_conclusion = evaluation_text
                elif branches:
                    # Fallback to first branch if no evaluation text
                    best_branch_id = branches[0]["branch_id"]
                    best_branch_conclusion = branches[0].get("conclusion", "")

            # Safely extract confidence value
            confidence_score = 0.7  # Default confidence
            if evaluation_result is not None and isinstance(evaluation_result, dict):
                confidence_data = evaluation_result.get("confidence")
                if isinstance(confidence_data, dict) and "combined" in confidence_data:
                    confidence_score = confidence_data.get("combined", 0.7)
                elif isinstance(confidence_data, (float, int)):
                    confidence_score = float(confidence_data)

            return {
                "evaluation_text": evaluation_text,
                "best_branch_id": best_branch_id,
                "best_branch_conclusion": best_branch_conclusion,
                "model_id": self._evaluation_model,
                "raw_result": evaluation_result,
                "confidence": confidence_score
            }

        except Exception as e:
            # More robust error handling to provide default values even on failure
            logger.error(f"Error during branch evaluation: {str(e)}")

            # Provide default fallback values
            best_branch_id = branches[0]["branch_id"] if branches else None
            best_branch_conclusion = branches[0].get("conclusion", "") if branches else ""

            return {
                "evaluation_text": f"Evaluation failed: {str(e)}",
                "best_branch_id": best_branch_id,
                "best_branch_conclusion": best_branch_conclusion,
                "model_id": self._evaluation_model,
                "raw_result": None,
                "confidence": 0.5,  # Lower confidence due to failure
                "error": str(e)
            }
