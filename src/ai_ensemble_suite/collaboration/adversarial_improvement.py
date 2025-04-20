
"""Adversarial Improvement collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set, Tuple
import time
import asyncio

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class AdversarialImprovement(BaseCollaborationPhase):
    """Adversarial Improvement collaboration phase.
    
    Models improve a solution by actively seeking its weaknesses
    and addressing them in an iterative process.
    """
    
    def __init__(
        self, 
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the adversarial improvement phase.
        
        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
            
        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)
        
        # Get number of iterations
        self._iterations = self._config.get("iterations", 3)
        if self._iterations < 1:
            logger.warning(
                f"Invalid iterations ({self._iterations}) for phase '{phase_name}', "
                "using default: 3"
            )
            self._iterations = 3
            
        # Get template names
        self._initial_template = self._config.get("initial_template", "adversarial_initial")
        self._critique_template = self._config.get("critique_template", "adversarial_critique")
        self._improvement_template = self._config.get("improvement_template", "adversarial_improvement")
        
        # Get model roles
        self._initial_model = self._config.get("initial_model")
        self._critique_model = self._config.get("critique_model")
        self._improvement_model = self._config.get("improvement_model")
        
        # If roles not specified, assign them based on available models
        if not self._initial_model and self._model_ids:
            self._initial_model = self._model_ids[0]
            
        if not self._critique_model and len(self._model_ids) > 1:
            self._critique_model = self._model_ids[1]
        elif not self._critique_model:
            self._critique_model = self._initial_model
            
        if not self._improvement_model and len(self._model_ids) > 2:
            self._improvement_model = self._model_ids[2]
        elif not self._improvement_model:
            self._improvement_model = self._initial_model
            
        logger.debug(
            f"Initialized AdversarialImprovement phase '{phase_name}' with "
            f"{self._iterations} iterations"
        )
    
    async def execute(
        self, 
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Adversarial Improvement phase.
        
        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.
            
        Returns:
            Dictionary containing:
                output: The final improved response.
                context: Updated context with iteration history.
            
        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()
        
        try:
            # Get inputs from previous phases
            inputs = self._get_inputs_from_context(context)
            
            # Step 1: Generate initial solution
            initial_solution = await self._generate_initial_solution(
                query, inputs, trace_collector
            )
            
            # Initialize iteration history
            iterations = []
            current_solution = initial_solution
            
            # Step 2: Iterative improvement
            for i in range(self._iterations):
                logger.debug(f"Starting adversarial iteration {i + 1}/{self._iterations}")
                
                # Generate critique
                critique = await self._generate_critique(
                    query, current_solution, trace_collector
                )
                
                # Generate improvement
                improved_solution = await self._generate_improvement(
                    query, current_solution, critique, trace_collector
                )
                
                # Store iteration
                iterations.append({
                    "iteration": i + 1,
                    "solution": current_solution,
                    "critique": critique,
                    "improvement": improved_solution
                })
                
                # Update current solution for next iteration
                current_solution = improved_solution
                
            execution_time = time.time() - start_time
            
            # Log completion
            logger.info(
                f"AdversarialImprovement phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={"iterations": self._iterations}
            )
            
            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={"query": query, "inputs": inputs},
                    output_data={
                        "initial_solution": initial_solution,
                        "iterations": iterations,
                        "final_solution": current_solution
                    },
                    execution_time=execution_time,
                    phase_parameters=self._config
                )
                
            # Return results
            return {
                "output": current_solution,
                "initial_solution": initial_solution,
                "iterations": iterations,
                "confidence": 0.7 + (0.05 * min(self._iterations, 5))  # Confidence increases with iterations
            }
            
        except Exception as e:
            raise CollaborationError(
                f"AdversarialImprovement phase '{self._phase_name}' failed: {str(e)}"
            )
    
    async def _generate_initial_solution(
        self,
        query: str,
        inputs: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> str:
        """Generate the initial solution.
        
        Args:
            query: The user query.
            inputs: Inputs from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Initial solution text.
            
        Raises:
            CollaborationError: If generation fails.
        """
        logger.debug("Generating initial solution")
        
        # Check if initial solution came from previous phase
        if self._input_from and inputs:
            for source in self._input_from:
                if source in inputs:
                    source_data = inputs[source]
                    
                    if isinstance(source_data, str):
                        logger.debug(f"Using solution from previous phase: {source}")
                        return source_data
                    elif isinstance(source_data, dict) and "output" in source_data:
                        logger.debug(f"Using solution from previous phase: {source}")
                        return source_data["output"]
        
        # If no solution from previous phases, generate new one
        if not self._initial_model:
            raise CollaborationError("No model available for generating initial solution")
            
        # Format prompt for initial solution
        try:
            context = {"query": query, **inputs}
            initial_prompt = self.render_template(self._initial_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format initial solution prompt: {str(e)}")
            
        # Run initial model
        try:
            initial_result = await self._model_manager.run_inference(
                model_id=self._initial_model,
                prompt=initial_prompt
            )
            
            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_model_trace(
                    model_id=self._initial_model,
                    input_prompt=initial_prompt,
                    output=initial_result,
                    execution_time=initial_result.get("generation_time", 0),
                    parameters={"role": "initial_solution"}
                )
                
            return initial_result.get("text", "")
            
        except Exception as e:
            raise CollaborationError(f"Failed to generate initial solution: {str(e)}")
    
    async def _generate_critique(
        self,
        query: str,
        solution: str,
        trace_collector: Optional[TraceCollector] = None
    ) -> str:
        """Generate a critique of the current solution.
        
        Args:
            query: The user query.
            solution: The current solution.
            trace_collector: Optional trace collector.
            
        Returns:
            Critique text.
            
        Raises:
            CollaborationError: If critique generation fails.
        """
        logger.debug("Generating critique")
        
        if not self._critique_model:
            raise CollaborationError("No model available for generating critique")
            
        # Format prompt for critique
        try:
            context = {"query": query, "solution": solution}
            critique_prompt = self.render_template(self._critique_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format critique prompt: {str(e)}")
            
        # Run critique model
        try:
            critique_result = await self._model_manager.run_inference(
                model_id=self._critique_model,
                prompt=critique_prompt
            )
            
            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_model_trace(
                    model_id=self._critique_model,
                    input_prompt=critique_prompt,
                    output=critique_result,
                    execution_time=critique_result.get("generation_time", 0),
                    parameters={"role": "critique"}
                )
                
            return critique_result.get("text", "")
            
        except Exception as e:
            raise CollaborationError(f"Failed to generate critique: {str(e)}")
    
    async def _generate_improvement(
        self,
        query: str,
        solution: str,
        critique: str,
        trace_collector: Optional[TraceCollector] = None
    ) -> str:
        """Generate an improved solution based on critique.
        
        Args:
            query: The user query.
            solution: The current solution.
            critique: The critique of the current solution.
            trace_collector: Optional trace collector.
            
        Returns:
            Improved solution text.
            
        Raises:
            CollaborationError: If improvement generation fails.
        """
        logger.debug("Generating improvement")
        
        if not self._improvement_model:
            raise CollaborationError("No model available for generating improvement")
            
        # Format prompt for improvement
        try:
            context = {"query": query, "solution": solution, "critique": critique}
            improvement_prompt = self.render_template(self._improvement_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format improvement prompt: {str(e)}")
            
        # Run improvement model
        try:
            improvement_result = await self._model_manager.run_inference(
                model_id=self._improvement_model,
                prompt=improvement_prompt
            )
            
            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_model_trace(
                    model_id=self._improvement_model,
                    input_prompt=improvement_prompt,
                    output=improvement_result,
                    execution_time=improvement_result.get("generation_time", 0),
                    parameters={"role": "improvement"}
                )
                
            return improvement_result.get("text", "")
            
        except Exception as e:
            raise CollaborationError(f"Failed to generate improvement: {str(e)}")
