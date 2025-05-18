# ai_ensemble_suite/collaboration/structured_debate/base_debate.py

"""Base class for structured debate collaboration phases."""

from typing import Dict, Any, Optional, List, Set, Tuple
import time
import re

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class BaseDebate(BaseCollaborationPhase):
    """Base class for structured debate collaboration phases.
    
    Provides common functionality for different debate patterns.
    """
    
    def __init__(
        self, 
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the base debate phase.
        
        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
            
        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)
        
        # Get debate-specific configuration
        self._subtype = self._config.get("subtype", "critique")
        
        # Get debate rounds (default to 1)
        self._rounds = self._config.get("rounds", 1)
        
        # Validate rounds
        if not isinstance(self._rounds, int) or self._rounds < 1:
            logger.warning(
                f"Invalid rounds value for phase '{phase_name}', defaulting to 1"
            )
            self._rounds = 1
            
        # Track rounds in debugging
        logger.debug(
            f"Initialized debate phase '{phase_name}' with subtype '{self._subtype}' "
            f"and {self._rounds} rounds"
        )
    
    async def execute(
        self, 
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the debate phase.
        
        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.
            
        Returns:
            Dictionary containing:
                output: The debate output (typically from the final round).
                context: Updated context including intermediate debate exchanges.
            
        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()
        
        try:
            # Get inputs from previous phases
            inputs = self._get_inputs_from_context(context)
            
            # Validate inputs
            if not inputs and self._input_from:
                raise CollaborationError(
                    f"Debate phase '{self._phase_name}' requires inputs from "
                    f"previous phases: {', '.join(self._input_from)}"
                )
            
            # Get or create the initial response that will be critiqued
            initial_response = self._get_initial_response(inputs)
            if not initial_response:
                raise CollaborationError(
                    f"Debate phase '{self._phase_name}' could not determine "
                    "an initial response to critique"
                )
                
            # Execute the appropriate debate pattern
            debate_results = await self._execute_debate_pattern(
                query, 
                initial_response, 
                inputs,
                trace_collector
            )
            
            execution_time = time.time() - start_time
            
            # Log completion
            logger.info(
                f"Debate phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={
                    "debate_type": self._subtype,
                    "rounds": self._rounds,
                    "phase": self._phase_name
                }
            )
            
            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={
                        "query": query, 
                        "initial_response": initial_response,
                        "inputs": inputs
                    },
                    output_data=debate_results,
                    execution_time=execution_time,
                    phase_parameters=self._config
                )
            
            # Return results - ensure we include an "output" key for consistent interface
            if "output" not in debate_results:
                # Default to the final critique or perspective as the output
                if "critique" in debate_results:
                    debate_results["output"] = debate_results["critique"]
                elif "final_perspective" in debate_results:
                    debate_results["output"] = debate_results["final_perspective"]
                elif "exchanges" in debate_results and debate_results["exchanges"]:
                    # Take the last exchange as the output
                    debate_results["output"] = debate_results["exchanges"][-1]["content"]
                else:
                    # Fallback to empty string
                    debate_results["output"] = ""
                    
            return debate_results
            
        except Exception as e:
            raise CollaborationError(
                f"Debate phase '{self._phase_name}' failed: {str(e)}"
            )
    
    def _get_initial_response(self, inputs: Dict[str, Any]) -> str:
        """Get the initial response that will be critiqued.
        
        Args:
            inputs: Inputs from previous phases.
            
        Returns:
            The initial response as a string.
        """
        # Check input_from configuration to determine source
        if not self._input_from:
            return ""
            
        # Typically the first input source is the initial response
        source_phase = self._input_from[0]
        
        # Check if the source phase exists in inputs
        if source_phase not in inputs:
            logger.warning(
                f"Initial response source '{source_phase}' not found in inputs"
            )
            return ""
            
        # Extract the response from the source phase
        source = inputs[source_phase]
        
        # Handle different input formats
        if isinstance(source, str):
            # Direct string response
            return source
        elif isinstance(source, dict):
            # Dictionary response - try to get the output field
            if "output" in source:
                return source["output"]
            # If no output field, try to find any string field
            for key, value in source.items():
                if isinstance(value, str):
                    return value
                    
        # Fallback to empty string if no suitable response found
        logger.warning(
            f"Could not extract initial response from source '{source_phase}'"
        )
        return ""
    
    @staticmethod
    def _extract_key_points(text: str, max_points: int = 5) -> List[str]:
        """Extract key points from a text.
        
        This is a simple implementation that finds sentences containing strong indicators.
        
        Args:
            text: The text to extract points from.
            max_points: Maximum number of points to extract.
            
        Returns:
            List of key points extracted from the text.
        """
        # List of indicator phrases for key points
        indicators = [
            r'important',
            r'significant',
            r'key',
            r'critical',
            r'essential',
            r'main',
            r'primary',
            r'fundamental',
            r'primarily',
            r'notably',
            r'specifically',
            r'particularly',
            r'must',
            r'should',
            r'need to',
            r'\d+\.',  # Numbered points like "1."
            r'firstly',
            r'secondly',
            r'lastly',
            r'finally',
            r'in conclusion',
        ]
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter sentences containing indicators
        key_sentences = []
        for sentence in sentences:
            for indicator in indicators:
                if re.search(r'\b' + indicator + r'\b', sentence, re.IGNORECASE):
                    key_sentences.append(sentence.strip())
                    break
                    
        # Limit to max_points
        points = key_sentences[:max_points]
        
        # If no points found, take the first few sentences
        if not points and sentences:
            points = sentences[:min(max_points, len(sentences))]
            
        return points
    
    async def _execute_debate_pattern(
        self,
        query: str,
        initial_response: str,
        inputs: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the specific debate pattern defined by the subtype.
        
        This method should be implemented by subclasses.
        
        Args:
            query: The user query to process.
            initial_response: The initial response to critique.
            inputs: Inputs from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary containing debate results.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            f"Debate pattern '{self._subtype}' not implemented in base class"
        )
