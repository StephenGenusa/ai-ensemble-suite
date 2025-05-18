# ai_ensemble_suite/collaboration/competitive_evaluation.py

"""Competitive Evaluation collaboration phase implementation."""

from typing import Dict, Any, Optional, List, Set, Tuple
import time
import re

from ai_ensemble_suite.collaboration.base import BaseCollaborationPhase
from ai_ensemble_suite.exceptions import CollaborationError, ConfigurationError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.tracing import TraceCollector


class CompetitiveEvaluation(BaseCollaborationPhase):
    """Competitive Evaluation collaboration phase.
    
    Models are pitted against each other in a competition, with each evaluating
    the others' outputs and a winner being determined.
    """
    
    def __init__(
        self, 
        model_manager: "ModelManager",
        config_manager: "ConfigManager",
        phase_name: str
    ) -> None:
        """Initialize the competitive evaluation phase.
        
        Args:
            model_manager: The ModelManager instance.
            config_manager: The ConfigManager instance.
            phase_name: The name of the phase for configuration lookup.
            
        Raises:
            ConfigurationError: If phase configuration is invalid.
        """
        super().__init__(model_manager, config_manager, phase_name)
        
        # Get evaluation criteria
        self._evaluation_criteria = self._config.get("evaluation_criteria", [
            "accuracy", "reasoning", "completeness", "clarity"
        ])
        
        # Get competitors (default to all models)
        self._competitors = self._config.get("competitors", self._model_ids)
        
        # Get judge (default to first model)
        self._judge = self._config.get("judge")
        if not self._judge and self._model_ids:
            self._judge = self._model_ids[0]
            
        # Get template names
        self._competitor_template = self._config.get("competitor_template", "competitor")
        self._evaluation_template = self._config.get("evaluation_template", "evaluation")
        
        logger.debug(
            f"Initialized CompetitiveEvaluation phase '{phase_name}' with "
            f"{len(self._competitors)} competitors and judge: {self._judge}"
        )
    
    async def execute(
        self, 
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, Any]:
        """Execute the Competitive Evaluation phase.
        
        Args:
            query: The user query to process.
            context: Context information from previous phases.
            trace_collector: Optional trace collector for gathering execution details.
            
        Returns:
            Dictionary containing:
                output: The winning response.
                context: Updated context with competition results.
            
        Raises:
            CollaborationError: If phase execution fails.
        """
        start_time = time.time()
        
        try:
            # Step 1: Have competitors generate responses
            competitor_responses = await self._generate_competitor_responses(
                query, context, trace_collector
            )
            
            # Step 2: Judge evaluates the responses
            evaluation_results = await self._evaluate_responses(
                query, competitor_responses, context, trace_collector
            )
            
            # Step 3: Determine the winner
            winner, scores = self._determine_winner(
                competitor_responses, evaluation_results
            )
            
            execution_time = time.time() - start_time
            
            # Log completion
            logger.info(
                f"CompetitiveEvaluation phase '{self._phase_name}' completed in {execution_time:.2f}s",
                extra={"winner": winner, "competitors": len(competitor_responses)}
            )
            
            # Add phase trace if collector is provided
            if trace_collector:
                trace_collector.add_phase_trace(
                    phase_name=self._phase_name,
                    input_data={
                        "query": query, 
                        "context": context
                    },
                    output_data={
                        "competitor_responses": competitor_responses,
                        "evaluation_results": evaluation_results,
                        "winner": winner,
                        "scores": scores
                    },
                    execution_time=execution_time,
                    phase_parameters=self._config
                )
                
            # Get winning response
            winning_response = competitor_responses.get(winner, "")
            
            # Return results
            return {
                "output": winning_response,
                "winner": winner,
                "competitor_responses": competitor_responses,
                "evaluation_results": evaluation_results,
                "scores": scores,
                "confidence": scores.get(winner, 0.7)  # Use winner's score as confidence
            }
            
        except Exception as e:
            raise CollaborationError(
                f"CompetitiveEvaluation phase '{self._phase_name}' failed: {str(e)}"
            )
    
    async def _generate_competitor_responses(
        self,
        query: str,
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, str]:
        """Generate responses from all competitors.
        
        Args:
            query: The user query.
            context: Context information from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary mapping competitor IDs to their responses.
            
        Raises:
            CollaborationError: If generation fails.
        """
        logger.debug(f"Generating responses from {len(self._competitors)} competitors")
        
        # Get inputs from previous phases
        inputs = self._get_inputs_from_context(context)
        
        # Get competitor template
        if not self._competitor_template:
            logger.warning(
                f"No competitor template specified, using {self._prompt_template or 'single_query'}"
            )
            self._competitor_template = self._prompt_template or "single_query"
            
        # Format prompt
        try:
            context = {"query": query, **inputs}
            competitor_prompt = self.render_template(self._competitor_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format competitor prompt: {str(e)}")
            
        # Run competitors
        model_results = await self._run_models(
            prompt=competitor_prompt,
            model_ids=self._competitors,
            trace_collector=trace_collector
        )
        
        # Extract responses
        competitor_responses = {}
        for model_id, result in model_results.items():
            competitor_responses[model_id] = result.get("text", "")
            
        return competitor_responses
    
    async def _evaluate_responses(
        self,
        query: str,
        competitor_responses: Dict[str, str],
        context: Dict[str, Any],
        trace_collector: Optional[TraceCollector] = None
    ) -> Dict[str, str]:
        """Evaluate competitor responses using the judge.
        
        Args:
            query: The user query.
            competitor_responses: Responses from competitors.
            context: Context information from previous phases.
            trace_collector: Optional trace collector.
            
        Returns:
            Dictionary with evaluation results.
            
        Raises:
            CollaborationError: If evaluation fails.
        """
        logger.debug(f"Evaluating competitor responses using judge: {self._judge}")
        
        # Check if judge is available
        if not self._judge:
            raise CollaborationError("No judge specified for evaluation")
            
        # Format the responses for evaluation
        formatted_responses = ""
        for model_id, response in competitor_responses.items():
            formatted_responses += f"\n\n## Response from Competitor {model_id}\n\n{response}"
            
        # Create criteria string
        criteria_str = ", ".join(self._evaluation_criteria)
        
        # Format evaluation prompt
        try:
            context = {
                "query": query,
                "responses": formatted_responses,
                "criteria": criteria_str
            }
            evaluation_prompt = self.render_template(self._evaluation_template, context)
        except (ConfigurationError, KeyError) as e:
            raise CollaborationError(f"Failed to format evaluation prompt: {str(e)}")
            
        # Run judge
        try:
            judge_results = await self._model_manager.run_inference(
                model_id=self._judge,
                prompt=evaluation_prompt
            )
            
            # Add trace if collector is provided
            if trace_collector:
                trace_collector.add_model_trace(
                    model_id=self._judge,
                    input_prompt=evaluation_prompt,
                    output=judge_results,
                    execution_time=judge_results.get("generation_time", 0),
                    parameters={}
                )
                
            return {
                "evaluation_text": judge_results.get("text", ""),
                "judge_id": self._judge,
                "raw_result": judge_results
            }
            
        except Exception as e:
            raise CollaborationError(f"Failed to run judge evaluation: {str(e)}")
    
    def _determine_winner(
        self,
        competitor_responses: Dict[str, str],
        evaluation_results: Dict[str, Any]
    ) -> Tuple[str, Dict[str, float]]:
        """Determine the winner based on evaluation results.
        
        Args:
            competitor_responses: Responses from competitors.
            evaluation_results: Evaluation results from judge.
            
        Returns:
            Tuple of (winner_id, score_dict) where score_dict maps
            competitor IDs to their scores.
            
        Raises:
            CollaborationError: If winner determination fails.
        """
        evaluation_text = evaluation_results.get("evaluation_text", "")
        
        # Parse scores for each competitor
        scores: Dict[str, float] = {}
        
        for competitor_id in competitor_responses.keys():
            # Look for score patterns for this competitor
            score_pattern = rf'(?:Competitor|Model|Response)\s+{competitor_id}[^\d]*?(\d+(?:\.\d+)?)/10'
            match = re.search(score_pattern, evaluation_text)
            
            if match:
                # Found explicit score
                scores[competitor_id] = float(match.group(1)) / 10.0  # Normalize to 0-1 range
            else:
                # Try alternative pattern (looking for overall rating)
                alt_pattern = rf'(?:Competitor|Model|Response)\s+{competitor_id}.*?overall.*?(\d+(?:\.\d+)?)/10'
                match = re.search(alt_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
                
                if match:
                    scores[competitor_id] = float(match.group(1)) / 10.0
                else:
                    # No explicit score found, try to determine from text
                    competitor_mentions = len(re.findall(
                        rf'(?:Competitor|Model|Response)\s+{competitor_id}', 
                        evaluation_text
                    ))
                    
                    # Set a default score based on mentions
                    scores[competitor_id] = 0.5 + (0.1 * min(competitor_mentions, 5))
        
        # Look for explicit winner declaration
        winner_pattern = r'(?:winner|best|highest)[^\w]+(Competitor|Model|Response)\s+([a-zA-Z0-9_]+)'
        winner_match = re.search(winner_pattern, evaluation_text, re.IGNORECASE)
        
        if winner_match:
            winner_id = winner_match.group(2)
            # Ensure winner_id is a valid competitor
            if winner_id in competitor_responses:
                # Ensure winner has the highest score
                scores[winner_id] = max(scores.values()) + 0.1
            else:
                logger.warning(f"Declared winner '{winner_id}' is not a valid competitor")
        
        # If no scores were found, assign default scores
        if not scores:
            logger.warning("No scores could be extracted from evaluation text")
            scores = {competitor_id: 0.5 for competitor_id in competitor_responses.keys()}
            
        # Determine winner (highest score)
        winner = max(scores.items(), key=lambda x: x[1])[0] if scores else ""
        
        if not winner and competitor_responses:
            # Fallback to first competitor
            winner = next(iter(competitor_responses.keys()))
            
        return winner, scores
