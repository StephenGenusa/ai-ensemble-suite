#!/usr/bin/env python3
"""Debate Master for structured debate-based response improvement."""

import asyncio
import json
import os
import re
import sys
import textwrap
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add the src directory to the path if running from the examples directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from ai_ensemble_suite import Ensemble
from ai_ensemble_suite.utils.logging import logger

# Default config with templates
DEFAULT_CONFIG = {
    "models": {
        "deepseek": {
            "path": str(project_root / "models" / "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf"),
            "role": "critic",
            "parameters": {
                "temperature": 0.75,
                "top_p": 0.9,
                "max_tokens": 4096,
                "n_ctx": 32768,
                "n_gpu_layers": -1
            }
        },
        "mistral": {
            "path": str(project_root / "models" / "Mistral-7B-Instruct-v0.3-Q6_K.gguf"),
            "role": "critic",
            "parameters": {
                "temperature": 0.75,
                "top_p": 0.9,
                "max_tokens": 4096,
                "n_ctx": 32768,
                "n_gpu_layers": -1
            }
        }
    },
    "collaboration": {
        "mode": "structured_debate",
        "phases": [
            {
                "name": "initial_response",
                "type": "async_thinking",
                "models": ["mistral"],
                "prompt_template": "debate_initial"
            },
            {
                "name": "critique",
                "type": "structured_debate",
                "subtype": "critique",
                "models": ["deepseek"],
                "input_from": "initial_response",
                "prompt_template": "debate_critique"
            },
            {
                "name": "defense",
                "type": "structured_debate",
                "subtype": "synthesis",
                "models": ["deepseek"],
                "input_from": ["initial_response", "critique"],
                "prompt_template": "debate_defense"
            },
            {
                "name": "synthesis",
                "type": "integration",
                "models": ["mistral"],
                "input_from": ["initial_response", "critique", "defense"],
                "prompt_template": "debate_synthesis"
            },
            {
                "name": "improved_response",
                "type": "integration",
                "models": ["deepseek"],
                "input_from": ["initial_response", "critique", "defense", "synthesis"],
                "prompt_template": "debate_improvement"
            },
            {
                "name": "evaluation",
                "type": "async_thinking",
                "models": ["mistral"],
                "input_from": ["improved_response"],
                "prompt_template": "debate_evaluation"
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "improved_response"
    },
    "templates": {
        "debate_initial": """You are an AI assistant with expertise in providing balanced, thoughtful responses. 
Address the following query with a well-reasoned response:

QUERY: {{ query }}

Provide a comprehensive but concise response that considers multiple perspectives.""",

        "debate_critique": """You are a thoughtful critic with expertise in critical analysis and reasoning.

Review the following response to this question:

ORIGINAL QUESTION: {{ query }}

RESPONSE TO EVALUATE:
{{ initial_response }}

Critically evaluate this response by reasoning through:
1. Factual accuracy - Are there any errors or misleading statements?
5. Clarity - Is the response clear and well-organized?
2. Comprehensiveness - Does it address all relevant aspects of the question?
3. Logical reasoning - Is the argument structure sound and coherent?
4. Fairness - Does it present a balanced view or show bias?
6. Use of Evidence - Are sources credible, up-to-date, and properly cited? Is the evidence relevant and effectively used to support claims?
7. Consistency - Are the argument's claims consistent with each other, or are there contradictions?

Be rigorous and thorough in your critique. Finding flaws is crucial for improvement.
Provide concise, actionable feedback for improvement.""",

        "debate_defense": """You are the original responder to a question that has received critique.

ORIGINAL QUESTION: {{ query }}

YOUR ORIGINAL RESPONSE:
{{ initial_response }}

CRITIC'S FEEDBACK:
{{ critique }}

Respond to these criticisms by either:
1. Defending your original points with additional evidence and reasoning, or
2. Acknowledging valid criticisms and refining your position

Keep your response concise and focused on the key issues.""",

        "debate_synthesis": """You are a neutral synthesizer reviewing a debate on the following question:

ORIGINAL QUESTION: {{ query }}

INITIAL RESPONSE:
{{ initial_response }}

CRITIQUE:
{{ critique }}

DEFENSE:
{{ defense }}

Based on this exchange, provide a balanced synthesis that:
1. Identifies areas of agreement between the perspectives
2. Acknowledges legitimate differences
3. Presents the strongest version of the final answer

Your synthesis should be concise and focused on the most important points.""",

        "debate_improvement": """You are an expert integrator tasked with creating an improved response.

ORIGINAL QUESTION: {{ query }}

INITIAL RESPONSE:
{{ initial_response }}

CRITIQUE:
{{ critique }}

DEFENSE:
{{ defense }}

SYNTHESIS:
{{ synthesis }}

Create a new comprehensive response that incorporates all the valid criticisms and feedback. Your response should be a complete, standalone answer to the original question that is better than the initial response.

IMPORTANT: FORMAT YOUR RESPONSE AS PLAIN TEXT ONLY. Do not include any meta-commentary, section headings, evaluation criteria, or notes about what you changed. Just provide the final improved answer as if you were directly answering the original question for the first time.""",

        "debate_evaluation": """You are an expert evaluator assessing the following response:

ORIGINAL QUESTION: {{ query }}

RESPONSE TO EVALUATE:
{{ improved_response }}

Evaluate this response on a scale of 1-10, where:
1 = Poor, inadequate response with significant issues
5 = Adequate response with some strengths and weaknesses
10 = Exceptional, comprehensive response with no notable flaws

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
RATING: [Your numerical rating from 1-10]
EXPLANATION: [Your explanation for the rating]

Be honest and critical in your assessment. Do NOT inflate the rating. A perfect 10 should be rare and only for truly exceptional responses. If the response does not meet the highest standards of quality, accuracy, comprehensiveness, and clarity, it should not receive a 10.""",
    }
}


class DebateMaster:
    """A system that facilitates structured debates to improve responses to a query.

    This class manages an iterative debate process that aims to improve responses
    to a query until a target rating is achieved or a maximum number of iterations
    is reached.

    Attributes:
        config: Configuration dictionary for the ensemble.
        config_path: Path to save the configuration file.
        target_rating: The target rating to achieve.
        max_iterations: Maximum number of iterations to run.
        debate_history: History of all debate iterations.
        best_response: Best response achieved so far.
        best_rating: Rating of the best response.
        output_dir: Directory to save output files.
        results_path: Path to save results JSON file.
    """

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
        target_rating: float = 10.0,
        max_iterations: int = 5
    ) -> None:
        """Initialize the debate master.

        Args:
            config_dict: Dictionary containing configuration for the ensemble.
                Defaults to a pre-defined config if None.
            config_path: Path to save the configuration file. If None, will use
                a default path in the script directory.
            target_rating: The target rating to achieve (default: 10.0).
            max_iterations: Maximum number of iterations to run (default: 5).
        """
        self.config = config_dict if config_dict else DEFAULT_CONFIG
        self.config_path = config_path
        self.target_rating = target_rating
        self.max_iterations = max_iterations
        self.debate_history = []
        self.best_response = None
        self.best_rating = 0
        self._init_paths()

    def _init_paths(self) -> None:
        """Initialize file and directory paths."""
        # Initialize config directory and path
        if self.config_path is None:
            config_dir = script_dir / "config"
            config_dir.mkdir(exist_ok=True)
            self.config_path = config_dir / "debate_master_config.yaml"

        # Initialize output directory
        self.output_dir = script_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.results_path = self.output_dir / "debate_master_results.json"

    async def validate_models(self) -> None:
        """Validate that all required model files exist.

        Raises:
            FileNotFoundError: If any model file is missing.
        """
        missing_models = []
        for model_id, model_config in self.config["models"].items():
            model_path = Path(model_config["path"])
            if not model_path.exists():
                missing_models.append(model_path)

        if missing_models:
            models_str = "\n  - ".join(str(m) for m in missing_models)
            logger.error(f"The following model files were not found:\n  - {models_str}")
            logger.error("Please ensure you have downloaded the required models to the 'models' directory.")
            raise FileNotFoundError(f"Missing model files: {models_str}")

    async def save_config(self) -> None:
        """Save the configuration to a file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Using config file: {self.config_path}")

    @staticmethod
    def extract_rating(evaluation_text: str) -> float:
        """Extract numerical rating from evaluation text.

        Args:
            evaluation_text: The evaluation text to parse.

        Returns:
            float: The numerical rating extracted from the text.
        """
        if not evaluation_text:
            return 0

        # Try to find the rating pattern
        rating_pattern = r"RATING:\s*(\d+(?:\.\d+)?)"
        match = re.search(rating_pattern, evaluation_text, re.IGNORECASE)

        if match:
            try:
                rating = float(match.group(1))
                return rating
            except (ValueError, TypeError):
                return 0

        # If no pattern match, try to find any number between 1-10
        number_pattern = r"(\d+(?:\.\d+)?)\s*(?:\/\s*10|out of 10)"
        match = re.search(number_pattern, evaluation_text)
        if match:
            try:
                rating = float(match.group(1))
                return rating
            except (ValueError, TypeError):
                return 0

        # Final attempt - find any digit from 1-10
        digit_pattern = r"(?:^|\s)([1-9]|10)(?:\s|$|\.|\/)"
        match = re.search(digit_pattern, evaluation_text)
        if match:
            try:
                rating = float(match.group(1))
                return rating
            except (ValueError, TypeError):
                return 0

        return 0

    @staticmethod
    def extract_content_safely(data: Any, phase_name: Optional[str] = None) -> str:
        """Extract content from complex data structures.

        Args:
            data: The data to extract content from.
            phase_name: Optional name of the phase for phase-specific extraction.

        Returns:
            str: The extracted content.
        """
        # Base cases for recursion
        if data is None:
            return ""

        if isinstance(data, str):
            return data

        if not isinstance(data, (dict, list)):
            return str(data)

        # Handle dictionaries: look for content in common key names
        if isinstance(data, dict):
            # For improved_response phase, look for actual response content
            # rather than meta-commentary
            if phase_name == "improved_response":
                # Look specifically for the improved content, often in these keys
                content_keys = ['response', 'text', 'content', 'output', 'answer',
                                'final_answer', 'improvement']
                for key in content_keys:
                    if key in data and data[key] and isinstance(data[key], str) and len(data[key]) > 100:
                        return data[key]

                # If we don't find in the above keys, look recursively
                for v in data.values():
                    if isinstance(v, str) and len(v) > 200:  # Substantial text
                        return v
                    if isinstance(v, dict):
                        result = DebateMaster.extract_content_safely(v, phase_name)
                        if result and len(result) > 200:
                            return result

            # Direct content keys with descending priority
            content_keys = ['text', 'content', 'output', 'response', 'result',
                           'critique', 'defense', 'synthesis']
            for key in content_keys:
                if key in data and data[key]:
                    result = data[key]
                    if isinstance(result, str):
                        return result
                    # If it's a nested structure, recurse
                    return DebateMaster.extract_content_safely(result, phase_name)

            # Collection keys that might contain model outputs
            collection_keys = ['outputs', 'model_outputs', 'model_critiques', 'model_responses']
            for key in collection_keys:
                if key in data and data[key]:
                    collection = data[key]
                    if isinstance(collection, dict) and collection:
                        # Get the first value from the collection
                        first_value = next(iter(collection.values()), "")
                        return DebateMaster.extract_content_safely(first_value, phase_name)

            # Last resort: check all values recursively
            for v in data.values():
                content = DebateMaster.extract_content_safely(v, phase_name)
                if content and isinstance(content, str) and len(content) > 50:
                    return content

        # Handle lists: look for content in items
        if isinstance(data, list) and data:
            # Try to find the first non-empty string or dict with substantial content
            for item in data:
                content = DebateMaster.extract_content_safely(item, phase_name)
                if content and isinstance(content, str) and len(content) > 50:
                    return content

        # If all else fails
        if isinstance(data, dict):
            return str(data)
        elif isinstance(data, list):
            return str(data)
        else:
            return str(data)

    @staticmethod
    def clean_text(text: str, phase_name: Optional[str] = None) -> str:
        """Clean text by removing delimiters, formatting, and artifacts.

        Args:
            text: The text to clean.
            phase_name: Optional name of the phase for phase-specific cleaning.

        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str):
            text = str(text)

        # Basic cleaning
        text = text.strip()
        text = text.replace('\\n', '\n')

        # Remove common prefixes
        prefixes_to_remove = [
            r'^RESPONSE:\s*',
            r'^ANSWER:\s*',
            r'^Final answer:\s*',
            r'^Final Answer:\s*',
            r'^---+\s*',
            r'^</?think>\s*',
            r'^USER\'S REFLECTION:[\s"]*',
            r'^Revised Response:\s*',
            r'^REVISED RESPONSE:\s*',
            r'^IMPROVED RESPONSE:\s*'
        ]

        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove anything inside <think> </think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove code block markers
        text = re.sub(r'```(?:plaintext|json|)\s*', '', text)
        text = re.sub(r'```\s*$', '', text)

        # Remove common JSON artifacts
        json_artifacts = [r'"\}\s*$', r'\}\s*$', r'"\s*$', r'\]\s*$', r'"\]\s*$']
        for artifact in json_artifacts:
            text = re.sub(artifact, '', text)

        # For improved response, need to be more aggressive
        if phase_name == "improved_response":
            # Remove meta-commentary, section headings, or evaluation references
            text = re.sub(r'(?i)^.*?addressing criticism.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'(?i)^\*\*.*?\*\*\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'(?i)^[0-9]+\.\s+\*\*.*?\*\*.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'(?i)^By integrating these considerations.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'(?i)^After evaluating.*?response\.?\s*$', '', text, flags=re.DOTALL)

            # Remove numbered list items with asterisks (critique responses)
            text = re.sub(r'(?i)^[0-9]+\.\s+\*\*.*?\*\*.*$', '', text, flags=re.MULTILINE)

            # Remove sections like "Final Revised Response:"
            text = re.sub(r'(?i)^.*?Final Revised Response:.*$', '', text, flags=re.MULTILINE)

            # Remove any horizontal separators
            text = re.sub(r'^[-_=]{3,}\s*$', '', text, flags=re.MULTILINE)

            # Remove lines that refer to criticism or feedback
            text = re.sub(r'(?i)^.*?criticism.*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'(?i)^.*?feedback.*$', '', text, flags=re.MULTILINE)

        # Clean up multiple consecutive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Final strip
        text = text.strip()

        return text

    @staticmethod
    def display_wrapped_text(text: str, width: int = 70, title: Optional[str] = None) -> None:
        """Display text with proper wrapping.

        Args:
            text: The text to display.
            width: The width to wrap text at.
            title: Optional title to display above the text.
        """
        if title:
            print(f"\n{title}:")
            print("-" * len(title))

        if not text or (isinstance(text, str) and not text.strip()):
            print("[No meaningful content generated]")
            return

        # Clean text
        text = DebateMaster.clean_text(text)

        # Split by actual newlines and wrap each paragraph
        paragraphs = text.split('\n')
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():  # Skip empty paragraphs
                wrapped = textwrap.fill(paragraph, width=width,
                                       break_long_words=False,
                                       replace_whitespace=False)
                print(wrapped)
                # Add a newline between paragraphs, but not after the last one
                if i < len(paragraphs) - 1:
                    print()

    async def get_improved_response(
        self,
        ensemble: Ensemble,
        query: str,
        current_iteration: int
    ) -> Dict[str, Any]:
        """Get an improved response for the query.

        Args:
            ensemble: The ensemble to use for the query.
            query: The query to get a response for.
            current_iteration: The current iteration number.

        Returns:
            Dict[str, Any]: Result data including the improved response and rating.
        """
        logger.info(f"Executing debate process for iteration {current_iteration}")
        logger.info(f"The query is: {query}")
        response_data = await ensemble.ask(query, trace=True)
        execution_time = response_data.get('execution_time', 0)
        logger.info(f"Debate process for iteration {current_iteration} completed in {execution_time:.2f} seconds")

        result = {
            'execution_time': execution_time,
            'improved_response': "[No improved response generated]",
            'evaluation_text': "[No evaluation generated]",
            'rating': 0,
            'phases_completed': 0
        }

        # Extract the improved response
        if 'trace' in response_data and 'phases' in response_data['trace']:
            phases = response_data['trace']['phases']
            result['phases_completed'] = len(phases)

            # Extract improved response
            if 'improved_response' in phases:
                improved_output = phases['improved_response'].get('output_data', {})
                improved_response = self.extract_content_safely(improved_output, 'improved_response')
                improved_response = self.clean_text(improved_response, 'improved_response')
                result['improved_response'] = improved_response
                content_length = len(improved_response)
                logger.info(f"Improved response extracted ({content_length} chars)")
            else:
                logger.warning(f"No improved response found in iteration {current_iteration}")

            # Extract evaluation text
            if 'evaluation' in phases:
                evaluation_output = phases['evaluation'].get('output_data', {})
                evaluation_text = self.extract_content_safely(evaluation_output, 'evaluation')
                evaluation_text = self.clean_text(evaluation_text)
                result['evaluation_text'] = evaluation_text

                # Extract numerical rating
                rating = self.extract_rating(evaluation_text)
                result['rating'] = rating
                logger.info(f"Iteration {current_iteration} - Response rated: {rating}/10")
            else:
                logger.warning(f"No evaluation found in iteration {current_iteration}")
        else:
            logger.warning(f"No phases found in trace for iteration {current_iteration}")

        return result

    async def run_debate(self, query: str) -> Dict[str, Any]:
        """Run the debate process on the given query.

        Args:
            query: The query to debate.

        Returns:
            Dict[str, Any]: Results of the debate process including best response and metrics.
        """
        await self.validate_models()
        await self.save_config()

        logger.info(f"Starting debate master with target rating: {self.target_rating}/10")

        print(f"\nQuery: {query}\n")
        print(f"Starting debate until a rating of {self.target_rating}/10 is achieved "
              f"(or max iterations reached)...\n")
        logger.info(f"Starting debate on query: {query}")

        # Initialize debate state
        current_iteration = 1
        current_query = query
        total_phases_completed = 0

        async with Ensemble(config_dict=self.config) as ensemble:
            # Run the debate cycle until we reach the target rating or max iterations
            while current_iteration <= self.max_iterations:
                logger.info(f"Starting debate iteration {current_iteration}/{self.max_iterations}")
                print(f"\n===== ITERATION {current_iteration} =====")

                # Add history to query for subsequent iterations
                if current_iteration > 1 and self.best_response:
                    # For subsequent iterations, include the previous best response to improve
                    logger.info(f"Building on previous best response with rating {self.best_rating}/10")
                    current_query = f"""
{query}

PREVIOUS BEST RESPONSE (rated {self.best_rating}/10):
{self.best_response}

Your task is to create a new response that addresses the weaknesses identified in the previous response
and aims to achieve a perfect {self.target_rating}/10 rating.
"""

                # Execute the debate process
                result = await self.get_improved_response(ensemble, current_query, current_iteration)
                total_phases_completed += result['phases_completed']

                # Add iteration information to the result
                iteration_result = {
                    'iteration': current_iteration,
                    'query': current_query if current_iteration > 1 else query,
                    'improved_response': result['improved_response'],
                    'evaluation': result['evaluation_text'],
                    'rating': result['rating'],
                    'execution_time': result['execution_time'],
                    'phases_completed': result['phases_completed']
                }
                self.debate_history.append(iteration_result)

                # Display the results for this iteration
                self.display_wrapped_text(result['improved_response'],
                                         title=f"Improved Response (Iteration {current_iteration})")
                self.display_wrapped_text(result['evaluation_text'], title="Evaluation")
                print(f"\nRating: {result['rating']}/10")

                # Update best response if this is better
                if self.best_response is None or result['rating'] > self.best_rating:
                    self.best_response = result['improved_response']
                    self.best_rating = result['rating']
                    logger.info(f"New best response achieved! Rating: {result['rating']}/10")
                    print(f"\n*** New best response achieved! Rating: {result['rating']}/10 ***")

                # Check if we've reached the target rating
                if result['rating'] >= self.target_rating:
                    logger.info(f"Target rating {self.target_rating}/10 achieved! Stopping debate.")
                    print(f"\nTarget rating {self.target_rating} achieved! Stopping debate.")
                    # Ensure we count this iteration before breaking
                    current_iteration += 1
                    break

                # Increment iteration counter
                current_iteration += 1

            # Prepare the results for saving - use the actual count of completed iterations
            iterations_completed = len(self.debate_history)
            final_results = {
                'query': query,
                'iterations_completed': iterations_completed,
                'total_phases_completed': total_phases_completed,
                'best_rating': self.best_rating,
                'target_rating': self.target_rating,
                'target_reached': self.best_rating >= self.target_rating,
                'best_response': self.best_response,
                'debate_history': self.debate_history
            }

            # Save results as pretty-printed JSON
            with open(self.results_path, 'w') as f:
                json.dump(final_results, f, indent=2)

            logger.info(f"Results saved to {self.results_path}")

            return {
                'query': query,
                'iterations_completed': iterations_completed,
                'total_phases_completed': total_phases_completed,
                'best_rating': self.best_rating,
                'target_rating': self.target_rating,
                'target_reached': self.best_rating >= self.target_rating,
                'best_response': self.best_response,
                'results_path': self.results_path
            }


async def main() -> None:
    """Run the debate master example."""
    try:
        # Create a debate master instance
        debate_master = DebateMaster()

        # Ask a nuanced question
        query = "Is artificial general intelligence (AGI) likely to be achieved in the next decade, and what might be the societal implications?"

        # Run the debate
        results = await debate_master.run_debate(query)

        # Display final results
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)

        print(f"Best rating achieved: {results['best_rating']}/10")
        print(f"Target rating reached: {'Yes' if results['target_reached'] else 'No'}")
        print(f"Total iterations: {results['iterations_completed']}")
        print(f"Total phases completed: {results['total_phases_completed']}")

        # Display the best response
        DebateMaster.display_wrapped_text(results['best_response'], title="BEST RESPONSE")

        print(f"\nResults saved to {results['results_path']}")

    except asyncio.TimeoutError:
        logger.error("The debate processing timed out")
        print("\nThe process timed out. This could be due to:")
        print("1. Context window or token limit issues")
        print("2. Resource constraints on your system")
        print("3. The model getting stuck in generation")
        print("\nTry running with smaller context windows and/or token limits.")
        print("\nUse a utility to view GPU usage to determine what resources are available and how much are in use")

    except Exception as e:
        logger.error(f"Error in debate master example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
