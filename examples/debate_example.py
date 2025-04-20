"""Example demonstrating structured debate collaboration."""

import asyncio
import os
import sys
import yaml
from pathlib import Path
import textwrap

# Add the src directory to the path if running from the examples directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from ai_ensemble_suite import Ensemble
from ai_ensemble_suite.utils.logging import logger

# Get the models directory
models_dir = project_root / "models"

# Configuration with structured_debate collaboration mode
DEBATE_CONFIG = {
    "models": {
        "deepseek": {
            "path": str(models_dir / "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf"),
            "role": "critic",
            "parameters": {
                "temperature": 0.75,
                "top_p": 0.9,
                "max_tokens": 2048,
                "n_ctx": 16384,  # Increased context window
                "n_gpu_layers": -1

            }
        },
        "mistral": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q6_K.gguf"),
            "role": "critic",
            "parameters": {
                "temperature": 0.75,
                "top_p": 0.9,
                "max_tokens": 4096,
                "n_ctx": 16384,
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
                "name": "final_synthesis",
                "type": "integration",
                "models": ["mistral"],
                "input_from": ["initial_response", "critique", "defense"],
                "prompt_template": "debate_synthesis"
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "final_synthesis"
    },
    "templates": {
        "debate_initial": """You are an AI assistant with expertise in providing balanced, thoughtful responses. 
    Address the following query with a well-reasoned response:

    QUERY: {{ query }}

    Provide a comprehensive response that considers multiple perspectives.""",

        "debate_critique": """You are a thoughtful critic with expertise in critical analysis and reasoning.

    Review the following response to this question:

    ORIGINAL QUESTION: {{ query }}

    RESPONSE TO EVALUATE:
    {{ initial_response }}

    Critically evaluate this response by reasoning through:
    1. Factual accuracy - Are there any errors or misleading statements?
    2. Comprehensiveness - Does it address all relevant aspects of the question?
    3. Logical reasoning - Is the argument structure sound and coherent?
    4. Fairness - Does it present a balanced view or show bias?
    5. Clarity - Is the response clear and well-organized?

    First, reason step-by-step through each aspect of the response, analyzing its strengths and weaknesses.
    Then, provide specific, actionable feedback for improvement.""",

        "debate_defense": """You are the original responder to a question that has received critique.

    ORIGINAL QUESTION: {{ query }}

    YOUR ORIGINAL RESPONSE:
    {{ initial_response }}

    CRITIC'S FEEDBACK:
    {{ critique }}

    Respond to these criticisms by either:
    1. Defending your original points with additional evidence and reasoning, or
    2. Acknowledging valid criticisms and refining your position

    Provide a thoughtful, balanced response to the critiques while maintaining intellectual integrity.""",

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
    3. Presents the strongest version of the final answer that incorporates valid points from all sides
    4. Notes any remaining uncertainties or areas where further information would be valuable

    Your goal is to produce the most accurate and balanced perspective possible.""",

        "self_evaluation": """You previously provided this response:

    {{ response }}

    Evaluate your confidence in this response on a scale from 1-10, where:
    1 = Completely uncertain or guessing
    10 = Absolutely certain based on verified facts

    Provide ONLY a numeric rating, nothing else."""
    }
}

async def main():
    """Run debate example."""
    # Check if models exist
    for model_id, model_config in DEBATE_CONFIG["models"].items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Please ensure you have downloaded the required models to the 'models' directory.")
            sys.exit(1)

    # Check if config file exists
    config_dir = script_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "debate_config.yaml"

    # If config file exists, use it; otherwise use our default config
    if config_path.exists():
        logger.info(f"Using config file: {config_path}")
        ensemble_kwargs = {'config_path': str(config_path)}
    else:
        logger.info("Using default config")
        # Save default config to file for reference
        with open(config_path, 'w') as f:
            yaml.dump(DEBATE_CONFIG, f, default_flow_style=False)
        ensemble_kwargs = {'config_dict': DEBATE_CONFIG}

    try:
        # Initialize the ensemble with proper kwargs
        async with Ensemble(**ensemble_kwargs) as ensemble:

            # Ask a nuanced question
            query = "Is artificial general intelligence (AGI) likely to be achieved in the next decade, and what might be the societal implications?"

            print(f"\nQuery: {query}\n")
            print("Processing debate...")

            # Get response with trace
            response_data = await ensemble.ask(query, trace=True)

            def display_wrapped_text(text, width=60):
                if not text or not text.strip():
                    print("[No meaningful content generated]")
                    return

                # Replace literal "\n" with actual newlines if present
                if isinstance(text, str):
                    text = text.replace('\\n', '\n')

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

                print(f"\n[Response length: {len(text)} characters]")

            # Print response
            print("\nSynthesized Response:")
            print("=" * 80)
            display_wrapped_text(response_data['response'])
            print("=" * 80)


            # In the main display section:
            if 'trace' in response_data and 'phases' in response_data['trace']:
                phases = response_data['trace']['phases']

                if 'initial_response' in phases:
                    print("\nInitial Response:")
                    print("-" * 50)
                    initial_data = phases['initial_response']['output_data']
                    initial = initial_data.get('output', '')
                    if not initial and 'outputs' in initial_data:
                        initial = next(iter(initial_data['outputs'].values()), '')
                    display_wrapped_text(initial)

                if 'critique' in phases:
                    print("\nCritique:")
                    print("-" * 50)
                    critique_data = phases['critique']['output_data']
                    critique = critique_data.get('critique', '')
                    if not critique and 'model_critiques' in critique_data:
                        critique = critique_data.get('model_critiques', {}).get('llama', '')
                    display_wrapped_text(critique)

                if 'defense' in phases:
                    print("\nDefense:")
                    print("-" * 50)
                    defense_data = phases['defense']['output_data']
                    defense = defense_data.get('output', '')
                    display_wrapped_text(defense)

            # Print execution statistics
            print(f"\nExecution time: {response_data['execution_time']:.2f} seconds")

            # Save trace to file
            output_dir = script_dir / "output"
            output_dir.mkdir(exist_ok=True)
            trace_path = output_dir / "debate_trace.json"

            # Save trace as pretty-printed JSON
            import json
            with open(trace_path, 'w') as f:
                json.dump(response_data['trace'], f, indent=2)

            print(f"Trace saved to {trace_path}")

    except Exception as e:
        logger.error(f"Error in debate example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
