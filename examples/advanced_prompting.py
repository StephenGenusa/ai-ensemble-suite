"""Example demonstrating advanced prompting techniques."""

import asyncio
import os
import sys
import yaml
from pathlib import Path
import textwrap
import json

# Add the src directory to the path if running from the examples directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from ai_ensemble_suite import Ensemble
from ai_ensemble_suite.utils.logging import logger

# Get the models directory
models_dir = project_root / "models"

# Advanced prompting configuration
ADVANCED_PROMPTING_CONFIG = {
    "models": {
        "model1": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
            "role": "primary",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
                "n_ctx": 8192  # Increased context window
            }
        }
    },
    "collaboration": {
        "mode": "advanced_prompting",
        "phases": [
            {
                "name": "basic_response",
                "type": "async_thinking",
                "models": ["model1"],
                "prompt_template": "expert_prompt"
            },
            {
                "name": "cot_response",
                "type": "async_thinking",
                "models": ["model1"],
                "prompt_template": "cot_prompt"
            },
            {
                "name": "comparative_analysis",
                "type": "integration",
                "models": ["model1"],
                "input_from": ["basic_response", "cot_response"],
                "prompt_template": "comparative_analysis"
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "comparative_analysis"
    },
    "templates": {
        "expert_prompt": "You are an expert on the following topic. Provide a comprehensive and accurate answer:\n\n{query}",

        "cot_prompt": "Think through this question step by step, reasoning carefully:\n\n{query}\n\nLet's break this down systematically:",

        # FIXED: Use direct indexing for output values rather than dot notation
        "comparative_analysis": """You've provided two responses to the query:

{query}

Response 1 (Standard approach):
{basic_response}

Response 2 (Step-by-step reasoning):
{cot_response}

Now, compare these approaches and create a final response that combines the strengths of both. Focus on accuracy, clarity, and depth of explanation.""",

        # Add a self-evaluation template for the config validation to pass
        "self_evaluation": """You previously provided this response:

{response}

Evaluate your confidence in this response on a scale from 1-10, where:
1 = Completely uncertain or guessing
10 = Absolutely certain based on verified facts

Provide ONLY a numeric rating, nothing else."""
    }
}

async def main():
    """Run advanced prompting example."""
    # Check if config file exists and delete it to ensure we use our updated version
    config_dir = script_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "prompting_config.yaml"

    # Force using our new config by deleting the old one
    if config_path.exists():
        os.remove(config_path)
        logger.info(f"Removed old config file to ensure proper template formatting")

    # Save default config to file for reference
    with open(config_path, 'w') as f:
        yaml.dump(ADVANCED_PROMPTING_CONFIG, f, default_flow_style=False)
    logger.info(f"Updated config file saved to: {config_path}")

    ensemble_kwargs = {'config_path': str(config_path)}

    try:
        # Initialize the ensemble with proper kwargs
        async with Ensemble(**ensemble_kwargs) as ensemble:

            # Ask a question that benefits from different prompting strategies
            query = "Explain the prisoner's dilemma and its relevance to real-world scenarios."

            print(f"\nQuery: {query}\n")
            print("Processing with different prompting strategies...")

            # Get response with trace
            response_data = await ensemble.ask(query, trace=True)

            # Add text wrapping function from debate_example
            def display_wrapped_text(text, width=80):
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

            # Print final response
            print("\nFinal Response:")
            print("=" * 80)
            display_wrapped_text(response_data['response'])
            print("=" * 80)

            # Print individual phase outputs from trace
            if 'trace' in response_data and 'phases' in response_data['trace']:
                phases = response_data['trace']['phases']

                if 'basic_response' in phases and 'output_data' in phases['basic_response']:
                    print("\nStandard Expert Prompt Response:")
                    print("-" * 50)
                    output_data = phases['basic_response']['output_data']
                    if 'outputs' in output_data and 'model1' in output_data['outputs']:
                        basic = output_data['outputs']['model1']
                    else:
                        basic = output_data.get('output', '')
                    # Display at most the first 500 characters for brevity
                    display_wrapped_text(basic[:500] + "..." if len(basic) > 500 else basic)

                if 'cot_response' in phases and 'output_data' in phases['cot_response']:
                    print("\nChain of Thought Prompt Response:")
                    print("-" * 50)
                    output_data = phases['cot_response']['output_data']
                    if 'outputs' in output_data and 'model1' in output_data['outputs']:
                        cot = output_data['outputs']['model1']
                    else:
                        cot = output_data.get('output', '')
                    # Display at most the first 500 characters for brevity
                    display_wrapped_text(cot[:500] + "..." if len(cot) > 500 else cot)

            # Print execution statistics
            print(f"\nExecution time: {response_data['execution_time']:.2f} seconds")

            # Save trace to file
            output_dir = script_dir / "output"
            output_dir.mkdir(exist_ok=True)
            trace_path = output_dir / "prompting_trace.json"

            # Save trace as pretty-printed JSON
            with open(trace_path, 'w') as f:
                json.dump(response_data['trace'], f, indent=2)

            print(f"Trace saved to {trace_path}")

    except Exception as e:
        logger.error(f"Error in advanced prompting example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
