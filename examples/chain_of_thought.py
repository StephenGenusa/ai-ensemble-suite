"""Example demonstrating chain of thought branching collaboration."""

import asyncio
import os
import sys
import yaml
from pathlib import Path

# Add the src directory to the path if running from the examples directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from ai_ensemble_suite import Ensemble
from ai_ensemble_suite.utils.logging import logger

# Get the models directory
models_dir = project_root / "models"

# Configuration with chain_of_thought collaboration mode
COT_CONFIG = {
    "models": {
        "model1": {
            "path": str(models_dir / "gemma-2-9b-it-Q6_K.gguf"),
            "role": "thinker",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 4096
            }
        },
        "model2": {
            "path": str(models_dir / "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf"),
            "role": "evaluator",
            "parameters": {
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 8192
            }
        }
    },
    "collaboration": {
        "mode": "chain_of_thought",
        "phases": [
            {
                "name": "branching_cot",
                "type": "chain_of_thought",
                "models": ["model1", "model2"],
                "branch_count": 3,
                "branch_depth": 2,
                "evaluation_model": "model2",
                "initial_template": "cot_initial",
                "branch_template": "cot_branch",
                "evaluation_template": "cot_evaluation"
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "branching_cot"
    },
    "templates": {
        "cot_initial": "You are thinking about the following problem:\n\n{query}\n\nExplore your initial thoughts and possible approaches to solving this problem. Consider different angles and methodologies.",
        "cot_branch": "You are continuing a chain of thought reasoning for this problem:\n\n{query}\n\nInitial thoughts:\n{initial_thoughts}\n\nPrevious steps:\n{previous_steps}\n\nContinue the reasoning for Step {step_number}, exploring this line of thinking further. Be detailed in your reasoning process.",
        "cot_evaluation": "You are evaluating different reasoning branches for this problem:\n\n{query}\n\nThe reasoning branches are:\n\n{branches}\n\nEvaluate each branch for logical soundness, completeness, and relevance to the original problem. Select the branch that provides the most compelling solution, and explain why. Then provide the final answer based on the best reasoning path."
    }
}


async def main():
    """Run chain of thought example."""
    # Check if models exist
    for model_id, model_config in COT_CONFIG["models"].items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Please ensure you have downloaded the required models to the 'models' directory.")
            sys.exit(1)

    # Check if config file exists
    config_dir = script_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "cot_config.yaml"

    # If config file exists, use it; otherwise use our default config
    if config_path.exists():
        logger.info(f"Using config file: {config_path}")
        ensemble_kwargs = {"config_path": str(config_path)}
    else:
        logger.info("Using default config")
        # Save default config to file for reference
        with open(config_path, 'w') as f:
            yaml.dump(COT_CONFIG, f, default_flow_style=False)
        ensemble_kwargs = {"config_dict": COT_CONFIG}

    try:
        # Initialize the ensemble
        async with Ensemble(**ensemble_kwargs) as ensemble:

            # Ask a reasoning problem
            query = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"

            print(f"\nQuery: {query}\n")
            print("Processing chain of thought reasoning...")

            # Get response with trace
            response_data = await ensemble.ask(query, trace=True)

            # Print response
            print("\nFinal Response:")
            print("=" * 80)
            print(response_data['response'])
            print("=" * 80)

            # Print reasoning branches from trace
            if 'trace' in response_data and 'phases' in response_data['trace'] and 'branching_cot' in \
                    response_data['trace']['phases']:
                cot_data = response_data['trace']['phases']['branching_cot']['output_data']

                if 'initial_thoughts' in cot_data:
                    print("\nInitial Thoughts:")
                    print("-" * 50)
                    initial = cot_data['initial_thoughts'].get('initial_thoughts', '')
                    print(initial[:300] + "..." if len(initial) > 300 else initial)

                if 'branches' in cot_data:
                    for i, branch in enumerate(cot_data['branches']):
                        print(f"\nBranch {i + 1}:")
                        print("-" * 50)

                        # Print the conclusion from this branch
                        conclusion = branch.get('conclusion', '')
                        print(conclusion[:300] + "..." if len(conclusion) > 300 else conclusion)

                if 'evaluation_results' in cot_data:
                    print("\nEvaluation:")
                    print("-" * 50)
                    eval_text = cot_data['evaluation_results'].get('evaluation_text', '')
                    print(eval_text[:300] + "..." if len(eval_text) > 300 else eval_text)

            # Print execution statistics
            print(f"\nExecution time: {response_data['execution_time']:.2f} seconds")

            # Save trace to file
            output_dir = script_dir / "output"
            output_dir.mkdir(exist_ok=True)
            trace_path = output_dir / "cot_trace.json"

            # Save trace as pretty-printed JSON
            import json
            with open(trace_path, 'w') as f:
                json.dump(response_data['trace'], f, indent=2)

            print(f"Trace saved to {trace_path}")

    except Exception as e:
        logger.error(f"Error in chain of thought example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
