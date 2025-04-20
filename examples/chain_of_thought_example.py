"""Example demonstrating chain of thought branching collaboration."""

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

# Configuration with chain_of_thought collaboration mode
COT_CONFIG = {
    "models": {
        "model1": {
            "path": str(models_dir / "gemma-2-9b-it-Q6_K.gguf"),
            "role": "thinker",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
                "n_ctx": 8192, # Example context size
                "n_gpu_layers": -1
            }
        },
        "model2": {
            "path": str(models_dir / "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf"),
            "role": "evaluator",
            "parameters": {
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 2048,
                "n_ctx": 32768,
                "n_gpu_layers": -1
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
        "cot_initial": "You are analyzing the following problem:\n\n{{ query }}\n\nBegin your solution with a structured mathematical approach:\n\n1. IDENTIFY: What are the key variables and quantities in this problem? What exactly are you trying to find?\n2. FORMULATE: Express the relationships between these variables as equations or logical statements.\n3. SOLVE: Develop a step-by-step plan to solve these equations or deduce the answer.\n4. VERIFY: Consider how you'll check your solution against the original conditions.\n\nBe precise and rigorous in your mathematical reasoning. Show your thinking process explicitly and avoid vague statements like \"let me know what you think\" - instead, begin developing concrete mathematical relationships and solution strategies.",
        "cot_branch": "You are continuing the mathematical reasoning for this problem:\n\n{{ query }}\n\nInitial thoughts:\n{{ initial_thoughts }}\n\nPrevious steps:\n{{ previous_steps }}\n\nContinue with Step {{ step_number }} below. IMPORTANT: Even if you believe you've already found the solution in earlier steps, you must still complete this step thoroughly by:\n\n1. VERIFICATION: Mathematically prove that your proposed solution satisfies all conditions in the original problem by substituting values and checking all constraints.\n2. ALTERNATIVE APPROACH: Solve the problem using a different method or perspective to confirm your answer independently.\n3. EDGE CASES: Examine any potential special cases or assumptions that might affect your solution.\n\nDo not leave this step empty. Thorough verification is essential for mathematical rigor and confidence in your solution. If you've proposed an answer of x, show explicitly why x must be correct and cannot be any other value.",
        "cot_evaluation": "You are evaluating different mathematical reasoning branches for this problem:\n\n{{ query }}\n\nThe reasoning branches are:\n\n{{ branches }}\n\nPerform a comprehensive evaluation following this exact structure:\n\n1. BRANCH ASSESSMENT:\n   • Branch 1 Analysis: Evaluate the mathematical correctness, completeness, and logical rigor of Branch 1.\n   • Branch 2 Analysis: Evaluate the mathematical correctness, completeness, and logical rigor of Branch 2.\n   • Branch 3 Analysis: Evaluate the mathematical correctness, completeness, and logical rigor of Branch 3.\n\n2. SELECTION JUSTIFICATION:\n   After careful consideration, I determine that Branch [INSERT SPECIFIC BRANCH NUMBER: 1, 2, or 3] provides the most reliable solution because [provide specific mathematical reasons].\n\n3. FINAL SOLUTION:\n   • Mathematical Approach: [Summarize the key equations/method from the best branch]\n   • Calculation Steps: [Show the key mathematical steps]\n   • Answer: [State the final answer with appropriate units]\n   • Verification: [Demonstrate how this answer satisfies all original conditions]\n\nYou MUST explicitly identify which branch number contains the best reasoning and provide a complete solution with all required elements. Do not output empty sections or placeholders like \"Step 1:\" without content."
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

            def display_wrapped_text(text, width=70, max_length=None):
                """Display text with proper wrapping and handling empty content."""
                if not text or not text.strip():
                    print("[No meaningful content generated]")
                    return

                # Apply maximum length truncation if specified
                if max_length and len(text) > max_length:
                    display_text = text[:max_length] + "... [truncated]"
                    print(f"[Content truncated to {max_length} characters. Full content in trace file.]")
                else:
                    display_text = text

                # Replace literal "\n" with actual newlines if present
                if isinstance(display_text, str):
                    display_text = display_text.replace('\\n', '\n')

                # Split by actual newlines and wrap each paragraph
                paragraphs = display_text.split('\n')
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():  # Skip empty paragraphs
                        wrapped = textwrap.fill(paragraph, width=width,
                                                break_long_words=False,
                                                replace_whitespace=False)
                        print(wrapped)
                        # Add a newline between paragraphs, but not after the last one
                        if i < len(paragraphs) - 1:
                            print()

            # Print response
            print("\nFinal Response:")
            print("=" * 80)
            display_wrapped_text(response_data['response'])
            print("=" * 80)

            # Print reasoning branches from trace
            if 'trace' in response_data and 'phases' in response_data['trace'] and 'branching_cot' in \
                    response_data['trace']['phases']:
                cot_data = response_data['trace']['phases']['branching_cot']['output_data']

                # Display initial thoughts
                if 'initial_thoughts' in cot_data:
                    print("\nInitial Thoughts:")
                    print("-" * 50)
                    initial = cot_data['initial_thoughts'].get('initial_thoughts', '')
                    display_wrapped_text(initial)

                    # Show token count for diagnostic purposes
                    if 'raw_result' in cot_data['initial_thoughts']:
                        token_count = cot_data['initial_thoughts']['raw_result'].get('token_count', 0)
                        if token_count == 0:
                            print("[Note: Model generated 0 tokens for initial thoughts]")

                # Display branches with full steps
                if 'branches' in cot_data:
                    for i, branch in enumerate(cot_data['branches']):
                        print(f"\nBranch {i + 1}:")
                        print("-" * 50)

                        # Display all steps in the branch
                        for j, step in enumerate(branch.get('steps', [])):
                            step_content = step.get('content', '')
                            if step_content.strip():
                                print(f"Step {j+1}:")
                                display_wrapped_text(step_content)
                                print()  # Add space between steps

                        # Display conclusion if it exists
                        conclusion = branch.get('conclusion', '')
                        if conclusion.strip():
                            print("Conclusion:")
                            display_wrapped_text(conclusion)

                # Display evaluation results
                if 'evaluation_results' in cot_data:
                    print("\nEvaluation:")
                    print("-" * 50)

                    # Try to extract best branch ID from the evaluation text
                    eval_text = cot_data['evaluation_results'].get('evaluation_text', '')

                    # Check if best_branch_id was explicitly set
                    best_branch_id = cot_data['evaluation_results'].get('best_branch_id')
                    if best_branch_id:
                        print(f"Selected Branch: {best_branch_id}\n")

                    # Display the evaluation text
                    display_wrapped_text(eval_text)

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
