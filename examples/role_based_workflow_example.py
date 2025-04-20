"""Example demonstrating role-based workflow collaboration."""

import asyncio
import os
import sys
import yaml
import textwrap
from pathlib import Path
import json
import re

# Add the src directory to the path if running from the examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

try:
    from ai_ensemble_suite import Ensemble
    from ai_ensemble_suite.utils.logging import logger
    from ai_ensemble_suite.exceptions import AiEnsembleSuiteError, ConfigurationError, ModelError
except ImportError as e:
    print("Error: Could not import ai_ensemble_suite. "
          "Ensure it's installed or the src directory is in the Python path.")
    print(f"Current sys.path: {sys.path}")
    print(f"Import error details: {e}")
    sys.exit(1)


# Define model paths relative to the project root or examples dir
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Configuration with role_based_workflow collaboration mode
WORKFLOW_CONFIG = {
    "models": {
        "model1": {
            "path": str(MODEL_DIR / "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf"),
            "role": "researcher",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 4096,
                "n_ctx": 8192,
                "n_gpu_layers": -1  # Default to GPU in this config
            }
        },
        "model2": {
            "path": str(MODEL_DIR / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
            "role": "analyst",
            "parameters": {
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 4096,
                "n_ctx": 8192,
                "n_gpu_layers": -1  # Default to GPU in this config
            }
        },
        "model3": {
            "path": str(MODEL_DIR / "vicuna-7b-v1.5.Q6_K.gguf"),
            "role": "writer",
            "parameters": {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 3072,
                "n_ctx": 4096,
                "n_batch": 512,
                "n_gpu_layers": -1  # Default to GPU in this config
            }
        }
    },
    "collaboration": {
        "mode": "role_based_workflow",
        "phases": [
            {
                "name": "workflow",
                "type": "role_based_workflow",
                "models": ["model1", "model2", "model3"],
                "workflow_steps": [
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
                        "template": "role_analyst",
                        "input_from": "research"  # Explicitly state the input source
                    },
                    {
                        "role": "writer",
                        "task": "write",
                        "description": "Create a well-structured response based on the analysis",
                        "template": "role_writer",
                        "input_from": ["research", "analyze"]  # Explicitly state the input sources
                    }
                ]
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "workflow"
    },
    "templates": {
        # Adding DEBUG markers to make it easy to spot if templates aren't being filled
        "role_researcher": "You are an AI research assistant. Synthesize comprehensive information about: {{ query }}. Provide detailed facts, context, and different viewpoints. Structure your findings clearly.",

        "role_analyst": """You are an AI critical analyst. 
Analyze the specific research provided below about "{{ query }}".
Your task is to critically examine THIS research.

DEBUG-MARKER: If you see this text, it means the variable substitution is working correctly.

RESEARCH FINDINGS:
{{ research }}

YOUR ANALYSIS:""",

        "role_writer": """You are an AI writer creating a comprehensive response about: "{{ query }}"
Use the information from the research and analysis to create a detailed, factual response.

DEBUG-MARKER: If you see this text, it means the variable substitution is working correctly.

QUERY: {{ query }}

RESEARCH:
{{ research }}

ANALYSIS:
{{ analyze }}

COMPREHENSIVE RESPONSE:""",

        "self_evaluation": "On a scale of 1-10, how confident are you in this response: {{ response }}? Only provide the number."
    },
    # Enable debug logging
    "debug": {
        "log_level": "DEBUG"
    }
}

# Define text-wrapping function
def display_wrapped_text(text, width=80):
    """Display text with proper word wrapping while preserving paragraph structure."""
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

# Add debugging functions for the templates
def check_for_template_variables(text):
    """Check if there are any template variables left unfilled in the text."""
    if not isinstance(text, str):
        return False

    template_vars = re.findall(r'\{([^{}]+)\}', text)
    return bool(template_vars)

def check_debug_marker(text):
    """Check if the DEBUG-MARKER is present in the text."""
    if not isinstance(text, str):
        return False

    return "DEBUG-MARKER" in text

async def main():
    """Run role-based workflow example."""
    logger.set_level("DEBUG")  # Set to DEBUG to see more information

    # Define config and output paths
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    config_dir = Path(__file__).resolve().parent / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "role_based_workflow_config.yaml"

    # Save the config
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(WORKFLOW_CONFIG, f, default_flow_style=False, sort_keys=False)
        print(f"Saved clean workflow config to {config_path}")
    except Exception as e:
        logger.error(f"Could not save config to {config_path}: {e}")
        logger.warning("Proceeding with in-memory config.")

    # Check if model files exist
    all_models_exist = True
    for model_id, model_cfg in WORKFLOW_CONFIG['models'].items():
        if not Path(model_cfg.get('path', '')).exists():
            logger.error(f"Model file not found for '{model_id}': {model_cfg.get('path')}")
            all_models_exist = False

    if not all_models_exist:
        logger.error("One or more model files are missing. Please ensure models are downloaded and paths are correct.")
        sys.exit(1)

    try:
        logger.info("Initializing Ensemble...")
        # Initialize using the dictionary directly to ensure we use our exact config
        async with Ensemble(config_dict=WORKFLOW_CONFIG) as ensemble:
            logger.info("Ensemble initialized.")

            query = "Analyze the primary environmental and socio-economic implications of large-scale adoption of solar power versus wind power."

            print(f"\nQuery:\n{query}\n")
            print("Processing workflow, this may take some time...")

            # Get response with trace enabled
            response_data = await ensemble.ask(query, trace=True)

            # Print Final Response with word-wrapping
            print("\nFinal Response:")
            print("=" * 80)
            final_response = "[No Response Received]"
            if isinstance(response_data, dict) and 'response' in response_data:
                final_response = response_data.get('response', final_response)
            elif isinstance(response_data, str):
                final_response = response_data

            display_wrapped_text(final_response)

            # Check for template variables in the final response
            if check_for_template_variables(final_response):
                print("\n⚠️ WARNING: Final response contains unfilled template variables!")
                print("This suggests the template substitution is not working correctly.")
            print("=" * 80)

            # Print Workflow Steps from Trace with word-wrapping
            print("\n--- Workflow Steps (from Trace) ---")
            try:
                if isinstance(response_data, dict) and 'trace' in response_data:
                    trace = response_data['trace']
                    phases = trace.get('phases', {})
                    workflow = phases.get('workflow', {})
                    output_data = workflow.get('output_data', {})

                    # Get the steps data (could be in 'steps' or directly in the task keys)
                    if 'steps' in output_data and isinstance(output_data['steps'], dict):
                        steps = output_data['steps']
                    else:
                        # Try to get from task keys directly
                        steps = {
                            'research': output_data.get('research', ''),
                            'analyze': output_data.get('analyze', ''),
                            'write': output_data.get('write', '')
                        }

                    # Print extra debug info about model inputs
                    print("\n--- MODEL INPUTS/OUTPUTS DEBUG INFO ---")
                    models_trace = trace.get('models', {})
                    for model_id, model_calls in models_trace.items():
                        for call in model_calls:
                            print(f"\nModel: {model_id}")
                            print(f"Role: {call.get('parameters', {}).get('role', 'unknown')}")
                            print(f"Task: {call.get('parameters', {}).get('task', 'unknown')}")

                            # Check for the DEBUG-MARKER in the input prompt
                            prompt = call.get('input_prompt', '')
                            if check_debug_marker(prompt):
                                print("✅ DEBUG-MARKER found in input prompt")
                            else:
                                print("❌ DEBUG-MARKER NOT found in input prompt")

                            # Check for template variables in the input prompt
                            if check_for_template_variables(prompt):
                                print("⚠️ Input prompt contains unfilled template variables!")

                            # Print input prompt length
                            print(f"Input prompt length: {call.get('input_prompt_length', 0)} characters")

                            # Print the first 200 characters of the prompt for inspection
                            if isinstance(prompt, str) and prompt:
                                print(f"Prompt (first 200 chars): {prompt[:200]}...")

                    print("\n--- WORKFLOW STEPS OUTPUT ---")
                    for step_name, content in steps.items():
                        if content:  # Only print if there's content
                            print(f"\nStep: {step_name.capitalize()}")
                            print("-" * 50)
                            if isinstance(content, str):
                                display_wrapped_text(content)

                                # Check for template variables in the output
                                if check_for_template_variables(content):
                                    print("\n⚠️ WARNING: This output contains unfilled template variables!")
                            else:
                                try:
                                    display_wrapped_text(str(content))
                                except Exception:
                                    print("[Cannot display content]")
                else:
                    print("Unable to retrieve workflow steps from trace data.")
            except Exception as e:
                print(f"Error displaying workflow steps: {e}")

            # Print execution stats
            if isinstance(response_data, dict):
                if 'execution_time' in response_data:
                    print(f"\nTotal execution time: {response_data['execution_time']:.2f} seconds")
                if 'confidence' in response_data:
                    print(f"Final confidence score: {response_data['confidence']:.3f}")

            # Save trace to file
            trace_path = output_dir / "role_based_workflow_config.json"
            try:
                if isinstance(response_data, dict) and 'trace' in response_data:
                    with open(trace_path, 'w', encoding='utf-8') as f:
                        json.dump(response_data['trace'], f, indent=2)
                    print(f"\nTrace saved successfully to {trace_path}")
                else:
                    print("\nCould not save trace: Trace data missing in response.")
            except Exception as e:
                print(f"\nError saving trace to {trace_path}: {e}")

    except (ConfigurationError, ModelError, AiEnsembleSuiteError) as e:
        logger.error(f"Ensemble Error: {str(e)}", exc_info=True)
    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}. Check model paths and config paths.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
    print("\nWorkflow example finished.")
