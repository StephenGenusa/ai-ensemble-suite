"""Example demonstrating perspective chain amplification."""

import asyncio
import os
import sys
import yaml
import textwrap
from pathlib import Path
import json

# Add the src directory to the path if running from the examples directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

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

# Get the models directory
models_dir = project_root / "models"

# Perspective Chain Amplification Configuration
PERSPECTIVE_CHAIN_CONFIG = {
    "models": {
        "optimist_model": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
            "role": "optimist",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000,
                "n_gpu_layers": -1 # Use GPU if available
            }
        },
        "pessimist_model": {
            "path": str(models_dir / "openhermes-2.5-mistral-7b.Q6_K.gguf"),
            "role": "pessimist",
            "parameters": {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1000,
                "n_gpu_layers": -1 # Use GPU if available
            }
        },
        "pragmatist_model": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
            "role": "pragmatist",
            "parameters": {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1000,
                "n_gpu_layers": -1 # Use GPU if available
            }
        },
        "synthesis_model": {
            "path": str(models_dir / "openhermes-2.5-mistral-7b.Q6_K.gguf"),
            "role": "synthesizer",
            "parameters": {
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 2000,
                "n_gpu_layers": -1 # Use GPU if available
            }
        }
    },
    "collaboration": {
        "mode": "chain",
        "phases": [
            {
                "name": "optimist_phase",
                "type": "async_thinking",
                "models": ["optimist_model"],
                "prompt_template": "optimist_template"
            },
            {
                "name": "pessimist_phase",
                "type": "async_thinking",
                "models": ["pessimist_model"],
                "prompt_template": "pessimist_template",
                "input_from": "optimist_phase"
            },
            {
                "name": "pragmatist_phase",
                "type": "async_thinking",
                "models": ["pragmatist_model"],
                "prompt_template": "pragmatist_template",
                "input_from": ["optimist_phase", "pessimist_phase"]
            },
            {
                "name": "synthesis_phase",
                "type": "async_thinking",
                "models": ["synthesis_model"],
                "prompt_template": "synthesis_template",
                "input_from": ["optimist_phase", "pessimist_phase", "pragmatist_phase"]
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "synthesis_phase"
    },
    "templates": {
        "optimist_template": """You are an OPTIMIST AI assistant. Please analyze the following question with a focus on positive outcomes, benefits, and opportunities.
Be thorough but maintain your optimistic perspective.

Question: {{ query }}

Your optimistic analysis:""",

        "pessimist_template": """You are a PESSIMIST AI assistant. You will now analyze the same question, but focus on potential risks, downsides, and challenges.

The question was: {{ query }}

The OPTIMIST'S perspective was:
{{ optimist_phase.outputs.optimist_model }}

Now provide your PESSIMIST analysis:""",

        "pragmatist_template": """You are a PRAGMATIST AI assistant. You will now analyze the same question with a balanced, practical perspective.

The question was: {{ query }}

The OPTIMIST'S perspective was:
{{ optimist_phase.outputs.optimist_model }}

The PESSIMIST'S perspective was:
{{ pessimist_phase.outputs.pessimist_model }}

Now provide your PRAGMATIST analysis:""",

        "synthesis_template": """You are a SYNTHESIS AI assistant. Your job is to create a comprehensive, balanced response that truly integrates all three perspectives.

IMPORTANT INSTRUCTIONS:
1. DO NOT copy or primarily favor any single perspective.
2. You MUST create an original comprehensive synthesis that draws equally from all three views.
3. Your response should be structured in THREE DISTINCT SECTIONS:
   - First section: The opportunities and benefits with elaboration
   - Second section: The risks and challenges with elaboration 
   - Third section: Balanced approaches and practical considerations with elaboration 
4. Your conclusion should reflect a true integration of all three viewpoints.

Original question: {{ query }}

OPTIMIST perspective (benefits and opportunities):
{{ optimist_phase.outputs.optimist_model }}

PESSIMIST perspective (risks and challenges):
{{ pessimist_phase.outputs.pessimist_model }}

PRAGMATIST perspective (balanced view and practical considerations):
{{ pragmatist_phase.outputs.pragmatist_model }}

Your synthesized response MUST represent a thoughtful, ORIGINAL integration that doesn't favor any single perspective 
and does not plagiarize the content of any view:"""
    }
}

async def main():
    """Run perspective chain amplification example."""
    logger.set_level("INFO")

    # Define paths relative to this script's location
    config_dir = script_dir / "config"
    output_dir = script_dir / "output"
    config_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    config_path = config_dir / "perspective_chain_config.yaml"

    # Delete existing config file if it exists
    if config_path.exists():
        try:
            config_path.unlink()
            print(f"Deleted existing config file: {config_path}")
        except Exception as e:
            print(f"Warning: Could not delete existing config file {config_path}: {e}")

    # Check if models exist
    for model_id, model_config in PERSPECTIVE_CHAIN_CONFIG["models"].items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Please ensure you have downloaded the required models to the 'models' directory.")
            sys.exit(1)

    # Always use the config dict since we've deleted any existing config file
    ensemble_kwargs = {'config_dict': PERSPECTIVE_CHAIN_CONFIG}
    logger.info(f"Using default PERSPECTIVE_CHAIN_CONFIG defined in script and saving.")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(PERSPECTIVE_CHAIN_CONFIG, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved default config to {config_path}")
    except Exception as e:
        logger.error(f"Could not save default config to {config_path}: {e}")

    # Define text-wrapping function
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

        print(f"\n[Response length: {len(text)} characters]")

    try:
        logger.info("Initializing Ensemble for Perspective Chain Amplification...")
        # Initialize the ensemble using determined kwargs
        async with Ensemble(**ensemble_kwargs) as ensemble:
            logger.info("Ensemble initialized.")

            # Display Hardware Config
            print("\n--- Model Hardware Configuration ---")
            try:
                model_ids = ensemble.model_manager.get_model_ids()
                if not model_ids: print("No models initialized.")
                else:
                    for model_id in model_ids:
                        try:
                            model_config = ensemble.config_manager.get_model_config(model_id)
                            params = model_config.get('parameters', {})
                            n_gpu_layers = params.get('n_gpu_layers', 0)
                            usage = "GPU (Attempting Max Layers)" if n_gpu_layers == -1 else (f"GPU ({n_gpu_layers} Layers)" if n_gpu_layers > 0 else "CPU")
                            print(f"- Model '{model_id}' ({model_config.get('role', 'unknown')}): Configured for {usage}")
                        except (ConfigurationError, KeyError, ModelError) as cfg_err:
                           print(f"- Model '{model_id}': Error retrieving config - {cfg_err}")
            except Exception as e: print(f"Error retrieving model hardware config: {e}")
            print("------------------------------------")

            # Define a question that benefits from multiple perspectives
            query = "What are the implications of artificial general intelligence for society?"

            print(f"\nQuery: {query}\n")
            print("Processing through the perspective chain...")

            # Get response with trace enabled
            response_data = await ensemble.ask(query, trace=True)


            # Print Execution Stats
            if isinstance(response_data, dict):
                 if 'execution_time' in response_data:
                     print(f"\nTotal execution time: {response_data['execution_time']:.2f} seconds")

            # Print Individual Model Perspectives
            if isinstance(response_data, dict) and 'trace' in response_data:
                trace_data = response_data['trace']
                phases_trace = trace_data.get('phases', {})

                print("\n--- Individual Perspectives ---")
                # Extract and display perspectives from each phase
                for phase_name in ['optimist_phase', 'pessimist_phase', 'pragmatist_phase']:
                    phase_data = phases_trace.get(phase_name, {})
                    phase_output = phase_data.get('output_data', {}).get('outputs', {})

                    if phase_output:
                        # Get the model name for this phase
                        model_id = list(phase_output.keys())[0] if phase_output else "unknown"
                        perspective_role = phase_name.split('_')[0].upper()

                        print(f"\n{perspective_role} PERSPECTIVE (from model '{model_id}'):")
                        print("-" * 50)

                        # Extract the perspective text
                        model_output = phase_output.get(model_id)
                        if isinstance(model_output, str):
                            display_wrapped_text(model_output)
                        elif isinstance(model_output, dict) and 'text' in model_output:
                            display_wrapped_text(model_output['text'])
                        else:
                            print(f"[Could not extract text from model output of type: {type(model_output)}]")

                # Print Final Synthesized Response with word-wrapping
                print("\nFinal Synthesized Response:")
                print("=" * 80)
                final_response = "[No Response Received]"
                if isinstance(response_data, dict) and 'response' in response_data:
                    final_response = response_data.get('response', final_response)
                elif isinstance(response_data, str):
                    final_response = response_data
                display_wrapped_text(final_response)
                print("=" * 80)
                print("-" * 80)

            # Save Trace
            trace_path = output_dir / "perspective_chain_trace.json"
            try:
                if isinstance(response_data, dict) and 'trace' in response_data:
                    # Custom JSON encoder to handle potential numpy types
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            try:
                                # If NumPy is available, handle its types
                                import numpy as np
                                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                                    np.int16, np.int32, np.int64, np.uint8,
                                                    np.uint16, np.uint32, np.uint64)):
                                    return int(obj)
                                elif isinstance(obj, (np.float_, np.float16, np.float32,
                                                      np.float64)):
                                    return float(obj)
                                elif isinstance(obj, (np.ndarray,)):
                                    return obj.tolist()
                            except ImportError:
                                pass
                            return super().default(obj)

                    with open(trace_path, 'w', encoding='utf-8') as f:
                        json.dump(response_data['trace'], f, indent=2, cls=NumpyEncoder)
                    print(f"\nTrace saved successfully to {trace_path}")
                else:
                    print("\nCould not save trace: Trace data missing or response format invalid.")
            except Exception as e:
                print(f"\nError saving trace to {trace_path}: {e}")


    except (ConfigurationError, ModelError, AiEnsembleSuiteError) as e:
        logger.error(f"Ensemble Error in perspective chain example: {str(e)}", exc_info=True)
    except FileNotFoundError as e:
         logger.error(f"File Not Found Error: {e}. Check model/config paths.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the perspective chain example: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
    print("\nPerspective Chain Amplification example finished.")
