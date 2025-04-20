"""Example demonstrating ensemble fusion aggregation."""

import asyncio
import os
import sys
import yaml
import textwrap  # Added for word-wrapping
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

# Default Configuration (Embedded)
FUSION_CONFIG = {
    "models": {
        "model1": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
            "role": "primary",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1500,
                "n_gpu_layers": -1 # Use GPU if available
            }
        },
        "model2": {
            "path": str(models_dir / "openhermes-2.5-mistral-7b.Q6_K.gguf"),
            "role": "assistant",
            "parameters": {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1500,
                "n_gpu_layers": -1 # Use GPU if available
            }
        },
        "model3": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
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
        "mode": "async_thinking", # Use a standard mode that's guaranteed to work
        "phases": [
            {
                "name": "perspectives",
                "type": "async_thinking",
                "models": ["model1", "model2"], # Models to generate initial perspectives
                "prompt_template": "single_query"
            }
        ]
    },
    "aggregation": {
        "strategy": "ensemble_fusion",
        "fusion_model": "model3",
        "fusion_template": "ensemble_fusion",
        "source_phase": "perspectives",
        "final_phase": "perspectives"  # Set to a valid phase name to satisfy validation
    },
    "templates": {
        "single_query": "You are a helpful AI assistant. Please answer the following question:\n\n{{ query }}",
        "ensemble_fusion": """You are a helpful AI assistant tasked with synthesizing multiple perspectives into a single, coherent, and high-quality response.
Analyze the different viewpoints provided below, identify the most accurate information, clearest explanations, and strongest arguments.
Then, create a unified response that integrates the best aspects of each input. Maintain a consistent tone and logical structure.

Original Query:
{{ query }}

Perspectives Provided:
{{ fusion_input }}

Your Synthesized Response:
""",
        "self_evaluation": "On a scale of 1-10, how confident are you in this response: {{ response }}? Only provide the number.",
    }
}

async def main():
    """Run ensemble fusion example."""
    logger.set_level("INFO")

    # Define paths relative to this script's location
    config_dir = script_dir / "config"
    output_dir = script_dir / "output"
    config_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    config_path = config_dir / "fusion_config.yaml"

    # Delete existing config file if it exists
    if config_path.exists():
        try:
            config_path.unlink()
            print(f"Deleted existing config file: {config_path}")
        except Exception as e:
            print(f"Warning: Could not delete existing config file {config_path}: {e}")

    # Check if models exist
    for model_id, model_config in FUSION_CONFIG["models"].items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Please ensure you have downloaded the required models to the 'models' directory.")
            sys.exit(1)

    # Always use the config dict since we've deleted any existing config file
    ensemble_kwargs = {'config_dict': FUSION_CONFIG}
    logger.info(f"Using default FUSION_CONFIG defined in script and saving.")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(FUSION_CONFIG, f, default_flow_style=False, sort_keys=False)
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
        logger.info("Initializing Ensemble for Fusion...")
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
                            print(f"- Model '{model_id}': Configured for {usage}")
                        except (ConfigurationError, KeyError, ModelError) as cfg_err:
                           print(f"- Model '{model_id}': Error retrieving config - {cfg_err}")
            except Exception as e: print(f"Error retrieving model hardware config: {e}")
            print("------------------------------------")

            # Ask a question that can benefit from multiple perspectives
            query = "What are the potential benefits and risks of quantum computing for cybersecurity?"

            print(f"\nQuery: {query}\n")
            print("Processing with multiple models and ensemble fusion...")

            # Get response with trace enabled
            response_data = await ensemble.ask(query, trace=True)

            # Print Fused Response with word-wrapping
            print("\nFused Response:")
            print("=" * 80)
            final_response = "[No Response Received]"
            if isinstance(response_data, dict) and 'response' in response_data:
                 final_response = response_data.get('response', final_response)
            elif isinstance(response_data, str):
                 final_response = response_data
            display_wrapped_text(final_response)
            print("=" * 80)

            # Print Execution Stats
            if isinstance(response_data, dict):
                 if 'execution_time' in response_data:
                     print(f"\nTotal execution time: {response_data['execution_time']:.2f} seconds")
                 if 'confidence' in response_data:
                     confidence_val = response_data.get('confidence')
                     print(f"Final confidence score: {confidence_val:.3f}" if confidence_val is not None else "Final confidence score: N/A")

            # Print Individual Model Outputs with word-wrapping
            if isinstance(response_data, dict) and 'trace' in response_data:
                trace_data = response_data['trace']
                try:
                    # Navigate through the trace structure - FIX HERE
                    phases_trace = trace_data.get('phases', {})
                    perspectives_phase = phases_trace.get('perspectives', {})

                    # Correctly access the outputs in phase_output_data
                    phase_output_data = perspectives_phase.get('output_data', {})
                    model_outputs = phase_output_data.get('outputs', {})

                    if model_outputs:
                        print("\n--- Individual Model Perspectives (from Phase 'perspectives') ---")
                        for model_id, model_output in model_outputs.items():
                            print(f"\nModel: {model_id}")
                            print("-" * 50)
                            # In this structure, model_output is directly the string
                            if isinstance(model_output, str):
                                display_wrapped_text(model_output)
                            elif isinstance(model_output, dict) and 'text' in model_output:
                                display_wrapped_text(model_output['text'])
                            else:
                                print(f"[Could not extract text from model output of type: {type(model_output)}]")
                        print("--------------------------------------------------------------")
                    else:
                        logger.warning("Could not find model outputs in the trace data for 'perspectives' phase.")

                except Exception as trace_parse_e:
                    logger.warning(f"Could not parse trace to show individual model outputs: {trace_parse_e}")
                    import traceback
                    traceback.print_exc()

            # Save Trace
            trace_path = output_dir / "fusion_trace.json"
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
        logger.error(f"Ensemble Error in fusion example: {str(e)}", exc_info=True)
    except FileNotFoundError as e:
         logger.error(f"File Not Found Error: {e}. Check model/config paths.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the fusion example: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
    print("\nFusion example finished.")
