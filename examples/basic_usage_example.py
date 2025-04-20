"""Basic usage example of the ai-ensemble-suite library."""

import asyncio
import os
import sys
import yaml
from pathlib import Path
import json
import textwrap

# Add the src directory to the path if running from the examples directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from ai_ensemble_suite.utils.tracing import NumpyEncoder

# Now import from the library
try:
    from ai_ensemble_suite import Ensemble
    from ai_ensemble_suite.utils.logging import logger
    from ai_ensemble_suite.exceptions import AiEnsembleSuiteError, ConfigurationError, ModelError # Keep ModelError
except ImportError as e:
    print("Error: Could not import ai_ensemble_suite. "
          "Ensure it's installed or the src directory is in the Python path.")
    print(f"Current sys.path: {sys.path}")
    print(f"Import error details: {e}")
    sys.exit(1)


# Define model paths relative to the project root or examples dir
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Basic configuration with async_thinking collaboration mode
# NOTE: This embedded config is used only if the YAML file doesn't exist.
# Modify the YAML file (e.g., config/basic_config.yaml) for persistent changes.
BASIC_CONFIG = {
    "models": {
        "mistral": {
            "path": str(MODEL_DIR / "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf"), # Ensure model exists here
            "role": "primary",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "n_ctx": 2048, # Example context size
                "max_tokens": 1024,
                # --- GPU Setting Example (falls back to 0/CPU if not present)
                # --- llama-cpp-python must be built or installed with GPU support ---
                "n_gpu_layers": -1 # Default to GPU in this config
            }
        }
    },
    "collaboration": {
        "mode": "simple_response", # A simple mode name
        "phases": [
            {
                "name": "initial_response",
                "type": "async_thinking", # Use AsyncThinking for single response
                "models": ["mistral"], # Specify the model to use
                "prompt_template": "single_query" # Reference template below
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement", # Selects output from final phase
        "final_phase": "initial_response" # The only phase is the final one here
    },
    "templates": {
        "single_query": "You are a helpful AI assistant. Please answer the following question:\n\n{{ query }}",
        "self_evaluation": "On a scale of 1-10, how confident are you in this response: {{ response }}? Only provide the number.",
    }
}

async def main():
    """Run basic example."""
    logger.set_level("INFO")

    # Define paths relative to this script's location
    config_dir = Path(__file__).resolve().parent / "config"
    output_dir = Path(__file__).resolve().parent / "output"
    config_path = config_dir / "basic_config.yaml"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine configuration source
    ensemble_kwargs = {}
    if config_path.exists():
        logger.info(f"Using config file: {config_path}")
        ensemble_kwargs['config_path'] = str(config_path)
    else:
        logger.info(f"Config file '{config_path.name}' not found in '{config_dir}'. Using default config defined in script and saving.")
        config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(BASIC_CONFIG, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Could not save default config to {config_path}: {e}")
        ensemble_kwargs['config_dict'] = BASIC_CONFIG

    # --- Check if model files exist before initializing ---
    config_source = ensemble_kwargs.get('config_dict') or {}
    if 'config_path' in ensemble_kwargs:
         try:
             with open(ensemble_kwargs['config_path'], 'r', encoding='utf-8') as f:
                 loaded_yaml = yaml.safe_load(f)
                 if loaded_yaml: # Check if file is not empty
                     config_source = loaded_yaml
                 else:
                     logger.warning(f"Config file {ensemble_kwargs['config_path']} is empty or invalid. Using script defaults.")
                     config_source = BASIC_CONFIG # Fallback if YAML is empty/invalid
         except Exception as e:
             logger.error(f"Could not read config file {ensemble_kwargs['config_path']} for model check: {e}. Using script defaults.")
             config_source = BASIC_CONFIG # Fallback if reading fails

    all_models_exist = True
    if isinstance(config_source, dict) and 'models' in config_source:
         for model_id, model_cfg in config_source['models'].items():
             model_file_path = model_cfg.get('path', '')
             if not Path(model_file_path).exists():
                 logger.error(f"Model file not found for '{model_id}': {model_file_path}")
                 all_models_exist = False
    if not all_models_exist:
         logger.error("One or more model files are missing. Please ensure models are downloaded and paths are correct in the config.")
         sys.exit(1)
    # --- End Model Check ---

    try:
        logger.info("Initializing Ensemble...")
        async with Ensemble(**ensemble_kwargs) as ensemble:
            logger.info("Ensemble initialized.")

            # --- Display Hardware Configuration --- # <--- NEW SECTION
            print("\n--- Model Hardware Configuration ---")
            try:
                # Get IDs of models actually managed (use model_manager for initialized models)
                model_ids = ensemble.model_manager.get_model_ids()
                if not model_ids:
                    print("No models were successfully initialized or configured.")
                # else:
                #     for model_id in model_ids:
                        # try:
                            # Get the fully resolved config for the model, including defaults
                            # model_config = ensemble.config_manager.get_model_config(model_id)
                            # params = model_config.get('parameters', {})
                            # ConfigManager applies defaults, but double-check n_gpu_layers
                            # The default in gguf_model.py is 0 if not specified anywhere.
                            # n_gpu_layers = params.get('n_gpu_layers', 0)

                            # if n_gpu_layers == -1:
                            #     usage = "GPU (Attempting Max Layers)"
                            # elif n_gpu_layers > 0:
                            #     usage = f"GPU ({n_gpu_layers} Layers)"
                            # else:
                            #     usage = "CPU"
                            # print(f"- Model '{model_id}': Configured for {usage}")
                        # except (ConfigurationError, KeyError, ModelError) as config_err:
                           # Catch errors if a model failed init but ID exists, or config is bad
                           # print(f"- Model '{model_id}': Error retrieving/interpreting config - {config_err}")
            except Exception as e:
                # Catch errors getting model IDs etc.
                print(f"Error retrieving model hardware configuration: {e}")
            print("------------------------------------")
            # --- End Hardware Configuration Display --- #

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

            # Ask a question
            query = "Explain the concept of quantum entanglement in simple terms."

            print(f"\nQuery:\n{query}\n")
            print("Processing...")

            # Get response with trace enabled
            response_data = await ensemble.ask(query, trace=True)

            # --- Print Response ---
            print("\nResponse:")
            print("=" * 80)
            final_response = "[No Response Received]"
            if isinstance(response_data, dict) and 'response' in response_data:
                 final_response = response_data.get('response', final_response)
            elif isinstance(response_data, str):
                 final_response = response_data
            display_wrapped_text(final_response)
            print("=" * 80)

            # --- Print Execution Stats ---
            if isinstance(response_data, dict):
                if 'execution_time' in response_data:
                    print(f"\nTotal execution time: {response_data['execution_time']:.2f} seconds")
                if 'confidence' in response_data:
                    confidence_val = response_data.get('confidence') # Use get for safety
                    if confidence_val is not None:
                        print(f"Final confidence score: {confidence_val:.3f}")
                    else:
                        print("Final confidence score: N/A")

            # --- Save Trace ---
            trace_path = output_dir / "basic_trace.json"
            try:
                if isinstance(response_data, dict) and 'trace' in response_data:
                    with open(trace_path, 'w', encoding='utf-8') as f:
                        json.dump(response_data['trace'], f, indent=2, cls=NumpyEncoder)
                    print(f"\nTrace saved successfully to {trace_path}")
                else:
                    print("\nCould not save trace: Trace data missing or response format invalid.")
            except Exception as e:
                print(f"\nError saving trace to {trace_path}: {e}")

    except (ConfigurationError, ModelError, AiEnsembleSuiteError) as e:
        logger.error(f"Ensemble Error: {str(e)}", exc_info=True)
    except FileNotFoundError as e:
         logger.error(f"File Not Found Error: {e}. Check model/config paths.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the basic usage example: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
    print("\nBasic usage example finished.")

