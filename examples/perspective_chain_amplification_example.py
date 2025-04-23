#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cyclical Perspective Chain Amplification Example.

This module demonstrates a cyclical implementation of Perspective Chain Amplification (PCA),
a technique where multiple mental perspectives are sequentially consulted and refined
to develop a more comprehensive understanding of a complex question.

The cyclical PCA process involves:
1. Initial independent perspectives (Optimist, Pessimist, Realist)
2. Refinement round where each perspective responds to others
3. Final round where perspectives give their concluding thoughts
4. Synthesis of all perspectives into a comprehensive response

This implementation follows the principle that deliberately cycling through different
mental perspectives allows for a more thorough analysis of complex questions.
"""

import asyncio
import json
import sys
import textwrap
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add the src directory to the path if running from the examples directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

try:
    from ai_ensemble_suite import Ensemble
    from ai_ensemble_suite.utils.logging import logger
    from ai_ensemble_suite.exceptions import (
        AiEnsembleSuiteError,
        ConfigurationError,
        ModelError
    )
except ImportError as e:
    print("Error: Could not import ai_ensemble_suite. "
          "Ensure it's installed or the src directory is in the Python path.")
    print(f"Current sys.path: {sys.path}")
    print(f"Import error details: {e}")
    sys.exit(1)

# Get the models directory
models_dir = project_root / "models"

# Cyclical Perspective Chain Amplification Configuration
CYCLICAL_PCA_CONFIG = {
    "models": {
        # Each model will take a specific perspective role in the chain
        "optimist_model": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
            "role": "optimist",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000,
                "n_gpu_layers": -1  # Use GPU if available
            }
        },
        "pessimist_model": {
            "path": str(models_dir / "openhermes-2.5-mistral-7b.Q6_K.gguf"),
            "role": "pessimist",
            "parameters": {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1000,
                "n_gpu_layers": -1  # Use GPU if available
            }
        },
        "realist_model": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
            "role": "realist",
            "parameters": {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1000,
                "n_gpu_layers": -1  # Use GPU if available
            }
        },
        "synthesis_model": {
            "path": str(models_dir / "openhermes-2.5-mistral-7b.Q6_K.gguf"),
            "role": "synthesizer",
            "parameters": {
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 2000,
                "n_gpu_layers": -1  # Use GPU if available
            }
        }
    },
    "collaboration": {
        "mode": "chain",
        "phases": [
            # ====== CYCLE 1: Initial independent perspectives ======
            # Each perspective gives their initial view without seeing others
            {
                "name": "optimist_initial",
                "type": "async_thinking",
                "models": ["optimist_model"],
                "prompt_template": "optimist_initial_template"
            },
            {
                "name": "pessimist_initial",
                "type": "async_thinking",
                "models": ["pessimist_model"],
                "prompt_template": "pessimist_initial_template"
            },
            {
                "name": "realist_initial",
                "type": "async_thinking",
                "models": ["realist_model"],
                "prompt_template": "realist_initial_template"
            },

            # ====== CYCLE 2: Refinement based on other perspectives ======
            # Each perspective responds to the others' initial views
            {
                "name": "optimist_refinement",
                "type": "async_thinking",
                "models": ["optimist_model"],
                "prompt_template": "optimist_refinement_template",
                "input_from": ["pessimist_initial", "realist_initial"]
            },
            {
                "name": "pessimist_refinement",
                "type": "async_thinking",
                "models": ["pessimist_model"],
                "prompt_template": "pessimist_refinement_template",
                "input_from": ["optimist_initial", "realist_initial"]
            },
            {
                "name": "realist_refinement",
                "type": "async_thinking",
                "models": ["realist_model"],
                "prompt_template": "realist_refinement_template",
                "input_from": ["optimist_initial", "pessimist_initial"]
            },

            # ====== CYCLE 3: Final perspectives after seeing all refinements ======
            # Each perspective gives their final view after seeing how others refined their thinking
            {
                "name": "optimist_final",
                "type": "async_thinking",
                "models": ["optimist_model"],
                "prompt_template": "optimist_final_template",
                "input_from": ["pessimist_refinement", "realist_refinement"]
            },
            {
                "name": "pessimist_final",
                "type": "async_thinking",
                "models": ["pessimist_model"],
                "prompt_template": "pessimist_final_template",
                "input_from": ["optimist_refinement", "realist_refinement"]
            },
            {
                "name": "realist_final",
                "type": "async_thinking",
                "models": ["realist_model"],
                "prompt_template": "realist_final_template",
                "input_from": ["optimist_refinement", "pessimist_refinement"]
            },

            # ====== FINAL PHASE: Synthesis of all evolved perspectives ======
            # Final model integrates all perspectives after they've evolved through dialogue
            {
                "name": "synthesis_phase",
                "type": "async_thinking",
                "models": ["synthesis_model"],
                "prompt_template": "synthesis_template",
                "input_from": ["optimist_final", "pessimist_final", "realist_final"]
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "synthesis_phase"  # The final output comes from the synthesis phase
    },
    "templates": {
        # ====== CYCLE 1 TEMPLATES: Initial independent perspectives ======
        "optimist_initial_template": """You are an OPTIMIST AI assistant. Please analyze the following question, focusing exclusively on positive outcomes, benefits, and opportunities.

Question: {{ query }}

Your optimistic analysis:""",

        "pessimist_initial_template": """You are a PESSIMIST AI assistant. Please analyze the following question, focusing exclusively on potential risks, downsides, and challenges.

Question: {{ query }}

Your pessimistic analysis:""",

        "realist_initial_template": """You are a REALIST AI assistant. Please analyze the following question, focusing on a practical, balanced assessment that considers both pros and cons with a pragmatic perspective.

Question: {{ query }}

Your realistic analysis:""",

        # ====== CYCLE 2 TEMPLATES: Refinement based on other perspectives ======
        "optimist_refinement_template": """You are an OPTIMIST AI assistant. You've been presented with pessimist and realist perspectives on this question. 

Original question: {{ query }}

PESSIMIST perspective: 
{{ pessimist_initial.outputs.pessimist_model }}

REALIST perspective:
{{ realist_initial.outputs.realist_model }}

Now, refine your optimistic perspective. Consider the points raised by others, but maintain your optimistic lens. What benefits and opportunities still exist? Which criticisms can you constructively address?

Your refined optimistic analysis:""",

        "pessimist_refinement_template": """You are a PESSIMIST AI assistant. You've been presented with optimist and realist perspectives on this question. 

Original question: {{ query }}

OPTIMIST perspective: 
{{ optimist_initial.outputs.optimist_model }}

REALIST perspective:
{{ realist_initial.outputs.realist_model }}

Now, refine your pessimistic perspective. Consider the points raised by others, but maintain your pessimistic lens. What risks and challenges are still relevant? Which optimistic claims should be tempered?

Your refined pessimistic analysis:""",

        "realist_refinement_template": """You are a REALIST AI assistant. You've been presented with optimist and pessimist perspectives on this question. 

Original question: {{ query }}

OPTIMIST perspective: 
{{ optimist_initial.outputs.optimist_model }}

PESSIMIST perspective:
{{ pessimist_initial.outputs.pessimist_model }}

Now, refine your realistic perspective. Consider the points raised by others while maintaining your balanced, pragmatic lens. How can you integrate valid points from both perspectives?

Your refined realistic analysis:""",

        # ====== CYCLE 3 TEMPLATES: Final perspectives after seeing all refinements ======
        "optimist_final_template": """You are an OPTIMIST AI assistant in the final round of perspective refinement.

Original question: {{ query }}

The perspectives have evolved through discussion. Here are the refined perspectives from others:

REFINED PESSIMIST perspective:
{{ pessimist_refinement.outputs.pessimist_model }}

REFINED REALIST perspective:
{{ realist_refinement.outputs.realist_model }}

Provide your final optimistic perspective that acknowledges these refined views while still highlighting the core opportunities and benefits. Focus on the most compelling positive aspects that remain valid even after considering the other perspectives:""",

        "pessimist_final_template": """You are a PESSIMIST AI assistant in the final round of perspective refinement.

Original question: {{ query }}

The perspectives have evolved through discussion. Here are the refined perspectives from others:

REFINED OPTIMIST perspective:
{{ optimist_refinement.outputs.optimist_model }}

REFINED REALIST perspective:
{{ realist_refinement.outputs.realist_model }}

Provide your final pessimistic perspective that acknowledges these refined views while still highlighting the core risks and challenges. Focus on the most important concerns that remain valid even after considering the other perspectives:""",

        "realist_final_template": """You are a REALIST AI assistant in the final round of perspective refinement.

Original question: {{ query }}

The perspectives have evolved through discussion. Here are the refined perspectives from others:

REFINED OPTIMIST perspective:
{{ optimist_refinement.outputs.optimist_model }}

REFINED PESSIMIST perspective:
{{ pessimist_refinement.outputs.pessimist_model }}

Provide your final realistic perspective that integrates insights from all viewpoints. What is a balanced, practical assessment that acknowledges both the opportunities and challenges while offering a pragmatic middle ground?""",

        # ====== FINAL SYNTHESIS TEMPLATE ======
        "synthesis_template": """You are a SYNTHESIS AI assistant. Your job is to create a comprehensive response that truly integrates all three perspectives after they have engaged in a dialogue and evolved their viewpoints.

IMPORTANT INSTRUCTIONS:
1. Your synthesis should show how each perspective has EVOLVED through the dialogue process
2. Highlight areas of convergence and persistent differences between perspectives
3. Create a response that demonstrates the value of the cyclical perspective-taking process
4. Structure your response to reflect the dialectical nature of the process

Original question: {{ query }}

FINAL OPTIMIST perspective (after two rounds of refinement):
{{ optimist_final.outputs.optimist_model }}

FINAL PESSIMIST perspective (after two rounds of refinement):
{{ pessimist_final.outputs.pessimist_model }}

FINAL REALIST perspective (after two rounds of refinement):
{{ realist_final.outputs.realist_model }}

Your synthesized response MUST show how these perspectives evolved through dialogue and what we can learn from this cyclical process:"""
    }
}


def display_wrapped_text(text: str, width: int = 80) -> None:
    """
    Display text with proper wrapping for readability.

    Args:
        text: The text to display with wrapping
        width: Maximum line width for wrapped text (default: 80 characters)
    """
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


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy data types in trace output.

    Extends the standard JSONEncoder to properly serialize NumPy data types
    when generating trace files.
    """

    def default(self, obj: Any) -> Any:
        """
        Convert NumPy types to standard Python types for JSON serialization.

        Args:
            obj: The object to encode

        Returns:
            JSON-serializable version of the object

        Raises:
            TypeError: If object cannot be serialized to JSON
        """
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


async def run_cyclical_pca() -> None:
    """
    Run the cyclical Perspective Chain Amplification example.

    This function:
    1. Initializes the ensemble with cyclical PCA configuration
    2. Runs the query through the perspective cycle process
    3. Displays the evolution of each perspective through three cycles
    4. Presents the final synthesis
    5. Saves execution traces for analysis

    Raises:
        ConfigurationError: If there's an issue with the ensemble configuration
        ModelError: If there's an issue with model loading or execution
        AiEnsembleSuiteError: For other AI Ensemble Suite related errors
    """
    logger.set_level("INFO")

    # Define paths relative to this script's location
    config_dir = script_dir / "config"
    output_dir = script_dir / "output"
    config_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    config_path = config_dir / "cyclical_pca_config.yaml"

    # Delete existing config file if it exists
    if config_path.exists():
        try:
            config_path.unlink()
            print(f"Deleted existing config file: {config_path}")
        except Exception as e:
            print(f"Warning: Could not delete existing config file {config_path}: {e}")

    # Check if models exist
    for model_id, model_config in CYCLICAL_PCA_CONFIG["models"].items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error("Please ensure you have downloaded the required models to the 'models' directory.")
            sys.exit(1)

    # Always use the config dict since we've deleted any existing config file
    ensemble_kwargs = {'config_dict': CYCLICAL_PCA_CONFIG}
    logger.info("Using defined CYCLICAL_PCA_CONFIG and saving to file.")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(CYCLICAL_PCA_CONFIG, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved config to {config_path}")
    except Exception as e:
        logger.error(f"Could not save config to {config_path}: {e}")

    try:
        logger.info("Initializing Ensemble for Cyclical Perspective Chain Amplification...")
        # Initialize the ensemble using determined kwargs
        async with Ensemble(**ensemble_kwargs) as ensemble:
            logger.info("Ensemble initialized.")

            # Display Hardware Config
            print("\n--- Model Hardware Configuration ---")
            try:
                model_ids = ensemble.model_manager.get_model_ids()
                if not model_ids:
                    print("No models initialized.")
                else:
                    for model_id in model_ids:
                        try:
                            model_config = ensemble.config_manager.get_model_config(model_id)
                            params = model_config.get('parameters', {})
                            n_gpu_layers = params.get('n_gpu_layers', 0)
                            usage = ("GPU (Attempting Max Layers)" if n_gpu_layers == -1
                                    else (f"GPU ({n_gpu_layers} Layers)" if n_gpu_layers > 0 else "CPU"))
                            print(f"- Model '{model_id}' ({model_config.get('role', 'unknown')}): Configured for {usage}")
                        except (ConfigurationError, KeyError, ModelError) as cfg_err:
                           print(f"- Model '{model_id}': Error retrieving config - {cfg_err}")
            except Exception as e:
                print(f"Error retrieving model hardware config: {e}")
            print("------------------------------------")

            # Define a question that benefits from multiple perspectives
            query = "What are the implications of artificial general intelligence for society?"

            print(f"\nQuery: {query}\n")
            print("Processing through the cyclical perspective chain...")
            print("This implementation demonstrates how perspectives evolve through multiple rounds of consideration.")

            # Get response with trace enabled for detailed analysis
            response_data = await ensemble.ask(query, trace=True)

            # Print Execution Stats
            if isinstance(response_data, dict) and 'execution_time' in response_data:
                print(f"\nTotal execution time: {response_data['execution_time']:.2f} seconds")

            # Extract trace data for analysis of the perspective evolution
            if isinstance(response_data, dict) and 'trace' in response_data:
                trace_data = response_data['trace']
                phases_trace = trace_data.get('phases', {})

                # Display the evolution of perspectives through the three cycles
                print("\n--- Evolution of Perspectives Through Cyclical Process ---")

                # Dictionary mapping perspective types to their phase names in each round
                perspective_phases = {
                    "OPTIMIST": ["optimist_initial", "optimist_refinement", "optimist_final"],
                    "PESSIMIST": ["pessimist_initial", "pessimist_refinement", "pessimist_final"],
                    "REALIST": ["realist_initial", "realist_refinement", "realist_final"]
                }

                # Display each perspective's evolution through the rounds
                for perspective, phases in perspective_phases.items():
                    print(f"\n{perspective} PERSPECTIVE EVOLUTION:")
                    print("=" * 50)

                    for i, phase_name in enumerate(phases):
                        round_names = ["INITIAL VIEW", "REFINEMENT AFTER DIALOGUE", "FINAL PERSPECTIVE"]
                        phase_data = phases_trace.get(phase_name, {})
                        phase_output = phase_data.get('output_data', {}).get('outputs', {})

                        if phase_output:
                            # Get the model name for this phase
                            model_id = list(phase_output.keys())[0] if phase_output else "unknown"

                            print(f"\n--- CYCLE {i+1}: {round_names[i]} ---")

                            # Extract the perspective text
                            model_output = phase_output.get(model_id)
                            if isinstance(model_output, str):
                                display_wrapped_text(model_output)
                            elif isinstance(model_output, dict) and 'text' in model_output:
                                display_wrapped_text(model_output['text'])
                            else:
                                print(f"[Could not extract text from model output]")

                # Print Final Synthesized Response
                print("\nFINAL SYNTHESIZED RESPONSE (INTEGRATING ALL EVOLVED PERSPECTIVES):")
                print("=" * 80)
                final_response = "[No Response Received]"
                if isinstance(response_data, dict) and 'response' in response_data:
                    final_response = response_data.get('response', final_response)
                elif isinstance(response_data, str):
                    final_response = response_data
                display_wrapped_text(final_response)
                print("=" * 80)

            # Save Trace for detailed analysis
            trace_path = output_dir / "cyclical_pca_trace.json"
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
        logger.error(f"Ensemble Error in cyclical PCA example: {str(e)}", exc_info=True)
    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}. Check model/config paths.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the cyclical PCA example: {str(e)}", exc_info=True)


async def main() -> None:
    """
    Main entry point for the Cyclical PCA example.

    Calls the run_cyclical_pca function and handles top-level program flow.
    """
    await run_cyclical_pca()
    print("\nCyclical Perspective Chain Amplification example completed.")
    print("\nThis implementation demonstrates how deliberately cycling through different")
    print("mental perspectives in multiple rounds allows for deeper consideration")
    print("and more comprehensive understanding of complex questions.")


if __name__ == "__main__":
    asyncio.run(main())
