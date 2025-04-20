"""Example demonstrating uncertainty-based collaboration for ensemble AI."""

import asyncio
import os
import sys
import yaml
import json
import random
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Add the src directory to the path if running from the examples directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from ai_ensemble_suite import Ensemble
from ai_ensemble_suite.utils.logging import logger

# Get the models directory
models_dir = project_root / "models"


def get_random_query():
    """Returns a randomly selected question that effectively demonstrates uncertainty-based collaboration."""
    queries = [
        # Questions with multiple valid interpretations
        "How should I approach developing a learning strategy?",

        # Open-ended questions where models might take different approaches
        "What are the ethical considerations in genetic engineering?",

        # Questions with potentially competing viewpoints
        "What are the pros and cons of working remotely versus in an office?",

        # Questions requiring nuanced reasoning
        "How should parents balance screen time restrictions with digital literacy?",

        # Questions with inherent uncertainty
        "How might artificial intelligence impact employment over the next decade?",

        # Questions where confidence calibration matters
        "What investment strategy is best for a mid-career professional?",

        # Questions where different expertise might lead to different answers
        "How should cities address homelessness?",

        # Questions where style and substance both matter
        "What makes for effective leadership in times of crisis?"
    ]

    return random.choice(queries)


# Configuration with uncertainty-based collaboration
UNCERTAINTY_CONFIG = {
    "models": {
        "mistral": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
            "role": "primary",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024,
                "n_ctx": 8192,
                "n_gpu_layers": -1
            }
        },
        "deepseek": {
            "path": str(models_dir / "DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf"),
            "role": "primary",
            "parameters": {
                "temperature": 0.75,
                "top_p": 0.9,
                "max_tokens": 2048,
                "n_ctx": 8192,
                "n_gpu_layers": -1
            }
        },
        "llama": {
            "path": str(models_dir / "llama-2-7b-chat.Q6_K.gguf"),
            "role": "primary",
            "parameters": {
                "temperature": 0.80,
                "top_p": 0.9,
                "max_tokens": 2048,
                "n_ctx": 4096,
                "n_gpu_layers": -1,
                "repeat_penalty": 1.18,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.2,
            },
            "prompt_template": "<s>[INST] <<SYS>>\nYou are a helpful, detailed assistant. Always provide comprehensive answers to questions. NEVER refuse to answer a question or ask for more information.\n<</SYS>>\n\n{prompt} [/INST]",
        }
    },
    "collaboration": {
        "mode": "custom",
        "phases": [
            {
                "name": "model_responses",
                "type": "bagging",
                "models": ["mistral", "deepseek", "llama"],
                "prompt_template": "response_prompt",
                "variation_strategy": "instruction_variation",
                "sample_ratio": 0.9,
                "num_variations": 3,
                "aggregation_method": "none"  # Don't aggregate within bagging
            },
            {
                "name": "uncertainty_collaboration",
                "type": "uncertainty_based",  # Using our new collaboration type
                "models": ["mistral", "deepseek", "llama"],
                "input_from": ["model_responses"],
                "prompt_template": "uncertainty_prompt",
                "uncertainty_metric": "disagreement",
                "selection_method": "consensus",
                "confidence_threshold": 0.7,
                "refinement_iterations": 2,
                "adaptive_selection": True
            }
        ]
    },
    "aggregation": {
        "strategy": "passthrough",  # Pass through the final phase output
        "final_phase": "uncertainty_collaboration"
    },
    "templates": {
        "response_prompt": """You are an expert AI assistant. Please provide a comprehensive, accurate, and insightful response to the following query:

QUERY: {{ query }}

Analyze the question thoroughly before answering. Be specific and include relevant details in your response.""",

        "uncertainty_prompt": """You are evaluating multiple AI responses to a query to determine the best answer.

ORIGINAL QUERY: {{ query }}

The model_responses phase has provided several different potential answers:

{% for var_id, output in model_responses.outputs.items() %}
--- MODEL RESPONSE {{ loop.index }} ---
{{ output }}

{% endfor %}

Your task:
1. Evaluate the quality and consistency of the responses
2. Identify any areas of disagreement or uncertainty
3. Provide a refined, comprehensive response that addresses any inconsistencies
4. Focus on accuracy and clarity in your answer"""}
}


def generate_model_performance_chart(trace_data, config, output_dir):
    """Generate a comparative chart of model performance metrics with max token context."""
    if not trace_data or 'models' not in trace_data:
        return

    plt.figure(figsize=(12, 8))

    # Extract model data
    model_stats = {}
    model_outputs = {}
    primary_models = []
    max_tokens = {}

    # Get max_tokens from config
    for model_id, model_config in config.get('models', {}).items():
        max_tokens[model_id] = model_config.get('parameters', {}).get('max_tokens', 1024)

    for model_id, runs in trace_data['models'].items():
        # Skip derived models (like bagging variations)
        if '_bagging_' in model_id:
            continue

        primary_models.append(model_id)

        # Get statistics for this model
        model_stats[model_id] = {
            'response_time': runs[0]['execution_time_ms'] / 1000,  # Convert to seconds
            'token_count': runs[0]['output'].get('token_count', 0) if 'output' in runs[0] else 0,
            'max_tokens': max_tokens.get(model_id, 1024),  # Default if not found
            'status': runs[0]['status']
        }

        # Calculate token utilization percentage
        if model_stats[model_id]['max_tokens'] > 0:
            model_stats[model_id]['token_utilization'] = (
                    model_stats[model_id]['token_count'] / model_stats[model_id]['max_tokens'] * 100
            )
        else:
            model_stats[model_id]['token_utilization'] = 0

        # Store output text for content analysis
        if 'output' in runs[0] and 'text' in runs[0]['output']:
            model_outputs[model_id] = runs[0]['output']['text']

    # Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

    # 1. Response time chart
    times = [model_stats[model]['response_time'] for model in primary_models]
    bars = ax1.bar(primary_models, times, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Response Time (s)')
    ax1.set_title('Model Response Time Comparison')

    # Add value labels
    for bar, value in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f'{value:.2f}s',
                 ha='center', va='bottom')

    # 2. Token count chart with max_tokens context
    token_data = []
    for model in primary_models:
        token_data.append([model_stats[model]['token_count'], model_stats[model]['max_tokens']])

    token_data = np.array(token_data)
    x = np.arange(len(primary_models))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, token_data[:, 0], width, label='Tokens Used', color='#3498db')
    bars2 = ax2.bar(x + width / 2, token_data[:, 1], width, label='Max Tokens', color='#95a5a6', alpha=0.5)

    ax2.set_ylabel('Token Count')
    ax2.set_title('Token Usage vs. Maximum Allowed')
    ax2.set_xticks(x)
    ax2.set_xticklabels(primary_models)
    ax2.legend()

    # Add value labels
    for bar, value in zip(bars1, token_data[:, 0]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f'{int(value)}',
                 ha='center', va='bottom')

    for bar, value in zip(bars2, token_data[:, 1]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50,
                 f'{int(value)}',
                 ha='center', va='bottom')

    # 3. Token utilization percentage
    utilization = [model_stats[model]['token_utilization'] for model in primary_models]
    bars = ax3.bar(primary_models, utilization, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax3.set_ylabel('Token Utilization (%)')
    ax3.set_title('Model Token Utilization Percentage')
    ax3.set_ylim(0, 100)

    # Add value labels
    for bar, value in zip(bars, utilization):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f'{value:.1f}%',
                 ha='center', va='bottom')

    # 4. Content analysis - measure lexical richness or uniqueness
    vocab_richness = {}

    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        stops = set(stopwords.words('english'))

        for model_id, text in model_outputs.items():
            if not text:
                vocab_richness[model_id] = 0
                continue

            words = nltk.word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in stops]

            if words:
                unique_words = len(set(words))
                total_words = len(words)
                vocab_richness[model_id] = unique_words / max(1, total_words)
            else:
                vocab_richness[model_id] = 0
    except Exception as e:
        print(f"NLTK processing error: {e}")
        # Fallback if NLTK fails
        for model_id, text in model_outputs.items():
            if not text:
                vocab_richness[model_id] = 0
                continue

            words = text.lower().split()
            if words:
                unique_words = len(set(words))
                total_words = len(words)
                vocab_richness[model_id] = unique_words / max(1, total_words)
            else:
                vocab_richness[model_id] = 0

    # Plot lexical diversity
    richness = [vocab_richness.get(model, 0) for model in primary_models]
    bars = ax4.bar(primary_models, richness, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax4.set_ylabel('Lexical Diversity')
    ax4.set_title('Model Response Vocabulary Richness')
    ax4.set_ylim(0, 1)

    # Add value labels
    for bar, value in zip(bars, richness):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f'{value:.2f}',
                 ha='center', va='bottom')

    # Layout adjustments
    plt.tight_layout()

    # Save the chart
    chart_path = output_dir / "model_performance_comparison.png"
    plt.savefig(chart_path)
    print(f"Model performance comparison saved to {chart_path}")
    plt.close()


def generate_uncertainty_visualization(uncertainty_metrics, selection_info, output_dir):
    """Generate visualizations of uncertainty metrics and selection information."""
    if not uncertainty_metrics:
        return

    # Create figure with metrics visualization
    plt.figure(figsize=(10, 6))

    # Display uncertainty metrics as a bar chart
    metrics_to_plot = ["disagreement", "entropy", "variance", "confidence"]
    values = [uncertainty_metrics.get(metric, 0) for metric in metrics_to_plot]

    colors = ['#ff9999', '#ffcc99', '#ffffcc', '#99ff99']  # Red to green
    bars = plt.bar(metrics_to_plot, values, color=colors)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + 0.05,
                 f'{value:.2f}',
                 ha='center', va='bottom')

    plt.ylim(0, 1.1)  # Limit y-axis for better visualization
    plt.title('Uncertainty Metrics for Collaborative Refinement')
    plt.ylabel('Score (0-1)')

    # Add annotation explaining selection
    best_model = selection_info.get("best_model", "unknown")
    best_iteration = selection_info.get("best_iteration", 0)
    confidence = selection_info.get("confidence", 0)
    refinement_count = selection_info.get("refinement_count", 0)

    explanation = (
        f"Best Model: {best_model}\n"
        f"Best Iteration: {best_iteration}\n"
        f"Refinement Cycles: {refinement_count}\n"
        f"Confidence: {confidence:.2f}"
    )

    plt.annotate(
        explanation,
        xy=(0.5, 0.02),
        xycoords='figure fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        ha='center'
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for annotation

    # Save the visualization
    chart_path = output_dir / "uncertainty_collaboration_metrics.png"
    plt.savefig(chart_path)
    print(f"Uncertainty visualization saved to {chart_path}")
    plt.close()


async def main():
    """Run uncertainty-based collaboration example."""
    # Check if models exist
    for model_id, model_config in UNCERTAINTY_CONFIG["models"].items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Please ensure you have downloaded the required models to the 'models' directory.")
            sys.exit(1)

    # Check if config file exists
    config_dir = script_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "uncertaintybased_config.yaml"

    # Delete existing config file to ensure we use the updated one
    if config_path.exists():
        os.remove(config_path)
        logger.info(f"Deleted existing config file: {config_path}")

    # Save our configuration
    with open(config_path, 'w') as f:
        yaml.dump(UNCERTAINTY_CONFIG, f, default_flow_style=False)
    logger.info(f"Created new config file: {config_path}")

    # Use the config file
    ensemble_kwargs = {'config_path': str(config_path)}

    try:
        # Initialize the ensemble with proper kwargs
        async with Ensemble(**ensemble_kwargs) as ensemble:

            # Get a random query
            query = get_random_query()

            print(f"\nQuery: {query}\n")
            print("Processing with uncertainty-based collaboration ensemble...")

            # Get response with trace
            response_data = await ensemble.ask(query, trace=True)

            # Print response
            print("\nFinal Response:")
            print("=" * 80)
            print(response_data['response'])
            print("=" * 80)

            def display_wrapped_text(text, width=70):
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

            # Create output directory for visualizations
            output_dir = script_dir / "output"
            output_dir.mkdir(exist_ok=True)

            # Display phase outputs
            if 'trace' in response_data and 'phases' in response_data['trace']:
                phases = response_data['trace']['phases']

                # Display model responses from bagging phase
                if 'model_responses' in phases:
                    bagging_data = phases['model_responses']['output_data']
                    variation_outputs = bagging_data.get('variation_outputs', {})

                    print("\nIndividual Model Responses from Bagging:")
                    print("-" * 80)

                    for i, (var_id, output) in enumerate(variation_outputs.items()):
                        print(f"\nResponse {i + 1} ({var_id}):")
                        print("-" * 50)
                        display_wrapped_text(output)

                # Display uncertainty collaboration results
                if 'uncertainty_collaboration' in phases:
                    print("\nUncertainty-Based Collaboration Results:")
                    print("-" * 80)

                    collab_data = phases['uncertainty_collaboration']['output_data']

                    # Display basic information
                    best_model = collab_data.get('best_model', 'unknown')
                    best_iteration = collab_data.get('best_iteration', 0)
                    refinement_count = collab_data.get('refinement_count', 0)
                    confidence = collab_data.get('confidence', 0)

                    print(f"Best Model: {best_model}")
                    print(f"Best Iteration: {best_iteration}")
                    print(f"Refinement Cycles: {refinement_count}")
                    print(f"Final Confidence: {confidence:.4f}")

                    # Display uncertainty metrics
                    uncertainty_metrics = collab_data.get('uncertainty_metrics', {})
                    if uncertainty_metrics:
                        print("\nUncertainty Metrics (Final):")
                        for metric, value in uncertainty_metrics.items():
                            print(f"  {metric.capitalize()}: {value:.4f}")

                    # Display iteration information
                    iterations = collab_data.get('iterations', [])
                    if iterations:
                        print("\nIteration Summary:")
                        for i, iteration in enumerate(iterations):
                            iter_confidence = iteration.get('uncertainty_metrics', {}).get('confidence', 0)
                            num_models = len(iteration.get('outputs', {}))
                            print(f"  Iteration {i}: {num_models} models, confidence: {iter_confidence:.4f}")

                    # Generate uncertainty visualization
                    generate_uncertainty_visualization(
                        uncertainty_metrics,
                        collab_data,
                        output_dir
                    )

            # Print execution statistics
            print(f"\nExecution time: {response_data['execution_time']:.2f} seconds")
            print(f"Confidence: {response_data.get('confidence', 'N/A')}")

            # Save trace to file
            trace_path = output_dir / "uncertainty_collaboration_trace.json"

            # Save trace as pretty-printed JSON
            with open(trace_path, 'w') as f:
                json.dump(response_data['trace'], f, indent=2)

            print(f"Trace saved to {trace_path}")

            generate_model_performance_chart(response_data['trace'], UNCERTAINTY_CONFIG, output_dir)

    except Exception as e:
        logger.error(f"Error in uncertainty-based collaboration example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
