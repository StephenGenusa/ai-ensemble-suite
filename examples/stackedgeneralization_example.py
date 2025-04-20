"""Example demonstrating stacked generalization collaboration for ensemble AI."""

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


def get_random_technical_query():
    """Returns a randomly selected technical question for the stacking example."""
    technical_queries = [
        # Microservices vs monolithic
        "What are the key differences between microservices and monolithic architectures in modern software development, and what factors should engineers consider when choosing between them?",

        # Database architecture
        "What are the comparative advantages of SQL vs NoSQL databases, and what specific use cases are each best suited for?",

        # Systems design
        "How should engineers approach capacity planning and scaling for high-traffic web applications, and what monitoring metrics are most important?",

        # Software optimization
        "What techniques are most effective for optimizing Python code performance in data-intensive applications, and what are their trade-offs?",

        # API design
        "What are considered best practices in REST API design, and how do GraphQL and gRPC compare as alternatives?",

        # DevOps practices
        "How do containerization and orchestration tools like Docker and Kubernetes solve deployment challenges, and what are their operational complexities?",

        # Security architecture
        "What are the most critical security considerations when designing a modern cloud-native application architecture?",

        # Technical debt question
        "What strategies are most effective for managing technical debt in large codebases, and how should teams prioritize refactoring efforts?"
    ]

    return random.choice(technical_queries)


# Configuration with stacked generalization collaboration mode
STACKING_CONFIG = {
    "models": {
        "mistral": {
            "path": str(models_dir / "Mistral-7B-Instruct-v0.3-Q6_K.gguf"),
            "role": "primary",
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024,
                "n_ctx": 4096,
                # --- GPU Setting Example (falls back to 0/CPU if not present)
                # --- llama-cpp-python must be built or installed with GPU support ---
                "n_gpu_layers": -1  # Default to GPU in this config
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
                # --- GPU Setting Example (falls back to 0/CPU if not present)
                # --- llama-cpp-python must be built or installed with GPU support ---
                "n_gpu_layers": -1  # Default to GPU in this config
            }
        },
        "llama": {
            "path": str(models_dir / "llama-2-7b-chat.Q4_K_M.gguf"),
            "role": "primary",
            "parameters": {
                "temperature": 0.72,
                "top_p": 0.9,
                "max_tokens": 1024,
                "n_ctx": 4096,
                # --- GPU Setting Example (falls back to 0/CPU if not present)
                # --- llama-cpp-python must be built or installed with GPU support ---
                "n_gpu_layers": -1  # Default to GPU in this config
            }
        }
    },
    "collaboration": {
        "mode": "custom",
        "phases": [
            {
                "name": "stacked_generation",
                "type": "stacked_generalization",
                "combination_strategy": "weighted",
                "base_models": ["mistral", "deepseek", "llama"],
                "meta_model": "mistral",
                "prompt_template": "base_model_prompt",
                "models": ["mistral", "deepseek", "llama"],
                "max_rounds": 2,
                "use_feedback_loop": True,
                "meta_prompt_template": "meta_model_prompt"
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "stacked_generation"
    },
    "templates": {
    "base_model_prompt": """You are an expert AI assistant. Please provide a comprehensive, accurate, and insightful response to the following query:

QUERY: {{ query }}

Analyze the question thoroughly before answering. Be specific and include relevant details in your response.""",

    "meta_model_prompt": """You are a meta-analyzer tasked with synthesizing multiple AI responses to the same query.

ORIGINAL QUERY: {{ original_prompt }}

The following responses were generated by different AI models. Your job is to combine them into a single, coherent, and comprehensive response.

{% for key, output in base_outputs.items() %}
--- MODEL {{ key }} RESPONSE ---
{{ output }}

{% endfor %}

Your task:
1. Identify the key points of agreement across all models' responses
2. Note any meaningful disagreements or variations in perspective
3. Evaluate the quality and correctness of each response
4. Create a synthesized response that leverages the strengths of each model while addressing any weaknesses
5. Ensure your final response is comprehensive, well-structured, and directly answers the original query

This is round {{ round }} of the synthesis process.

{% if round == 2 %}
You should refine and improve upon the previous synthesis to create an even more accurate and valuable response.
{% endif %}

Present your synthesis below:"""}
}


def generate_model_comparison_chart(base_outputs, meta_output, output_dir):
    """Generate a visual comparison of base models and meta model."""
    if not base_outputs:
        return

    # Analyze response characteristics
    model_stats = {}

    # Calculate basic metrics for each model output
    for model_id, output in base_outputs.items():
        words = output.split()
        sentences = output.split('.')

        model_stats[model_id] = {
            'word_count': len(words),
            'sentence_count': len(sentences) - 1,  # Adjust for potential empty sentences
            'avg_word_length': sum(len(word) for word in words) / max(1, len(words))
        }

    # Add meta model stats
    if meta_output:
        words = meta_output.split()
        sentences = meta_output.split('.')

        model_stats['meta_synthesis'] = {
            'word_count': len(words),
            'sentence_count': len(sentences) - 1,
            'avg_word_length': sum(len(word) for word in words) / max(1, len(words))
        }

    # Create comparison charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Word count chart
    model_ids = list(model_stats.keys())
    word_counts = [stats['word_count'] for stats in model_stats.values()]
    sentence_counts = [stats['sentence_count'] for stats in model_stats.values()]

    # Set positions for x-ticks (important to fix warnings)
    x_positions = np.arange(len(model_ids))

    # Set different color for meta model
    colors = ['skyblue'] * len(base_outputs) + ['orange']

    # Word count chart
    bars1 = ax1.bar(x_positions, word_counts, color=colors)
    ax1.set_title('Word Count Comparison')
    ax1.set_ylabel('Number of Words')
    ax1.set_xticks(x_positions)  # Set tick positions before labels
    ax1.set_xticklabels(model_ids, rotation=45, ha='right')

    # Add count labels
    for bar, count in zip(bars1, word_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{count}', ha='center', va='bottom')

    # Sentence count chart
    bars2 = ax2.bar(x_positions, sentence_counts, color=colors)
    ax2.set_title('Sentence Count Comparison')
    ax2.set_ylabel('Number of Sentences')
    ax2.set_xticks(x_positions)  # Set tick positions before labels
    ax2.set_xticklabels(model_ids, rotation=45, ha='right')

    # Add count labels
    for bar, count in zip(bars2, sentence_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{count}', ha='center', va='bottom')

    plt.tight_layout()
    chart_path = output_dir / "stacking_comparison.png"
    plt.savefig(chart_path)
    print(f"Model comparison chart saved to {chart_path}")
    plt.close()


def analyze_content_similarity(base_outputs, meta_output):
    """Analyze how much content from each base model is included in the meta output."""
    if not base_outputs or not meta_output:
        return {}

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))

    # Function to get significant words from a text
    def get_significant_words(text):
        words = text.lower().split()
        return {word for word in words if word not in stop_words
                and len(word) > 3 and word.isalpha()}

    # Get significant words from each output
    meta_words = get_significant_words(meta_output)
    model_words = {model_id: get_significant_words(output)
                   for model_id, output in base_outputs.items()}

    # Calculate overlap statistics
    similarity_stats = {}
    for model_id, words in model_words.items():
        if not words:  # Avoid division by zero
            similarity_stats[model_id] = 0
            continue

        # Words from this model that appear in meta output
        overlap = words.intersection(meta_words)

        # Calculate what percentage of this model's unique content made it to the meta output
        similarity = len(overlap) / len(words)
        similarity_stats[model_id] = similarity

    return similarity_stats


async def main():
    """Run stacked generalization example."""
    # Check if models exist
    for model_id, model_config in STACKING_CONFIG["models"].items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Please ensure you have downloaded the required models to the 'models' directory.")
            sys.exit(1)

    # Check if config file exists
    config_dir = script_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "stackedgeneralization_config.yaml"

    # Delete existing config file to ensure we use the updated one
    if config_path.exists():
        os.remove(config_path)
        logger.info(f"Deleted existing config file: {config_path}")

    # Save our configuration
    with open(config_path, 'w') as f:
        yaml.dump(STACKING_CONFIG, f, default_flow_style=False)
    logger.info(f"Created new config file: {config_path}")

    # Use the config file
    ensemble_kwargs = {'config_path': str(config_path)}

    try:
        # Initialize the ensemble with proper kwargs
        async with Ensemble(**ensemble_kwargs) as ensemble:

            # Get a random technical question
            query = get_random_technical_query()

            print(f"\nQuery: {query}\n")
            print("Processing with stacked generalization ensemble...")

            # Get response with trace
            response_data = await ensemble.ask(query, trace=True)

            # Print response
            print("\nFinal Synthesized Response:")
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

            # Display each base model's response and meta model synthesis
            if 'trace' in response_data and 'phases' in response_data['trace']:
                phases = response_data['trace']['phases']

                if 'stacked_generation' in phases:
                    stacking_data = phases['stacked_generation']['output_data']
                    base_outputs = stacking_data.get('base_outputs', {})
                    meta_output = stacking_data.get('meta_output', '')
                    all_round_outputs = stacking_data.get('all_round_outputs', {})

                    print("\nBase Model Outputs:")
                    print("-" * 80)

                    for model_id, output in base_outputs.items():
                        print(f"\nModel: {model_id}")
                        print("-" * 50)
                        display_wrapped_text(output)

                    # Show outputs from each round if multiple rounds were performed
                    if len(all_round_outputs) > 1:
                        print("\nEvolution of Synthesis Across Rounds:")
                        print("-" * 80)

                        for round_id, round_output in all_round_outputs.items():
                            print(f"\n{round_id.replace('_', ' ').title()}:")
                            print("-" * 50)
                            display_wrapped_text(round_output)

                    # Generate visual comparison
                    generate_model_comparison_chart(base_outputs, meta_output, output_dir)

                    # Analyze content similarity
                    similarity_stats = analyze_content_similarity(base_outputs, meta_output)

                    if similarity_stats:
                        print("\nContent Contribution Analysis:")
                        print("-" * 80)
                        print("What percentage of each model's unique content appears in the final synthesis:")

                        for model_id, similarity in similarity_stats.items():
                            print(f"{model_id}: {similarity * 100:.1f}%")

            # Print execution statistics
            print(f"\nExecution time: {response_data['execution_time']:.2f} seconds")

            # Calculate peak memory usage
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                print(f"Peak memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
            except (ImportError, AttributeError):
                print("Memory usage information not available (psutil not installed)")

            # Save trace to file
            trace_path = output_dir / "stackedgeneralization_config.json"

            # Save trace as pretty-printed JSON
            with open(trace_path, 'w') as f:
                json.dump(response_data['trace'], f, indent=2)

            print(f"Trace saved to {trace_path}")

    except Exception as e:
        logger.error(f"Error in stacked generalization example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
