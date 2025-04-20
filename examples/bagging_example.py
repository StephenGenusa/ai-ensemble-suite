# examples/bagging_example.py

"""Example demonstrating bagging collaboration for ensemble AI."""

import asyncio
import os
import sys
import yaml
import json
import random
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
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

def get_random_technical_query():
    """Returns a randomly selected technical question for the bagging example."""
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

# Configuration with bagging collaboration mode
BAGGING_CONFIG = {
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
            "path": str(models_dir / "llama-2-7b-chat.Q4_K_M.gguf"),
            "role": "primary",
            "parameters": {
                "temperature": 0.72,
                "top_p": 0.9,
                "max_tokens": 1024,
                "n_ctx": 8192,
                "n_gpu_layers": -1
            }
        }
    },
    "collaboration": {
        "mode": "custom",
        "phases": [
            {
                "name": "bagged_responses",
                "type": "bagging",
                "models": ["mistral", "deepseek", "llama"],
                "prompt_template": "bagging_prompt",
                "variation_strategy": "instruction_variation",
                "sample_ratio": 0.9,
                "aggregation_method": "voting",
                "num_variations": 4 # <-------
            },
            {
                "name": "meta_analysis",
                "type": "integration",
                "models": ["mistral"],
                "input_from": ["bagged_responses"],
                "prompt_template": "meta_analysis_prompt"
            }
        ]
    },
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "meta_analysis"
    },
    "templates": {
    "bagging_prompt": """You are an expert AI assistant. Please provide a comprehensive, accurate, and insightful response to the following query:

QUERY: {{ query }}

Analyze the question thoroughly before answering. Be specific and include relevant details in your response.""",

    "meta_analysis_prompt": """You are a meta-analyzer tasked with synthesizing multiple AI responses to the same query.

ORIGINAL QUERY: {{ query }}

The following summaries were generated using different variations of the query or by different models.
I'll provide key excerpts from each response due to space constraints:

{% for output in bagged_responses.outputs %}
--- MODEL RESPONSE {{ loop.index }} ---
{{ output|substring(0, 1200) }}...

{% endfor %}

Your task:
1. Identify the key points of agreement across the responses
2. Note any meaningful disagreements or variations in perspective
3. Assess the overall confidence in the aggregated information
4. Provide a comprehensive synthesis that represents the collective knowledge
5. Highlight any areas where the responses seem uncertain or where further information would be valuable

Present your synthesis in a clear, well-structured format that captures the most reliable information from all responses."""
}
}

def generate_agreement_chart(variation_outputs, output_dir):
    """Generate a visual representation of response agreement with stopwords filtering."""
    if not variation_outputs or len(variation_outputs) < 2:
        return

    # Initialize stopwords for filtering
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    # Add any additional words you consider irrelevant
    additional_stopwords = {'like', 'also', 'using', 'can', 'may', 'must', 'will', 'would',
                           'one', 'make', 'use', 'set', 'get', 'well', 'need', 'take'}
    stop_words.update(additional_stopwords)

    # Extract words from each output
    word_sets = []
    for output in variation_outputs.values():
        # Process all words to keep analysis comprehensive
        words = output.lower().split()
        # Filter out stopwords and very short words
        filtered_words = {word for word in words
                         if word not in stop_words
                         and len(word) > 3
                         and word.isalpha()}  # Only alphabetic words
        word_sets.append(filtered_words)

    # Find common words across responses
    all_words = set()
    for words in word_sets:
        all_words.update(words)

    # Count occurrences of each word
    word_counts = {}
    for word in all_words:
        count = sum(1 for words in word_sets if word in words)
        word_counts[word] = count

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[:15]  # Show top 15 words

    # Create bar chart
    labels = [word for word, _ in top_words]
    values = [count for _, count in top_words]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='skyblue')
    plt.xlabel('Common Relevant Words')
    plt.ylabel('Number of Responses Containing Word')
    plt.title('Word Agreement Across Bagged Responses')
    plt.xticks(rotation=45, ha='right')

    # Add count labels on top of bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                str(value), ha='center', va='bottom')

    plt.tight_layout()
    chart_path = output_dir / "bagging_agreement.png"
    plt.savefig(chart_path)
    print(f"Agreement chart saved to {chart_path}")
    plt.close()

async def main():
    """Run bagging example."""
    # Check if models exist
    for model_id, model_config in BAGGING_CONFIG["models"].items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Please ensure you have downloaded the required models to the 'models' directory.")
            sys.exit(1)

    # Check if config file exists
    config_dir = script_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "bagging_config.yaml"

    # Delete existing config file to ensure we use the updated one
    if config_path.exists():
        os.remove(config_path)
        logger.info(f"Deleted existing config file: {config_path}")

    # Save our configuration
    with open(config_path, 'w') as f:
        yaml.dump(BAGGING_CONFIG, f, default_flow_style=False)
    logger.info(f"Created new config file: {config_path}")

    # Use the config file
    ensemble_kwargs = {'config_path': str(config_path)}

    try:
        # Initialize the ensemble with proper kwargs
        async with Ensemble(**ensemble_kwargs) as ensemble:

            # Get a random technical question
            query = get_random_technical_query()

            print(f"\nQuery: {query}\n")
            print("Processing with bagging ensemble...")

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

            # Display each variation and its response
            if 'trace' in response_data and 'phases' in response_data['trace']:
                phases = response_data['trace']['phases']

                # Create output directory for visualizations
                output_dir = script_dir / "output"
                output_dir.mkdir(exist_ok=True)

                if 'bagged_responses' in phases:
                    bagging_data = phases['bagged_responses']['output_data']
                    variation_outputs = bagging_data.get('variation_outputs', {})

                    print("\nIndividual Bagging Variations:")
                    print("-" * 80)

                    for i, (var_id, output) in enumerate(variation_outputs.items()):
                        print(f"\nVariation {i+1} ({var_id}):")
                        print("-" * 50)
                        display_wrapped_text(output)

                    # Calculate and display agreement metrics with stopword filtering
                    if len(variation_outputs) > 1:
                        # Generate visual agreement chart (uses stopword filtering internally)
                        generate_agreement_chart(variation_outputs, output_dir)

                        # Initialize stopwords
                        try:
                            nltk.data.find('corpora/stopwords')
                        except LookupError:
                            nltk.download('stopwords', quiet=True)

                        stop_words = set(stopwords.words('english'))
                        additional_stopwords = {'like', 'also', 'using', 'can', 'may', 'must', 'will', 'would'}
                        stop_words.update(additional_stopwords)

                        # Calculate filtered agreement score
                        word_sets = []
                        for output in variation_outputs.values():
                            words = output.lower().split()
                            filtered = {word for word in words
                                      if word not in stop_words
                                      and len(word) > 3
                                      and word.isalpha()}
                            word_sets.append(filtered)

                        union_words = set().union(*word_sets)
                        word_counts = {}
                        for word in union_words:
                            word_counts[word] = sum(1 for words in word_sets if word in words)

                        # Words that appear in majority of responses
                        consensus_words = sum(1 for count in word_counts.values()
                                             if count > len(variation_outputs) / 2)

                        print("\nResponse Agreement Analysis (Stopwords Filtered):")
                        print("-" * 50)
                        print(f"Total unique relevant words: {len(union_words)}")
                        print(f"Words in majority consensus: {consensus_words}")
                        if union_words:
                            print(f"Agreement ratio: {consensus_words / len(union_words):.2f}")
                        else:
                            print(f"Agreement ratio: N/A (No relevant words found)")

                if 'meta_analysis' in phases:
                    print("\nMeta-Analysis:")
                    print("-" * 80)
                    meta_data = phases['meta_analysis']['output_data']
                    # Use primary_output instead of output
                    meta_output = meta_data.get('primary_output', '')
                    display_wrapped_text(meta_output)

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
            trace_path = output_dir / "bagging_trace.json"

            # Save trace as pretty-printed JSON
            with open(trace_path, 'w') as f:
                json.dump(response_data['trace'], f, indent=2)

            print(f"Trace saved to {trace_path}")

    except Exception as e:
        logger.error(f"Error in bagging example: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
