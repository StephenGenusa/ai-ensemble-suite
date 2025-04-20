
"""Example showing how to use a custom configuration file."""

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

# Example custom configuration template
CUSTOM_CONFIG_TEMPLATE = """
# Custom configuration for ai-ensemble-suite
# Replace the model paths with your actual model paths

models:
  model1:
    path: "models/your-first-model.gguf"
    role: "primary"
    parameters:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 1500
      n_gpu_layers: -1
  
  model2:
    path: "models/your-second-model.gguf"
    role: "critic"
    parameters:
      temperature: 0.5
      top_p: 0.9
      max_tokens: 1500
      n_gpu_layers: -1

collaboration:
  mode: "structured_critique"
  phases:
    - name: "initial_response"
      type: "async_thinking"
      models: ["model1"]
      prompt_template: "single_query"
    
    - name: "critique"
      type: "structured_debate"
      subtype: "critique"
      models: ["model2"]
      input_from: "initial_response"
      prompt_template: "debate_critique"
    
    - name: "refinement"
      type: "integration"
      models: ["model1"]
      input_from: ["initial_response", "critique"]
      prompt_template: "refinement"

aggregation:
  strategy: "sequential_refinement"
  final_phase: "refinement"
  
templates:
   ## Your Template or Templates Go Here
   {{ query }}  
"""

async def main():
    """Run custom config example."""
    # Create custom config file template
    config_dir = script_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "custom_config.yaml"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Write template if file doesn't exist
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write(CUSTOM_CONFIG_TEMPLATE)
    
        print(f"This program has just created a custom config template at {config_path}")
        print("Please edit this file with your actual model paths before running this example.")
        print("Then run this script again.")
        return
    
    # Check if the file still has the placeholder paths
    with open(config_path, 'r') as f:
        config_content = f.read()
        
    if "your-first-model.gguf" in config_content or "your-second-model.gguf" in config_content:
        print(f"Please edit {config_path} with your actual model paths.")
        print("The file still contains placeholder paths.")
        return
    
    print(f"Using custom configuration from {config_path}")
    
    try:
        # Initialize the ensemble with custom config
        async with Ensemble(config_path=config_path) as ensemble:
            
            # Ask a question
            query = "What are the most promising approaches to addressing climate change?"
            
            print(f"\nQuery: {query}\n")
            print("Processing...")
            
            # Get response
            response = await ensemble.ask(query)
            
            print("\nResponse:")
            print("=" * 80)
            print(response)
            print("=" * 80)
            
    except Exception as e:
        logger.error(f"Error in custom config example: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
