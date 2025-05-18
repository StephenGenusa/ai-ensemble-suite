# ai_ensemble_suite/config/defaults.py

"""Default configuration values for ai-ensemble-suite."""

DEFAULT_CONFIG = {
    # Model default parameters
    "model_defaults": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 2048,
        "repeat_penalty": 1.1,
        "n_ctx": 4096
    },
    
    # Default collaboration configuration
    "collaboration": {
        "mode": "structured_debate",
        "phases": [
            {
                "name": "initial_response",
                "type": "async_thinking",
                "models": ["primary"],
                "prompt_template": "initial_response"
            },
            {
                "name": "critique",
                "type": "structured_debate",
                "subtype": "critique",
                "models": ["critic"],
                "input_from": "initial_response",
                "prompt_template": "critique"
            },
            {
                "name": "refinement",
                "type": "integration",
                "models": ["primary"],
                "input_from": ["initial_response", "critique"],
                "prompt_template": "refinement"
            }
        ]
    },
    
    # Default aggregation configuration
    "aggregation": {
        "strategy": "sequential_refinement",
        "final_phase": "refinement",
        "fallback": {
            "strategy": "confidence_based",
            "threshold": 0.6
        }
    },
    
    # Confidence estimation configuration
    "confidence": {
        "default_method": "combined",
        "token_prob_weight": 0.6,
        "self_eval_weight": 0.4,
        "consistency_samples": 3
    },
    
    # Logging configuration
    "logging": {
        "level": "INFO",
        "sanitize_prompts": True
    },
    
    # Prompt templates
    "templates": {
        "initial_response": "You are an expert assistant. Answer the following question:\n\n{query}",
        "critique": "You are a thoughtful critic. Review the following response to the question and provide constructive criticism:\n\nQuestion:\n{query}\n\nResponse:\n{initial_response}\n\nProvide criticism focusing on accuracy, clarity, completeness, and logical coherence.",
        "refinement": "You are an expert assistant. Refine your previous response based on the critic's feedback:\n\nQuestion:\n{query}\n\nYour initial response:\n{initial_response}\n\nCritic's feedback:\n{critique}\n\nImproved response:",
        "self_evaluation": "You previously provided this response:\n\n{response}\n\nOn a scale of 1-10, provide ONLY a numeric rating of your confidence in this response. Just the number, nothing else."
    }
}
