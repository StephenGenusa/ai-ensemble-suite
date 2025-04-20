# src/ai_ensemble_suite/models/confidence.py

"""Confidence estimation methods for model outputs."""

import asyncio
import math
import re
import statistics
from typing import Dict, Any, List, Optional, Union, TypeVar, Callable, Tuple, cast, TYPE_CHECKING, Set, Protocol

# Moved import to top level
from ai_ensemble_suite.utils.prompt_utils import format_prompt
from ai_ensemble_suite.exceptions import ValidationError
from ai_ensemble_suite.utils.logging import logger

# Forward reference for type hints
if TYPE_CHECKING:
    # Keep GGUFModel import within TYPE_CHECKING only
    from ai_ensemble_suite.models.gguf_model import GGUFModel
    # Add ConfigManager for type hints if needed by _get_self_evaluation_template
    from ai_ensemble_suite.config import ConfigManager


async def calculate_token_confidence(
    model_output: Dict[str, Any]
) -> Dict[str, float]:
    """Calculate confidence based on token probabilities.

    Args:
        model_output: The raw output from the model containing token information
            (e.g., from llama_cpp.Llama.create_completion).

    Returns:
        Dictionary with different confidence metrics:
        - mean: Average token probability
        - min: Minimum token probability
        - geometric_mean: Geometric mean of token probabilities
        - median: Median token probability
    """
    # Extract token probabilities if available
    # Llama-cpp returns logprobs in the 'logprobs' key (if requested) which contains 'top_logprobs' per token
    # Or sometimes simple 'tokens' might have probabilities (less common for generation)
    # We need to handle the structure returned by llama-cpp
    logprobs_data = model_output.get("logprobs")
    probabilities = []

    if logprobs_data and "token_logprobs" in logprobs_data:
        # Prefer token_logprobs if available
        token_logprobs = logprobs_data["token_logprobs"]
        # Filter out None values (e.g., for the first token) and negative infinity
        valid_logprobs = [lp for lp in token_logprobs if lp is not None and lp > -float('inf')]
        if valid_logprobs:
            probabilities = [math.exp(lp) for lp in valid_logprobs]

    # Fallback or alternative: Check 'tokens' list if it contains probabilities
    elif "tokens" in model_output:
        tokens = model_output.get("tokens", [])
        for token_info in tokens:
            # Check various possible keys for probability or log probability
            if "probability" in token_info and isinstance(token_info["probability"], (int, float)) and token_info["probability"] > 0:
                probabilities.append(float(token_info["probability"]))
            elif "prob" in token_info and isinstance(token_info["prob"], (int, float)) and token_info["prob"] > 0:
                probabilities.append(float(token_info["prob"]))
            elif "logprob" in token_info and isinstance(token_info["logprob"], (int, float)):
                try:
                    prob = math.exp(float(token_info["logprob"]))
                    if prob > 0:
                        probabilities.append(prob)
                except OverflowError:
                    pass # Ignore if logprob is too small

    # If no probabilities available, return default confidence
    if not probabilities:
        logger.warning("No valid token probabilities found in model output for confidence calculation")
        return {
            "mean": 0.7,  # Default medium-high confidence
            "min": 0.7,
            "geometric_mean": 0.7,
            "median": 0.7
        }

    # Filter out any zero or negative probabilities that might sneak in
    probabilities = [p for p in probabilities if p > 0]
    if not probabilities:
        logger.warning("All extracted probabilities were non-positive.")
        return { "mean": 0.5, "min": 0.5, "geometric_mean": 0.5, "median": 0.5 }

    # Calculate different confidence metrics
    mean_prob = statistics.mean(probabilities)
    min_prob = min(probabilities)

    # Geometric mean calculation
    try:
        # Ensure all values are positive for geometric mean (already done, but double check)
        positive_probs = [max(p, 1e-10) for p in probabilities] # Use a small epsilon
        geometric_mean = statistics.geometric_mean(positive_probs) if positive_probs else 0.0
    except (ValueError, statistics.StatisticsError) as e:
        logger.warning(f"Could not calculate geometric mean: {e}")
        geometric_mean = 0.0 # Or consider using mean as fallback

    # Median calculation
    median_prob = statistics.median(probabilities)

    return {
        "mean": mean_prob,
        "min": min_prob,
        "geometric_mean": geometric_mean,
        "median": median_prob
    }


async def _get_self_evaluation_template(
    model: 'GGUFModel',
    prompt_template: str = "self_evaluation"
) -> str:
    """Get the appropriate template for self-evaluation.

    Args:
        model: The model to evaluate confidence.
        prompt_template: The template name for self-evaluation prompting.

    Returns:
        Template string for self-evaluation.
    """
    # Get self-evaluation template from model's config manager
    # Access config manager through the model instance
    config_manager = getattr(model, "_config_manager", None)
    default_template = (
        "You previously provided this response:\n\n{response}\n\n"
        "On a scale of 1-10, provide ONLY a numeric rating of your confidence "
        "in this response. Just the number, nothing else."
    )

    if config_manager is None:
        logger.warning("Model does not have a config manager, using default self-evaluation prompt")
        return default_template

    try:
        # Assume ConfigManager has a method to get templates
        template = config_manager.get_template(prompt_template)
        if template is None:
            logger.warning(f"Self-evaluation template '{prompt_template}' not found, using default")
            return default_template
        return template
    except AttributeError:
         logger.warning(f"Model's config manager missing 'get_template' method, using default")
         return default_template
    except Exception as e:
        logger.warning(f"Error accessing template '{prompt_template}': {str(e)}, using default")
        return default_template


async def _parse_self_evaluation_response(eval_text: str) -> float:
    """Parse the model's self-evaluation response.

    Args:
        eval_text: The model's evaluation response.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    # Try numeric extraction first
    match = re.search(r'(\d+(?:\.\d+)?)', eval_text)
    if match:
        try:
            rating = float(match.group(1))
            # Normalize to 0-1 range (assuming scale 1-10)
            confidence = min(max((rating - 1) / 9.0, 0.0), 1.0) if rating >= 1 else 0.0
            # Alternative normalization if scale is 0-10
            # confidence = min(max(rating / 10.0, 0.0), 1.0)
            return confidence
        except ValueError:
             logger.warning(f"Could not convert extracted rating '{match.group(1)}' to float.")

    # Try word-to-number conversion
    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    for word, num in number_words.items():
        # Use word boundaries for better matching
        if re.search(r'\b' + word + r'\b', eval_text.lower()):
            # Normalize 1-10 to 0-1
            return (num - 1) / 9.0

    logger.warning(f"Failed to extract numeric rating from self-evaluation: '{eval_text}'")
    return 0.5  # Default medium confidence


async def get_model_self_evaluation(
    model: 'GGUFModel',
    response: str,
    prompt_template: str = "self_evaluation"
) -> float:
    """Have model evaluate its own confidence.

    Args:
        model: The model to evaluate confidence.
        response: The response to evaluate.
        prompt_template: The template name for self-evaluation prompting.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    try:
        # Get template and format prompt
        template = await _get_self_evaluation_template(model, prompt_template)
        # Ensure response is passed correctly to formatting
        prompt = format_prompt(template, response=response)

        # Set parameters for confidence evaluation
        # Short max_tokens, low temperature needed
        params = {
            "temperature": 0.1,  # Low temperature for more deterministic rating
            "max_tokens": 10,     # Slightly more tokens in case it adds context
            "logprobs": None,    # We don't need logprobs for the rating itself
            "echo": False
        }

        # Generate self-evaluation response
        # Use the model's own generate method, assuming it handles locking
        eval_result = await model.generate(prompt, **params)

        eval_text = eval_result.get("text", "").strip()

        # Parse the response
        return await _parse_self_evaluation_response(eval_text)

    except Exception as e:
        logger.error(f"Error during self-evaluation for model {model.get_id()}: {str(e)}")
        return 0.5  # Default medium confidence


async def measure_consistency_confidence(
    model: 'GGUFModel',
    prompt: str,
    num_samples: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 150 # Added max_tokens for consistency
) -> float:
    """Generate multiple responses and measure consistency.

    Args:
        model: The model to generate responses.
        prompt: The prompt to send to the model.
        num_samples: Number of samples to generate.
        temperature: Temperature to use for generation (should allow some variance).
        max_tokens: Max tokens for sampled responses.

    Returns:
        Consistency score between 0.0 and 1.0.
    """
    if num_samples < 2:
        logger.warning("At least 2 samples are required for consistency measurement")
        return 0.5  # Default medium confidence

    responses = []

    # Generate multiple responses
    generation_params = {
        "temperature": temperature,
        "max_tokens": max_tokens, # Use passed or default max_tokens
        "logprobs": None,        # Don't need logprobs for consistency text
        "echo": False
    }

    try:
        # Generate responses concurrently
        async def generate_response():
            try:
                # Use the model's generate method
                result = await model.generate(prompt, **generation_params)
                return result.get("text", "").strip()
            except Exception as e:
                logger.warning(f"Error in consistency sample generation for model {model.get_id()}: {e}")
                return None

        # Execute tasks concurrently
        tasks = [generate_response() for _ in range(num_samples)]
        results = await asyncio.gather(*tasks)

        # Filter out None values (failed generations)
        responses = [r for r in results if r is not None and r] # Also ensure not empty

        if len(responses) < 2:
            logger.warning(f"Not enough valid responses ({len(responses)}/{num_samples}) for consistency calculation")
            return 0.5 if len(responses) == 0 else 0.7 # Slightly higher if 1 response

        # Calculate similarity scores between all pairs of responses
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = calculate_similarity(responses[i], responses[j])
                similarities.append(similarity)

        # Average similarity as consistency score
        consistency_score = statistics.mean(similarities) if similarities else 0.5
        return consistency_score

    except ValidationError as ve:
        logger.warning(f"Validation error during consistency check: {ve}")
        return 0.5
    except Exception as e:
        logger.error(f"Error measuring consistency for model {model.get_id()}: {str(e)}")
        return 0.5  # Default medium confidence


def calculate_similarity(
    text1: str,
    text2: str
) -> float:
    """Calculate semantic similarity between two texts.

    Uses a combined n-gram and word-level approach suitable for GGUF models
    without requiring external embeddings.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    # Quick exact match check
    if text1 == text2:
        return 1.0

    # Handle empty or very short texts
    if not text1 or not text2 or len(text1) < 5 or len(text2) < 5:
        # For very short or empty texts, simple comparison
        return 1.0 if text1 == text2 else 0.0

    # Check length difference - if too different, decrease similarity
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    # Consider a less harsh penalty than direct return
    # if len_ratio < 0.3:
    #     return 0.3 * len_ratio

    # Basic normalization
    text1_norm = text1.lower().strip()
    text2_norm = text2.lower().strip()

    # Simple word-level overlap calculation
    try:
        words1 = set(re.findall(r'\b\w+\b', text1_norm))
        words2 = set(re.findall(r'\b\w+\b', text2_norm))
    except Exception as e:
        logger.warning(f"Regex failed in calculate_similarity: {e}")
        return 0.5 # Fallback

    # Handle empty sets after regex
    if not words1 or not words2:
        return 0.0 if words1 != words2 else 1.0

    # For very small word sets, use direct set ratio
    # Increase threshold slightly
    if len(words1) < 5 or len(words2) < 5:
        intersection_len = len(words1.intersection(words2))
        max_len = max(len(words1), len(words2))
        return intersection_len / max_len if max_len > 0 else 0.0

    # Jaccard similarity (Word level)
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    jaccard = intersection / union if union > 0 else 0.0

    # N-gram similarity (adjust n based on text length?)
    def get_ngrams(text: str, n: int = 3) -> Set[str]:
        try:
            tokens = re.findall(r'\b\w+\b', text) # Use normalized text
            if len(tokens) < n:
                return set() # Not enough tokens for n-gram
            ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            return set(ngrams)
        except Exception as e:
            logger.warning(f"Regex/Ngram failed in get_ngrams: {e}")
            return set()

    # Calculate trigram similarity
    ngrams1 = get_ngrams(text1_norm, n=3)
    ngrams2 = get_ngrams(text2_norm, n=3)
    ngram_jaccard = 0.0
    if ngrams1 and ngrams2:
        intersection_ngram = len(ngrams1.intersection(ngrams2))
        union_ngram = len(ngrams1.union(ngrams2))
        ngram_jaccard = intersection_ngram / union_ngram if union_ngram > 0 else 0.0
    elif ngrams1 == ngrams2: # Both empty, means short text handled earlier or identical
         ngram_jaccard = 1.0

    # Compute a weighted similarity score
    # Increased weight for ngram similarity, slightly reduced length ratio impact
    # Weights: 35% word Jaccard, 45% n-gram Jaccard, 20% length ratio
    weighted_score = 0.35 * jaccard + 0.45 * ngram_jaccard + 0.20 * len_ratio
    return min(max(weighted_score, 0.0), 1.0) # Ensure score is within [0, 1]


def _validate_confidence_weights(
    token_weight: float,
    self_eval_weight: float,
    consistency_weight: float # Added consistency weight
) -> Tuple[float, float, float]:
    """Validate and normalize confidence weights.

    Args:
        token_weight: Weight for token probability confidence.
        self_eval_weight: Weight for self-evaluation confidence.
        consistency_weight: Weight for consistency confidence.

    Returns:
        Normalized weights as a tuple (token_weight, self_eval_weight, consistency_weight).
    """
    weights = [token_weight, self_eval_weight, consistency_weight]

    # Check if weights are valid numbers between 0 and 1
    if not all(isinstance(w, (int, float)) and 0 <= w <= 1 for w in weights):
        logger.warning("Invalid confidence weights (non-numeric or out of range 0-1), using defaults")
        # Default weights: token=0.5, self_eval=0.3, consistency=0.2
        return 0.5, 0.3, 0.2

    # Normalize weights if their sum is not 1.0 (and sum > 0)
    total = sum(weights)
    if total > 0 and not math.isclose(total, 1.0):
        logger.debug(f"Normalizing confidence weights from sum {total}")
        normalized_weights = tuple(w / total for w in weights)
        return normalized_weights
    elif total == 0:
        logger.warning("All confidence weights are zero, using equal distribution if components available later")
        # Return zeros, combination logic will handle this
        return 0.0, 0.0, 0.0
    else:
        # Weights sum to 1.0 already or are invalid (negative handled earlier)
        return token_weight, self_eval_weight, consistency_weight


def _combine_confidence_scores(
    scores: Dict[str, float],
    token_prob_weight: float,
    self_eval_weight: float,
    consistency_weight: float # Added consistency weight
) -> float:
    """Combine multiple confidence scores into a single score using normalized weights.

    Args:
        scores: Dictionary of available confidence scores (e.g., {"token_prob": 0.8, "self_eval": 0.7}).
        token_prob_weight: Normalized weight for token probability.
        self_eval_weight: Normalized weight for self-evaluation.
        consistency_weight: Normalized weight for consistency.

    Returns:
        Combined confidence score between 0.0 and 1.0.
    """
    combined_score = 0.0
    total_weight_used = 0.0

    # Get available scores
    available_scores = {k: v for k, v in scores.items() if isinstance(v, (int, float)) and 0 <= v <= 1}

    # Apply weights only for available scores
    if "token_prob" in available_scores:
        combined_score += token_prob_weight * available_scores["token_prob"]
        total_weight_used += token_prob_weight

    if "self_eval" in available_scores:
        combined_score += self_eval_weight * available_scores["self_eval"]
        total_weight_used += self_eval_weight

    if "consistency" in available_scores:
        combined_score += consistency_weight * available_scores["consistency"]
        total_weight_used += consistency_weight

    # Normalize the combined score based on the weights actually used
    if total_weight_used > 0:
        # Renormalize based on the components that were actually present
        final_score = combined_score / total_weight_used
    elif available_scores:
         # If weights were all zero but scores exist, average the scores
        logger.warning("Weights summed to zero, averaging available scores.")
        final_score = statistics.mean(available_scores.values())
    else:
        # No valid scores available
        logger.warning("No valid confidence scores available for combination, returning default 0.5")
        final_score = 0.5

    return min(max(final_score, 0.0), 1.0) # Clamp to [0, 1]


async def get_confidence_score(
    model: 'GGUFModel',
    prompt: str,
    response: str,
    model_output: Dict[str, Any],
    method: str = "combined",
    **kwargs: Any
) -> Dict[str, float]:
    """Get confidence score(s) using the specified method(s).

    Args:
        model: The model instance used for generation/evaluation.
        prompt: The prompt sent to the model for the original response.
        response: The response text generated by the model.
        model_output: Raw model output data (containing logprobs etc.).
        method: The confidence estimation method(s) to use:
            - "token_prob": Token probability aggregation.
            - "self_eval": Self-evaluation prompting.
            - "consistency": Response consistency check.
            - "combined": Calculate all available methods and combine them.
            You can also provide a list/tuple, e.g., ["token_prob", "self_eval"].
        **kwargs: Additional parameters for specific methods:
            - token_prob_weight: Weight for token probability (default: 0.5).
            - self_eval_weight: Weight for self-evaluation (default: 0.3).
            - consistency_weight: Weight for consistency (default: 0.2).
            - consistency_samples: Number of samples for consistency check (default: 3).
            - consistency_temperature: Temperature for consistency samples (default: 0.7).
            - consistency_max_tokens: Max tokens for consistency samples (default: 150).
            - token_metric: Which token metric to use ('geometric_mean', 'mean', 'median', 'min') (default: 'geometric_mean').

    Returns:
        Dictionary with confidence scores for each calculated method
        and potentially a "combined" score.
    """
    result: Dict[str, float] = {}
    methods_to_run: Set[str] = set()

    if isinstance(method, str):
        if method == "combined":
            methods_to_run = {"token_prob", "self_eval", "consistency"}
        elif method in {"token_prob", "self_eval", "consistency"}:
            methods_to_run = {method}
        else:
            logger.warning(f"Unknown single confidence method: {method}, defaulting to combined.")
            methods_to_run = {"token_prob", "self_eval", "consistency"}
    elif isinstance(method, (list, tuple, set)):
        valid_methods = {m for m in method if m in {"token_prob", "self_eval", "consistency"}}
        if not valid_methods:
             logger.warning(f"No valid methods provided in list: {method}, defaulting to combined.")
             methods_to_run = {"token_prob", "self_eval", "consistency"}
        else:
            methods_to_run = valid_methods
    else:
        logger.warning(f"Invalid type for confidence method: {type(method)}, defaulting to combined.")
        methods_to_run = {"token_prob", "self_eval", "consistency"}


    # Get and validate weights - use defaults matching the _validate function
    token_prob_weight, self_eval_weight, consistency_weight = _validate_confidence_weights(
        kwargs.get("token_prob_weight", 0.5),
        kwargs.get("self_eval_weight", 0.3),
        kwargs.get("consistency_weight", 0.2)
    )

    # --- Calculate individual scores based on methods_to_run ---

    if "token_prob" in methods_to_run:
        token_metrics = await calculate_token_confidence(model_output)
        # Use the specified metric or default to geometric mean
        token_metric = kwargs.get("token_metric", "geometric_mean")
        if token_metric not in token_metrics:
             logger.warning(f"Specified token_metric '{token_metric}' not found, using 'geometric_mean'.")
             token_metric = "geometric_mean"
        result["token_prob"] = token_metrics[token_metric]
        # Optionally include all token metrics
        # result.update({f"token_{k}": v for k, v in token_metrics.items()})

    if "self_eval" in methods_to_run:
        # Check if model has generate method needed for self-evaluation
        if hasattr(model, 'generate') and callable(model.generate):
             self_eval_score = await get_model_self_evaluation(model, response)
             result["self_eval"] = self_eval_score
        else:
            logger.warning(f"Model {model.get_id()} missing 'generate' method required for self-evaluation.")

    if "consistency" in methods_to_run:
         # Check if model has generate method needed for consistency
        if hasattr(model, 'generate') and callable(model.generate):
            consistency_samples = kwargs.get("consistency_samples", 3)
            consistency_temperature = kwargs.get("consistency_temperature", 0.7)
            consistency_max_tokens = kwargs.get("consistency_max_tokens", 150)
            consistency_score = await measure_consistency_confidence(
                model,
                prompt,
                num_samples=consistency_samples,
                temperature=consistency_temperature,
                max_tokens=consistency_max_tokens
            )
            result["consistency"] = consistency_score
        else:
            logger.warning(f"Model {model.get_id()} missing 'generate' method required for consistency check.")

    # --- Calculate combined score if requested or only one method was run ---
    if method == "combined" or len(methods_to_run) > 1:
         result["combined"] = _combine_confidence_scores(
            result,
            token_prob_weight,
            self_eval_weight,
            consistency_weight
        )
    elif len(methods_to_run) == 1:
        # If only one method ran, its score is the "combined" score
        single_method = list(methods_to_run)[0]
        result["combined"] = result.get(single_method, 0.5) # Use 0.5 if score somehow missing

    return result

async def get_combined_confidence(
    model: 'GGUFModel',
    prompt: str,
    response: str,
    model_output: Dict[str, Any],
    token_prob_weight: float = 0.5,
    self_eval_weight: float = 0.3,
    consistency_weight: float = 0.2,
    **kwargs: Any # Pass other kwargs through
) -> float:
    """Calculate a weighted confidence score using the 'combined' method.

    This function is deprecated. Use get_confidence_score with method='combined' instead.

    Args:
        model: The model that generated the response.
        prompt: The prompt sent to the model.
        response: The generated response.
        model_output: The raw model output with token information.
        token_prob_weight: Weight for token probability (default: 0.5).
        self_eval_weight: Weight for self-evaluation (default: 0.3).
        consistency_weight: Weight for consistency (default: 0.2).
        **kwargs: Additional args for underlying confidence methods.

    Returns:
        Combined confidence score between 0.0 and 1.0.
    """
    logger.warning("'get_combined_confidence' is deprecated. "
                   "Consider using 'get_confidence_score' with method='combined'.")

    # Pass weights and other kwargs to get_confidence_score
    confidence_scores = await get_confidence_score(
        model,
        prompt,
        response,
        model_output,
        method="combined",
        token_prob_weight=token_prob_weight,
        self_eval_weight=self_eval_weight,
        consistency_weight=consistency_weight,
        **kwargs # Pass through other relevant args like consistency_samples etc.
    )

    return confidence_scores.get("combined", 0.5)
