# ai_ensemble_suite/models/gguf_model.py

"""Wrapper for llama-cpp-python GGUF models."""

from typing import Dict, Any, Optional, TYPE_CHECKING, List
import time
import os
import asyncio
import copy
import psutil
from concurrent.futures import ThreadPoolExecutor
import math
import threading
from pathlib import Path  # Added import

try:
    from llama_cpp import Llama, LlamaGrammar, LogitsProcessorList, LogitsProcessor

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    print("--------------------------------------------------------------------")
    print("WARNING: llama-cpp-python is not installed.")
    print("You will not be able to use GGUF models.")
    print("Please install with hardware acceleration (choose one):")
    print("- CPU only: pip install llama-cpp-python")
    print("- NVIDIA GPU: CMAKE_ARGS=\"-DGGML_CUDA=on\" FORCE_CMAKE=1 pip install llama-cpp-python")
    print("- Apple Metal (Metal): CMAKE_ARGS=\"-DLLAMA_METAL=on\" FORCE_CMAKE=1 pip install llama-cpp-python")
    print("Visit https://github.com/abetlen/llama-cpp-python for more options.")
    print("--------------------------------------------------------------------")
    LLAMA_CPP_AVAILABLE = False
    # Define dummy types for type checking if install failed
    Llama = Any
    LlamaGrammar = Any
    LogitsProcessorList = Any
    LogitsProcessor = Any

from ai_ensemble_suite.exceptions import ModelError, ResourceError
from ai_ensemble_suite.utils.async_utils import run_in_threadpool
from ai_ensemble_suite.utils.logging import logger

if TYPE_CHECKING:
    from ai_ensemble_suite.config.config_manager import ConfigManager


def calculate_confidence_from_logprobs(logprobs: Optional[List[Dict[str, float]]]) -> Optional[float]:
    """Calculate average token probability from logprobs.

    Args:
        logprobs: List of dictionaries with token logprob information.

    Returns:
        Confidence score based on token probabilities, or None if not calculable.
    """
    if not logprobs:
        logger.warning("Cannot calculate confidence: logprobs data is missing or empty.")
        return None

    valid_probs = []
    for token_info in logprobs:
        # Check if token_info is a dict and has 'logprob' key
        if isinstance(token_info, dict) and 'logprob' in token_info and isinstance(token_info['logprob'], (int, float)):
            try:
                prob = math.exp(token_info['logprob'])
                valid_probs.append(prob)
            except (OverflowError, ValueError):
                logger.debug(f"Could not compute exp(logprob) for value: {token_info['logprob']}. Skipping token.")
        else:
            logger.debug(f"Skipping invalid token info for confidence: {token_info}")

    if not valid_probs:
        logger.warning("No valid token probabilities found in model output for confidence calculation")
        return None

    avg_prob = sum(valid_probs) / len(valid_probs)
    if not (0 <= avg_prob <= 1):
        logger.warning(f"Calculated average probability outside [0, 1] range: {avg_prob}. Returning None.")
        return None

    return avg_prob


class GGUFModel:
    """Wrapper for llama-cpp-python GGUF models."""

    def __init__(
            self,
            model_id: str,
            model_path: str,
            model_config: Dict[str, Any],
            config_manager: Optional["ConfigManager"] = None,
            executor: Optional[ThreadPoolExecutor] = None
    ) -> None:
        if not LLAMA_CPP_AVAILABLE:
            raise ResourceError(
                "llama-cpp-python is not installed. Please install it to use GGUF models."
                " See warning message above for installation instructions."
            )
        self._model_id = model_id
        self._model_path = str(model_path)  # Ensure path is string
        self._model_config_original = model_config
        self._config_manager = config_manager
        self._executor = executor
        self._llm: Optional[Llama] = None
        self._is_loaded = False
        self._inference_lock = asyncio.Lock()
        self._role = model_config.get("role")

        # --- Parameter Merging ---
        model_params_specific = model_config.get("parameters", {})
        if not isinstance(model_params_specific, dict):
            logger.warning(f"Model '{model_id}': 'parameters' section is not a dictionary. Ignoring.")
            model_params_specific = {}

        model_params_defaults = {}
        if self._config_manager:
            try:
                if hasattr(self._config_manager, 'get_default_model_parameters'):
                    model_params_defaults = self._config_manager.get_default_model_parameters()
                elif hasattr(self._config_manager, 'get_model_defaults'):  # Fallback check
                    logger.warning(
                        f"ConfigManager has 'get_model_defaults' not 'get_default_model_parameters'. Using it for defaults for {model_id}, but might indicate mismatch.")
                    model_params_defaults = self._config_manager.get_model_defaults()

                if not isinstance(model_params_defaults, dict):
                    logger.warning(
                        f"Default model parameters retrieved from ConfigManager for '{model_id}' is not a dict. Ignoring defaults.")
                    model_params_defaults = {}
            except AttributeError:
                logger.warning(
                    f"ConfigManager instance passed to GGUFModel '{model_id}' does not implement expected defaults method. Skipping defaults.")
            except Exception as e:
                logger.warning(f"Could not retrieve model defaults from ConfigManager for '{model_id}': {e}")

        self._parameters = {**model_params_defaults, **model_params_specific}

        # --- Path Validation ---
        try:
            resolved_path_obj = Path(self._model_path).resolve()
            if not resolved_path_obj.exists():
                logger.error(f"Model path does not exist: {resolved_path_obj}")
                raise ModelError(f"Model path does not exist: {resolved_path_obj}")
            self._model_path = str(resolved_path_obj)  # Store the resolved absolute path
        except Exception as path_e:
            logger.error(f"Error resolving or checking model path '{self._model_path}': {path_e}")
            raise ModelError(f"Invalid model path '{self._model_path}': {path_e}") from path_e

        logger.info(f"Initialized GGUFModel '{self._model_id}' (Path: {os.path.basename(self._model_path)})",
                    extra={"role": self._role})
        logger.debug(f"Effective parameters for '{self._model_id}': {self._parameters}")

    def _get_inference_lock(self) -> asyncio.Lock:
        # Provides access to the instance's lock for external coordinated use if needed
        return self._inference_lock

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text using the model's tokenizer.

        Args:
            text: The text to tokenize and count

        Returns:
            Number of tokens in the text

        Raises:
            ModelError: If the model isn't loaded
        """
        if not self.is_loaded() or self._llm is None:
            raise ModelError(f"Model {self._model_id} must be loaded to count tokens")

        try:
            # Use the model's tokenizer to get an accurate count
            tokens = self._llm.tokenize(text.encode('utf-8'))
            return len(tokens)
        except Exception as e:
            logger.warning(f"Error counting tokens with model tokenizer for {self._model_id}: {e}")
            # Fallback estimate (rough approximation for English text)
            words = len(text.split())
            est_tokens = int(words * 1.3)  # ~1.3 tokens per word is a rough estimate
            logger.warning(
                f"Using fallback token estimation for {self._model_id}: ~{est_tokens} tokens (based on {words} words)")
            return est_tokens

    def load(self) -> bool:
        """Loads the GGUF model using llama-cpp-python. Blocking call."""
        if self._llm is not None:
            logger.debug(f"Model {self._model_id} is already loaded.")
            return True
        if not os.path.exists(self._model_path):
            logger.error(f"Model path disappeared before loading: {self._model_path}")
            return False

        load_start_time = time.time()
        thread_name = threading.current_thread().name  # Get thread name for logging
        logger.info(f"[{thread_name}] Loading model {self._model_id} from {self._model_path}")
        llama_params = self._get_llama_constructor_params()
        try:
            logger.debug(
                f"[{thread_name}] Initializing llama_cpp.Llama for '{self._model_id}' with params: {llama_params}")
            self._llm = Llama(
                model_path=str(self._model_path),
                **llama_params
            )
            logger.debug(f"[{thread_name}] llama_cpp.Llama constructor RETURNED SUCCESSFULLY for '{self._model_id}'")

            self._is_loaded = True
            load_time = time.time() - load_start_time
            logger.info(f"[{thread_name}] Successfully loaded model {self._model_id} in {load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"[{thread_name}] EXCEPTION during llama_cpp.Llama constructor for '{self._model_id}'")
            self._llm = None
            self._is_loaded = False
            logger.error(f"[{thread_name}] Failed to load model {self._model_id} from {self._model_path}: {str(e)}",
                         exc_info=True)
            str_e = str(e).lower()
            if "ggml_assert" in str_e: logger.error(
                f"[{thread_name}] Hint: GGML_ASSERT errors often indicate an issue with the model file itself (corruption) or incompatibility with the llama.cpp version.")
            if "cublas" in str_e or "cuda" in str_e: logger.error(
                f"[{thread_name}] Hint: CUDA/cuBLAS related errors often point to driver issues or problems with the llama-cpp-python GPU build.")
            if "blas" in str_e and "cublas" not in str_e: logger.error(
                f"[{thread_name}] Hint: BLAS errors might indicate issues with the CPU backend library (OpenBLAS, etc.).")
            if "metal" in str_e: logger.error(
                f"[{thread_name}] Hint: Metal errors usually occur on macOS and might relate to GPU compatibility or build issues.")
            if "path" in str_e: logger.error(
                f"[{thread_name}] Hint: Double-check the model path is correct and accessible: {self._model_path}")
            return False
        # finally:
        #    logger.debug(f"[{thread_name}] Exiting load() method for {self._model_id}")

    def _get_llama_constructor_params(self) -> Dict[str, Any]:
        """Prepare parameters specifically for the Llama constructor."""
        known_llama_params = {
            # Model Params
            "n_ctx": int, "n_parts": int, "n_gpu_layers": int, "seed": int,
            "f16_kv": bool, "logits_all": bool, "vocab_only": bool, "use_mmap": bool, "use_mlock": bool,
            # Loading Params
            "lora_base": str, "lora_path": str,
            # Context Params
            "embedding": bool, "n_threads": int, "n_threads_batch": int, "n_batch": int,
            "last_n_tokens_size": int,
            # Misc
            "numa": bool, "verbose": bool, "chat_format": str,
            # Newer params examples
            # "rope_freq_base": float, "rope_freq_scale": float,
            # "main_gpu": int, "tensor_split": List[float],
        }

        constructor_params = {}
        model_params = self._parameters

        for param_key, param_type in known_llama_params.items():
            if param_key in model_params:
                value = model_params[param_key]
                try:
                    if value is None and param_key in ["lora_base", "lora_path", "chat_format"]:
                        constructor_params[param_key] = None
                        continue
                    elif value is None:
                        continue

                    original_value_repr = repr(value)
                    if param_type == bool and not isinstance(value, bool):
                        if isinstance(value, str):
                            value = value.lower() in ['true', '1', 't', 'y', 'yes', 'on']
                        else:
                            value = bool(value)
                    elif param_type == int and not isinstance(value, int):
                        value = int(float(value))
                    elif param_type == float and not isinstance(value, float):
                        value = float(value)
                    elif param_type == str and not isinstance(value, str):
                        value = str(value)
                    # Add list handling if needed (e.g., tensor_split)

                    constructor_params[param_key] = value
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Could not convert param '{param_key}' (value: {original_value_repr}) for Llama constructor for '{self._model_id}': {e}. Skipping parameter.")

        # Sensible Defaults & Overrides
        if constructor_params.get("verbose") is not True:
            constructor_params["verbose"] = False

        constructor_params["logits_all"] = True  # Required for logprobs

        if "n_ctx" not in constructor_params:
            constructor_params["n_ctx"] = 4096
            logger.debug(f"Using default n_ctx={constructor_params['n_ctx']} for {self._model_id}")

        if "n_gpu_layers" not in constructor_params:
            constructor_params["n_gpu_layers"] = 0
            logger.debug(f"Using default n_gpu_layers={constructor_params['n_gpu_layers']} for {self._model_id}")
        else:
            logger.debug(f"Using configured n_gpu_layers={constructor_params['n_gpu_layers']} for {self._model_id}")

        # Keep n_batch config or default from ConfigManager if present
        if "n_batch" in constructor_params:
            logger.debug(f"Using configured n_batch={constructor_params['n_batch']} for {self._model_id}")
        # NOTE: We removed the manual default setting for n_batch here based on previous findings.
        # If it's needed, it should come from the model defaults in the config.
        # else:
        #    logger.debug(f"n_batch not configured for {self._model_id}, llama-cpp will use its internal default.")

        return constructor_params

    # Changed to sync function based on previous analysis
    def unload(self) -> None:
        """Release the Llama model instance resources."""
        if not self._is_loaded or self._llm is None:
            logger.debug(f"Model {self._model_id} is not loaded or already unloaded")
            return

        logger.info(f"Unloading model {self._model_id}...")
        unload_start_time = time.time()

        llm_instance_ref = self._llm
        self._llm = None
        self._is_loaded = False

        try:
            del llm_instance_ref
            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug(f"Requested clearing CUDA cache during unload of {self._model_id}.")
            except ImportError:
                pass
            except Exception as cuda_e:
                logger.warning(f"Could not clear CUDA cache during unload of {self._model_id}: {cuda_e}")

            unload_duration = time.time() - unload_start_time
            logger.info(
                f"Model {self._model_id} instance released in {unload_duration:.2f}s (Resource cleanup might take slightly longer).")
        except Exception as e:
            logger.error(f"Error occurred during model {self._model_id} unload/cleanup: {str(e)}", exc_info=True)

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded and the instance exists."""
        return self._is_loaded and self._llm is not None

    def _prepare_generation_params(
            self,
            compute_confidence: bool = False,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """Merges default, model-specific, and call-specific generation parameters."""

        gen_params = {
            "temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": 2048,
            "repeat_penalty": 1.1, "echo": False, "logprobs": None
        }

        known_gen_keys = {
            "suffix", "max_tokens", "temperature", "top_p", "logprobs", "echo",
            "stop", "frequency_penalty", "presence_penalty", "repeat_penalty",
            "top_k", "stream", "seed", "tfs_z", "mirostat_mode", "mirostat_tau",
            "mirostat_eta", "grammar", "logits_processor"
        }

        model_gen_params = {k: v for k, v in self._parameters.items() if k in known_gen_keys}
        gen_params.update(model_gen_params)

        kwarg_overrides = {k: v for k, v in kwargs.items() if k in known_gen_keys}
        gen_params.update(kwarg_overrides)

        # --- Type checks and Post-processing ---

        if gen_params.get("max_tokens") is not None:
            try:
                max_t = int(gen_params["max_tokens"])
                gen_params["max_tokens"] = max_t if max_t > 0 else None
            except (ValueError, TypeError):
                logger.warning(
                    f"'max_tokens' value {gen_params['max_tokens']} is invalid for {self._model_id}. Setting to None (unlimited).")
                gen_params["max_tokens"] = None

        if "grammar" in gen_params and isinstance(gen_params["grammar"], str):
            grammar_str = gen_params["grammar"]
            if grammar_str:
                try:
                    gen_params["grammar"] = LlamaGrammar.from_string(grammar_str)
                    logger.debug(f"Successfully parsed grammar string for {self._model_id}.")
                except Exception as e:
                    logger.error(
                        f"Failed to parse grammar string for {self._model_id}: {e}. Removing grammar parameter.")
                    gen_params.pop("grammar", None)
            else:
                gen_params.pop("grammar", None)

        # Automatically enable logprobs if confidence calculation is explicitly requested FOR THIS CALL
        if compute_confidence and gen_params.get('logprobs') is None:
            # Setting to 1 implies getting logprobs for the *selected* token.
            # If top-k logprobs are needed, >1 should be used, but that complicates parsing.
            logprob_val = 1
            logger.debug(f"Confidence requested for {self._model_id}, setting 'logprobs={logprob_val}'.")
            gen_params['logprobs'] = logprob_val
        elif gen_params.get('logprobs') is not None:
            try:
                gen_params['logprobs'] = int(gen_params['logprobs'])
                if gen_params['logprobs'] <= 0:  # Ensure it's positive if set
                    logger.warning(f"'logprobs' value must be positive for {self._model_id}. Disabling logprobs.")
                    gen_params['logprobs'] = None
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid 'logprobs' value {gen_params['logprobs']} for {self._model_id}. Disabling logprobs.")
                gen_params['logprobs'] = None

        # 'stream' is handled differently; remove if present
        if 'stream' in gen_params:
            logger.debug(
                f"Removing 'stream' parameter for {self._model_id} as it's not supported by this async generate method's usage.")
            del gen_params['stream']

        logger.debug(f"Prepared final generation params for {self._model_id}: {gen_params}")
        return gen_params

    async def generate(
            self,
            prompt: str,
            **kwargs: Any  # Includes compute_confidence passed from ModelManager
    ) -> Dict[str, Any]:
        """Generate text using the loaded model, executed in a thread pool."""
        if not self.is_loaded() or self._llm is None:
            logger.error(f"Attempted to generate with unloaded model: {self._model_id}")
            raise ModelError(f"Model {self._model_id} is not loaded or instance is None")

        if self._executor is None:
            logger.error(f"Model {self._model_id} has no executor assigned for generation.")
            raise ModelError(f"Executor not available for model {self._model_id}")

        # Explicitly read compute_confidence flag for this specific call
        compute_confidence_flag = kwargs.get('compute_confidence', False)
        parameters_override = kwargs.get('parameters_override', {})

        kwargs_for_prepare = kwargs.copy()
        kwargs_for_prepare.pop('compute_confidence', None)  # Don't pass meta-flag down
        kwargs_for_prepare.pop('parameters_override', None)
        if isinstance(parameters_override, dict):
            kwargs_for_prepare.update(parameters_override)

        generation_params = self._prepare_generation_params(
            compute_confidence=compute_confidence_flag,  # Pass flag to enable logprobs if needed
            **kwargs_for_prepare
        )

        formatted_prompt = self._format_prompt(prompt)

        thread_name = threading.current_thread().name
        # Include role in log message
        model_id_with_role = f"{self._model_id} ({self._role})" if self._role else self._model_id

        # Get context window size
        context_window = self.get_context_window()

        # Count tokens and log prompt metrics
        token_count = 0
        try:
            if self._llm and hasattr(self._llm, 'tokenize'):
                tokens = self._llm.tokenize(formatted_prompt.encode('utf-8'))
                token_count = len(tokens)

                # Calculate usage percentage
                usage_pct = (token_count / context_window) * 100 if context_window > 0 else 0

                # Log comprehensive prompt metrics
                logger.info(
                    f"[{thread_name}] Prompt metrics for {model_id_with_role}: "
                    f"{len(formatted_prompt)} chars, {token_count}/{context_window} tokens "
                    f"({usage_pct:.1f}% of context window)"
                )

                # Warning if approaching context limit
                if token_count > context_window * 0.8 and token_count <= context_window:
                    logger.warning(
                        f"[{thread_name}] Prompt for {model_id_with_role} is using >{usage_pct:.1f}% "
                        f"of context window ({token_count}/{context_window} tokens)"
                    )
                # Error if exceeding context limit
                if token_count > context_window:
                    logger.error(
                        f"[{thread_name}] CONTEXT WINDOW EXCEEDED: Prompt token count ({token_count}) exceeds "
                        f"model's context window ({context_window}) for {model_id_with_role}"
                    )
                    raise ModelError(
                        f"Input exceeds model's context window ({token_count} > {context_window}). "
                        f"This can cause hanging or OOM errors."
                    )
            else:
                logger.warning(
                    f"[{thread_name}] Cannot count tokens for {model_id_with_role} - tokenize method not available")
                # Still log basic character count
                logger.debug(
                    f"[{thread_name}] Starting generation task for model {model_id_with_role}. "
                    f"Prompt length: {len(formatted_prompt)} characters. Context window: {context_window} tokens."
                )
        except ModelError as me:
            # Re-raise model errors (like context window exceeded)
            raise
        except Exception as token_e:
            # Log but continue for other token counting errors
            logger.warning(
                f"[{thread_name}] Error counting tokens for {model_id_with_role}: {token_e}. "
                f"Prompt has {len(formatted_prompt)} characters. Context window: {context_window} tokens. "
                f"Proceeding with generation anyway."
            )

        start_gen_time = time.time()

        try:
            raw_result = await run_in_threadpool(
                self._llm,
                _executor=self._executor,
                prompt=formatted_prompt,
                **generation_params
            )

            generation_time = time.time() - start_gen_time
            logger.debug(
                f"[{thread_name}] Raw generation result received for {model_id_with_role} (took {generation_time:.3f}s).")

            # --- Process the result ---
            generated_text = ""
            completion_tokens = 0
            logprobs_data = None  # Initialize

            if isinstance(raw_result, dict) and "choices" in raw_result and isinstance(raw_result["choices"], list) and \
                    raw_result["choices"]:
                first_choice = raw_result["choices"][0]
                if isinstance(first_choice, dict):
                    generated_text = first_choice.get("text", "").strip()
                    # Extract logprobs if they exist (structure might vary!)
                    logprobs_dict_or_list = first_choice.get("logprobs")

                    # --- ADD DETAILED LOGGING HERE ---
                    logger.debug(
                        f"[{thread_name}] Raw logprobs structure for {model_id_with_role}: {str(logprobs_dict_or_list)[:500]}...")  # Log snippet
                    # --- END ADDED LOGGING ---

                    # Attempt extraction - ADAPT THIS based on the logged structure!
                    # ... [logprobs extraction code - unchanged] ...

                else:
                    logger.warning(
                        f"[{thread_name}] First choice in 'choices' list is not a dictionary for model {model_id_with_role}: {first_choice}")
            else:
                logger.warning(
                    f"[{thread_name}] Unexpected raw_result structure or empty choices for model {model_id_with_role}. Cannot reliably extract text/logprobs. Result: {str(raw_result)[:200]}...")

            usage_stats = raw_result.get("usage", {}) if isinstance(raw_result, dict) else {}
            completion_tokens = usage_stats.get("completion_tokens", 0) if isinstance(usage_stats, dict) else 0
            prompt_tokens = usage_stats.get("prompt_tokens", 0) if isinstance(usage_stats, dict) else 0

            # Compare reported prompt tokens with our count if available
            if token_count > 0 and prompt_tokens > 0 and abs(token_count - prompt_tokens) > 5:
                logger.debug(
                    f"[{thread_name}] Token count discrepancy for {model_id_with_role}: "
                    f"pre-count={token_count}, reported={prompt_tokens}, diff={token_count - prompt_tokens}"
                )

            # Calculate confidence score ONLY IF flag is set and data potentially available
            confidence_score = None
            if compute_confidence_flag:
                # --- LOG BEFORE CALLING ---
                logger.debug(
                    f"[{thread_name}] Data being passed to calculate_confidence_from_logprobs for {model_id_with_role}: {str(logprobs_data)[:500]}...")  # Log snippet
                # --- END LOG ---
                confidence_score = calculate_confidence_from_logprobs(logprobs_data)
                if confidence_score is not None:
                    logger.debug(
                        f"[{thread_name}] Calculated confidence for {model_id_with_role}: {confidence_score:.4f}")
                else:
                    # --- ADD LOG IF CALCULATION FAILED ---
                    logger.warning(
                        f"[{thread_name}] calculate_confidence_from_logprobs returned None for {model_id_with_role}")

            response_data = {
                "text": generated_text,
                "generation_time": generation_time,
                "token_count": completion_tokens,
                "prompt_token_count": prompt_tokens if prompt_tokens > 0 else token_count,
                # Use our count if reported is missing
                # Store calculated score (might be None)
                "confidence": confidence_score,
                # Keep raw output for potential use by external confidence logic
                "raw_output": raw_result,
            }

            confidence_str = f"{confidence_score:.4f}" if confidence_score is not None else "N/A (Not Calculated or Failed)"
            # MODIFIED HERE: Include role in log message
            logger.info(
                f"[{thread_name}] Model {model_id_with_role} generated {completion_tokens} tokens in {generation_time:.2f}s. Confidence (Token Prob): {confidence_str}"
            )

            return response_data

        except Exception as e:
            gen_duration = time.time() - start_gen_time
            if isinstance(e, TypeError) and "NoneType.__format__" in str(e):
                logger.error(
                    f"[{thread_name}] Generation failed for model {model_id_with_role} after {gen_duration:.2f}s due to likely logging format error with None confidence. "
                    f"Original error: {type(e).__name__}: {str(e)}",
                    exc_info=False
                )
            else:
                logger.error(
                    f"[{thread_name}] Generation failed for model {model_id_with_role} after {gen_duration:.2f}s: {type(e).__name__}: {str(e)}",
                    exc_info=True
                )
            raise ModelError(f"Generation failed with model {model_id_with_role}: {str(e)}") from e

    def _format_prompt(self, prompt: str) -> str:
        """Applies system prompt and/or chat formatting if configured."""
        system_prompt = self._parameters.get("system_prompt")
        chat_formatter_name = self._parameters.get("chat_format")

        if chat_formatter_name and self._llm and hasattr(self._llm, 'chat_handler') and self._llm.chat_handler:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                # llama-cpp-python >=0.2.X uses create_chat_completion handlers
                if hasattr(self._llm, 'create_chat_completion'):
                    # This is complex as the handler isn't easily accessible just for prompt formatting
                    # We might need to manually apply formatting based on chat_formatter_name if possible,
                    # or rely on simpler system prompt prepending as fallback.
                    logger.warning(
                        f"Automatic chat formatting via create_chat_completion handler not fully implemented for prompt preparation in GGUFModel._format_prompt for {self._model_id}. Falling back.")
                # Handle older chat_handler style if necessary (less likely now)
                elif hasattr(self._llm, 'chat_handler'):
                    chat_handler = self._llm.chat_handler(messages=messages)
                    full_prompt = chat_handler.prompt()  # Deprecated?
                    logger.debug(
                        f"Formatted prompt using legacy Llama chat_handler ({chat_formatter_name}) for {self._model_id}")
                    return full_prompt

            except Exception as e:
                logger.warning(
                    f"Failed to use Llama chat formatting for {self._model_id} (format: {chat_formatter_name}). Falling back. Error: {e}")

        # Fallback: Basic system prompt prepending
        if system_prompt:
            logger.debug(f"Formatting prompt using basic system prompt prepend for {self._model_id}")
            return f"{system_prompt.strip()}\n\n{prompt.strip()}"
        else:
            logger.debug(f"No special prompt formatting applied for {self._model_id}")
            return prompt.strip()

    # --- Standard Getters ---
    def get_id(self) -> str:
        return self._model_id

    def get_path(self) -> str:
        return self._model_path

    def get_role(self) -> Optional[str]:
        return self._role

    def get_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._model_config_original)

    def get_context_window(self) -> int:
        return self._parameters.get("n_ctx", 4096)

    def _set_executor(self, executor: Optional[ThreadPoolExecutor]) -> None:
        self._executor = executor

    def get_effective_parameters(self) -> Dict[str, Any]:
        return copy.deepcopy(self._parameters)
