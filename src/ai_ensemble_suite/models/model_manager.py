# src/ai_ensemble_suite/models/model_manager.py

from typing import Dict, Any, Optional, List, Set, Protocol, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
import re
import psutil
import copy
import random # Import random for get_random_model
import threading # <<< Added Import
from pathlib import Path # Added for path resolution

# Correct import path assuming gguf_model.py is in the same directory or accessible via path
from ai_ensemble_suite.models.gguf_model import GGUFModel
from ai_ensemble_suite.models.confidence import get_confidence_score # Needs confidence.py
from ai_ensemble_suite.exceptions import ModelError, ConfigurationError, ResourceError
from ai_ensemble_suite.utils.logging import logger
from ai_ensemble_suite.utils.async_utils import run_in_threadpool, gather_with_concurrency

class ConfigProvider(Protocol):
    """Protocol for configuration providers."""
    def get_all_models_config(self) -> Dict[str, Dict[str, Any]]: ...
    def get_template(self, template_name: str) -> Optional[str]: ...
    def get_confidence_config(self) -> Dict[str, Any]: ...
    # Ensure this matches ConfigManager implementation
    def get_default_model_parameters(self) -> Dict[str, Any]: ...
    # Required by the loading logic with path resolution/config passing
    def get_model_config(self, model_id: str) -> Dict[str, Any]: ...
    # Optional: path of the config file for better relative path resolution
    # @property
    # def config_path(self) -> Optional[str]: ...

class ModelManager:
    """Manages the loading, execution, and lifecycle of GGUF models."""

    def __init__(
            self,
            config_manager: ConfigProvider,
            ensemble: Optional["Ensemble"] = None,  # Add ensemble parameter
            max_workers: Optional[int] = None
    ) -> None:
        """Initialize the ModelManager.

        Args:
            config_manager: Configuration manager instance that implements ConfigProvider.
            ensemble: Optional reference to the parent Ensemble instance.
            max_workers: Optional limit for thread pool size.

        Raises:
            TypeError: If config_manager doesn't implement required methods.
        """
        # Validate the ConfigProvider implementation
        required_methods = [
            'get_all_models_config', 'get_template', 'get_confidence_config',
            'get_default_model_parameters', 'get_model_config'
        ]
        if not all(hasattr(config_manager, method) for method in required_methods):
            missing = [m for m in required_methods if not hasattr(config_manager, m)]
            raise TypeError(f"config_manager must implement the ConfigProvider protocol methods. Missing: {missing}")

        self.config_manager = config_manager
        self.ensemble = ensemble  # Store reference to parent ensemble
        self.models: Dict[str, GGUFModel] = {}  # Stores successfully loaded GGUFModel instances
        self.initialized = False

        # Determine max workers for the thread pool
        if max_workers is None:
             try:
                 # Use physical cores as a baseline, clamp between reasonable limits
                 cores = psutil.cpu_count(logical=False) or 1
                 # Example: Limit to 8 workers max by default, even if more cores
                 max_workers = min(cores, 8)
                 logger.info(f"Determined default max_workers based on physical cores: {max_workers}")
             except Exception as e:
                 logger.warning(f"Could not detect CPU cores: {e}. Using default max_workers=4")
                 max_workers = 4 # Fallback default

        # Ensure at least 1 worker
        max_workers = max(1, max_workers)

        # Create the thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="GGUFWorker")

        # --- Concurrency Control ---
        # Limit concurrent loads (often IO/CPU bound init) more strictly than inference
        # Example: Allow half the workers for loading concurrently
        self._max_concurrent_loads: int = max(1, max_workers // 2)
        self._max_concurrent_loads = 1  # Force limit to 1
        self._load_semaphore = asyncio.Semaphore(self._max_concurrent_loads)

        # Limit concurrent inference calls (can be adjusted)
        self._max_concurrent_inference: int = max_workers # Allow all workers for inference by default

        logger.info(f"Initialized ModelManager with up to {max_workers} worker threads.")
        logger.info(f"Concurrency limits set: Loads={self._max_concurrent_loads}, Inference={self._max_concurrent_inference}")

    def _resolve_model_path(self, model_id: str, model_path_str: str) -> str:
        """Resolves a potentially relative model path to an absolute path.

        Tries to resolve relative to CWD as a fallback. More robust implementation
        would use the config file's directory from ConfigManager if available.
        """
        model_path = Path(model_path_str)
        if model_path.is_absolute():
            return str(model_path)

        # --- Attempt resolution relative to CWD (fallback) ---
        # TODO: Enhance this to use config file location from config_manager if possible
        # config_dir = Path(self.config_manager.config_path).parent if hasattr(self.config_manager, 'config_path') and self.config_manager.config_path else Path.cwd()
        config_dir = Path.cwd() # Current fallback
        resolved_path = (config_dir / model_path).resolve()

        logger.debug(f"Resolved relative path '{model_path_str}' for model '{model_id}' relative to '{config_dir}' -> '{resolved_path}'")
        return str(resolved_path)

    async def _load_model_async(self, model_id: str) -> Tuple[str, Optional[GGUFModel], Optional[Exception]]:
        """Loads a single model asynchronously using the thread pool executor and semaphore."""
        # Use async with pattern for proper semaphore handling
        async with self._load_semaphore:
            thread_id = threading.get_ident()
            logger.debug(f"[Thread-{thread_id}] Acquired load semaphore slot for {model_id}.")

            loop = asyncio.get_running_loop()
            try:
                # Get a fresh copy of the config for this specific model
                model_config = self.config_manager.get_model_config(model_id)
                if not model_config:
                    raise ConfigurationError(f"Could not retrieve config for model '{model_id}'")

                model_path_from_config = model_config.get("path")
                if not model_path_from_config:
                    raise ConfigurationError(f"Path missing in config for model '{model_id}'")

                # Resolve path before passing to thread
                resolved_path = self._resolve_model_path(model_id, model_path_from_config)
                # Add resolved path back to config dict for GGUFModel __init__
                model_config_with_resolved_path = model_config.copy()
                model_config_with_resolved_path['resolved_path'] = resolved_path

                logger.debug(f"[Thread-{thread_id}] Submitting loading task for {model_id} to executor.")

                # Use run_in_executor to run the synchronous loading part in the pool
                future = loop.run_in_executor(
                    self.executor,
                    lambda: self._perform_load_sync(model_id, model_config_with_resolved_path)
                )
                # Await the result from the thread pool
                result: Tuple[str, Optional[GGUFModel], Optional[Exception]] = await future
                logger.debug(f"[Thread-{thread_id}] Received result from executor for {model_id}.")
                return result

            except Exception as e:
                # Catch errors during task submission or config retrieval
                logger.error(f"[Thread-{thread_id}] Error in _load_model_async coordinating load for {model_id}: {e}",
                             exc_info=True)
                return model_id, None, e
                # No need to manually release semaphore - async with handles it

    def _perform_load_sync(self, model_id: str, model_config: Dict[str, Any]) -> Tuple[
        str, Optional[GGUFModel], Optional[Exception]]:
        """Synchronous part of model loading executed in the thread pool."""
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Attempting to load model {model_id}...")
        start_load_time = time.time()
        model_instance: Optional[GGUFModel] = None
        try:
            # Path should already be resolved and in the config dict
            resolved_path = model_config.get('resolved_path')
            if not resolved_path:
                raise ConfigurationError(f"Resolved path missing in config passed to _perform_load_sync for {model_id}")

            # ---- Instantiate the Model Wrapper ----
            logger.debug(f"[{thread_name}] Instantiating GGUFModel wrapper for {model_id}...")
            model_instance = GGUFModel(
                model_id=model_id,
                model_path=resolved_path,
                model_config=model_config,
                config_manager=self.config_manager,
                executor=self.executor
            )
            logger.debug(f"[{thread_name}] GGUFModel wrapper instantiated for {model_id}.")

            # ---- Perform the actual blocking load ----
            load_successful: bool = model_instance.load()
            logger.debug(f"[{thread_name}] GGUFModel.load() RETURNED {load_successful} for {model_id}")

            if not load_successful:
                logger.error(f"[{thread_name}] GGUFModel.load() returned False for model {model_id}. Load failed.")
                raise ModelError(f"GGUFModel.load() returned False for model {model_id}")

            # --- Load successful ---
            load_time = time.time() - start_load_time
            logger.info(f"[{thread_name}] Model {model_id} loaded successfully in {load_time:.2f}s.")
            return model_id, model_instance, None

        except Exception as e:
            logger.error(f"[{thread_name}] Error during sync load process for model {model_id}: {e}", exc_info=True)
            # Attempt cleanup if instance exists but load failed
            if model_instance and hasattr(model_instance, 'is_loaded') and model_instance.is_loaded():
                logger.debug(f"[{thread_name}] Attempting cleanup for failed model {model_id}")
                try:
                    # Mark model as not loaded to prevent future issues
                    model_instance._is_loaded = False
                    # Release any resources that can be directly released
                    if hasattr(model_instance, '_llm'):
                        model_instance._llm = None
                    logger.debug(f"[{thread_name}] Marked model as unloaded and cleared references")
                except Exception as unload_e:
                    logger.error(f"[{thread_name}] Error during cleanup for failed model {model_id}: {unload_e}")
            return model_id, None, e

    async def initialize(self) -> None:
        """Initialize the ModelManager: Instantiates models and loads them asynchronously."""
        if self.initialized:
            logger.warning("ModelManager already initialized. Skipping.")
            return

        logger.info("Initializing ModelManager...")
        init_start_time = time.time()
        model_configs: Dict[str, Dict[str, Any]] = {}

        try:
            # 1. Get configurations for all models
            model_configs = self.config_manager.get_all_models_config()
            if not model_configs:
                raise ConfigurationError("No models found in configuration")
            logger.info(f"Found {len(model_configs)} models in configuration.")

            # 2. Create asynchronous loading tasks for each model
            load_tasks = []
            task_to_model_id_map = {} # Map index to model_id for result processing
            valid_model_ids_to_load = list(model_configs.keys()) # Keep track

            for i, model_id in enumerate(valid_model_ids_to_load):
                logger.debug(f"Creating async load task {i} for model {model_id}")
                # Use the helper that handles semaphore and calls sync part in executor
                task = self._load_model_async(model_id)
                load_tasks.append(task)
                task_to_model_id_map[i] = model_id # Store mapping BEFORE awaiting

            if not load_tasks:
                raise ModelError("No load tasks were created (possibly no valid models in config).")

            # 3. Run loading tasks concurrently using asyncio.gather
            # The semaphore inside _load_model_async limits actual concurrency.
            logger.info(f"Loading {len(load_tasks)} models concurrently (Hardware limit via semaphore: {self._max_concurrent_loads})...")
            # We can use simple gather here as semaphore controls concurrency
            load_results = await asyncio.gather(*load_tasks, return_exceptions=True)

            # 4. Process the results
            temp_models = {} # Dict to store successfully loaded model instances
            failed_loads = [] # List to store IDs of models that failed to load

            # >>> NEW LOGGING LINE <<<
            logger.debug(f"Processing {len(load_results)} results from asyncio.gather in initialize()...")
            # ^^^^^^^^^^^^^^^^^^^^^^^^
            for i, result in enumerate(load_results):
                # Get the model ID corresponding to this result index
                model_id = task_to_model_id_map.get(i, f"Unknown Task {i}")
                # >>> NEW LOGGING LINE <<<
                logger.debug(f"Processing result index {i} for model_id '{model_id}'...")
                # ^^^^^^^^^^^^^^^^^^^^^^^^

                if isinstance(result, Exception):
                    # Task itself raised exception (e.g., during coordination in _load_model_async)
                    logger.error(f"Load task for model '{model_id}' failed directly in gather (exception type: {type(result).__name__}): {result}", exc_info=result)
                    failed_loads.append(model_id)
                elif isinstance(result, tuple) and len(result) == 3:
                    # Expected result format: (model_id, model_instance | None, exception | None)
                    res_id, model_instance, error = result
                    if model_id != res_id: # Sanity check
                         logger.warning(f"Model ID mismatch in result processing: expected '{model_id}', got '{res_id}'. Using '{res_id}'.")
                         model_id = res_id # Trust the ID returned from the task

                    if error:
                         # _perform_load_sync caught an exception or load returned False
                         logger.error(f"Load task for model '{model_id}' reported failure (exception type: {type(error).__name__}): {error}", exc_info=error if isinstance(error, Exception) else None)
                         failed_loads.append(model_id)
                    elif model_instance is not None and isinstance(model_instance, GGUFModel):
                         # Successfully loaded
                         # >>> NEW LOGGING LINE <<<
                         logger.debug(f"Successfully processed load result for model '{model_id}'. Adding to temp_models.")
                         # ^^^^^^^^^^^^^^^^^^^^^^^^
                         temp_models[model_id] = model_instance
                    else:
                         # Should not happen: success tuple but no instance
                         logger.error(f"Load task for model '{model_id}' returned success tuple but no valid model instance.")
                         failed_loads.append(model_id)
                else:
                     # Unexpected format from gather (should be Exception or Tuple)
                     logger.error(f"Unexpected result format '{type(result)}' from load task for model '{model_id}'.")
                     failed_loads.append(model_id)

            # >>> NEW LOGGING LINE <<<
            logger.debug("Finished processing results from asyncio.gather.")
             # ^^^^^^^^^^^^^^^^^^^^^^^^

            # 5. Finalize initialization state
            self.models = temp_models # Assign successfully loaded models to the manager

            if not self.models:
                # All models failed to load
                raise ModelError("Initialization failed: No models were loaded successfully.")

            self.initialized = True
            init_duration = time.time() - init_start_time
            logger.info(f"ModelManager initialized successfully with {len(self.models)} loaded models in {init_duration:.2f}s.")
            if failed_loads:
                 # Log warnings for models that failed
                 logger.warning(f"Models failed to load during initialization: {', '.join(failed_loads)}")


        except (ModelError, ConfigurationError, ResourceError) as e:
            # Catch specific known errors during initialization phase
            logger.error(f"ModelManager initialization failed: {e}. Triggering shutdown.", exc_info=True)
            await self.shutdown() # Attempt graceful cleanup
            raise # Re-raise the caught exception
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error during ModelManager initialization: {e}. Triggering shutdown.", exc_info=True)
            await self.shutdown() # Attempt graceful cleanup
            # Wrap in ModelError or re-raise depending on desired behavior
            raise ModelError(f"Unexpected initialization error: {e}") from e

    async def load_models(self, model_ids: Optional[List[str]] = None) -> Tuple[int, int]:
        """Loads specified models or attempts to load all un-loaded configured models.

        Uses the same async loading mechanism as initialize. Primarily for loading
        models that failed during init or were manually unloaded.

        Args:
            model_ids: List of model IDs to load. If None, tries all models from
                       config that are not currently loaded in the manager.

        Returns:
            Tuple (successful_loads, failed_loads).
        """
        if model_ids is None:
            # Load all models from config not currently present or loaded
            all_config_ids = set(self.config_manager.get_all_models_config().keys())
            loaded_ids = set(mid for mid, m in self.models.items() if m.is_loaded())
            ids_to_consider = list(all_config_ids - loaded_ids)
            logger.info(f"Attempting to load all models defined in config that are not currently loaded ({len(ids_to_consider)} models).")
        else:
            # Load specific requested models if they exist in config and aren't loaded
            all_config_ids = set(self.config_manager.get_all_models_config().keys())
            valid_ids_in_request = [mid for mid in model_ids if mid in all_config_ids]

            if len(valid_ids_in_request) != len(model_ids):
                 missing_in_config = set(model_ids) - set(valid_ids_in_request)
                 logger.warning(f"Requested models not defined in config and will be ignored: {list(missing_in_config)}")

            # Filter out models already loaded
            ids_to_consider = [mid for mid in valid_ids_in_request if mid not in self.models or not self.models[mid].is_loaded()]
            already_loaded_count = len(valid_ids_in_request) - len(ids_to_consider)
            if already_loaded_count > 0:
                 logger.info(f"{already_loaded_count} requested models are already loaded and will be skipped.")

        if not ids_to_consider:
            logger.info("No models specified or require loading.")
            # Return reflects status of originally requested (if any) vs. needs loading
            initial_request_count = len(model_ids) if model_ids is not None else 0
            return initial_request_count, 0 # Report 0 failures for *this* operation

        # --- Create and run load tasks ---
        load_tasks = []
        task_to_model_id = {} # Map index back to model_id
        logger.info(f"Preparing to load {len(ids_to_consider)} models...")
        for i, model_id in enumerate(ids_to_consider):
            task = self._load_model_async(model_id) # Use the core async loader
            load_tasks.append(task)
            task_to_model_id[i] = model_id

        if not load_tasks:
             logger.error("Internal error: No load tasks created despite having models to consider.")
             return 0, len(ids_to_consider) # All considered models failed to start task

        logger.info(f"Loading {len(load_tasks)} models concurrently (Hardware limit: {self._max_concurrent_loads})...")
        start_time = time.time()

        # Gather results (semaphore limits actual concurrency)
        load_results = await asyncio.gather(*load_tasks, return_exceptions=True)

        exec_time = time.time() - start_time
        logger.info(f"Finished loading models batch in {exec_time:.2f}s.")

        # --- Process results ---
        success_count = 0
        fail_count = 0
        newly_loaded_models = {} # Store successfully loaded models from THIS batch

        for i, result in enumerate(load_results):
            model_id_loaded = task_to_model_id.get(i, f"Unknown Task {i}")

            if isinstance(result, Exception):
                logger.error(f"Load task for model '{model_id_loaded}' failed directly in gather (exception type: {type(result).__name__}): {result}", exc_info=result)
                fail_count += 1
            elif isinstance(result, tuple) and len(result) == 3:
                 res_id, model_instance, error = result
                 if model_id_loaded != res_id: logger.warning(f"Model ID mismatch: expected '{model_id_loaded}', got '{res_id}'")

                 if error:
                      logger.error(f"Load task for model '{res_id}' reported failure: {error}", exc_info=error if isinstance(error, Exception) else None)
                      fail_count += 1
                 elif model_instance is not None and isinstance(model_instance, GGUFModel):
                      logger.debug(f"Successfully loaded model '{res_id}' in this batch.")
                      newly_loaded_models[res_id] = model_instance
                      success_count += 1
                 else:
                      logger.error(f"Load task for model '{res_id}' returned success tuple but no valid model instance.")
                      fail_count += 1
            else:
                 logger.error(f"Unexpected result format '{type(result)}' from load task for model '{model_id_loaded}'.")
                 fail_count += 1

        # Update the manager's dictionary ONLY with newly loaded models from this batch
        self.models.update(newly_loaded_models)

        if fail_count > 0:
             logger.warning(f"Model loading batch complete for requested IDs. Newly Loaded: {success_count}, Failures in batch: {fail_count}")
        else:
             logger.info(f"Model loading batch successful. Newly loaded: {success_count}")

        return success_count, fail_count # Return counts for THIS batch operation

    async def shutdown(self) -> None:
        """Shutdown the ModelManager: Unloads models and shuts down the executor."""
        if not self.models and self.executor is None and not self.initialized:
             # Avoid logging shutdown if nothing was ever really started
             logger.info("ModelManager already shut down or was not initialized.")
             return

        logger.info("Shutting down ModelManager...")
        shutdown_start_time = time.time()

        # --- Unload Models ---
        unload_tasks = []
        # Copy items to avoid modification during iteration if needed, though list() might suffice
        active_models = list(self.models.values())
        if active_models:
             logger.debug(f"Preparing to unload {len(active_models)} managed models...")
             for model in active_models:
                  if model.is_loaded():
                      # Run async unload (which likely calls sync unload in executor)
                      # Make unload itself potentially async if it helps
                      unload_tasks.append(run_in_threadpool(model.unload, _executor=self.executor)) # Run unload in pool
                  else:
                      # Clean up references if needed for unloaded models
                      if hasattr(model, '_set_executor'): model._set_executor(None)

        if unload_tasks:
             logger.debug(f"Executing {len(unload_tasks)} unload tasks concurrently...")
             unload_results = await asyncio.gather(*unload_tasks, return_exceptions=True)
             # Log any errors during unload
             for i, res in enumerate(unload_results):
                 # Find corresponding model (order should match active_models used)
                 # model_id_unloaded = active_models[i]._model_id if i < len(active_models) and hasattr(active_models[i], '_model_id') else f"Unknown Task {i}" # Original attempt
                 model_id_unloaded = active_models[i].get_id() if i < len(
                     active_models) else f"Unknown Task {i}"  # Use getter method
                 if isinstance(res, Exception):
                     logger.error(f"Error during unload task for model '{model_id_unloaded}': {res}", exc_info=res)
                 elif res is False:  # If unload() can return False on failure
                     logger.warning(f"Unload method returned False for model '{model_id_unloaded}'.")
             logger.debug("Finished executing unload tasks.")


        # --- Shutdown Executor ---
        # Should happen AFTER all tasks using it (including unload) are complete
        if self.executor:
            logger.debug("Shutting down thread pool executor...")
            try:
                # Wait for all submitted tasks to complete before shutting down
                self.executor.shutdown(wait=True, cancel_futures=False) # cancel_futures=False is safer usually
                logger.debug("Executor shut down successfully.")
            except Exception as e:
                 logger.error(f"Error shutting down executor: {e}")
            finally:
                 self.executor = None # Ensure it's marked as gone

        # --- Clear State ---
        self.models = {} # Clear the dictionary of model instances
        self.initialized = False # Mark as not initialized

        shutdown_duration = time.time() - shutdown_start_time
        logger.info(f"ModelManager shut down successfully in {shutdown_duration:.2f}s.")

    async def run_inference(
            self,
            model_id: str,
            prompt: str,
            compute_confidence: bool = True,
            parameters_override: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """Run inference on a specific model using its generate method via thread pool."""
        # Combine explicit args with kwargs for passing down, giving precedence to kwargs
        inference_kwargs = kwargs.copy()
        if parameters_override:
            inference_kwargs.update(parameters_override)  # Merge overrides
        # Ensure compute_confidence flag is correctly passed or overridden
        inference_kwargs['compute_confidence'] = compute_confidence

        if not self.initialized:
            # Check if we intend to allow inference before full init (e.g. loading on demand)
            # For now, strict check:
            raise ModelError("ModelManager not initialized. Call initialize() first.")

        # Retrieve the model instance
        if model_id not in self.models:
            raise ModelError(f"Model not found: {model_id}")

        model = self.models[model_id]

        # Ensure model is loaded
        if not model.is_loaded():
            # Option: Attempt to load it here? Or just fail? Currently fails.
            # await self.load_models([model_id]) # Example load-on-demand (needs error handling)
            # if not model.is_loaded(): raise ModelError(...)
            raise ModelError(f"Model '{model_id}' is not loaded.")

        # --- DIAGNOSTIC FLAG (Can be removed if confidence is stable) ---
        internal_compute_confidence = False
        role = model.get_role()
        model_id_with_role = f"{model_id} ({role})" if role else model_id

        if not internal_compute_confidence:
            # Ensure the flag passed down to generate doesn't falsely trigger logprobs if we skip calc
            inference_kwargs['compute_confidence'] = False
            logger.debug(f"Diagnostic: internal_compute_confidence override: False for model {model_id_with_role}")
        else:
            logger.debug(f"Diagnostic: internal_compute_confidence: True for model {model_id_with_role}")
        # --- END DIAGNOSTIC FLAG ---

        total_start_time = time.time()
        thread_name = threading.current_thread().name  # Coordinating thread

        # Use the model's specific lock to prevent concurrent generation ON THE SAME instance
        # Assumes GGUFModel has `_inference_lock = asyncio.Lock()`
        if not hasattr(model, '_inference_lock') or not isinstance(getattr(model, '_inference_lock', None),
                                                                   asyncio.Lock):
            logger.warning(
                f"Model {model_id_with_role} lacks a valid '_inference_lock'. Inference will proceed without instance-level locking.")
            lock_context = asyncio.Lock()  # Create temp lock CANCELED - this locks ALL, use dummy

            # Define a dummy async context manager that does nothing
            class NoLock:
                async def __aenter__(self): pass

                async def __aexit__(self, exc_type, exc, tb): pass

            lock_context = NoLock()
        else:
            lock_context = model._inference_lock  # Use the model's lock

        async with lock_context:
            logger.debug(f"[{thread_name}] Acquired inference lock for model {model_id_with_role}")
            try:
                # Call the GGUFModel's generate method (which uses run_in_threadpool)
                logger.debug(f"[{thread_name}] Calling model.generate for {model_id_with_role}")
                # Pass the prompt and all collected kwargs
                generation_result = await model.generate(
                    prompt=prompt,
                    **inference_kwargs  # Includes compute_confidence, overrides, etc.
                )
                logger.debug(f"[{thread_name}] model.generate completed for {model_id_with_role}")

                # --- Post-processing (if GGUFModel.generate doesn't do it all) ---
                # GGUFModel.generate should ideally return the full desired dict including metadata
                # If not, do minimal additions here:
                result_with_meta = copy.deepcopy(generation_result)  # Assume generate returns a dict
                result_with_meta["model_id"] = model_id  # Ensure model ID is present
                result_with_meta["role"] = model.get_role()  # Add role info

                # --- Confidence calculation (IF ENABLED INTERNALLY) ---
                if internal_compute_confidence:
                    logger.debug(f"[{thread_name}] Attempting confidence calculation for {model_id_with_role}")
                    # This part might need adjustment based on what 'model.generate' returns
                    # Assuming generate returns 'text' and 'raw_output' needed by get_confidence_score
                    try:
                        confidence_config = self.config_manager.get_confidence_config()
                        conf_method = inference_kwargs.get("confidence_method",
                                                           confidence_config.get("default_method", "combined"))
                        # Extract relevant keys for confidence calculation from kwargs or config
                        conf_kwargs_keys = {
                            "token_prob_weight", "self_eval_weight", "consistency_weight",
                            "consistency_samples", "consistency_temperature",
                            "consistency_max_tokens", "token_metric", "self_eval_template"
                        }
                        conf_kwargs = {k: confidence_config.get(k) for k in conf_kwargs_keys if k in confidence_config}
                        conf_kwargs.update({k: v for k, v in inference_kwargs.items() if k in conf_kwargs_keys})

                        confidence_scores = await get_confidence_score(
                            model=model,  # Pass the GGUFModel instance
                            prompt=prompt,  # Original prompt
                            response=generation_result.get("text", ""),  # Generated text
                            model_output=generation_result.get("raw_output", {}),
                            # Raw output potentially containing logprobs
                            method=conf_method,
                            **conf_kwargs
                        )
                        result_with_meta["confidence"] = confidence_scores  # Add/update confidence
                        logger.debug(f"[{thread_name}] Confidence calculation successful for {model_id_with_role}")
                    except Exception as conf_e:
                        logger.error(
                            f"[{thread_name}] Failed to compute confidence for model {model_id_with_role}: {conf_e}",
                            exc_info=True)
                        result_with_meta["confidence"] = {"error": str(conf_e), "combined": 0.5}  # Default error state
                else:
                    # Ensure key exists even if skipped (might be None or added by generate)
                    if "confidence" not in result_with_meta:
                        result_with_meta["confidence"] = None
                    logger.debug(
                        f"[{thread_name}] Confidence calculation skipped for {model_id_with_role} due to diagnostic flag.")
                # --- End Confidence Block ---

                total_time = time.time() - total_start_time
                result_with_meta["total_inference_time"] = total_time  # More specific name?

                logger.debug(
                    f"[{thread_name}] Completed run_inference coordinator for {model_id_with_role} in {total_time:.2f}s")
                return result_with_meta

            except ModelError as e:
                # Model-specific error already logged by generate, re-raise
                logger.error(f"[{thread_name}] ModelError during inference coordination for {model_id_with_role}: {e}",
                             exc_info=True)
                raise
            except Exception as e:
                # Catch unexpected errors during the locked section/coordination
                total_time = time.time() - total_start_time
                logger.error(
                    f"[{thread_name}] Unexpected error during inference coordination for model {model_id_with_role} after {total_time:.2f}s: {str(e)}",
                    exc_info=True)
                # Wrap in ModelError for consistency
                raise ModelError(
                    f"Inference coordination failed unexpectedly for model {model_id_with_role}: {str(e)}") from e
            finally:
                logger.debug(f"[{thread_name}] Releasing inference lock for model {model_id_with_role}")
                # Lock released automatically by 'async with'

    async def run_all_models(self,
                            prompt: str,
                            model_ids: Optional[List[str]] = None,
                            compute_confidence: bool = True,
                            parameters_override: Optional[Dict[str, Any]] = None,
                            **kwargs) -> Dict[str, Dict[str, Any]]:
        """Runs inference concurrently on specified or all loaded models."""
        if not self.initialized:
             logger.warning("ModelManager not initialized, returning empty results for run_all_models.")
             return {}

        start_time = time.time()

        # Determine which models to run
        if model_ids is None:
            # Run all currently loaded models
            ids_to_run = [mid for mid, m in self.models.items() if m.is_loaded()]
            logger.info(f"Running inference on all {len(ids_to_run)} loaded models.")
        else:
            # Run only specified models that are loaded
            ids_to_run = []
            ignored_ids = []
            for mid in model_ids:
                if mid in self.models and self.models[mid].is_loaded():
                    ids_to_run.append(mid)
                else:
                    ignored_ids.append(mid)
            if ignored_ids:
                 logger.warning(f"Ignoring requested models for run_all_models as they are not loaded or managed: {ignored_ids}")
            logger.info(f"Running inference on specified loaded models: {ids_to_run}")


        if not ids_to_run:
            logger.warning("No loaded models specified or available to run inference in run_all_models.")
            return {}

        # --- Create and run inference tasks ---
        concurrency_limit = self._max_concurrent_inference
        logger.info(f"Running inference tasks concurrently (Hardware limit: {concurrency_limit}).")

        results: Dict[str, Dict[str, Any]] = {} # Store results keyed by model_id
        tasks = []
        task_to_model_id = {} # Map task index to model ID for better error association

        for i, model_id in enumerate(ids_to_run):
            # Create coroutine for run_inference
            task_coro = self.run_inference(
                 model_id=model_id,
                 prompt=prompt,
                 compute_confidence=compute_confidence,
                 parameters_override=parameters_override,
                 **kwargs # Pass any other relevant kwargs
            )
            tasks.append(task_coro)
            task_to_model_id[i] = model_id # Store mapping

        logger.debug(f"Launching {len(tasks)} inference tasks...")
        # Use gather_with_concurrency to limit simultaneous executions
        inference_results_list = await gather_with_concurrency(
            concurrency_limit,
            *tasks,
            return_exceptions=True # Capture exceptions from run_inference tasks
        )
        logger.debug("All inference tasks completed or failed.")

        # --- Process results ---
        success_count = 0
        fail_count = 0
        for i, task_result in enumerate(inference_results_list):
            current_model_id = task_to_model_id.get(i, f"Unknown Task {i}") # Get model ID for this result

            if isinstance(task_result, Exception):
                 logger.error(f"Inference task for model '{current_model_id}' failed with exception (type: {type(task_result).__name__}): {task_result}", exc_info=task_result)
                 fail_count += 1
                 # Store error under the correct model ID
                 results[current_model_id] = {
                     "error": f"Task Exception: {type(task_result).__name__}: {str(task_result)}",
                     "model_id": current_model_id, # Include model_id in error dict
                     "role": self.models.get(current_model_id, None).get_role() if current_model_id in self.models else None # Add role if possible
                }

            elif isinstance(task_result, dict) and "model_id" in task_result:
                 # Result is a dictionary, likely success or partial failure from run_inference
                 model_id_res = task_result["model_id"]
                 if model_id_res != current_model_id:
                     logger.warning(f"Task index model ID '{current_model_id}' differs from result dict model ID '{model_id_res}'. Using result dict ID.")

                 if "error" in task_result: # Check if run_inference itself returned an error dict
                      logger.warning(f"Model '{model_id_res}' reported error during inference: {task_result['error']}")
                      fail_count += 1
                 else:
                      # Assume success if dict has model_id and no 'error' key
                      logger.debug(f"Model '{model_id_res}' inference successful.")
                      success_count += 1
                 results[model_id_res] = task_result # Store the dictionary result
            else:
                 # Unexpected result type from gather
                 logger.error(f"Unexpected result format '{type(task_result)}' from inference task for model '{current_model_id}'")
                 fail_count += 1
                 results[current_model_id] = {
                    "error": f"Unexpected task result format: {type(task_result)}",
                    "model_id": current_model_id,
                    "role": self.models.get(current_model_id, None).get_role() if current_model_id in self.models else None
                }

        total_exec_time = time.time() - start_time
        logger.info(f"Finished run_all_models in {total_exec_time:.2f}s. Successes: {success_count}, Failures: {fail_count}.")

        return results

    # --- Getter and Helper Methods ---

    def get_model(self, model_id: str) -> GGUFModel:
        """Retrieve a specific managed model instance by ID."""
        if model_id not in self.models:
             # Distinguish between not initialized and model simply not found/loaded
             if not self.initialized and not self.models:
                  raise ModelError("ModelManager not initialized")
             raise ModelError(f"Model not found or not loaded: {model_id}")
        return self.models[model_id]

    def get_models_by_role(self, role: str) -> List[GGUFModel]:
        """Get a list of loaded models matching a specific role."""
        if not self.initialized: return [] # Or raise error? Returning empty list is safer.
        if not role or not isinstance(role, str):
             logger.warning(f"Invalid role requested: {role}")
             return []
        # Filter models currently in the manager
        matching_models = [
             m for m in self.models.values()
             if m.is_loaded() and m.get_role() is not None and m.get_role().lower() == role.lower()
        ]
        logger.debug(f"Found {len(matching_models)} loaded models with role '{role}'.")
        return matching_models

    def get_model_ids(self) -> Set[str]:
        """Get the set of model IDs currently managed (loaded or not)."""
        return set(self.models.keys())

    def get_loaded_model_ids(self) -> Set[str]:
        """Get the set of model IDs that are currently successfully loaded."""
        return {mid for mid, m in self.models.items() if m.is_loaded()}

    def get_roles(self) -> Set[str]:
        """Get the set of unique roles assigned to the managed models."""
        if not self.initialized and not self.models: return set()
        # Use walrus operator (Python 3.8+) for conciseness
        roles = {role for m in self.models.values() if (role := m.get_role())}
        return roles

    def get_random_model(self, loaded_only: bool = True) -> Optional[GGUFModel]:
        """Get a random model instance, optionally restricted to loaded models."""
        if not self.models: return None # No models managed

        eligible_model_ids = []
        if loaded_only:
            eligible_model_ids = [mid for mid, m in self.models.items() if m.is_loaded()]
            if not eligible_model_ids:
                 logger.warning("No models are currently loaded to select a random one from.")
                 return None
        else:
            eligible_model_ids = list(self.models.keys())
            if not eligible_model_ids: return None # Should not happen if self.models is not empty

        try:
             random_model_id = random.choice(eligible_model_ids)
             return self.models[random_model_id]
        except IndexError: # Should not happen if eligible_model_ids is checked
             logger.error("Error selecting random model (IndexError despite checks).")
             return None
        except KeyError: # If ID somehow not in self.models after selection (concurrency?)
             logger.error(f"Error retrieving random model '{random_model_id}' after selection (KeyError).")
             return None

    def get_model_count(self, loaded_only: bool = False) -> int:
        """Get the count of managed models, optionally only loaded ones."""
        if loaded_only:
             # Count loaded models even if manager isn't fully "initialized" yet
             return sum(1 for m in self.models.values() if m.is_loaded())
        else:
             # Return count of all model wrappers held (reflects config)
             return len(self.models)

    def is_model_available(self, model_id: str) -> bool:
         """Checks if a model ID is managed AND currently loaded."""
         return model_id in self.models and self.models[model_id].is_loaded()
