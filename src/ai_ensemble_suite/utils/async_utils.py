# src/ai_ensemble_suite/utils/async_utils.py

import asyncio
from typing import Callable, Any, Optional, Coroutine, TypeVar, Union, List
import concurrent.futures
import functools
from ai_ensemble_suite.utils.logging import logger

# Define a generic type variable for gather_with_concurrency
T = TypeVar('T')


async def run_in_threadpool(
    func: Callable[..., Any],
    *args: Any,
     # Use keyword-only argument for the executor to avoid name clashes
    _executor: Optional[concurrent.futures.Executor] = None,
    **kwargs: Any
) -> Any:
    """Run a potentially blocking function in a separate thread using an executor.

    Args:
        func: The synchronous function or callable object to run.
        *args: Positional arguments for func.
        _executor: The concurrent.futures.Executor to use. If None, uses the
                  default executor of the current event loop.
        **kwargs: Keyword arguments for func.

    Returns:
        The result of the function call.

    Raises:
        Any exception raised by func.
    """
    loop = asyncio.get_running_loop()
    func_to_run = functools.partial(func, *args, **kwargs)

    try:
        result = await loop.run_in_executor(
            _executor,
            func_to_run
        )
        return result
    except Exception as e:
        # --- MODIFIED Error Logging ---
        # Get a descriptive name for the callable, falling back gracefully
        func_name = getattr(func, '__name__', None) # Try standard __name__
        if func_name is None:
            # If no __name__, try getting the type name
            func_name = type(func).__name__
        # --- END MODIFICATION ---

        logger.error(f"Error in threadpool execution of '{func_name}': {e}", exc_info=True)
        raise # Re-raise the exception to be handled by the caller


# --- gather_with_concurrency remains the same ---
async def gather_with_concurrency(
    limit: int,
    *tasks: Coroutine[Any, Any, T],
    return_exceptions: bool = False
) -> List[Union[T, BaseException]]:
    """Run asyncio tasks with a concurrency limit.

    Args:
        limit: The maximum number of tasks to run concurrently.
        *tasks: The coroutine tasks to run.
        return_exceptions: If True, exceptions are returned as results
                           instead of raising them. Defaults to False.

    Returns:
        A list containing the results or exceptions from the tasks, preserving
        the original order.
    """
    if limit <= 0:
        raise ValueError("Concurrency limit must be greater than 0")
    if not tasks:
        return []

    semaphore = asyncio.Semaphore(limit)
    results: List[Optional[Union[T, BaseException]]] = [None] * len(tasks) # Preallocate results list

    async def sem_task(task_index: int, task: Coroutine[Any, Any, T]) -> None:
        """Wrapper to run a task under semaphore control."""
        nonlocal results
        async with semaphore:
            try:
                result = await task
                results[task_index] = result # Store result at correct index
            except Exception as e:
                if return_exceptions:
                    logger.debug(f"Task {task_index} failed with exception (captured): {e}")
                    results[task_index] = e # Store exception at correct index
                else:
                    logger.error(f"Task {task_index} failed with exception (raising): {e}", exc_info=True)
                    raise # Re-raise if return_exceptions is False

    # Create wrapper tasks
    wrapper_tasks = [
        sem_task(i, task) for i, task in enumerate(tasks)
    ]

    # Run all wrapper tasks
    task_run_results = await asyncio.gather(*wrapper_tasks, return_exceptions=True)

    # Process potential errors from asyncio.gather itself or re-raised exceptions
    final_ordered_results: List[Union[T, BaseException]] = []
    for i, run_result in enumerate(task_run_results):
        original_result = results[i]
        if isinstance(run_result, Exception):
             if return_exceptions:
                  if original_result is None:
                      logger.warning(f"gather_with_concurrency: Gather reported exception for task {i}, but no result stored: {run_result}")
                      final_ordered_results.append(run_result)
                  elif isinstance(original_result, BaseException):
                      final_ordered_results.append(original_result)
                  else:
                      logger.error(f"gather_with_concurrency: Mismatch state for task {i}. Gather reported exc {run_result}, result was {original_result}")
                      final_ordered_results.append(run_result)
             else:
                 logger.error(f"gather_with_concurrency: Task {i} raised an exception (return_exceptions=False): {run_result}")
                 raise run_result
        elif original_result is not None:
              final_ordered_results.append(original_result)
        else:
             logger.error(f"gather_with_concurrency: Inconsistent state for task {i}. run_result={run_result}, original_result=None. Using default.")
             final_ordered_results.append(None)

    if len(final_ordered_results) != len(tasks):
        logger.error(f"gather_with_concurrency: Result list length mismatch ({len(final_ordered_results)} vs {len(tasks)}). Padding with Nones.")
        final_ordered_results.extend([None] * (len(tasks) - len(final_ordered_results)))

    return final_ordered_results
