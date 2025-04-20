# src/ai_ensemble_suite/utils/tracing.py

"""Tracing utilities for collecting and organizing execution information."""

from typing import Dict, Any, Optional, List, Callable, Union
import time
from dataclasses import dataclass, field
import json
import copy
from pathlib import Path
import numpy as np
import logging # Import standard logging
import os

# --- Modified JSON Encoder with Debug Logging ---
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types with debug logging """
    def default(self, obj):
        try:
            # --- Existing checks for NumPy types ---
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                # Handle NaN/Inf which are invalid in standard JSON
                if np.isnan(obj) or np.isinf(obj):
                    return str(obj)
                return float(obj)
            elif isinstance(obj, np.complex_):
                 return {'real': float(obj.real), 'imag': float(obj.imag)}
            elif isinstance(obj, np.ndarray):
                # Convert array to list; elements will be processed recursively by JSONEncoder
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.void):
                 return None
        # --- Attempt default serialization, log on failure ---
            # Let the base class default method raise the TypeError for unsupported types
            return super(NumpyEncoder, self).default(obj)
        except TypeError:
            # --- DEBUGGING: Log the problematic object ---
            # Use configured logger if available, otherwise print (as last resort)
            try:
                # Use a logger instance if available (e.g., passed or global)
                # This assumes 'logger' is accessible or use logging.getLogger
                logging.getLogger(__name__).debug(
                    f"NumpyEncoder failed on object: {repr(obj)} of type: {type(obj)}"
                )
            except Exception:
                # Fallback to print if logging fails for some reason
                print(f"DEBUG: NumpyEncoder failed on object: {repr(obj)} of type: {type(obj)}")
            # --- End Debugging ---
            # Re-raise the original TypeError so the calling code handles the failure
            raise
# --- End Modified JSON Encoder ---


# --- Dataclasses (ModelTrace, PhaseTrace, AggregationTrace, SessionTrace) remain unchanged ---
@dataclass
class ModelTrace:
    model_id: str
    input_prompt: str
    output: Dict[str, Any]
    execution_time: float
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    status: str = "success" # Added status
    error_message: Optional[str] = None # Added error message

    def to_dict(self) -> Dict[str, Any]:
        # Create a safe copy of the output, potentially truncating large fields
        safe_output = copy.deepcopy(self.output)
        # Example: Truncate token list if very long
        if "tokens" in safe_output and isinstance(safe_output["tokens"], list) and len(safe_output["tokens"]) > 20:
            token_count = len(safe_output["tokens"])
            safe_output["tokens"] = safe_output["tokens"][:10] + ["..."] + safe_output["tokens"][-10:]
            safe_output["tokens_truncated_info"] = f"Truncated, showing first/last 10 of {token_count} tokens"
        # Example: Truncate very long text fields
        if "text" in safe_output and isinstance(safe_output["text"], str) and len(safe_output["text"]) > 2000:
             safe_output["text"] = safe_output["text"][:1000] + "..." + safe_output["text"][-1000:]
             safe_output["text_truncated"] = True
        # Truncate long input prompt
        input_prompt_truncated = self.input_prompt[:1000] + "..." + self.input_prompt[-1000:] if len(self.input_prompt) > 2000 else self.input_prompt

        return {
            "model_id": self.model_id,
            "input_prompt": input_prompt_truncated,
            "input_prompt_length": len(self.input_prompt),
            "output": safe_output,
            "execution_time_ms": round(self.execution_time * 1000) if self.execution_time is not None else None,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "status": self.status,
            "error_message": self.error_message
        }


@dataclass
class PhaseTrace:
    phase_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    phase_parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    phase_type: Optional[str] = None # Added phase type
    status: str = "success" # Added status
    error_message: Optional[str] = None # Added error message

    def to_dict(self) -> Dict[str, Any]:
        # Deep copy inputs/outputs but potentially sanitize/truncate large values
        safe_input = copy.deepcopy(self.input_data)
        safe_output = copy.deepcopy(self.output_data)
        # Example Sanitization: Remove potentially sensitive info or large embedding lists
        if "embeddings" in safe_output: safe_output["embeddings"] = "[Truncated Embeddings]"
        if "raw_model_results" in safe_output: safe_output["raw_model_results"] = "[Truncated Raw Results]"

        return {
            "phase_name": self.phase_name,
            "phase_type": self.phase_type,
            "input_data": safe_input,
            "output_data": safe_output,
            "execution_time_ms": round(self.execution_time * 1000) if self.execution_time is not None else None,
            "phase_parameters": self.phase_parameters,
            "timestamp": self.timestamp,
            "status": self.status,
            "error_message": self.error_message
        }

@dataclass
class AggregationTrace:
    strategy_name: str
    inputs: Dict[str, Any]
    output: Dict[str, Any]
    execution_time: float
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    status: str = "success" # Added status
    error_message: Optional[str] = None # Added error message

    def to_dict(self) -> Dict[str, Any]:
        # Deep copy inputs/outputs but potentially sanitize/truncate
        safe_input = copy.deepcopy(self.inputs)
        safe_output = copy.deepcopy(self.output)
        # Example: Truncate long raw votes or weights lists
        if "raw_votes" in safe_input and len(str(safe_input["raw_votes"])) > 1000: safe_input["raw_votes"] = "[Truncated Raw Votes]"
        if "weights" in safe_output and len(str(safe_output["weights"])) > 1000: safe_output["weights"] = "[Truncated Weights]"

        return {
            "strategy_name": self.strategy_name,
            "inputs": safe_input,
            "output": safe_output,
            "execution_time_ms": round(self.execution_time * 1000) if self.execution_time is not None else None,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "status": self.status,
            "error_message": self.error_message
        }


@dataclass
class SessionTrace:
    query: str
    final_response: str
    total_execution_time: float
    configuration: Dict[str, Any] # Snapshot of relevant config
    timestamp: float = field(default_factory=time.time)
    status: str = "success" # Added status
    final_error: Optional[str] = None # Final error message if session failed

    def to_dict(self) -> Dict[str, Any]:
        # Deep copy config to avoid modifications
        safe_config = copy.deepcopy(self.configuration)
        # Ensure secrets are not in the trace config snapshot (should be done before passing to SessionTrace)

        return {
            "query": self.query,
            "final_response": self.final_response,
            "total_execution_time_ms": round(self.total_execution_time * 1000) if self.total_execution_time is not None else None,
            "configuration_snapshot": safe_config,
            "timestamp": self.timestamp,
            "status": self.status,
            "final_error": self.final_error
        }


# --- TraceCollector uses logger_instance passed in __init__ ---
class TraceCollector:
    """Collects and organizes trace information during execution."""

    def __init__(self, enabled: bool = True, logger_instance=None) -> None: # Add logger_instance parameter
        """Initialize the TraceCollector."""
        self.enabled = enabled
        if logger_instance:
            self.logger = logger_instance
        else:
            # Fallback to standard logger if none provided
            self.logger = logging.getLogger(__name__)
            # Optionally add a handler if no root config exists
            if not logging.getLogger().hasHandlers():
                 handler = logging.StreamHandler()
                 formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                 handler.setFormatter(formatter)
                 self.logger.addHandler(handler)
                 self.logger.setLevel(logging.DEBUG) # Set a default level
                 self.logger.warning(
                    "TraceCollector: No logger instance provided and root logger had no handlers. "
                    "Added default StreamHandler for '%s'.", __name__
                 )
            else:
                 self.logger.warning(
                    "TraceCollector initialized without a specific logger instance. "
                    "Using standard logger '%s'. Ensure it's configured correctly.", __name__
                 )


        self.model_traces: Dict[str, List[ModelTrace]] = {}
        self.phase_traces: Dict[str, PhaseTrace] = {}
        self.aggregation_trace: Optional[AggregationTrace] = None
        self.session_trace: Optional[SessionTrace] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.session_id: str = f"session_{int(time.time())}_{os.getpid()}" # Unique session ID

    def start_session(self) -> None:
        """Start a new tracing session, clearing previous data."""
        if not self.enabled: return
        self.clear() # Clear previous state
        self.session_id = f"session_{int(time.time())}_{os.getpid()}" # Generate new ID
        self.start_time = time.time()
        self.logger.debug(f"Trace session started. ID: {self.session_id}")

    def end_session(self, status: str = "success", final_error: Optional[str] = None) -> None:
        """End the current tracing session."""
        if not self.enabled or self.start_time is None: return
        if self.end_time is None: # Ensure end_time is only set once
             self.end_time = time.time()
             duration = self.end_time - self.start_time
             # Update session trace status if it exists
             if self.session_trace:
                  self.session_trace.status = status
                  self.session_trace.final_error = final_error
             self.logger.debug(f"Trace session ended. ID: {self.session_id}. Duration: {duration:.3f}s. Status: {status}.")

    # --- Modified add_* methods to include status/error ---
    def add_model_trace(self, model_id: str, input_prompt: str, output: Dict[str, Any], execution_time: float, parameters: Dict[str, Any], status: str = "success", error_message: Optional[str] = None) -> None:
        """Add a trace record for a model inference call."""
        if not self.enabled: return
        trace = ModelTrace(model_id=model_id, input_prompt=input_prompt, output=output, execution_time=execution_time, parameters=parameters, status=status, error_message=error_message)
        if model_id not in self.model_traces: self.model_traces[model_id] = []
        self.model_traces[model_id].append(trace);
        log_message = f"Trace added: Model '{model_id}' (Status: {status})"
        if error_message: log_message += f" Error: {error_message}"
        if status == "failure":
            self.logger.error(log_message)
        else:
            self.logger.debug(log_message)

    def add_phase_trace(self, phase_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any], execution_time: float, phase_parameters: Dict[str, Any], phase_type: Optional[str] = None, status: str = "success", error_message: Optional[str] = None) -> None:
        """Add a trace record for a collaboration phase execution."""
        if not self.enabled: return
        trace = PhaseTrace(phase_name=phase_name, phase_type=phase_type, input_data=input_data, output_data=output_data, execution_time=execution_time, phase_parameters=phase_parameters, status=status, error_message=error_message)
        self.phase_traces[phase_name] = trace;
        log_message = f"Trace added: Phase '{phase_name}' (Status: {status})"
        if error_message: log_message += f" Error: {error_message}"
        if status == "failure":
            self.logger.error(log_message)
        else:
            self.logger.debug(log_message)

    def add_aggregation_trace(self, strategy_name: str, inputs: Dict[str, Any], output: Dict[str, Any], execution_time: float, parameters: Dict[str, Any], status: str = "success", error_message: Optional[str] = None) -> None:
        """Add a trace record for the aggregation step."""
        if not self.enabled: return
        trace = AggregationTrace(strategy_name=strategy_name, inputs=inputs, output=output, execution_time=execution_time, parameters=parameters, status=status, error_message=error_message)
        self.aggregation_trace = trace;
        log_message = f"Trace added: Aggregation '{strategy_name}' (Status: {status})"
        if error_message: log_message += f" Error: {error_message}"
        if status == "failure":
            self.logger.error(log_message)
        else:
            self.logger.debug(log_message)

    def add_session_trace(self, query: str, final_response: str, total_execution_time: float, configuration: Dict[str, Any]) -> None:
        """Add the overall session summary trace record. Overwrites previous if called again."""
        if not self.enabled: return
        # Status/error updated via end_session
        trace = SessionTrace(query=query, final_response=final_response, total_execution_time=total_execution_time, configuration=configuration)
        self.session_trace = trace;
        self.logger.debug("Session trace data updated.")

    def get_trace_data(self) -> Dict[str, Any]:
        """Retrieve the collected trace data as a dictionary."""
        if not self.enabled: return {"error": "Tracing disabled"}
        # Ensure session ends if not explicitly called, using current status
        if self.start_time is not None and self.end_time is None:
            self.logger.warning("get_trace_data called before end_session. Ending session implicitly.")
            self.end_session() # End with default status

        result: Dict[str, Any] = {
            "session_id": self.session_id,
            "session": self.session_trace.to_dict() if self.session_trace else None,
            "phases": {name: trace.to_dict() for name, trace in self.phase_traces.items()},
            "aggregation": self.aggregation_trace.to_dict() if self.aggregation_trace else None,
            "models": {model_id: [trace.to_dict() for trace in traces] for model_id, traces in self.model_traces.items()},
            "statistics": self.calculate_statistics(),
            "trace_metadata": {
                "start_time_unix": self.start_time,
                "end_time_unix": self.end_time,
                "collection_duration_ms": round((self.end_time - self.start_time) * 1000) if self.start_time and self.end_time else None,
            }
        }
        return result

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics from the collected trace data."""
        stats: Dict[str, Any] = {
            "model_count_configured": 0, # Need config access for this realistically
            "phase_count_executed": len(self.phase_traces),
            "total_model_calls": sum(len(traces) for traces in self.model_traces.values()),
            "successful_model_calls": sum(1 for traces in self.model_traces.values() for trace in traces if trace.status == "success"),
            "failed_model_calls": sum(1 for traces in self.model_traces.values() for trace in traces if trace.status != "success"),
            "total_model_execution_time_ms": 0,
            "average_model_execution_time_ms": 0,
            "total_phase_execution_time_ms": 0,
            "average_phase_execution_time_ms": 0,
            "aggregation_execution_time_ms": 0,
            "model_execution_details": {},
            "phase_execution_times_ms": {}
        }

        all_model_times_ms = []
        for model_id, traces in self.model_traces.items():
            times_ms = [trace.execution_time * 1000 for trace in traces if trace.execution_time is not None and trace.status == "success"]
            total_ms = sum(times_ms) if times_ms else 0
            if times_ms:
                stats["model_execution_details"][model_id] = {
                    "total_ms": round(total_ms),
                    "average_ms": round(total_ms / len(times_ms)),
                    "min_ms": round(min(times_ms)),
                    "max_ms": round(max(times_ms)),
                    "success_calls": len(times_ms),
                    "total_calls": len(traces) # Include failures in total call count for the model
                }
                all_model_times_ms.extend(times_ms)
            elif traces: # If traces exist but none were successful/timed
                 stats["model_execution_details"][model_id] = {"total_ms": 0, "average_ms": 0, "success_calls": 0, "total_calls": len(traces)}

        if all_model_times_ms:
            stats["total_model_execution_time_ms"] = round(sum(all_model_times_ms))
            stats["average_model_execution_time_ms"] = round(sum(all_model_times_ms) / len(all_model_times_ms))

        all_phase_times_ms = []
        for phase_name, trace in self.phase_traces.items():
            if trace.execution_time is not None and trace.status == "success":
                time_ms = trace.execution_time * 1000
                stats["phase_execution_times_ms"][phase_name] = round(time_ms)
                all_phase_times_ms.append(time_ms)
            elif trace.status != "success": # Record failed/skipped phases as 0 time for stats
                 stats["phase_execution_times_ms"][phase_name] = 0

        if all_phase_times_ms:
            stats["total_phase_execution_time_ms"] = round(sum(all_phase_times_ms))
            stats["average_phase_execution_time_ms"] = round(sum(all_phase_times_ms) / len(all_phase_times_ms))

        if self.aggregation_trace and self.aggregation_trace.execution_time is not None and self.aggregation_trace.status == "success":
            stats["aggregation_execution_time_ms"] = round(self.aggregation_trace.execution_time * 1000)

        # Calculate perceived total time (sum of phase + aggregation times) vs actual session time
        perceived_total_ms = stats["total_phase_execution_time_ms"] + stats["aggregation_execution_time_ms"]
        stats["_perceived_total_execution_time_ms"] = perceived_total_ms

        # Add overall session time from session trace if available
        if self.session_trace and self.session_trace.total_execution_time is not None:
             stats["session_total_execution_time_ms"] = round(self.session_trace.total_execution_time * 1000)
        elif self.start_time and self.end_time: # Use collector's measured time if session trace missing total
             stats["session_total_execution_time_ms"] = round((self.end_time - self.start_time) * 1000)

        return stats

    def clear(self) -> None:
        """Clear all collected trace data and reset timers."""
        self.model_traces = {}
        self.phase_traces = {}
        self.aggregation_trace = None
        self.session_trace = None
        self.start_time = None
        self.end_time = None
        # Keep session_id or generate new? Let's keep existing if any, start_session generates new one.
        self.logger.debug("Trace data cleared.")

    def save_to_file(self, filepath: Union[str, Path]) -> bool:
        """Save the collected trace data to a JSON file.

        Args:
            filepath: The path to the file where the trace data should be saved.

        Returns:
            True if saving was successful, False otherwise.
        """
        if not self.enabled:
            self.logger.info("Tracing is disabled, cannot save trace.")
            return False

        filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            trace_data_dict = self.get_trace_data()

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trace_data_dict, f, indent=2, cls=NumpyEncoder) # Use the encoder

            self.logger.info(f"Trace data saved successfully to {filepath}")
            return True
        except TypeError as e: # This will be caught after the debug log in the encoder
            self.logger.error(f"Failed to serialize trace data to JSON for file '{filepath}': {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"Error saving trace data to file '{filepath}': {str(e)}", exc_info=True)
            return False

    def to_json(self) -> str:
        """Convert the collected trace data to a JSON string.

        Returns:
            A JSON string representation of the trace data, or a JSON error object.
        """
        if not self.enabled: return json.dumps({"error": "Tracing disabled"})
        trace_data_dict = self.get_trace_data()
        try:
            return json.dumps(trace_data_dict, indent=2, cls=NumpyEncoder) # Use the encoder
        except TypeError as e: # This will be caught after the debug log in the encoder
            self.logger.error(f"Error converting trace data to JSON string: {str(e)}")
            return json.dumps({"error": f"Failed to serialize trace data: {str(e)}"}, indent=2)
        except Exception as e:
            self.logger.error(f"Unexpected error converting trace data to JSON string: {str(e)}", exc_info=True)
            return json.dumps({"error": f"Unexpected error serializing trace data: {str(e)}"}, indent=2)
