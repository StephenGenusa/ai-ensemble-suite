# ai_ensemble_suite/utils/prompt_utils.py

"""Utilities for prompt formatting and handling."""

from typing import Dict, Any, Optional, List, Union, Callable
import string
import re
from ai_ensemble_suite.exceptions import ValidationError


def format_prompt(
    template: str, strict: bool = False, **kwargs: Any
) -> str:
    """Format a prompt template with the provided values.

    Supports both standard Python string formatting and custom markers.

    Args:
        template: The template string to format.
        strict: If True, raises an error for missing values.
        **kwargs: Values to format the template with.

    Returns:
        The formatted prompt.

    Raises:
        ValidationError: If strict=True and a required value is missing.
    """
    if not kwargs and not strict:
        return template

    # First, identify all required keys in the template
    required_keys = set()
    for match in re.finditer(r"\{([a-zA-Z0-9_]+)\}", template):
        required_keys.add(match.group(1))

    # Check if all required keys are provided when strict is True
    if strict:
        missing_keys = required_keys - set(kwargs.keys())
        if missing_keys:
            missing_keys_str = ", ".join(missing_keys)
            raise ValidationError(f"Missing required values: {missing_keys_str}")

    # Format using Python's string formatting
    try:
        formatted_prompt = template.format(**kwargs)
        return formatted_prompt
    except KeyError as e:
        if strict:
            raise ValidationError(f"Missing required value: {str(e)}")
        # Fall back to partial formatting when not strict
        return template


def truncate_text(text: str, max_length: int, truncation_marker: str = "...") -> str:
    """Truncate text to a maximum length.

    Args:
        text: The text to truncate.
        max_length: Maximum length of the text.
        truncation_marker: Marker to append to truncated text.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    
    # Account for truncation marker length
    available_length = max_length - len(truncation_marker)
    if available_length <= 0:
        return truncation_marker
    
    return text[:available_length] + truncation_marker


def create_system_message(system_prompt: str) -> Dict[str, str]:
    """Create a system message for model prompting.

    Args:
        system_prompt: The system prompt content.

    Returns:
        Dictionary containing the system message.
    """
    return {"role": "system", "content": system_prompt}


def create_user_message(user_prompt: str) -> Dict[str, str]:
    """Create a user message for model prompting.

    Args:
        user_prompt: The user prompt content.

    Returns:
        Dictionary containing the user message.
    """
    return {"role": "user", "content": user_prompt}


def create_assistant_message(assistant_response: str) -> Dict[str, str]:
    """Create an assistant message for model prompting.

    Args:
        assistant_response: The assistant response content.

    Returns:
        Dictionary containing the assistant message.
    """
    return {"role": "assistant", "content": assistant_response}


def create_chat_prompt(
    system_prompt: Optional[str],
    messages: List[Dict[str, str]],
) -> str:
    """Create a chat prompt from system prompt and messages.

    Args:
        system_prompt: Optional system prompt.
        messages: List of message dictionaries.

    Returns:
        Formatted chat prompt as string.
    """
    formatted_prompt = ""
    
    if system_prompt:
        formatted_prompt += f"<|system|>\n{system_prompt}\n\n"
    
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        
        if role == "system":
            formatted_prompt += f"<|system|>\n{content}\n\n"
        elif role == "user":
            formatted_prompt += f"<|user|>\n{content}\n\n"
        elif role == "assistant":
            formatted_prompt += f"<|assistant|>\n{content}\n\n"
    
    # Add a final assistant marker to indicate where the model should generate
    formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt
