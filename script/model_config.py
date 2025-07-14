"""
Model configuration file
Contains configuration information for all supported models and client initialization logic
"""

import os
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI

# Default API configuration
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL")

LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")

# Model configuration dictionary
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o-2024-11-20": {
        "model_id": "gpt-4o-2024-11-20",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "gpt-4o-mini-2024-07-18": {
        "model_id": "gpt-4o-mini-2024-07-18",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "o4-mini-2025-04-16": {
        "model_id": "o4-mini-2025-04-16",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "claude-3-5-sonnet-20240620": {
        "model_id": "claude-3-5-sonnet-20240620",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "gemini-2.0-flash": {
        "model_id": "gemini-2.0-flash",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "gemini-2.5-flash": {
        "model_id": "gemini-2.5-flash",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "gemini-2.5-pro-preview-05-06": {
        "model_id": "gemini-2.5-pro-preview-05-06",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "deepseek-chat": {
        "model_id": "deepseek-chat",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "deepseek-reasoner": {
        "model_id": "deepseek-reasoner",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    },
    "Llama-3.3-70B-Instruct": {
        "model_id": "Llama-3.3-70B-Instruct",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": LOCAL_API_KEY,
        "base_url": LOCAL_BASE_URL,
    },
    "Llama-3.1-70B-Instruct": {
        "model_id": "Llama-3.1-70B-Instruct",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": LOCAL_API_KEY,
        "base_url": LOCAL_BASE_URL,
    },
    "Qwen2.5-7B-Instruct": {
        "model_id": "qwen2.5-7b-instruct",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": LOCAL_API_KEY,
        "base_url": LOCAL_BASE_URL,
    },
    "Qwen2.5-32B-Instruct": {
        "model_id": "Qwen2.5-32B-Instruct",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": LOCAL_API_KEY,
        "base_url": LOCAL_BASE_URL,
    },
    "Qwen2.5-72B-Instruct": {
        "model_id": "qwen2.5-72b-instruct",
        "temperature": 0,
        "max_tokens": 2048,
        "supports_system_prompt": True,
        "api_key": LOCAL_API_KEY,
        "base_url": LOCAL_BASE_URL,
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for the specified model

    Args:
        model_name: Model name

    Returns:
        Model configuration dictionary

    Raises:
        ValueError: If the model does not exist
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Supported models: {', '.join(MODEL_CONFIGS.keys())}"
        )

    return MODEL_CONFIGS[model_name]


def get_client(model_name: str) -> OpenAI:
    """
    Get OpenAI client instance for the specified model

    Args:
        model_name: Model name for retrieving the corresponding API configuration

    Returns:
        OpenAI client instance

    Raises:
        EnvironmentError: If necessary API configuration is missing
    """
    # Get model configuration
    model_config = get_model_config(model_name)
    api_key = model_config.get("api_key")
    base_url = model_config.get("base_url")

    client_args = {}
    client_args["base_url"] = base_url.strip()
    print(f"Model {model_name} using API base URL: {base_url}")

    # Create client
    return OpenAI(api_key=api_key, **client_args)


def format_prompt(
    model_name: str,
    user_input: Union[str, List[Dict[str, Any]]],
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format prompt according to model

    Args:
        model_name: Model name
        user_input: User input content, can be a string or pre-formatted message list (for multimodal input)
        system_prompt: System prompt, some models may not support this

    Returns:
        Formatted prompt dictionary, can be used directly for API calls
    """
    model_config = get_model_config(model_name)
    supports_system_prompt = model_config.get("supports_system_prompt", True)

    # If user input is already a formatted message list (multimodal input), use it directly
    if isinstance(user_input, list):
        # In this case, the input is already a processed message list
        # system_prompt should have been handled in make_prompt and query_model
        return {"messages": user_input}

    # Process normal text input
    if supports_system_prompt and system_prompt:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        }
    else:
        # For models that don't support system prompts, combine system prompt and user input
        if system_prompt:
            combined_input = f"{system_prompt}\n\n{user_input}"
        else:
            combined_input = user_input

        return {"messages": [{"role": "user", "content": combined_input}]}


def get_available_models() -> List[str]:
    """
    Get list of all available models

    Returns:
        List of model names
    """
    return list(MODEL_CONFIGS.keys())
