#!/usr/bin/env python3
"""
Model connectivity test script
This script tests the connectivity of all specified models to ensure they are accessible
before running the main inference tasks.
"""

import sys
import os
from pathlib import Path

# Add the script directory to the path so we can import model_config
sys.path.append(str(Path(__file__).parent))

from model_config import get_client, format_prompt, get_model_config

def test_model_connectivity(model_name: str) -> bool:
    """
    Test connectivity to a single model
    
    Args:
        model_name: Name of the model to test
    
    Returns:
        bool: True if connectivity is successful, False otherwise
    """
    try:
        print(f"Testing connectivity for model: {model_name}...")
        
        # Get model configuration
        model_config = get_model_config(model_name)
        model_id = model_config["model_id"]
        temperature = model_config["temperature"]
        max_tokens = model_config["max_tokens"]
        # top_p = model_config.get("top_p", 1)
        
        # Get the client for the model
        client = get_client(model_name)
        
        # Create a simple test prompt
        prompt = format_prompt(
            model_name=model_name,
            user_input="Hello, this is a connectivity test",
            system_prompt="You are a helpful assistant"
        )
        
        # Make a minimal API call to test connectivity
        response = client.chat.completions.create(
            model=model_id,
            messages=prompt["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=top_p
        )
        
        # Check if we got a valid response
        if response and hasattr(response, 'choices') and response.choices:
            print(f"✅ Model {model_name} connected successfully")
            return True
        else:
            print(f"❌ Model {model_name} returned an invalid response")
            return False
            
    except Exception as e:
        print(f"❌ Model {model_name} connectivity failed: {str(e)}")
        return False

def main():
    """
    Main function to test connectivity for all models specified in command line arguments
    """
    if len(sys.argv) < 2:
        print("Usage: python test_model_connectivity.py <model1> <model2> ... <modelN>")
        sys.exit(1)
    
    models_to_test = sys.argv[1:]
    print(f"Testing connectivity for {len(models_to_test)} models...")
    print("=" * 50)
    
    failed_models = []
    for model in models_to_test:
        if not test_model_connectivity(model):
            failed_models.append(model)
        print()  # Empty line between tests
    
    print("=" * 50)
    if not failed_models:
        print("✅ All models connected successfully!")
        sys.exit(0)
    else:
        print(f"❌ Connectivity test failed for {len(failed_models)} out of {len(models_to_test)} models:")
        for model in failed_models:
            print(f"   - {model}")
        print("Please check the model configurations and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()