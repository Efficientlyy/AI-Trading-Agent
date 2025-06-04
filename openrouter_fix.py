#!/usr/bin/env python
"""
OpenRouter Integration Fix for Trading-Agent System

This module provides a compatibility layer for OpenRouter integration,
ensuring proper API access and error handling for the LLM Overseer component.
"""

import os
import sys
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("openrouter_fix.log")
    ]
)

logger = logging.getLogger("openrouter_fix")

# Import the fixed OpenRouter client
try:
    from fixed_openrouter_client import OpenRouterClient
    logger.info("Successfully imported OpenRouterClient from fixed_openrouter_client")
except ImportError as e:
    logger.error(f"Failed to import OpenRouterClient: {str(e)}")
    raise

# Create a module-level client for easy access
client = OpenRouterClient()

# Create a compatibility module for openrouter
class OpenRouter:
    """Compatibility class for OpenRouter integration"""
    
    def __init__(self, api_key=None):
        """Initialize OpenRouter
        
        Args:
            api_key: OpenRouter API key (optional)
        """
        self.client = OpenRouterClient(api_key=api_key)
        logger.info("OpenRouter compatibility class initialized")
    
    def chat_completion(self, messages, model="default", temperature=0.7, max_tokens=1000):
        """Get chat completion from OpenRouter
        
        Args:
            messages: List of message dictionaries
            model: Model name or key
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            dict: Response dictionary
        """
        return self.client.chat_completion(messages, model, temperature, max_tokens)

# Create a compatibility module
class Module:
    """Module class for openrouter package compatibility"""
    
    def __init__(self):
        """Initialize module"""
        self.OpenRouter = OpenRouter
        logger.info("OpenRouter module initialized")

# Create the openrouter module if it doesn't exist
if "openrouter" not in sys.modules:
    sys.modules["openrouter"] = Module()
    logger.info("Created openrouter module")
else:
    # Add OpenRouter to existing module
    sys.modules["openrouter"].OpenRouter = OpenRouter
    logger.info("Added OpenRouter to existing openrouter module")

# Set OpenRouter API key in environment if not already set
if "OPENROUTER_API_KEY" not in os.environ:
    # Check if API key is in .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    if key:
                        os.environ["OPENROUTER_API_KEY"] = key
                        logger.info("Set OPENROUTER_API_KEY from .env file")
    
    # Check if API key is in .env-secure/.env file
    secure_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env-secure", ".env")
    if os.path.exists(secure_env_path):
        with open(secure_env_path, "r") as f:
            for line in f:
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    if key:
                        os.environ["OPENROUTER_API_KEY"] = key
                        logger.info("Set OPENROUTER_API_KEY from .env-secure/.env file")

# Log current state
if "OPENROUTER_API_KEY" in os.environ:
    logger.info("OPENROUTER_API_KEY is set in environment")
else:
    logger.warning("OPENROUTER_API_KEY is not set in environment")

if __name__ == "__main__":
    # Test the fix
    try:
        import openrouter
        
        print("OpenRouter import successful")
        
        # Create instance
        client = openrouter.OpenRouter()
        
        print("OpenRouter instance created")
        
        # Test chat completion
        print("Testing chat completion...")
        messages = [
            {"role": "system", "content": "You are a trading assistant that analyzes market data and provides insights."},
            {"role": "user", "content": "Please provide a market analysis for BTC/USDC based on the following data: price=65000, 24h_change=+2.3%, volume=1.5B, bid-ask spread=0.05%"}
        ]
        
        response = client.chat_completion(messages)
        
        print("Chat completion response received:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Error testing OpenRouter: {str(e)}")
