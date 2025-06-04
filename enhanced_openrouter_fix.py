#!/usr/bin/env python
"""
Enhanced OpenRouter Integration Fix for Trading-Agent System

This module provides a comprehensive compatibility layer for OpenRouter integration,
ensuring proper API access, error handling, and argument compatibility for the LLM Overseer component.
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
        logging.FileHandler("enhanced_openrouter_fix.log")
    ]
)

logger = logging.getLogger("enhanced_openrouter_fix")

# Import the fixed OpenRouter client
try:
    from fixed_openrouter_client import OpenRouterClient
    logger.info("Successfully imported OpenRouterClient from fixed_openrouter_client")
except ImportError as e:
    logger.error(f"Failed to import OpenRouterClient: {str(e)}")
    raise

# Create a module-level client for easy access
client = OpenRouterClient()

# Create a compatibility class for OpenRouter
class OpenRouter:
    """Enhanced compatibility class for OpenRouter integration with flexible argument handling"""
    
    def __init__(self, api_key=None, http_client=None, **kwargs):
        """Initialize OpenRouter with flexible argument handling
        
        Args:
            api_key: OpenRouter API key (optional)
            http_client: HTTP client (ignored, for compatibility)
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        # Log all received arguments for debugging
        logger.info(f"OpenRouter.__init__ called with arguments: api_key={api_key is not None}, http_client={http_client is not None}, kwargs={list(kwargs.keys())}")
        
        # Initialize client with API key
        self.client = OpenRouterClient(api_key=api_key)
        
        # Store additional arguments for compatibility
        self.http_client = http_client
        self.kwargs = kwargs
        
        logger.info("Enhanced OpenRouter compatibility class initialized")
    
    def chat_completion(self, messages, model="default", temperature=0.7, max_tokens=1000, **kwargs):
        """Get chat completion from OpenRouter with flexible argument handling
        
        Args:
            messages: List of message dictionaries
            model: Model name or key
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments (ignored, for compatibility)
            
        Returns:
            dict: Response dictionary
        """
        # Log all received arguments for debugging
        logger.info(f"OpenRouter.chat_completion called with arguments: model={model}, temperature={temperature}, max_tokens={max_tokens}, kwargs={list(kwargs.keys())}")
        
        # Forward to client
        return self.client.chat_completion(messages, model, temperature, max_tokens)

# Create a compatibility module
class Module:
    """Enhanced module class for openrouter package compatibility"""
    
    def __init__(self):
        """Initialize module"""
        self.OpenRouter = OpenRouter
        logger.info("Enhanced OpenRouter module initialized")

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

# Create a mock API key if not found
if "OPENROUTER_API_KEY" not in os.environ:
    os.environ["OPENROUTER_API_KEY"] = "mock_api_key_for_testing"
    logger.warning("Created mock OPENROUTER_API_KEY for testing")

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
        
        # Create instance with various arguments
        client = openrouter.OpenRouter(
            api_key="test_key",
            http_client={"mock": True},
            base_url="https://api.example.com",
            timeout=30
        )
        
        print("OpenRouter instance created with various arguments")
        
        # Test chat completion
        print("Testing chat completion...")
        messages = [
            {"role": "system", "content": "You are a trading assistant that analyzes market data and provides insights."},
            {"role": "user", "content": "Please provide a market analysis for BTC/USDC based on the following data: price=65000, 24h_change=+2.3%, volume=1.5B, bid-ask spread=0.05%"}
        ]
        
        response = client.chat_completion(
            messages,
            model="default",
            temperature=0.7,
            max_tokens=1000,
            stream=False,
            stop=None
        )
        
        print("Chat completion response received:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Error testing OpenRouter: {str(e)}")
