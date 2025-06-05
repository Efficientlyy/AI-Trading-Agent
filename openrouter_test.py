#!/usr/bin/env python
"""
OpenRouter LLM Integration Test

This script tests connectivity to the OpenRouter API using the provided credentials.
"""

import os
import sys
import json
import logging
import time
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("openrouter_test.log")
    ]
)
logger = logging.getLogger("openrouter_test")

# Load environment variables
load_dotenv('.env-secure/.env')

# OpenRouter API credentials
API_KEY = os.getenv('OPENROUTER_API_KEY')

# OpenRouter API endpoint
API_URL = "https://openrouter.ai/api/v1"

def test_models_list():
    """Test retrieving the list of available models.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing OpenRouter models list API...")
    
    # Endpoint
    endpoint = "/models"
    url = API_URL + endpoint
    
    # Headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://github.com/Efficientlyy/AI-Trading-Agent",
        "X-Title": "AI Trading Agent"
    }
    
    try:
        # Send request
        response = requests.get(url, headers=headers)
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            logger.info("OpenRouter models list API test successful!")
            logger.info(f"Available models: {len(models)}")
            
            # Log some models
            for model in models[:5]:  # Show first 5 models
                logger.info(f"Model: {model.get('id')} - {model.get('name')}")
            
            return True
        else:
            logger.error(f"OpenRouter models list API test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"OpenRouter models list API test failed with exception: {e}")
        return False

def test_completion():
    """Test sending a completion request.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing OpenRouter completion API...")
    
    # Endpoint
    endpoint = "/chat/completions"
    url = API_URL + endpoint
    
    # Headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Efficientlyy/AI-Trading-Agent",
        "X-Title": "AI Trading Agent"
    }
    
    # Request body
    data = {
        "model": "openai/gpt-3.5-turbo",  # Using a reliable model for testing
        "messages": [
            {
                "role": "system",
                "content": "You are the System Overseer for a cryptocurrency trading bot. You provide concise, accurate information about market conditions and trading performance."
            },
            {
                "role": "user",
                "content": "What's the current status of the trading system?"
            }
        ],
        "max_tokens": 150
    }
    
    try:
        # Send request
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()
        
        # Check response
        if response.status_code == 200:
            response_data = response.json()
            completion = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            model_used = response_data.get("model", "")
            
            logger.info("OpenRouter completion API test successful!")
            logger.info(f"Model used: {model_used}")
            logger.info(f"Response time: {end_time - start_time:.2f} seconds")
            logger.info(f"Completion: {completion}")
            
            return True
        else:
            logger.error(f"OpenRouter completion API test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"OpenRouter completion API test failed with exception: {e}")
        return False

def test_system_overseer_prompt():
    """Test sending a System Overseer specific prompt.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing System Overseer specific prompt...")
    
    # Endpoint
    endpoint = "/chat/completions"
    url = API_URL + endpoint
    
    # Headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Efficientlyy/AI-Trading-Agent",
        "X-Title": "AI Trading Agent"
    }
    
    # Request body
    data = {
        "model": "openai/gpt-4o",  # Using GPT-4o for more advanced capabilities
        "messages": [
            {
                "role": "system",
                "content": """You are the System Overseer for a cryptocurrency trading bot with the following capabilities:
                
1. Monitor trading system performance and market conditions
2. Analyze trading signals and performance metrics
3. Provide insights and recommendations to the user
4. Manage trading parameters and settings
5. Alert the user to important events or anomalies

Your communication style is professional but conversational. You provide concise, accurate information with appropriate context. You are helpful and proactive, but not verbose.

Current trading pairs: BTC/USDC, ETH/USDC, SOL/USDC
Current risk level: moderate
"""
            },
            {
                "role": "user",
                "content": "I'm seeing unusual price movements for Bitcoin. What should I do?"
            }
        ],
        "max_tokens": 300
    }
    
    try:
        # Send request
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()
        
        # Check response
        if response.status_code == 200:
            response_data = response.json()
            completion = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            model_used = response_data.get("model", "")
            
            logger.info("System Overseer prompt test successful!")
            logger.info(f"Model used: {model_used}")
            logger.info(f"Response time: {end_time - start_time:.2f} seconds")
            logger.info(f"Completion: {completion}")
            
            return True
        else:
            logger.error(f"System Overseer prompt test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"System Overseer prompt test failed with exception: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting OpenRouter LLM integration tests...")
    
    # Check if OpenRouter API credentials are available
    if not API_KEY:
        logger.error("OpenRouter API key not found in environment variables")
        return False
    
    # Test models list
    models_list_success = test_models_list()
    
    # Test completion
    completion_success = test_completion()
    
    # Test System Overseer prompt
    system_overseer_success = test_system_overseer_prompt()
    
    # Summarize results
    logger.info("OpenRouter LLM integration test results:")
    logger.info(f"Models List: {'SUCCESS' if models_list_success else 'FAILED'}")
    logger.info(f"Completion: {'SUCCESS' if completion_success else 'FAILED'}")
    logger.info(f"System Overseer Prompt: {'SUCCESS' if system_overseer_success else 'FAILED'}")
    
    # Overall result
    overall_success = models_list_success and completion_success and system_overseer_success
    logger.info(f"Overall result: {'SUCCESS' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    main()
