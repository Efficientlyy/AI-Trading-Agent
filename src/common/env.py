"""
Environment utilities for loading and validating API credentials.

This module provides functions for safely loading environment variables
and checking if required API credentials are available.
"""

import os
import logging
from typing import Dict, List, Optional

# Try to import dotenv, but don't fail if it's not available
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass  # dotenv is optional

logger = logging.getLogger(__name__)

def load_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Load an environment variable with an optional default.
    
    Args:
        name: Name of the environment variable
        default: Default value if not found
        
    Returns:
        The environment variable value or the default
    """
    value = os.environ.get(name, default)
    if value is None:
        logger.debug(f"Environment variable {name} not found")
    return value

def check_twitter_credentials() -> bool:
    """Check if Twitter API credentials are available.
    
    Returns:
        True if all required credentials are available
    """
    required_vars = [
        "TWITTER_API_KEY",
        "TWITTER_API_SECRET",
        "TWITTER_ACCESS_TOKEN",
        "TWITTER_ACCESS_SECRET"
    ]
    
    missing = [var for var in required_vars if not load_env_var(var)]
    
    if missing:
        logger.warning(f"Missing Twitter credentials: {', '.join(missing)}")
        logger.info("Using mock Twitter data instead of real API")
        return False
        
    return True

def check_reddit_credentials() -> bool:
    """Check if Reddit API credentials are available.
    
    Returns:
        True if all required credentials are available
    """
    required_vars = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT"
    ]
    
    missing = [var for var in required_vars if not load_env_var(var)]
    
    if missing:
        logger.warning(f"Missing Reddit credentials: {', '.join(missing)}")
        logger.info("Using mock Reddit data instead of real API")
        return False
        
    return True

def check_exchange_credentials(exchange: str) -> bool:
    """Check if exchange API credentials are available.
    
    Args:
        exchange: Exchange name (e.g., "BINANCE")
        
    Returns:
        True if all required credentials are available
    """
    exchange = exchange.upper()
    required_vars = [
        f"{exchange}_API_KEY",
        f"{exchange}_API_SECRET"
    ]
    
    missing = [var for var in required_vars if not load_env_var(var)]
    
    if missing:
        logger.warning(f"Missing {exchange} credentials: {', '.join(missing)}")
        return False
        
    return True

def get_api_credentials(api_name: str) -> Dict[str, str]:
    """Get all credentials for a specific API.
    
    Args:
        api_name: API name (e.g., "TWITTER", "REDDIT")
        
    Returns:
        Dictionary with API credentials
    """
    api_name = api_name.upper()
    credentials = {}
    
    # Get all environment variables that start with the API name
    for key, value in os.environ.items():
        if key.startswith(f"{api_name}_"):
            # Convert to lowercase key without prefix
            credential_key = key[len(api_name)+1:].lower()
            credentials[credential_key] = value
    
    return credentials
