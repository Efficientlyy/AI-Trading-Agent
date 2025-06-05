#!/usr/bin/env python
"""
Debug Environment Variable Loader

This script checks if the environment variables are being loaded correctly
from the .env file for the Telegram bot.
"""

import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("env_debug")

def load_environment_variables(env_path=None):
    """Load environment variables from .env file
    
    Args:
        env_path: Path to .env file (optional)
        
    Returns:
        dict: Environment variables
    """
    env_vars = {}
    
    # Try to find .env file
    if env_path is None:
        possible_paths = [
            '.env-secure/.env',
            '.env',
            '../.env-secure/.env',
            '../.env'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                env_path = path
                logger.info(f"Found .env file at: {os.path.abspath(path)}")
                break
    
    # Check if .env file exists
    if env_path and os.path.exists(env_path):
        logger.info(f"Loading environment variables from: {os.path.abspath(env_path)}")
        
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            env_vars[key] = value
                            # Log loaded variables (mask sensitive values)
                            if 'TOKEN' in key or 'KEY' in key or 'SECRET' in key:
                                masked_value = value[:5] + '...' if len(value) > 5 else '***'
                                logger.info(f"Loaded {key}={masked_value}")
                            else:
                                logger.info(f"Loaded {key}={value}")
                        except ValueError:
                            logger.warning(f"Could not parse line: {line}")
        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")
    else:
        logger.error(f"Environment file not found: {env_path}")
    
    # Check for specific variables needed by Telegram
    for var in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_USER_ID', 'TELEGRAM_CHAT_ID']:
        if var in env_vars:
            logger.info(f"Found {var} in .env file")
        else:
            logger.warning(f"{var} not found in .env file")
    
    return env_vars

if __name__ == "__main__":
    logger.info("Starting environment variable debug")
    
    # Try to load from default locations
    env_vars = load_environment_variables()
    
    # Check current directory
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir('.')}")
    
    # Check .env-secure directory if it exists
    if os.path.exists('.env-secure'):
        logger.info(f".env-secure contents: {os.listdir('.env-secure')}")
    
    # Check environment variables specifically for Telegram
    telegram_token = env_vars.get('TELEGRAM_BOT_TOKEN')
    telegram_user_id = env_vars.get('TELEGRAM_USER_ID')
    telegram_chat_id = env_vars.get('TELEGRAM_CHAT_ID')
    
    logger.info(f"TELEGRAM_BOT_TOKEN: {'Found (masked)' if telegram_token else 'Not found'}")
    logger.info(f"TELEGRAM_USER_ID: {telegram_user_id if telegram_user_id else 'Not found'}")
    logger.info(f"TELEGRAM_CHAT_ID: {telegram_chat_id if telegram_chat_id else 'Not found'}")
    
    # Try to load directly from .env-secure/.env
    if os.path.exists('.env-secure/.env'):
        logger.info("Trying to load directly from .env-secure/.env")
        direct_env_vars = load_environment_variables('.env-secure/.env')
        
        # Check environment variables again
        telegram_token = direct_env_vars.get('TELEGRAM_BOT_TOKEN')
        telegram_user_id = direct_env_vars.get('TELEGRAM_USER_ID')
        telegram_chat_id = direct_env_vars.get('TELEGRAM_CHAT_ID')
        
        logger.info(f"Direct TELEGRAM_BOT_TOKEN: {'Found (masked)' if telegram_token else 'Not found'}")
        logger.info(f"Direct TELEGRAM_USER_ID: {telegram_user_id if telegram_user_id else 'Not found'}")
        logger.info(f"Direct TELEGRAM_CHAT_ID: {telegram_chat_id if telegram_chat_id else 'Not found'}")
