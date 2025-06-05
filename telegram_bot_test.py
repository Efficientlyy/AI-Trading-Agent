#!/usr/bin/env python
"""
Telegram Bot Command Test Script

This script tests all available commands for the Telegram bot.
"""

import os
import time
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_bot_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("telegram_bot_test")

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
    
    return env_vars

def send_telegram_message(token, chat_id, message):
    """Send message to Telegram
    
    Args:
        token: Telegram bot token
        chat_id: Telegram chat ID
        message: Message to send
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            logger.info(f"Message sent successfully: {message[:20]}...")
            return True
        else:
            logger.error(f"Error sending message: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        return False

def test_telegram_commands():
    """Test all Telegram commands"""
    logger.info("Starting Telegram command test")
    
    # Load environment variables
    env_vars = load_environment_variables('.env-secure/.env')
    token = env_vars.get('TELEGRAM_BOT_TOKEN')
    chat_id = env_vars.get('TELEGRAM_CHAT_ID') or env_vars.get('TELEGRAM_USER_ID')
    
    if not token or not chat_id:
        logger.error("Token or chat ID not set")
        return False
    
    # Send test message
    test_message = f"""
🧪 *TELEGRAM COMMAND TEST*
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Testing all available commands for the Trading-Agent Telegram bot.
"""
    
    if not send_telegram_message(token, chat_id, test_message):
        logger.error("Failed to send test message")
        return False
    
    # Test commands
    commands = [
        "/help",
        "/status",
        "/pairs",
        "/add_pair ETHUSDC",
        "/remove_pair ETHUSDC",
        "/notifications all"
    ]
    
    for command in commands:
        logger.info(f"Testing command: {command}")
        
        if not send_telegram_message(token, chat_id, command):
            logger.error(f"Failed to send command: {command}")
            continue
        
        # Wait for bot to process command
        time.sleep(2)
    
    # Send summary
    summary_message = f"""
📊 *COMMAND TEST SUMMARY*
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Tested {len(commands)} commands:
- {commands[0]}
- {commands[1]}
- {commands[2]}
- {commands[3]}
- {commands[4]}
- {commands[5]}

Please check if you received responses for all commands.
If any command did not work, please let me know.
"""
    
    send_telegram_message(token, chat_id, summary_message)
    
    logger.info("Telegram command test completed")
    return True

if __name__ == "__main__":
    test_telegram_commands()
