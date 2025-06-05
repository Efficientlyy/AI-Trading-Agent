#!/usr/bin/env python
"""
Telegram Bot Connectivity Test

This script tests connectivity to the Telegram Bot API using the provided credentials.
"""

import os
import sys
import logging
import time
from dotenv import load_dotenv
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("telegram_bot_test.log")
    ]
)
logger = logging.getLogger("telegram_bot_test")

# Load environment variables
load_dotenv('.env-secure/.env')

# Telegram Bot credentials
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Telegram Bot API endpoint
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

def test_get_me():
    """Test the getMe method to verify bot credentials.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing Telegram Bot API getMe method...")
    
    # Endpoint
    endpoint = "/getMe"
    url = API_URL + endpoint
    
    try:
        # Send request
        response = requests.get(url)
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                bot_info = data.get("result")
                logger.info("Telegram Bot API getMe test successful!")
                logger.info(f"Bot info: {json.dumps(bot_info, indent=2)}")
                return True
            else:
                logger.error(f"Telegram Bot API getMe test failed: {data.get('description')}")
                return False
        else:
            logger.error(f"Telegram Bot API getMe test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram Bot API getMe test failed with exception: {e}")
        return False

def test_get_updates():
    """Test the getUpdates method to check for recent messages.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing Telegram Bot API getUpdates method...")
    
    # Endpoint
    endpoint = "/getUpdates"
    url = API_URL + endpoint
    
    try:
        # Send request
        response = requests.get(url)
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                updates = data.get("result", [])
                logger.info("Telegram Bot API getUpdates test successful!")
                logger.info(f"Received {len(updates)} updates")
                
                # Log recent updates
                if updates:
                    for update in updates[-3:]:  # Show last 3 updates
                        logger.info(f"Update: {json.dumps(update, indent=2)}")
                
                return True
            else:
                logger.error(f"Telegram Bot API getUpdates test failed: {data.get('description')}")
                return False
        else:
            logger.error(f"Telegram Bot API getUpdates test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram Bot API getUpdates test failed with exception: {e}")
        return False

def test_send_message():
    """Test sending a message to the chat.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing Telegram Bot API sendMessage method...")
    
    # Endpoint
    endpoint = "/sendMessage"
    url = API_URL + endpoint
    
    # Message parameters
    params = {
        "chat_id": CHAT_ID,
        "text": "🔄 System Overseer Test: This is a connectivity test message. The System Overseer is being configured and tested with real credentials.",
        "parse_mode": "HTML"
    }
    
    try:
        # Send request
        response = requests.post(url, json=params)
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                message = data.get("result")
                logger.info("Telegram Bot API sendMessage test successful!")
                logger.info(f"Message sent: {json.dumps(message, indent=2)}")
                return True
            else:
                logger.error(f"Telegram Bot API sendMessage test failed: {data.get('description')}")
                return False
        else:
            logger.error(f"Telegram Bot API sendMessage test failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram Bot API sendMessage test failed with exception: {e}")
        return False

def test_send_system_status():
    """Test sending a formatted system status message.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing sending formatted system status message...")
    
    # Endpoint
    endpoint = "/sendMessage"
    url = API_URL + endpoint
    
    # Current time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Message parameters
    params = {
        "chat_id": CHAT_ID,
        "text": f"""
<b>📊 System Overseer Status Report</b>

<b>Time:</b> {current_time}
<b>Status:</b> <i>Initializing</i>

<b>Components:</b>
✅ MEXC API Connection: <code>ACTIVE</code>
✅ GitHub Repository: <code>SYNCED</code>
🔄 LLM Integration: <code>PENDING</code>
🔄 Trading System: <code>PENDING</code>

<b>System Overseer is being configured and tested.</b>
You will receive a full status report once all tests are complete.
""",
        "parse_mode": "HTML"
    }
    
    try:
        # Send request
        response = requests.post(url, json=params)
        
        # Check response
        if response.status_code == 200:
            data = response.json()
            if data.get("ok"):
                message = data.get("result")
                logger.info("Formatted system status message sent successfully!")
                return True
            else:
                logger.error(f"Sending formatted system status message failed: {data.get('description')}")
                return False
        else:
            logger.error(f"Sending formatted system status message failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Sending formatted system status message failed with exception: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting Telegram Bot connectivity tests...")
    
    # Check if Telegram Bot credentials are available
    if not BOT_TOKEN or not CHAT_ID:
        logger.error("Telegram Bot credentials not found in environment variables")
        return False
    
    # Test getMe method
    get_me_success = test_get_me()
    
    # Test getUpdates method
    get_updates_success = test_get_updates()
    
    # Test sendMessage method
    send_message_success = test_send_message()
    
    # Test sending formatted system status
    system_status_success = test_send_system_status()
    
    # Summarize results
    logger.info("Telegram Bot connectivity test results:")
    logger.info(f"getMe: {'SUCCESS' if get_me_success else 'FAILED'}")
    logger.info(f"getUpdates: {'SUCCESS' if get_updates_success else 'FAILED'}")
    logger.info(f"sendMessage: {'SUCCESS' if send_message_success else 'FAILED'}")
    logger.info(f"System Status Message: {'SUCCESS' if system_status_success else 'FAILED'}")
    
    # Overall result
    overall_success = get_me_success and get_updates_success and send_message_success and system_status_success
    logger.info(f"Overall result: {'SUCCESS' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    main()
