#!/usr/bin/env python
"""
Telegram Bot Verification Script

This script verifies that the Telegram bot can receive updates and messages,
and process commands correctly.
"""

import os
import time
import logging
import threading
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("telegram_verification")

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

class TelegramVerifier:
    """Telegram bot verification utility"""
    
    def __init__(self, env_path=None):
        """Initialize Telegram verifier
        
        Args:
            env_path: Path to .env file (optional)
        """
        # Load API credentials
        env_vars = load_environment_variables(env_path)
        self.token = env_vars.get('TELEGRAM_BOT_TOKEN')
        
        # Use TELEGRAM_USER_ID if TELEGRAM_CHAT_ID is not available
        self.chat_id = env_vars.get('TELEGRAM_CHAT_ID') or env_vars.get('TELEGRAM_USER_ID')
        
        logger.info(f"Loaded Telegram credentials - Token: {self.token[:5] if self.token else None}... Chat ID: {self.chat_id}")
        
        # Last update ID for polling
        self.last_update_id = 0
        
        # Running flag
        self.running = False
        
        # Update polling thread
        self.polling_thread = None
        
        # Updates received
        self.updates_received = 0
        
        # Commands received
        self.commands_received = 0
        
        # Last command received
        self.last_command = None
    
    def start_polling(self):
        """Start polling for updates"""
        if self.running:
            return
        
        self.running = True
        
        # Start polling thread
        self.polling_thread = threading.Thread(target=self._poll_updates)
        self.polling_thread.daemon = True
        self.polling_thread.start()
        
        logger.info("Started polling for updates")
    
    def stop_polling(self):
        """Stop polling for updates"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for polling thread to finish
        if self.polling_thread:
            self.polling_thread.join(timeout=1.0)
        
        logger.info("Stopped polling for updates")
    
    def _poll_updates(self):
        """Poll for updates from Telegram"""
        while self.running:
            try:
                # Get updates from Telegram
                updates = self._get_updates()
                
                # Process updates
                for update in updates:
                    self._process_update(update)
                
                # Sleep to avoid hitting rate limits
                time.sleep(1.0)
            except Exception as e:
                if self.running:  # Only log if still running
                    logger.error(f"Error polling updates: {str(e)}")
                    time.sleep(5.0)  # Sleep longer on error
        
        logger.info("Update polling thread stopped")
    
    def _get_updates(self):
        """Get updates from Telegram
        
        Returns:
            list: Updates
        """
        if not self.token:
            return []
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 30
            }
            
            response = requests.get(url, params=params, timeout=35)
            if response.status_code == 200:
                data = response.json()
                if data["ok"]:
                    updates = data["result"]
                    
                    # Update last update ID
                    if updates:
                        self.last_update_id = max(update["update_id"] for update in updates)
                        self.updates_received += len(updates)
                        logger.info(f"Received {len(updates)} updates, total: {self.updates_received}")
                    
                    return updates
            
            return []
        except Exception as e:
            logger.error(f"Error getting updates: {str(e)}")
            return []
    
    def _process_update(self, update):
        """Process update from Telegram
        
        Args:
            update: Update from Telegram
        """
        try:
            # Check if update contains message
            if "message" not in update:
                return
            
            message = update["message"]
            
            # Check if message contains text
            if "text" not in message:
                return
            
            text = message["text"]
            
            # Check if message is from authorized chat
            if str(message["chat"]["id"]) != str(self.chat_id):
                logger.warning(f"Unauthorized message from chat ID: {message['chat']['id']}")
                return
            
            # Process command
            if text.startswith('/'):
                self.commands_received += 1
                self.last_command = text
                logger.info(f"Received command: {text}, total commands: {self.commands_received}")
            else:
                logger.info(f"Received message: {text}")
        except Exception as e:
            logger.error(f"Error processing update: {str(e)}")
    
    def send_message(self, message):
        """Send message to Telegram
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.token or not self.chat_id:
            logger.error("Token or chat ID not set")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
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
    
    def get_bot_info(self):
        """Get bot information
        
        Returns:
            dict: Bot information or None if failed
        """
        if not self.token:
            logger.error("Token not set")
            return None
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/getMe"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data["ok"]:
                    logger.info(f"Bot info retrieved: {data['result']['username']}")
                    return data["result"]
            
            logger.error(f"Error getting bot info: {response.status_code}, {response.text}")
            return None
        except Exception as e:
            logger.error(f"Error getting bot info: {str(e)}")
            return None

def verify_telegram_bot():
    """Verify Telegram bot functionality"""
    logger.info("Starting Telegram bot verification")
    
    # Create verifier
    verifier = TelegramVerifier('.env-secure/.env')
    
    # Get bot info
    bot_info = verifier.get_bot_info()
    if not bot_info:
        logger.error("Failed to get bot info, verification failed")
        return False
    
    logger.info(f"Bot info: {bot_info}")
    
    # Start polling for updates
    verifier.start_polling()
    
    # Send test message
    test_message = f"""
🔍 *TELEGRAM BOT VERIFICATION*
Bot: @{bot_info['username']}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is a test message to verify that the Telegram bot is working correctly.
Please send a command (e.g., /help) to verify command handling.

Available commands:
/help - Show help message
/status - Show system status
/pairs - Show active trading pairs
"""
    
    if not verifier.send_message(test_message):
        logger.error("Failed to send test message, verification failed")
        verifier.stop_polling()
        return False
    
    # Wait for response
    logger.info("Waiting for commands (30 seconds)...")
    start_time = time.time()
    while time.time() - start_time < 30:
        if verifier.commands_received > 0:
            logger.info(f"Received {verifier.commands_received} commands, verification successful!")
            logger.info(f"Last command: {verifier.last_command}")
            break
        time.sleep(1)
    
    # Check if commands were received
    if verifier.commands_received == 0:
        logger.warning("No commands received during verification period")
    
    # Stop polling
    verifier.stop_polling()
    
    # Send summary
    summary = f"""
📊 *VERIFICATION SUMMARY*
Updates received: {verifier.updates_received}
Commands received: {verifier.commands_received}
Last command: {verifier.last_command or 'None'}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'✅ Verification successful!' if verifier.updates_received > 0 else '⚠️ No updates received during verification period'}
"""
    
    verifier.send_message(summary)
    
    logger.info("Telegram bot verification completed")
    return verifier.updates_received > 0

if __name__ == "__main__":
    verify_telegram_bot()
