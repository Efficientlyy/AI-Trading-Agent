#!/usr/bin/env python
"""
Enhanced Telegram Notifications with Settings Command

This module provides a Telegram notification system for the Trading-Agent
with support for controlling trading pairs through commands.
"""

import os
import json
import time
import logging
import threading
import requests
from queue import Queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("telegram_notifications")

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

class TelegramSettings:
    """Settings manager for Telegram bot"""
    
    def __init__(self, settings_file="telegram_settings.json"):
        """Initialize settings manager
        
        Args:
            settings_file: Path to settings file
        """
        self.settings_file = settings_file
        self.settings = self._load_settings()
        
        # Set default settings if not present
        if "active_pairs" not in self.settings:
            self.settings["active_pairs"] = ["BTCUSDC"]
        
        if "notification_level" not in self.settings:
            self.settings["notification_level"] = "all"  # all, signals, trades, errors, none
        
        # Save settings to ensure file exists with defaults
        self._save_settings()
    
    def _load_settings(self):
        """Load settings from file
        
        Returns:
            dict: Settings
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error loading settings: {str(e)}")
            return {}
    
    def _save_settings(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
    
    def get_active_pairs(self):
        """Get active trading pairs
        
        Returns:
            list: Active trading pairs
        """
        return self.settings.get("active_pairs", ["BTCUSDC"])
    
    def set_active_pairs(self, pairs):
        """Set active trading pairs
        
        Args:
            pairs: List of trading pairs
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.settings["active_pairs"] = pairs
            self._save_settings()
            return True
        except Exception as e:
            logger.error(f"Error setting active pairs: {str(e)}")
            return False
    
    def add_active_pair(self, pair):
        """Add trading pair to active pairs
        
        Args:
            pair: Trading pair to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            active_pairs = self.get_active_pairs()
            if pair not in active_pairs:
                active_pairs.append(pair)
                self.settings["active_pairs"] = active_pairs
                self._save_settings()
            return True
        except Exception as e:
            logger.error(f"Error adding active pair: {str(e)}")
            return False
    
    def remove_active_pair(self, pair):
        """Remove trading pair from active pairs
        
        Args:
            pair: Trading pair to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            active_pairs = self.get_active_pairs()
            if pair in active_pairs:
                active_pairs.remove(pair)
                self.settings["active_pairs"] = active_pairs
                self._save_settings()
            return True
        except Exception as e:
            logger.error(f"Error removing active pair: {str(e)}")
            return False
    
    def get_notification_level(self):
        """Get notification level
        
        Returns:
            str: Notification level
        """
        return self.settings.get("notification_level", "all")
    
    def set_notification_level(self, level):
        """Set notification level
        
        Args:
            level: Notification level (all, signals, trades, errors, none)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            valid_levels = ["all", "signals", "trades", "errors", "none"]
            if level in valid_levels:
                self.settings["notification_level"] = level
                self._save_settings()
                return True
            else:
                logger.error(f"Invalid notification level: {level}")
                return False
        except Exception as e:
            logger.error(f"Error setting notification level: {str(e)}")
            return False

class EnhancedTelegramNotifier:
    """Enhanced Telegram notification system with settings command support"""
    
    def __init__(self, env_path=None):
        """Initialize Telegram notifier
        
        Args:
            env_path: Path to .env file (optional)
        """
        # Load API credentials
        env_vars = load_environment_variables(env_path)
        self.token = env_vars.get('TELEGRAM_BOT_TOKEN')
        
        # Use TELEGRAM_USER_ID if TELEGRAM_CHAT_ID is not available
        self.chat_id = env_vars.get('TELEGRAM_CHAT_ID') or env_vars.get('TELEGRAM_USER_ID')
        
        logger.info(f"Loaded Telegram credentials - Token: {self.token[:5] if self.token else None}... Chat ID: {self.chat_id}")
        
        # Initialize settings manager
        self.settings = TelegramSettings()
        
        # Initialize notification queue
        self.notification_queue = Queue()
        self.notification_thread = None
        self.running = False
        
        # Command handlers
        self.command_handlers = {
            "/help": self._handle_help_command,
            "/status": self._handle_status_command,
            "/pairs": self._handle_pairs_command,
            "/add_pair": self._handle_add_pair_command,
            "/remove_pair": self._handle_remove_pair_command,
            "/notifications": self._handle_notifications_command
        }
        
        # Last update ID for polling
        self.last_update_id = 0
        
        # Command polling thread
        self.command_thread = None
    
    def start(self):
        """Start notification system"""
        if self.running:
            return
        
        self.running = True
        
        # Start notification thread
        self.notification_thread = threading.Thread(target=self._process_notifications)
        self.notification_thread.daemon = True
        self.notification_thread.start()
        
        # Start command polling thread
        self.command_thread = threading.Thread(target=self._poll_commands)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        logger.info("Telegram notification system started")
        
        # Send startup notification
        self.send_system_notification("Trading-Agent system started")
    
    def stop(self):
        """Stop notification system"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for threads to finish
        if self.notification_thread:
            self.notification_thread.join(timeout=1.0)
        
        if self.command_thread:
            self.command_thread.join(timeout=1.0)
        
        logger.info("Telegram notification system stopped")
    
    def _process_notifications(self):
        """Process notifications from queue"""
        while self.running:
            try:
                # Get notification from queue with timeout
                notification_type, data = self.notification_queue.get(timeout=1.0)
                
                # Check notification level
                notification_level = self.settings.get_notification_level()
                if notification_level == "none":
                    continue
                
                if notification_level != "all":
                    if notification_type == "signal" and notification_level != "signals":
                        continue
                    if notification_type == "trade" and notification_level != "trades":
                        continue
                    if notification_type == "error" and notification_level != "errors":
                        continue
                
                # Process notification based on type
                if notification_type == "signal":
                    self._send_signal_notification(data)
                elif notification_type == "trade":
                    self._send_trade_notification(data)
                elif notification_type == "error":
                    self._send_error_notification(data)
                elif notification_type == "system":
                    self._send_system_notification(data)
                
                # Mark task as done
                self.notification_queue.task_done()
                
                # Log notification
                logger.info(f"Notification sent: {notification_type}")
            except Exception as e:
                if self.running:  # Only log if still running
                    logger.error(f"Error processing notification: {str(e)}")
        
        logger.info("Notification processing thread stopped")
    
    def _poll_commands(self):
        """Poll for commands from Telegram"""
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
                    logger.error(f"Error polling commands: {str(e)}")
                    time.sleep(5.0)  # Sleep longer on error
        
        logger.info("Command polling thread stopped")
    
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
            self._process_command(text)
        except Exception as e:
            logger.error(f"Error processing update: {str(e)}")
    
    def _process_command(self, text):
        """Process command from Telegram
        
        Args:
            text: Command text
        """
        try:
            # Split command and arguments
            parts = text.split()
            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            # Check if command is supported
            if command in self.command_handlers:
                # Call command handler
                self.command_handlers[command](args)
            else:
                # Send help message for unknown command
                self.send_system_notification(f"Unknown command: {command}\nUse /help for available commands")
        except Exception as e:
            logger.error(f"Error processing command: {str(e)}")
            self.send_system_notification(f"Error processing command: {str(e)}")
    
    def _handle_help_command(self, args):
        """Handle help command
        
        Args:
            args: Command arguments
        """
        help_text = """
Available commands:
/help - Show this help message
/status - Show system status
/pairs - Show active trading pairs
/add_pair SYMBOL - Add trading pair (e.g., /add_pair ETHUSDC)
/remove_pair SYMBOL - Remove trading pair (e.g., /remove_pair ETHUSDC)
/notifications LEVEL - Set notification level (all, signals, trades, errors, none)
"""
        self.send_system_notification(help_text)
    
    def _handle_status_command(self, args):
        """Handle status command
        
        Args:
            args: Command arguments
        """
        active_pairs = self.settings.get_active_pairs()
        notification_level = self.settings.get_notification_level()
        
        status_text = f"""
System Status:
- Active pairs: {', '.join(active_pairs)}
- Notification level: {notification_level}
- System time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_system_notification(status_text)
    
    def _handle_pairs_command(self, args):
        """Handle pairs command
        
        Args:
            args: Command arguments
        """
        active_pairs = self.settings.get_active_pairs()
        
        if active_pairs:
            pairs_text = f"Active trading pairs: {', '.join(active_pairs)}"
        else:
            pairs_text = "No active trading pairs"
        
        self.send_system_notification(pairs_text)
    
    def _handle_add_pair_command(self, args):
        """Handle add_pair command
        
        Args:
            args: Command arguments
        """
        if not args:
            self.send_system_notification("Please specify a trading pair to add (e.g., /add_pair ETHUSDC)")
            return
        
        pair = args[0].upper()
        
        # Validate pair format
        if not self._validate_pair_format(pair):
            self.send_system_notification(f"Invalid pair format: {pair}\nPair should be in format BTCUSDC, ETHUSDC, etc.")
            return
        
        # Add pair
        if self.settings.add_active_pair(pair):
            self.send_system_notification(f"Added trading pair: {pair}")
        else:
            self.send_system_notification(f"Failed to add trading pair: {pair}")
    
    def _handle_remove_pair_command(self, args):
        """Handle remove_pair command
        
        Args:
            args: Command arguments
        """
        if not args:
            self.send_system_notification("Please specify a trading pair to remove (e.g., /remove_pair ETHUSDC)")
            return
        
        pair = args[0].upper()
        
        # Check if pair is active
        active_pairs = self.settings.get_active_pairs()
        if pair not in active_pairs:
            self.send_system_notification(f"Trading pair not active: {pair}")
            return
        
        # Don't allow removing all pairs
        if len(active_pairs) == 1 and pair in active_pairs:
            self.send_system_notification(f"Cannot remove last active pair: {pair}")
            return
        
        # Remove pair
        if self.settings.remove_active_pair(pair):
            self.send_system_notification(f"Removed trading pair: {pair}")
        else:
            self.send_system_notification(f"Failed to remove trading pair: {pair}")
    
    def _handle_notifications_command(self, args):
        """Handle notifications command
        
        Args:
            args: Command arguments
        """
        if not args:
            notification_level = self.settings.get_notification_level()
            self.send_system_notification(f"Current notification level: {notification_level}")
            self.send_system_notification("Available levels: all, signals, trades, errors, none")
            return
        
        level = args[0].lower()
        
        # Validate level
        valid_levels = ["all", "signals", "trades", "errors", "none"]
        if level not in valid_levels:
            self.send_system_notification(f"Invalid notification level: {level}")
            self.send_system_notification(f"Available levels: {', '.join(valid_levels)}")
            return
        
        # Set level
        if self.settings.set_notification_level(level):
            self.send_system_notification(f"Notification level set to: {level}")
        else:
            self.send_system_notification(f"Failed to set notification level: {level}")
    
    def _validate_pair_format(self, pair):
        """Validate trading pair format
        
        Args:
            pair: Trading pair
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Simple validation for now
        return len(pair) >= 6 and pair.isalnum()
    
    def send_signal_notification(self, signal):
        """Send signal notification
        
        Args:
            signal: Signal data
        """
        # Check if pair is active
        symbol = signal.get('symbol')
        active_pairs = self.settings.get_active_pairs()
        
        if symbol and symbol not in active_pairs:
            logger.info(f"Skipping notification for inactive pair: {symbol}")
            return
        
        # Queue notification
        self.notification_queue.put(("signal", signal))
    
    def send_trade_notification(self, trade):
        """Send trade notification
        
        Args:
            trade: Trade data
        """
        # Check if pair is active
        symbol = trade.get('symbol')
        active_pairs = self.settings.get_active_pairs()
        
        if symbol and symbol not in active_pairs:
            logger.info(f"Skipping notification for inactive pair: {symbol}")
            return
        
        # Queue notification
        self.notification_queue.put(("trade", trade))
    
    def send_error_notification(self, error):
        """Send error notification
        
        Args:
            error: Error message
        """
        # Queue notification
        self.notification_queue.put(("error", error))
    
    def send_system_notification(self, message):
        """Send system notification
        
        Args:
            message: System message
        """
        # Queue notification
        self.notification_queue.put(("system", message))
    
    def _send_signal_notification(self, signal):
        """Send signal notification to Telegram
        
        Args:
            signal: Signal data
        """
        if not self.token or not self.chat_id:
            return
        
        try:
            # Format signal message
            symbol = signal.get('symbol', 'UNKNOWN')
            direction = signal.get('direction', 'UNKNOWN')
            strength = signal.get('strength', 0.0)
            price = signal.get('price', 0.0)
            source = signal.get('source', 'UNKNOWN')
            timestamp = signal.get('timestamp', 0)
            
            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(timestamp / 1000) if timestamp else datetime.now()
            
            message = f"""
🔔 *TRADING SIGNAL*
Symbol: `{symbol}`
Direction: `{direction}`
Strength: `{strength:.2f}`
Price: `{price:.2f}`
Source: `{source}`
Time: `{dt.strftime('%Y-%m-%d %H:%M:%S')}`
"""
            
            # Send message
            self._send_telegram_message(message)
        except Exception as e:
            logger.error(f"Error sending signal notification: {str(e)}")
    
    def _send_trade_notification(self, trade):
        """Send trade notification to Telegram
        
        Args:
            trade: Trade data
        """
        if not self.token or not self.chat_id:
            return
        
        try:
            # Format trade message
            symbol = trade.get('symbol', 'UNKNOWN')
            action = trade.get('action', 'UNKNOWN')
            quantity = trade.get('quantity', 0.0)
            price = trade.get('price', 0.0)
            status = trade.get('status', 'UNKNOWN')
            timestamp = trade.get('timestamp', 0)
            
            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(timestamp / 1000) if timestamp else datetime.now()
            
            # Calculate total value
            total = quantity * price
            
            message = f"""
💰 *TRADE EXECUTED*
Symbol: `{symbol}`
Action: `{action}`
Quantity: `{quantity:.8f}`
Price: `{price:.2f}`
Total: `{total:.2f}`
Status: `{status}`
Time: `{dt.strftime('%Y-%m-%d %H:%M:%S')}`
"""
            
            # Send message
            self._send_telegram_message(message)
        except Exception as e:
            logger.error(f"Error sending trade notification: {str(e)}")
    
    def _send_error_notification(self, error):
        """Send error notification to Telegram
        
        Args:
            error: Error message
        """
        if not self.token or not self.chat_id:
            return
        
        try:
            # Format error message
            message = f"""
❌ *ERROR*
{error}
Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
            
            # Send message
            self._send_telegram_message(message)
        except Exception as e:
            logger.error(f"Error sending error notification: {str(e)}")
    
    def _send_system_notification(self, message):
        """Send system notification to Telegram
        
        Args:
            message: System message
        """
        if not self.token or not self.chat_id:
            return
        
        try:
            # Format system message
            formatted_message = f"""
ℹ️ *SYSTEM*
{message}
"""
            
            # Send message
            self._send_telegram_message(formatted_message)
        except Exception as e:
            logger.error(f"Error sending system notification: {str(e)}")
    
    def _send_telegram_message(self, message):
        """Send message to Telegram
        
        Args:
            message: Message to send
        """
        if not self.token or not self.chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                logger.error(f"Error sending Telegram message: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

# For backward compatibility
TelegramNotifier = EnhancedTelegramNotifier

# Test function
def test_telegram_notifier():
    """Test Telegram notifier"""
    notifier = EnhancedTelegramNotifier('.env-secure/.env')
    notifier.start()
    
    # Send test notifications
    notifier.send_system_notification("Telegram notification system test")
    
    # Test signal notification
    signal = {
        'id': f"SIG-{int(time.time())}",
        'symbol': 'BTCUSDC',
        'direction': 'BUY',
        'strength': 0.75,
        'price': 50000.0,
        'source': 'technical',
        'timestamp': int(time.time() * 1000)
    }
    notifier.send_signal_notification(signal)
    
    # Test trade notification
    trade = {
        'id': f"TRD-{int(time.time())}",
        'symbol': 'BTCUSDC',
        'action': 'BUY',
        'quantity': 0.1,
        'price': 50000.0,
        'timestamp': int(time.time() * 1000),
        'status': 'FILLED'
    }
    notifier.send_trade_notification(trade)
    
    # Test error notification
    notifier.send_error_notification("Test error message")
    
    # Wait for notifications to be sent
    time.sleep(5)
    
    # Stop notifier
    notifier.stop()

if __name__ == "__main__":
    test_telegram_notifier()
