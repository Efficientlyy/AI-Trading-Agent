#!/usr/bin/env python
"""
Telegram Bot Service

This script runs the Telegram bot as a background service,
handling commands and notifications for the Trading-Agent system.
"""

import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_bot_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("telegram_bot_service")

# Import the improved Telegram settings command handler
try:
    from improved_telegram_settings_command import EnhancedTelegramNotifier
except ImportError:
    try:
        from telegram_settings_command import EnhancedTelegramNotifier
    except ImportError:
        logger.error("Failed to import EnhancedTelegramNotifier")
        sys.exit(1)

class TelegramBotService:
    """Telegram bot service"""
    
    def __init__(self, env_path=None):
        """Initialize Telegram bot service
        
        Args:
            env_path: Path to .env file (optional)
        """
        self.env_path = env_path
        self.notifier = None
        self.running = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, sig, frame):
        """Handle signals
        
        Args:
            sig: Signal number
            frame: Frame
        """
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
    
    def start(self):
        """Start Telegram bot service"""
        if self.running:
            logger.warning("Telegram bot service already running")
            return
        
        logger.info("Starting Telegram bot service...")
        
        try:
            # Initialize notifier
            self.notifier = EnhancedTelegramNotifier(self.env_path)
            
            # Start notifier
            self.notifier.start()
            
            self.running = True
            
            logger.info("Telegram bot service started")
            
            # Send startup notification
            self.notifier.send_system_notification(f"Trading-Agent Telegram bot service started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Keep running until stopped
            while self.running:
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error starting Telegram bot service: {str(e)}")
            self.stop()
    
    def stop(self):
        """Stop Telegram bot service"""
        if not self.running:
            return
        
        logger.info("Stopping Telegram bot service...")
        
        try:
            # Stop notifier
            if self.notifier:
                self.notifier.stop()
            
            self.running = False
            
            logger.info("Telegram bot service stopped")
        
        except Exception as e:
            logger.error(f"Error stopping Telegram bot service: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Telegram Bot Service")
    parser.add_argument("--env", help="Path to .env file")
    args = parser.parse_args()
    
    # Find .env file
    env_path = args.env
    if not env_path:
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
    
    # Start service
    service = TelegramBotService(env_path)
    service.start()

if __name__ == "__main__":
    main()
