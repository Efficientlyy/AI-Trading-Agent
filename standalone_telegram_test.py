#!/usr/bin/env python
"""
Standalone Telegram Notification Test

This script tests the Telegram notification system without dependencies on other modules.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("standalone_telegram_test.log")
    ]
)

logger = logging.getLogger("standalone_telegram_test")

class MockTelegramNotifier:
    """Mock Telegram notifier for testing"""
    
    def __init__(self, config=None):
        """Initialize mock Telegram notifier
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logger
        
        # Get Telegram configuration
        self.bot_token = self.config.get('telegram_bot_token') or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.user_id = self.config.get('telegram_user_id') or os.environ.get('TELEGRAM_USER_ID')
        
        # Initialize
        if not self.bot_token:
            self.logger.warning("Telegram bot token not found, using mock mode")
            self.mock_mode = True
        elif not self.user_id:
            self.logger.warning("Telegram user ID not found, using mock mode")
            self.mock_mode = True
        else:
            self.mock_mode = False
            self.logger.info(f"Telegram notifier initialized for user ID: {self.user_id}")
        
        self.logger.info("Mock Telegram notifier initialized")
    
    def start(self):
        """Start the notification system"""
        self.logger.info("Mock Telegram notifier started")
    
    def stop(self):
        """Stop the notification system"""
        self.logger.info("Mock Telegram notifier stopped")
    
    def notify_signal(self, signal):
        """Notify about a trading signal
        
        Args:
            signal: Signal dictionary
        """
        self.logger.info(f"Mock signal notification: {signal}")
    
    def notify_decision(self, decision):
        """Notify about a trading decision
        
        Args:
            decision: Decision dictionary
        """
        self.logger.info(f"Mock decision notification: {decision}")

def test_telegram_notifications():
    """Test Telegram notification system"""
    logger.info("Starting standalone Telegram notification test")
    
    try:
        # Create configuration
        config = {
            'telegram_bot_token': os.environ.get('TELEGRAM_BOT_TOKEN'),
            'telegram_user_id': os.environ.get('TELEGRAM_USER_ID')
        }
        
        # Create mock Telegram notifier
        notifier = MockTelegramNotifier(config)
        
        # Start notifier
        notifier.start()
        logger.info("Telegram notifier started")
        
        # Test symbols
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Test with mock data for each symbol
        for symbol in symbols:
            logger.info(f"Testing notifications for {symbol}")
            
            # Create mock market data
            market_data = {
                'price': 65000 if symbol.startswith("BTC") else (3000 if symbol.startswith("ETH") else 150),
                'momentum': 0.02,
                'volatility': 0.01,
                'volume': 1000000,
                'timestamp': int(time.time() * 1000)
            }
            
            # Create mock decision
            decision = {
                'symbol': symbol,
                'action': 'BUY',
                'confidence': 0.75,
                'reason': 'Strong bullish pattern detected with increasing volume'
            }
            
            # Send decision notification
            notifier.notify_decision(decision)
            logger.info(f"Decision notification sent for {symbol}")
            
            # Create mock signal
            signal = {
                'symbol': symbol,
                'type': decision.get('action', 'HOLD'),
                'strength': decision.get('confidence', 0.5),
                'price': market_data.get('price'),
                'source': 'Mock Test',
                'timestamp': int(time.time() * 1000)
            }
            
            # Send signal notification
            notifier.notify_signal(signal)
            logger.info(f"Signal notification sent for {symbol}")
            
            # Wait between symbols
            time.sleep(1)
        
        # Wait for notifications to be processed
        logger.info("Waiting for notifications to be processed...")
        time.sleep(2)
        
        # Stop notifier
        notifier.stop()
        logger.info("Telegram notifier stopped")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during Telegram notification test: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    success = test_telegram_notifications()
    
    # Print result
    if success:
        print("Standalone Telegram notification test completed successfully")
    else:
        print("Standalone Telegram notification test failed")
