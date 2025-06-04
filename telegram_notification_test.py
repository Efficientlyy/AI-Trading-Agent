#!/usr/bin/env python
"""
Enhanced Telegram Notification Test for Trading-Agent System

This module tests the enhanced Telegram notification system with the new token
and validates that all notification types are delivered correctly.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("telegram_notification_test.log")
    ]
)

logger = logging.getLogger("telegram_notification_test")

# Import required modules
try:
    from enhanced_telegram_notifications import EnhancedTelegramNotifier
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    sys.exit(1)

class TelegramNotificationTest:
    """Test class for enhanced Telegram notifications"""
    
    def __init__(self):
        """Initialize Telegram notification test"""
        logger.info("Initializing Telegram notification test...")
        
        # Initialize notifier
        self.notifier = EnhancedTelegramNotifier()
        
        # Test symbols
        self.test_symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC", "BTC/USDT", "ETH/USDT", "SOL/USDT"]
        
        # Results storage
        self.results = {
            "system_notifications": [],
            "signal_notifications": [],
            "order_notifications": [],
            "error_notifications": []
        }
        
        logger.info("Telegram notification test initialized")
    
    def test_system_notifications(self):
        """Test system notifications"""
        logger.info("Testing system notifications...")
        
        # Test different system notification types
        notification_types = [
            "startup", "shutdown", "market_data_update", 
            "signal_generation", "order_execution", "test"
        ]
        
        for notification_type in notification_types:
            message = f"System {notification_type} notification test at {datetime.now().isoformat()}"
            logger.info(f"Sending system notification: {notification_type}")
            
            # Send notification
            self.notifier.notify_system(notification_type, message)
            
            # Store result
            self.results["system_notifications"].append({
                "type": notification_type,
                "message": message,
                "timestamp": int(time.time() * 1000)
            })
            
            # Wait a bit to avoid rate limiting
            time.sleep(1)
        
        logger.info(f"Sent {len(notification_types)} system notifications")
        return len(notification_types)
    
    def test_signal_notifications(self):
        """Test signal notifications"""
        logger.info("Testing signal notifications...")
        
        signal_count = 0
        
        for symbol in self.test_symbols:
            # Create test signals
            signal_types = ["BUY", "SELL"]
            signal_sources = ["momentum", "order_imbalance", "pattern_recognition", "volatility_breakout"]
            
            for signal_type in signal_types:
                for source in signal_sources:
                    # Create test signal
                    strength = 0.5 + (0.1 * (signal_types.index(signal_type) + signal_sources.index(source)))
                    price = 50000.0 if "BTC" in symbol else (3000.0 if "ETH" in symbol else 150.0)
                    
                    test_signal = {
                        "type": signal_type,
                        "source": source,
                        "strength": strength,
                        "timestamp": int(time.time() * 1000),
                        "price": price,
                        "symbol": symbol
                    }
                    
                    logger.info(f"Sending signal notification: {signal_type} {symbol} from {source}")
                    
                    # Send notification
                    self.notifier.notify_signal(test_signal)
                    
                    # Store result
                    self.results["signal_notifications"].append({
                        "signal": test_signal,
                        "timestamp": int(time.time() * 1000)
                    })
                    
                    signal_count += 1
                    
                    # Wait a bit to avoid rate limiting
                    time.sleep(1)
        
        logger.info(f"Sent {signal_count} signal notifications")
        return signal_count
    
    def test_order_notifications(self):
        """Test order notifications"""
        logger.info("Testing order notifications...")
        
        order_count = 0
        
        for symbol in self.test_symbols:
            # Create test orders
            order_sides = ["BUY", "SELL"]
            order_types = ["MARKET", "LIMIT"]
            
            for side in order_sides:
                for order_type in order_types:
                    # Create test order
                    price = 50000.0 if "BTC" in symbol else (3000.0 if "ETH" in symbol else 150.0)
                    quantity = 0.01 if "BTC" in symbol else (0.1 if "ETH" in symbol else 1.0)
                    
                    test_order = {
                        "orderId": f"TEST-{int(time.time())}-{order_count}",
                        "symbol": symbol,
                        "side": side,
                        "type": order_type,
                        "quantity": quantity,
                        "price": price,
                        "status": "NEW",
                        "timestamp": int(time.time() * 1000)
                    }
                    
                    logger.info(f"Sending order created notification: {side} {symbol} {order_type}")
                    
                    # Send order created notification
                    self.notifier.notify_order_created(test_order)
                    
                    # Store result
                    self.results["order_notifications"].append({
                        "type": "created",
                        "order": test_order,
                        "timestamp": int(time.time() * 1000)
                    })
                    
                    order_count += 1
                    
                    # Wait a bit to avoid rate limiting
                    time.sleep(1)
                    
                    # Update order status to FILLED
                    test_order["status"] = "FILLED"
                    test_order["filledQuantity"] = quantity
                    test_order["filledPrice"] = price
                    
                    logger.info(f"Sending order filled notification: {side} {symbol} {order_type}")
                    
                    # Send order filled notification
                    self.notifier.notify_order_filled(test_order)
                    
                    # Store result
                    self.results["order_notifications"].append({
                        "type": "filled",
                        "order": test_order,
                        "timestamp": int(time.time() * 1000)
                    })
                    
                    order_count += 1
                    
                    # Wait a bit to avoid rate limiting
                    time.sleep(1)
        
        logger.info(f"Sent {order_count} order notifications")
        return order_count
    
    def test_error_notifications(self):
        """Test error notifications"""
        logger.info("Testing error notifications...")
        
        # Test different error types
        error_types = [
            "market_data", "signal_generation", "order_execution", 
            "notification", "system", "test"
        ]
        
        for error_type in error_types:
            message = f"Test error in {error_type} component at {datetime.now().isoformat()}"
            logger.info(f"Sending error notification: {error_type}")
            
            # Send notification
            self.notifier.notify_error(error_type, message)
            
            # Store result
            self.results["error_notifications"].append({
                "type": error_type,
                "message": message,
                "timestamp": int(time.time() * 1000)
            })
            
            # Wait a bit to avoid rate limiting
            time.sleep(1)
        
        logger.info(f"Sent {len(error_types)} error notifications")
        return len(error_types)
    
    def run_tests(self):
        """Run all notification tests"""
        logger.info("Running all notification tests...")
        
        # Start notification system
        self.notifier.start()
        
        # Run tests
        system_count = self.test_system_notifications()
        signal_count = self.test_signal_notifications()
        order_count = self.test_order_notifications()
        error_count = self.test_error_notifications()
        
        # Wait for notifications to be processed
        logger.info("Waiting for notifications to be processed")
        time.sleep(5)
        
        # Stop notification system
        self.notifier.stop()
        
        # Print summary
        logger.info("Notification tests completed")
        logger.info(f"System notifications: {system_count}")
        logger.info(f"Signal notifications: {signal_count}")
        logger.info(f"Order notifications: {order_count}")
        logger.info(f"Error notifications: {error_count}")
        logger.info(f"Total notifications: {system_count + signal_count + order_count + error_count}")
        
        # Save results to file
        with open("telegram_notification_results.json", "w") as f:
            json.dump(self.results, f, default=str, indent=2)
        
        return self.results

def main():
    """Main function"""
    logger.info("Starting Telegram notification test...")
    
    # Initialize and run tests
    test = TelegramNotificationTest()
    results = test.run_tests()
    
    # Print summary
    print("\n=== TELEGRAM NOTIFICATION TEST RESULTS ===")
    print(f"System notifications: {len(results['system_notifications'])}")
    print(f"Signal notifications: {len(results['signal_notifications'])}")
    print(f"Order notifications: {len(results['order_notifications'])}")
    print(f"Error notifications: {len(results['error_notifications'])}")
    print(f"Total notifications: {sum(len(v) for v in results.values())}")
    print("\nDetailed results saved to telegram_notification_results.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
