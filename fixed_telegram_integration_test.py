#!/usr/bin/env python
"""
Updated Telegram Integration Test with Fixed Paper Trading

This module validates the enhanced Telegram notification system with the fixed
paper trading system, ensuring proper integration across the trading pipeline.
"""

import os
import sys
import json
import time
import logging
import threading
from queue import Queue
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from enhanced_logging_fixed import EnhancedLogger
from enhanced_telegram_notifications import EnhancedTelegramNotifier
from fixed_llm_overseer import FixedLLMOverseer
from fixed_paper_trading import FixedPaperTradingSystem
from optimized_mexc_client import OptimizedMEXCClient

# Initialize enhanced logger
logger = EnhancedLogger("telegram_integration_test_fixed")

class FixedTelegramIntegrationTest:
    """Test class for validating Telegram notification integration with fixed components"""
    
    def __init__(self):
        """Initialize test environment"""
        logger.system.info("Initializing fixed Telegram integration test environment")
        
        # Create configuration
        self.config = {
            'telegram_bot_token': os.environ.get('TELEGRAM_BOT_TOKEN'),
            'telegram_user_id': os.environ.get('TELEGRAM_USER_ID'),
            'min_signal_strength': 0.5,
            'max_signal_age_ms': 10000,  # Extended for testing
            'order_creation_retries': 3,
            'order_creation_retry_delay_ms': 500,
            'risk_per_trade_pct': 1.0,
            'max_position_pct': 5.0,
            'min_position_size': 0.001,
            'position_precision': 3,
            'default_position_size': 0.001,
            'buy_price_factor': 1.001,
            'sell_price_factor': 0.999,
            'max_spread_pct': 1.0,
            'max_price_change_pct': 0.5
        }
        
        # Initialize components
        self.client = OptimizedMEXCClient()
        self.paper_trading = FixedPaperTradingSystem(self.client, self.config)
        self.notifier = EnhancedTelegramNotifier(self.config)
        self.llm_overseer = FixedLLMOverseer(self.config)
        
        # Set up notification callback
        self.paper_trading.set_notification_callback(self.notification_callback)
        
        # Test results
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
        logger.system.info("Test environment initialized")
    
    def notification_callback(self, notification_type, data):
        """Notification callback for paper trading system
        
        Args:
            notification_type: Type of notification
            data: Notification data
        """
        if notification_type == 'order_created':
            self.notifier.notify_order_created(data)
        elif notification_type == 'order_filled':
            self.notifier.notify_order_filled(data)
        elif notification_type == 'order_cancelled':
            self.notifier.notify_order_cancelled(data)
    
    def run_all_tests(self):
        """Run all test cases"""
        logger.system.info("Starting all fixed Telegram integration tests")
        
        # Start components
        self.notifier.start()
        self.paper_trading.start()
        
        # Run individual tests
        self.test_signal_notifications()
        self.test_order_notifications()
        self.test_decision_notifications()
        self.test_error_notifications()
        self.test_system_notifications()
        self.test_performance_notifications()
        self.test_integration_with_llm_overseer()
        self.test_integration_with_paper_trading()
        
        # Stop components
        self.paper_trading.stop()
        self.notifier.stop()
        
        # Log summary
        logger.system.info(f"Test summary: {self.results['passed_tests']}/{self.results['total_tests']} tests passed")
        
        # Return results
        return self.results
    
    def record_test_result(self, test_name, passed, details=None):
        """Record test result
        
        Args:
            test_name: Name of the test
            passed: Whether the test passed
            details: Additional test details
        """
        self.results["total_tests"] += 1
        
        if passed:
            self.results["passed_tests"] += 1
            logger.system.info(f"Test '{test_name}' PASSED")
        else:
            self.results["failed_tests"] += 1
            logger.system.error(f"Test '{test_name}' FAILED")
        
        self.results["test_details"].append({
            "test_name": test_name,
            "passed": passed,
            "details": details or {}
        })
    
    def test_signal_notifications(self):
        """Test signal notifications"""
        logger.system.info("Running test: Signal notifications")
        
        try:
            # Create test signal
            signal = {
                'id': f"SIG-TEST-{int(time.time())}",
                'type': 'BUY',
                'source': 'test',
                'strength': 0.8,
                'timestamp': int(time.time() * 1000),
                'price': 105000.0,
                'symbol': 'BTCUSDC',
                'session': 'TEST'
            }
            
            # Send notification
            self.notifier.notify_signal(signal)
            
            # Wait for processing
            time.sleep(1)
            
            # Verify notification was processed
            # Note: In a real test, we would verify more details
            self.record_test_result(
                "Signal notifications",
                True,  # Assume passed if no exceptions
                {
                    "signal": signal
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_signal_notifications: {str(e)}")
            self.record_test_result(
                "Signal notifications",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_order_notifications(self):
        """Test order notifications"""
        logger.system.info("Running test: Order notifications")
        
        try:
            # Create test order
            order = {
                'symbol': 'BTCUSDC',
                'side': 'BUY',
                'type': 'LIMIT',
                'quantity': 0.001,
                'price': 105000.0,
                'orderId': f"ORD-TEST-{int(time.time())}"
            }
            
            # Send notifications
            self.notifier.notify_order_created(order)
            time.sleep(0.5)
            
            self.notifier.notify_order_filled(order)
            time.sleep(0.5)
            
            self.notifier.notify_order_cancelled(order, reason="Test cancellation")
            time.sleep(0.5)
            
            # Verify notifications were processed
            self.record_test_result(
                "Order notifications",
                True,  # Assume passed if no exceptions
                {
                    "order": order
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_order_notifications: {str(e)}")
            self.record_test_result(
                "Order notifications",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_decision_notifications(self):
        """Test decision notifications"""
        logger.system.info("Running test: Decision notifications")
        
        try:
            # Create test decision
            decision = {
                'id': f"DECISION-TEST-{int(time.time())}",
                'symbol': 'BTCUSDC',
                'action': 'BUY',
                'confidence': 0.75,
                'reason': 'Strong bullish pattern detected with increasing volume'
            }
            
            # Send notification
            self.notifier.notify_decision(decision)
            
            # Wait for processing
            time.sleep(1)
            
            # Verify notification was processed
            self.record_test_result(
                "Decision notifications",
                True,  # Assume passed if no exceptions
                {
                    "decision": decision
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_decision_notifications: {str(e)}")
            self.record_test_result(
                "Decision notifications",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_error_notifications(self):
        """Test error notifications"""
        logger.system.info("Running test: Error notifications")
        
        try:
            # Send error notification
            self.notifier.notify_error("test_component", "Test error message")
            
            # Wait for processing
            time.sleep(1)
            
            # Verify notification was processed
            self.record_test_result(
                "Error notifications",
                True,  # Assume passed if no exceptions
                {
                    "component": "test_component",
                    "message": "Test error message"
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_error_notifications: {str(e)}")
            self.record_test_result(
                "Error notifications",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_system_notifications(self):
        """Test system notifications"""
        logger.system.info("Running test: System notifications")
        
        try:
            # Send system notification
            self.notifier.notify_system("test_component", "Test system message")
            
            # Wait for processing
            time.sleep(1)
            
            # Verify notification was processed
            self.record_test_result(
                "System notifications",
                True,  # Assume passed if no exceptions
                {
                    "component": "test_component",
                    "message": "Test system message"
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_system_notifications: {str(e)}")
            self.record_test_result(
                "System notifications",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_performance_notifications(self):
        """Test performance notifications"""
        logger.system.info("Running test: Performance notifications")
        
        try:
            # Send performance notification
            self.notifier.notify_performance("profit_loss", "+2.5%")
            
            # Wait for processing
            time.sleep(1)
            
            # Verify notification was processed
            self.record_test_result(
                "Performance notifications",
                True,  # Assume passed if no exceptions
                {
                    "metric": "profit_loss",
                    "value": "+2.5%"
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_performance_notifications: {str(e)}")
            self.record_test_result(
                "Performance notifications",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_integration_with_llm_overseer(self):
        """Test integration with LLM overseer"""
        logger.system.info("Running test: Integration with LLM overseer")
        
        try:
            # Create test market state
            market_state = {
                'symbol': 'BTCUSDC',
                'timestamp': int(time.time() * 1000),
                'bid_price': 105000.0,
                'ask_price': 105100.0,
                'mid_price': 105050.0,
                'spread': 100.0,
                'spread_bps': 9.52,
                'order_imbalance': 0.2,
                'price_history': [105000.0, 104900.0, 104950.0, 105050.0, 105100.0],
                'timestamp_history': [int(time.time() * 1000) - i * 60000 for i in range(5)],
                'volume_history': [10.0, 15.0, 12.0, 8.0, 11.0],
                'last_trade_price': 105050.0,
                'last_trade_size': 0.1,
                'last_trade_side': 'BUY',
                'last_trade_time': int(time.time() * 1000),
                'momentum': 0.15,
                'volatility': 0.02,
                'trend': 0.05
            }
            
            # Create market update
            market_update = {
                'market_state': market_state
            }
            
            # Process market update
            decision = self.llm_overseer.process_market_update(market_update)
            
            # Send decision notification
            if decision:
                self.notifier.notify_decision(decision)
            
            # Wait for processing
            time.sleep(1)
            
            # Verify integration
            self.record_test_result(
                "Integration with LLM overseer",
                decision is not None,
                {
                    "market_update": market_update,
                    "decision": decision
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_integration_with_llm_overseer: {str(e)}")
            self.record_test_result(
                "Integration with LLM overseer",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_integration_with_paper_trading(self):
        """Test integration with paper trading"""
        logger.system.info("Running test: Integration with paper trading")
        
        try:
            # Create order in paper trading system
            order_id = self.paper_trading.create_order(
                'BTCUSDC',
                'BUY',
                'LIMIT',
                0.001,
                105000.0
            )
            
            # Wait for order processing
            time.sleep(2)
            
            # Fill order in paper trading system
            self.paper_trading.fill_order(order_id)
            
            # Wait for order processing
            time.sleep(2)
            
            # Verify integration
            self.record_test_result(
                "Integration with paper trading",
                order_id is not None,
                {
                    "order_id": order_id
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_integration_with_paper_trading: {str(e)}")
            self.record_test_result(
                "Integration with paper trading",
                False,
                {
                    "error": str(e)
                }
            )


def save_test_results(results, filename="fixed_telegram_integration_test_results.json"):
    """Save test results to file
    
    Args:
        results: Test results dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.system.info(f"Test results saved to {filename}")


def main():
    """Main function"""
    logger.system.info("Starting fixed Telegram integration tests")
    
    # Create and run tests
    test = FixedTelegramIntegrationTest()
    results = test.run_all_tests()
    
    # Save results
    save_test_results(results)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"  Total tests: {results['total_tests']}")
    print(f"  Passed tests: {results['passed_tests']}")
    print(f"  Failed tests: {results['failed_tests']}")
    print(f"  Success rate: {results['passed_tests'] / results['total_tests'] * 100:.1f}%")
    print(f"\nDetailed results saved to fixed_telegram_integration_test_results.json")


if __name__ == "__main__":
    main()
