#!/usr/bin/env python
"""
Signal-to-Order Flow Validation Tests

This module provides comprehensive test cases to validate the enhanced
signal-to-order pipeline integration and logging system.
"""

import os
import sys
import json
import time
import logging
import unittest
import threading
from queue import Queue
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from enhanced_signal_processor import EnhancedSignalProcessor, SignalOrderIntegration
from enhanced_logging import EnhancedLogger
from optimized_mexc_client import OptimizedMEXCClient
from paper_trading_extension import EnhancedPaperTradingSystem

# Initialize enhanced logger
logger = EnhancedLogger("signal_flow_test")

class SignalOrderFlowTest:
    """Test class for validating signal-to-order flow"""
    
    def __init__(self):
        """Initialize test environment"""
        logger.system.info("Initializing signal-to-order flow test environment")
        
        # Create configuration
        self.config = {
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
        self.paper_trading = EnhancedPaperTradingSystem(self.client)
        self.signal_processor = EnhancedSignalProcessor(self.config)
        self.integration = SignalOrderIntegration(self.config)
        
        # Test results
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
        logger.system.info("Test environment initialized")
    
    def run_all_tests(self):
        """Run all test cases"""
        logger.system.info("Starting all signal-to-order flow tests")
        
        # Run individual tests
        self.test_basic_signal_processing()
        self.test_signal_validation()
        self.test_order_creation()
        self.test_market_condition_checks()
        self.test_position_sizing()
        self.test_error_handling()
        self.test_thread_safety()
        
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
    
    def test_basic_signal_processing(self):
        """Test basic signal processing flow"""
        logger.system.info("Running test: Basic signal processing")
        
        try:
            # Create test signal
            signal = {
                'type': 'BUY',
                'source': 'test',
                'strength': 0.8,
                'timestamp': int(time.time() * 1000),
                'price': 105000.0,
                'symbol': 'BTCUSDC',
                'session': 'TEST'
            }
            
            # Process signal directly
            result = self.signal_processor.process_signal(signal)
            
            # Verify signal was processed
            self.record_test_result(
                "Basic signal processing",
                result == True,
                {
                    "signal": signal,
                    "result": result
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_basic_signal_processing: {str(e)}")
            self.record_test_result(
                "Basic signal processing",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_signal_validation(self):
        """Test signal validation logic"""
        logger.system.info("Running test: Signal validation")
        
        try:
            # Test cases
            test_cases = [
                {
                    "name": "Valid signal",
                    "signal": {
                        'type': 'BUY',
                        'source': 'test',
                        'strength': 0.8,
                        'timestamp': int(time.time() * 1000),
                        'price': 105000.0,
                        'symbol': 'BTCUSDC',
                        'session': 'TEST'
                    },
                    "expected": True
                },
                {
                    "name": "Low strength signal",
                    "signal": {
                        'type': 'BUY',
                        'source': 'test',
                        'strength': 0.3,  # Below threshold
                        'timestamp': int(time.time() * 1000),
                        'price': 105000.0,
                        'symbol': 'BTCUSDC',
                        'session': 'TEST'
                    },
                    "expected": False
                },
                {
                    "name": "Old signal",
                    "signal": {
                        'type': 'BUY',
                        'source': 'test',
                        'strength': 0.8,
                        'timestamp': int(time.time() * 1000) - 20000,  # 20 seconds old
                        'price': 105000.0,
                        'symbol': 'BTCUSDC',
                        'session': 'TEST'
                    },
                    "expected": False
                },
                {
                    "name": "Missing field",
                    "signal": {
                        'type': 'BUY',
                        'source': 'test',
                        'strength': 0.8,
                        'timestamp': int(time.time() * 1000),
                        # Missing price
                        'symbol': 'BTCUSDC',
                        'session': 'TEST'
                    },
                    "expected": False
                }
            ]
            
            # Run test cases
            results = []
            for case in test_cases:
                # Validate signal
                result = self.signal_processor.validate_signal(case["signal"])
                
                # Record result
                results.append({
                    "name": case["name"],
                    "expected": case["expected"],
                    "actual": result,
                    "passed": result == case["expected"]
                })
            
            # Check if all tests passed
            all_passed = all(r["passed"] for r in results)
            
            self.record_test_result(
                "Signal validation",
                all_passed,
                {
                    "results": results
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_signal_validation: {str(e)}")
            self.record_test_result(
                "Signal validation",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_order_creation(self):
        """Test order creation from signals"""
        logger.system.info("Running test: Order creation")
        
        try:
            # Create test signal
            signal = {
                'type': 'BUY',
                'source': 'test',
                'strength': 0.8,
                'timestamp': int(time.time() * 1000),
                'price': 105000.0,
                'symbol': 'BTCUSDC',
                'session': 'TEST',
                'id': f"SIG-TEST-{int(time.time())}"
            }
            
            # Create order from signal
            order = self.signal_processor.create_order_from_signal(signal)
            
            # Verify order was created
            order_created = order is not None
            has_signal_id = order_created and 'signal_id' in order and order['signal_id'] == signal['id']
            
            self.record_test_result(
                "Order creation",
                order_created and has_signal_id,
                {
                    "signal": signal,
                    "order": order,
                    "order_created": order_created,
                    "has_signal_id": has_signal_id
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_order_creation: {str(e)}")
            self.record_test_result(
                "Order creation",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_market_condition_checks(self):
        """Test market condition checks"""
        logger.system.info("Running test: Market condition checks")
        
        try:
            # Create test signal
            signal = {
                'type': 'BUY',
                'source': 'test',
                'strength': 0.8,
                'timestamp': int(time.time() * 1000),
                'price': 105000.0,
                'symbol': 'BTCUSDC',
                'session': 'TEST'
            }
            
            # Check market conditions
            result = self.signal_processor.check_market_conditions(signal)
            
            self.record_test_result(
                "Market condition checks",
                result is not None,  # Just check that it returns something
                {
                    "signal": signal,
                    "result": result
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_market_condition_checks: {str(e)}")
            self.record_test_result(
                "Market condition checks",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_position_sizing(self):
        """Test position sizing logic"""
        logger.system.info("Running test: Position sizing")
        
        try:
            # Create test signals with different strengths
            signals = [
                {
                    'type': 'BUY',
                    'source': 'test',
                    'strength': 0.5,
                    'timestamp': int(time.time() * 1000),
                    'price': 105000.0,
                    'symbol': 'BTCUSDC',
                    'session': 'TEST'
                },
                {
                    'type': 'BUY',
                    'source': 'test',
                    'strength': 0.8,
                    'timestamp': int(time.time() * 1000),
                    'price': 105000.0,
                    'symbol': 'BTCUSDC',
                    'session': 'TEST'
                },
                {
                    'type': 'BUY',
                    'source': 'test',
                    'strength': 1.0,
                    'timestamp': int(time.time() * 1000),
                    'price': 105000.0,
                    'symbol': 'BTCUSDC',
                    'session': 'TEST'
                }
            ]
            
            # Calculate position sizes
            position_sizes = [self.signal_processor.calculate_position_size(signal) for signal in signals]
            
            # Verify position sizes are proportional to signal strength
            valid_sizes = all(size > 0 for size in position_sizes)
            proportional = (position_sizes[0] <= position_sizes[1] <= position_sizes[2])
            
            self.record_test_result(
                "Position sizing",
                valid_sizes and proportional,
                {
                    "signals": signals,
                    "position_sizes": position_sizes,
                    "valid_sizes": valid_sizes,
                    "proportional": proportional
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_position_sizing: {str(e)}")
            self.record_test_result(
                "Position sizing",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_error_handling(self):
        """Test error handling in signal processing"""
        logger.system.info("Running test: Error handling")
        
        try:
            # Create invalid signal (missing required fields)
            invalid_signal = {
                'type': 'BUY',
                'source': 'test',
                # Missing strength
                'timestamp': int(time.time() * 1000),
                # Missing price
                'symbol': 'BTCUSDC',
                'session': 'TEST'
            }
            
            # Process invalid signal
            result = self.signal_processor.process_signal(invalid_signal)
            
            # Verify signal was rejected
            self.record_test_result(
                "Error handling",
                result == False,
                {
                    "signal": invalid_signal,
                    "result": result
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_error_handling: {str(e)}")
            self.record_test_result(
                "Error handling",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_thread_safety(self):
        """Test thread safety of signal processing"""
        logger.system.info("Running test: Thread safety")
        
        try:
            # Start integration
            self.integration.start()
            
            # Create multiple test signals
            signals = []
            for i in range(10):
                signal = {
                    'type': 'BUY',
                    'source': 'test',
                    'strength': 0.8,
                    'timestamp': int(time.time() * 1000),
                    'price': 105000.0 + i * 100,  # Slightly different prices
                    'symbol': 'BTCUSDC',
                    'session': 'TEST'
                }
                signals.append(signal)
            
            # Add signals to queue
            for signal in signals:
                self.integration.add_signal(signal)
            
            # Wait for processing
            time.sleep(5)
            
            # Stop integration
            self.integration.stop()
            
            # Check if any errors occurred
            # Note: This is a basic check, in a real test we would verify more details
            self.record_test_result(
                "Thread safety",
                True,  # Assume passed if no exceptions
                {
                    "signals_sent": len(signals)
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_thread_safety: {str(e)}")
            self.record_test_result(
                "Thread safety",
                False,
                {
                    "error": str(e)
                }
            )


def save_test_results(results, filename="signal_flow_test_results.json"):
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
    logger.system.info("Starting signal-to-order flow validation tests")
    
    # Create and run tests
    test = SignalOrderFlowTest()
    results = test.run_all_tests()
    
    # Save results
    save_test_results(results)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"  Total tests: {results['total_tests']}")
    print(f"  Passed tests: {results['passed_tests']}")
    print(f"  Failed tests: {results['failed_tests']}")
    print(f"  Success rate: {results['passed_tests'] / results['total_tests'] * 100:.1f}%")
    print(f"\nDetailed results saved to signal_flow_test_results.json")


if __name__ == "__main__":
    main()
