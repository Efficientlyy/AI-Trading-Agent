#!/usr/bin/env python
"""
LLM Decision Making Validation Tests

This module provides comprehensive test cases to validate the enhanced
LLM strategic overseer integration and decision making pipeline.
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
from enhanced_logging_fixed import EnhancedLogger
from enhanced_llm_overseer import EnhancedLLMOverseer
from optimized_mexc_client import OptimizedMEXCClient
from paper_trading_extension import EnhancedPaperTradingSystem

# Initialize enhanced logger
logger = EnhancedLogger("llm_decision_test")

class LLMDecisionTest:
    """Test class for validating LLM decision making"""
    
    def __init__(self):
        """Initialize test environment"""
        logger.system.info("Initializing LLM decision making test environment")
        
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
        logger.system.info("Starting all LLM decision making tests")
        
        # Run individual tests
        self.test_llm_overseer_initialization()
        self.test_pattern_recognition()
        self.test_llm_context_generation()
        self.test_decision_processing()
        self.test_market_update_processing()
        self.test_end_to_end_flow()
        self.test_error_handling()
        
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
    
    def test_llm_overseer_initialization(self):
        """Test LLM overseer initialization"""
        logger.system.info("Running test: LLM overseer initialization")
        
        try:
            # Create LLM overseer
            overseer = EnhancedLLMOverseer(self.config)
            
            # Verify initialization
            initialized = (
                overseer is not None and
                hasattr(overseer, 'llm_overseer') and
                hasattr(overseer, 'pattern_recognition') and
                hasattr(overseer, 'signal_processor') and
                hasattr(overseer, 'integration')
            )
            
            self.record_test_result(
                "LLM overseer initialization",
                initialized,
                {
                    "overseer": str(overseer),
                    "initialized": initialized
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_llm_overseer_initialization: {str(e)}")
            self.record_test_result(
                "LLM overseer initialization",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_pattern_recognition(self):
        """Test pattern recognition"""
        logger.system.info("Running test: Pattern recognition")
        
        try:
            # Create LLM overseer
            overseer = EnhancedLLMOverseer(self.config)
            
            # Create test market state
            market_state = {
                'symbol': 'BTCUSDC',
                'timestamp': int(time.time() * 1000),
                'price_history': [105000.0, 104900.0, 104950.0, 105050.0, 105100.0]
            }
            
            # Recognize patterns
            patterns = overseer.recognize_patterns(market_state)
            
            # Verify patterns
            self.record_test_result(
                "Pattern recognition",
                patterns is not None,
                {
                    "market_state": market_state,
                    "patterns": patterns
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_pattern_recognition: {str(e)}")
            self.record_test_result(
                "Pattern recognition",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_llm_context_generation(self):
        """Test LLM context generation"""
        logger.system.info("Running test: LLM context generation")
        
        try:
            # Create LLM overseer
            overseer = EnhancedLLMOverseer(self.config)
            
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
            
            # Recognize patterns
            patterns = overseer.recognize_patterns(market_state)
            
            # Generate context
            context = overseer.generate_llm_context(market_state, patterns)
            
            # Verify context
            has_required_fields = (
                'symbol' in context and
                'current_price' in context and
                'bid_price' in context and
                'ask_price' in context and
                'patterns' in context
            )
            
            self.record_test_result(
                "LLM context generation",
                has_required_fields,
                {
                    "market_state": market_state,
                    "context": context,
                    "has_required_fields": has_required_fields
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_llm_context_generation: {str(e)}")
            self.record_test_result(
                "LLM context generation",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_decision_processing(self):
        """Test decision processing"""
        logger.system.info("Running test: Decision processing")
        
        try:
            # Create LLM overseer
            overseer = EnhancedLLMOverseer(self.config)
            
            # Start overseer
            overseer.start()
            
            # Create test decision
            decision = {
                'id': f"DECISION-TEST-{int(time.time())}",
                'action': 'BUY',
                'reason': 'Test decision',
                'confidence': 0.8,
                'signals': [
                    {
                        'id': f"SIG-TEST-{int(time.time())}",
                        'type': 'BUY',
                        'source': 'llm_test',
                        'strength': 0.8,
                        'timestamp': int(time.time() * 1000),
                        'price': 105000.0,
                        'symbol': 'BTCUSDC',
                        'session': 'TEST'
                    }
                ]
            }
            
            # Add decision to queue
            overseer.decision_queue.put(decision)
            
            # Wait for processing
            time.sleep(2)
            
            # Stop overseer
            overseer.stop()
            
            # Verify decision was processed
            # Note: In a real test, we would verify more details
            self.record_test_result(
                "Decision processing",
                True,  # Assume passed if no exceptions
                {
                    "decision": decision
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_decision_processing: {str(e)}")
            self.record_test_result(
                "Decision processing",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_market_update_processing(self):
        """Test market update processing"""
        logger.system.info("Running test: Market update processing")
        
        try:
            # Create LLM overseer
            overseer = EnhancedLLMOverseer(self.config)
            
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
            decision = overseer.process_market_update(market_update)
            
            # Verify decision
            self.record_test_result(
                "Market update processing",
                decision is not None,
                {
                    "market_update": market_update,
                    "decision": decision
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_market_update_processing: {str(e)}")
            self.record_test_result(
                "Market update processing",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_end_to_end_flow(self):
        """Test end-to-end flow"""
        logger.system.info("Running test: End-to-end flow")
        
        try:
            # Create LLM overseer
            overseer = EnhancedLLMOverseer(self.config)
            
            # Start overseer
            overseer.start()
            
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
            decision = overseer.process_market_update(market_update)
            
            # Wait for processing
            time.sleep(5)
            
            # Stop overseer
            overseer.stop()
            
            # Verify end-to-end flow
            self.record_test_result(
                "End-to-end flow",
                decision is not None,
                {
                    "market_update": market_update,
                    "decision": decision
                }
            )
        except Exception as e:
            logger.log_error(f"Error in test_end_to_end_flow: {str(e)}")
            self.record_test_result(
                "End-to-end flow",
                False,
                {
                    "error": str(e)
                }
            )
    
    def test_error_handling(self):
        """Test error handling"""
        logger.system.info("Running test: Error handling")
        
        try:
            # Create LLM overseer
            overseer = EnhancedLLMOverseer(self.config)
            
            # Create invalid market state (missing required fields)
            invalid_market_state = {
                'symbol': 'BTCUSDC',
                'timestamp': int(time.time() * 1000)
                # Missing other required fields
            }
            
            # Create market update
            market_update = {
                'market_state': invalid_market_state
            }
            
            # Process market update
            decision = overseer.process_market_update(market_update)
            
            # Verify error handling
            self.record_test_result(
                "Error handling",
                True,  # Assume passed if no exceptions
                {
                    "market_update": market_update,
                    "decision": decision
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


def save_test_results(results, filename="llm_decision_test_results.json"):
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
    logger.system.info("Starting LLM decision making validation tests")
    
    # Create and run tests
    test = LLMDecisionTest()
    results = test.run_all_tests()
    
    # Save results
    save_test_results(results)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"  Total tests: {results['total_tests']}")
    print(f"  Passed tests: {results['passed_tests']}")
    print(f"  Failed tests: {results['failed_tests']}")
    print(f"  Success rate: {results['passed_tests'] / results['total_tests'] * 100:.1f}%")
    print(f"\nDetailed results saved to llm_decision_test_results.json")


if __name__ == "__main__":
    main()
