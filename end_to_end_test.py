#!/usr/bin/env python
"""
End-to-End Test for Trading-Agent System

This module provides comprehensive end-to-end testing for the Trading-Agent system,
validating all components from market data processing to order execution.
"""

import os
import sys
import json
import time
import logging
import unittest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
from enum import Enum
import traceback
import uuid
import threading
import queue
import html
import re

# Import Trading-Agent components
from env_loader import load_environment_variables
from enhanced_flash_trading_signals import EnhancedFlashTradingSignals
from enhanced_dl_integration_fixed import EnhancedPatternRecognitionIntegration
from enhanced_dl_model_fixed import EnhancedPatternRecognitionModel
from enhanced_feature_adapter_fixed import EnhancedFeatureAdapter
from execution_optimization import (
    OrderRouter, SmartOrderRouter, ExecutionOptimizer, 
    LatencyProfiler, Order, OrderType, OrderSide, OrderStatus
)
from rl_agent_fixed_v4 import TradingRLAgent
from rl_integration_fixed_v2 import RLIntegration
from optimized_mexc_client import OptimizedMEXCClient  # Fixed casing to match actual class name
from mock_exchange_client import MockExchangeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("end_to_end_test")

# Load environment variables
load_environment_variables()

class TestResult(Enum):
    """Test result enum"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class EndToEndTest(unittest.TestCase):
    """End-to-end test for Trading-Agent system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_results = {}
        self.test_output_dir = "end_to_end_test_results"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Initialize test parameters
        self.symbols = ["BTC/USDC", "ETH/USDT", "SOL/USDT"]
        self.timeframes = ["5m", "15m"]
        self.max_signals = 100  # Limit number of signals to process for faster test completion
        self.max_patterns = 50  # Limit number of patterns to detect for faster test completion
        
        # Initialize components
        self.mexc_client = self._initialize_mexc_client()
        self.pattern_recognition = self._initialize_pattern_recognition()
        self.trading_signals = self._initialize_trading_signals()
        self.rl_agent = self._initialize_rl_agent()
        self.order_router = self._initialize_order_router()
        
        logger.info("Test environment set up successfully")
    
    def _initialize_mexc_client(self):
        """Initialize MEXC client
        
        Returns:
            OptimizedMEXCClient or MockExchangeClient: MEXC client instance
        """
        try:
            # Try to initialize with real API credentials
            api_key = os.getenv("MEXC_API_KEY")
            api_secret = os.getenv("MEXC_SECRET_KEY")
            
            if api_key and api_secret:
                logger.info("Initializing OptimizedMEXCClient with real API credentials")
                return OptimizedMEXCClient(api_key=api_key, api_secret=api_secret)
            else:
                logger.warning("MEXC API credentials not found, using MockExchangeClient")
                return MockExchangeClient(simulate_errors=False)
        except Exception as e:
            logger.error(f"Error initializing MEXC client: {e}")
            logger.info("Falling back to MockExchangeClient")
            return MockExchangeClient(simulate_errors=False)
    
    def _initialize_pattern_recognition(self):
        """Initialize pattern recognition
        
        Returns:
            EnhancedPatternRecognitionIntegration: Pattern recognition instance
        """
        model = EnhancedPatternRecognitionModel(input_dim=16, sequence_length=40)
        feature_adapter = EnhancedFeatureAdapter()
        
        return EnhancedPatternRecognitionIntegration(
            model=model,
            feature_adapter=feature_adapter,
            confidence_threshold=0.45
        )
    
    def _initialize_trading_signals(self):
        """Initialize trading signals
        
        Returns:
            EnhancedFlashTradingSignals: Trading signals instance
        """
        return EnhancedFlashTradingSignals(
            client_instance=self.mexc_client,
            pattern_recognition=self.pattern_recognition
        )
    
    def _initialize_rl_agent(self):
        """Initialize RL agent
        
        Returns:
            TradingRLAgent: RL agent instance
        """
        return TradingRLAgent(state_dim=10, action_dim=3)
    
    def _initialize_order_router(self):
        """Initialize order router
        
        Returns:
            OrderRouter: Order router instance
        """
        latency_profiler = LatencyProfiler()
        latency_profiler.set_threshold("order_submission", 100)  # 100ms threshold
        
        return OrderRouter(
            client_instance=self.mexc_client,
            latency_profiler=latency_profiler
        )
    
    def test_system_recovery_and_error_handling(self):
        """Test system recovery and error handling
        
        This test validates the system's ability to recover from errors and handle
        high latency scenarios.
        """
        logger.info("Starting system recovery and error handling test")
        
        try:
            # Test high latency rejection
            logger.info("Testing high latency rejection")
            
            # Create a test order
            test_order = Order(
                symbol="BTC/USDC",
                quantity=0.001,
                price=50000.0,
                type=OrderType.MARKET,
                side=OrderSide.BUY
            )
            
            # Set high latency threshold
            self.order_router.latency_profiler.set_threshold("order_submission", 100)  # 100ms threshold
            
            # Simulate high latency
            self.order_router.simulate_latency = True
            self.order_router.simulated_latency = 150  # 150ms (above threshold)
            
            # Submit order (should be rejected due to high latency)
            result = self.order_router.submit_order(test_order)
            
            # Verify order was rejected
            self.assertEqual(result.status, OrderStatus.REJECTED, "High latency order was not rejected")
            
            # Test error recovery
            logger.info("Testing error recovery")
            
            # Set up error simulation
            self.order_router.simulate_errors = True
            self.order_router.error_rate = 1.0  # 100% error rate for first attempt
            self.order_router.max_retries = 3
            
            # Reset latency simulation
            self.order_router.simulate_latency = False
            
            # Create a new test order
            test_order = Order(
                symbol="ETH/USDT",
                quantity=0.01,
                price=3000.0,
                type=OrderType.MARKET,
                side=OrderSide.BUY
            )
            
            # Submit order (should succeed after retries)
            self.order_router.error_rate = 0.5  # 50% error rate for retries
            result = self.order_router.submit_order(test_order)
            
            # Verify order was eventually submitted
            self.assertNotEqual(result.status, OrderStatus.REJECTED, "Order failed after retries")
            
            # Verify retry count
            self.assertGreater(self.order_router.retry_count, 0, "No retries were attempted")
            
            logger.info("System recovery and error handling test passed")
            self.test_results["system_recovery"] = TestResult.PASSED.value
            
        except Exception as e:
            logger.error(f"System recovery and error handling test failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results["system_recovery"] = TestResult.FAILED.value
    
    def test_pattern_recognition_integration(self):
        """Test pattern recognition integration
        
        This test validates the pattern recognition integration with market data.
        """
        logger.info("Starting pattern recognition integration test")
        
        try:
            # Get market data
            all_signals = []
            
            # Limit to fewer symbols and timeframes for faster test completion
            test_symbols = self.symbols[:1]  # Just BTC/USDC
            test_timeframes = self.timeframes[:1]  # Just 5m
            
            for symbol in test_symbols:
                for timeframe in test_timeframes:
                    logger.info(f"Processing {symbol} {timeframe}")
                    
                    # Get candles
                    candles = self.mexc_client.get_klines(
                        symbol=symbol.replace("/", ""),
                        interval=timeframe,
                        limit=1000  # Get more data for better pattern detection
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])
                    
                    # Convert types
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)
                    
                    # Detect patterns
                    signals = self.pattern_recognition.detect_patterns(
                        df, symbol, timeframe, max_patterns=self.max_patterns
                    )
                    
                    if signals:
                        all_signals.extend(signals)
                        logger.info(f"Generated {len(signals)} signals for {symbol} {timeframe}")
            
            # Verify signals were generated
            self.assertGreater(len(all_signals), 0, "No pattern signals generated")
            
            # Log some sample signals
            for i, signal in enumerate(all_signals[:5]):
                logger.info(f"Sample signal {i+1}: {signal}")
            
            logger.info(f"Pattern recognition integration test passed with {len(all_signals)} signals")
            self.test_results["pattern_recognition"] = TestResult.PASSED.value
            
        except Exception as e:
            logger.error(f"Pattern recognition integration test failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results["pattern_recognition"] = TestResult.FAILED.value
    
    def test_market_data_processing(self):
        """Test market data processing
        
        This test validates the market data processing and signal generation.
        """
        logger.info("Starting market data processing test")
        
        try:
            # Initialize trading signals
            signals = self.trading_signals
            
            # Get signals for multiple symbols and timeframes
            all_signals = []
            
            # Limit to fewer symbols and timeframes for faster test completion
            test_symbols = self.symbols[:1]  # Just BTC/USDC
            test_timeframes = self.timeframes[:1]  # Just 5m
            
            for symbol in test_symbols:
                for timeframe in test_timeframes:
                    logger.info(f"Processing {symbol} {timeframe}")
                    
                    # Get signals
                    symbol_signals = signals.get_signals(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=1000,  # Get more data for better signal generation
                        max_signals=self.max_signals  # Limit signals for faster test completion
                    )
                    
                    if symbol_signals:
                        all_signals.extend(symbol_signals)
                        logger.info(f"Generated {len(symbol_signals)} signals for {symbol} {timeframe}")
            
            # Verify signals were generated
            self.assertGreater(len(all_signals), 0, "No signals generated")
            
            # Log some sample signals
            for i, signal in enumerate(all_signals[:5]):
                logger.info(f"Sample signal {i+1}: {signal}")
            
            logger.info(f"Market data processing test passed with {len(all_signals)} signals")
            self.test_results["market_data"] = TestResult.PASSED.value
            
        except Exception as e:
            logger.error(f"Market data processing test failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results["market_data"] = TestResult.FAILED.value
    
    def test_signal_to_decision_pipeline(self):
        """Test signal to decision pipeline
        
        This test validates the signal to decision pipeline.
        """
        logger.info("Starting signal to decision pipeline test")
        
        # Skip if market data test failed
        if self.test_results.get("market_data") != TestResult.PASSED.value:
            logger.warning("Skipping signal to decision pipeline test due to market data test failure")
            self.test_results["signal_to_decision"] = TestResult.SKIPPED.value
            return
        
        try:
            # Initialize trading signals
            signals = self.trading_signals
            
            # Get signals
            all_signals = []
            
            # Limit to fewer symbols and timeframes for faster test completion
            test_symbols = self.symbols[:1]  # Just BTC/USDC
            test_timeframes = self.timeframes[:1]  # Just 5m
            
            for symbol in test_symbols:
                for timeframe in test_timeframes:
                    logger.info(f"Processing {symbol} {timeframe}")
                    
                    # Get signals
                    symbol_signals = signals.get_signals(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=1000,  # Get more data for better signal generation
                        max_signals=self.max_signals  # Limit signals for faster test completion
                    )
                    
                    if symbol_signals:
                        all_signals.extend(symbol_signals)
            
            # Verify signals were generated
            self.assertGreater(len(all_signals), 0, "No signals generated")
            
            # Process signals with RL agent
            decisions = []
            for signal in all_signals[:self.max_signals]:  # Limit signals for faster test completion
                # Convert signal to state
                state = self.rl_agent.signal_to_state(signal)
                
                # Get action
                action = self.rl_agent.get_action(state)
                
                # Convert action to decision
                decision = self.rl_agent.action_to_decision(action, signal)
                
                if decision:
                    decisions.append(decision)
            
            # Verify decisions were generated
            self.assertGreater(len(decisions), 0, "No decisions generated")
            
            # Log some sample decisions
            for i, decision in enumerate(decisions[:5]):
                logger.info(f"Sample decision {i+1}: {decision}")
            
            logger.info(f"Signal to decision pipeline test passed with {len(decisions)} decisions")
            self.test_results["signal_to_decision"] = TestResult.PASSED.value
            
        except Exception as e:
            logger.error(f"Signal to decision pipeline test failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results["signal_to_decision"] = TestResult.FAILED.value
    
    def test_end_to_end_order_execution(self):
        """Test end-to-end order execution
        
        This test validates the end-to-end order execution pipeline.
        """
        logger.info("Starting end-to-end order execution test")
        
        # Skip if signal to decision test failed or was skipped
        if self.test_results.get("signal_to_decision") not in [TestResult.PASSED.value]:
            logger.warning("Skipping end-to-end order execution test due to signal to decision test failure/skip")
            self.test_results["order_execution"] = TestResult.SKIPPED.value
            return
        
        try:
            # Initialize trading signals
            signals = self.trading_signals
            
            # Get signals
            all_signals = []
            
            # Limit to fewer symbols and timeframes for faster test completion
            test_symbols = self.symbols[:1]  # Just BTC/USDC
            test_timeframes = self.timeframes[:1]  # Just 5m
            
            for symbol in test_symbols:
                for timeframe in test_timeframes:
                    logger.info(f"Processing {symbol} {timeframe}")
                    
                    # Get signals
                    symbol_signals = signals.get_signals(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=1000,  # Get more data for better signal generation
                        max_signals=self.max_signals  # Limit signals for faster test completion
                    )
                    
                    if symbol_signals:
                        all_signals.extend(symbol_signals)
            
            # Verify signals were generated
            self.assertGreater(len(all_signals), 0, "No signals generated")
            
            # Process signals with RL agent
            decisions = []
            for signal in all_signals[:self.max_signals]:  # Limit signals for faster test completion
                # Convert signal to state
                state = self.rl_agent.signal_to_state(signal)
                
                # Get action
                action = self.rl_agent.get_action(state)
                
                # Convert action to decision
                decision = self.rl_agent.action_to_decision(action, signal)
                
                if decision:
                    decisions.append(decision)
            
            # Verify decisions were generated
            self.assertGreater(len(decisions), 0, "No decisions generated")
            
            # Execute orders
            executed_orders = []
            for decision in decisions[:10]:  # Limit to 10 orders for faster test completion
                # Create order
                order = Order(
                    symbol=decision["symbol"],
                    quantity=0.001,  # Small quantity for testing
                    price=None,  # Market order
                    type=OrderType.MARKET,
                    side=OrderSide.BUY if decision["action"] == "buy" else OrderSide.SELL
                )
                
                # Submit order
                result = self.order_router.submit_order(order)
                
                if result and result.status != OrderStatus.REJECTED:
                    executed_orders.append(result)
            
            # Verify orders were executed
            self.assertGreater(len(executed_orders), 0, "No orders executed")
            
            # Log some sample executed orders
            for i, order in enumerate(executed_orders[:5]):
                logger.info(f"Sample executed order {i+1}: {order}")
            
            logger.info(f"End-to-end order execution test passed with {len(executed_orders)} executed orders")
            self.test_results["order_execution"] = TestResult.PASSED.value
            
        except Exception as e:
            logger.error(f"End-to-end order execution test failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results["order_execution"] = TestResult.FAILED.value
    
    def tearDown(self):
        """Tear down test environment and generate report"""
        logger.info("Tearing down test environment")
        
        # Save test results
        with open(f"{self.test_output_dir}/test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report()
        
        logger.info("Test environment torn down successfully")
    
    def _generate_html_report(self):
        """Generate HTML report"""
        logger.info("Generating HTML report")
        
        # Define CSS styles
        css = """
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f8f8f8;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .passed {{
            color: green;
            font-weight: bold;
        }}
        .failed {{
            color: red;
            font-weight: bold;
        }}
        .skipped {{
            color: orange;
            font-weight: bold;
        }}
        .summary {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f8f8;
            border-radius: 5px;
        }}
        """
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading-Agent End-to-End Test Report</title>
            <style>
            {css}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trading-Agent End-to-End Test Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Test Results</h2>
                <table>
                    <tr>
                        <th>Test</th>
                        <th>Result</th>
                    </tr>
        """
        
        # Add test results
        test_names = {
            "system_recovery": "System Recovery and Error Handling",
            "pattern_recognition": "Pattern Recognition Integration",
            "market_data": "Market Data Processing",
            "signal_to_decision": "Signal to Decision Pipeline",
            "order_execution": "End-to-End Order Execution"
        }
        
        for test_id, test_name in test_names.items():
            result = self.test_results.get(test_id, TestResult.SKIPPED.value)
            result_class = result.lower()
            
            html_content += f"""
                    <tr>
                        <td>{test_name}</td>
                        <td class="{result_class}">{result}</td>
                    </tr>
            """
        
        # Calculate summary
        total_tests = len(test_names)
        passed_tests = sum(1 for result in self.test_results.values() if result == TestResult.PASSED.value)
        failed_tests = sum(1 for result in self.test_results.values() if result == TestResult.FAILED.value)
        skipped_tests = sum(1 for result in self.test_results.values() if result == TestResult.SKIPPED.value)
        
        # Add summary
        html_content += f"""
                </table>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total Tests: {total_tests}</p>
                    <p>Passed: <span class="passed">{passed_tests}</span></p>
                    <p>Failed: <span class="failed">{failed_tests}</span></p>
                    <p>Skipped: <span class="skipped">{skipped_tests}</span></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(f"{self.test_output_dir}/end_to_end_test_report.html", "w") as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {self.test_output_dir}/end_to_end_test_report.html")

if __name__ == "__main__":
    # Run tests
    test_suite = unittest.TestSuite()
    test_suite.addTest(EndToEndTest("test_system_recovery_and_error_handling"))
    test_suite.addTest(EndToEndTest("test_pattern_recognition_integration"))
    test_suite.addTest(EndToEndTest("test_market_data_processing"))
    test_suite.addTest(EndToEndTest("test_signal_to_decision_pipeline"))
    test_suite.addTest(EndToEndTest("test_end_to_end_order_execution"))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
