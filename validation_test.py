#!/usr/bin/env python
"""
Validation and Testing Script for Enhanced Flash Trading Signals

This script performs comprehensive validation and testing of the enhanced
flash trading signals implementation, including technical indicators,
multi-timeframe analysis, dynamic thresholding, and liquidity awareness.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from threading import Thread, Event

# Import modules to test
from indicators import TechnicalIndicators
from enhanced_flash_trading_signals import EnhancedFlashTradingSignals, EnhancedMarketState
from optimized_mexc_client import OptimizedMEXCClient
from error_handling_utils import safe_get, log_exception

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("validation_test")

class ValidationTest:
    """Validation and testing for enhanced flash trading signals"""
    
    def __init__(self, env_path=None):
        """Initialize validation test
        
        Args:
            env_path: Path to .env file
        """
        self.env_path = env_path
        self.api_client = None
        self.signal_generator = None
        self.test_results = {
            "indicators": {},
            "market_state": {},
            "signals": {},
            "decisions": {},
            "performance": {},
            "errors": []
        }
        self.stop_event = Event()
    
    def setup(self):
        """Set up test environment"""
        try:
            logger.info("Setting up test environment")
            
            # Initialize API client
            self.api_client = OptimizedMEXCClient(env_path=self.env_path)
            
            # Initialize signal generator
            self.signal_generator = EnhancedFlashTradingSignals(client_instance=self.api_client)
            
            logger.info("Test environment set up successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up test environment: {str(e)}")
            self.test_results["errors"].append({
                "phase": "setup",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            })
            return False
    
    def test_indicators(self):
        """Test technical indicators implementation"""
        try:
            logger.info("Testing technical indicators")
            
            # Generate sample data
            np.random.seed(42)
            n = 200
            close_prices = np.cumsum(np.random.normal(0, 1, n)) + 100
            high_prices = close_prices + np.random.uniform(0, 2, n)
            low_prices = close_prices - np.random.uniform(0, 2, n)
            volumes = np.random.uniform(1000, 5000, n)
            timestamps = np.array([1622505600000 + i * 60000 for i in range(n)])  # 1-minute intervals
            
            # Test individual indicators
            indicators_to_test = [
                ("sma", lambda: TechnicalIndicators.calculate_sma(close_prices, period=20)),
                ("ema", lambda: TechnicalIndicators.calculate_ema(close_prices, period=20)),
                ("rsi", lambda: TechnicalIndicators.calculate_rsi(close_prices)),
                ("macd", lambda: TechnicalIndicators.calculate_macd(close_prices)),
                ("bollinger_bands", lambda: TechnicalIndicators.calculate_bollinger_bands(close_prices)),
                ("vwap", lambda: TechnicalIndicators.calculate_vwap(close_prices, volumes, reset_daily=True, timestamp=timestamps)),
                ("atr", lambda: TechnicalIndicators.calculate_atr(high_prices, low_prices, close_prices))
            ]
            
            for name, func in indicators_to_test:
                start_time = time.time()
                result = func()
                end_time = time.time()
                
                self.test_results["indicators"][name] = {
                    "success": result is not None,
                    "execution_time": end_time - start_time,
                    "sample_result": result
                }
                
                logger.info(f"Indicator {name}: {'Success' if result is not None else 'Failed'} in {end_time - start_time:.6f} seconds")
            
            # Test calculate_all_indicators
            market_data = {
                'close': close_prices,
                'high': high_prices,
                'low': low_prices,
                'volume': volumes,
                'timestamp': timestamps
            }
            
            start_time = time.time()
            all_indicators = TechnicalIndicators.calculate_all_indicators(market_data)
            end_time = time.time()
            
            self.test_results["indicators"]["all_indicators"] = {
                "success": len(all_indicators) > 0,
                "execution_time": end_time - start_time,
                "indicator_count": len(all_indicators)
            }
            
            logger.info(f"All indicators: {'Success' if len(all_indicators) > 0 else 'Failed'} in {end_time - start_time:.6f} seconds")
            
            # Test error handling
            empty_result = TechnicalIndicators.calculate_rsi(np.array([]))
            self.test_results["indicators"]["error_handling"] = {
                "empty_array": empty_result is None
            }
            
            logger.info("Technical indicators testing completed")
            return True
        except Exception as e:
            logger.error(f"Error testing technical indicators: {str(e)}")
            self.test_results["errors"].append({
                "phase": "test_indicators",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            })
            return False
    
    def test_market_state(self):
        """Test enhanced market state implementation"""
        try:
            logger.info("Testing enhanced market state")
            
            # Create sample market state
            market_state = EnhancedMarketState("BTCUSDC")
            
            # Generate sample order book
            bids = [[str(100 - i), str(1.0)] for i in range(10)]
            asks = [[str(100 + i), str(0.5)] for i in range(10)]
            
            # Test update_order_book
            start_time = time.time()
            update_result = market_state.update_order_book(bids, asks)
            end_time = time.time()
            
            self.test_results["market_state"]["update_order_book"] = {
                "success": update_result,
                "execution_time": end_time - start_time
            }
            
            logger.info(f"Market state update: {'Success' if update_result else 'Failed'} in {end_time - start_time:.6f} seconds")
            
            # Test multi-timeframe updates
            for i in range(100):
                # Simulate price changes
                bid_change = np.random.normal(0, 0.1)
                ask_change = np.random.normal(0, 0.1)
                
                bids = [[str(float(bids[j][0]) + bid_change), str(1.0)] for j in range(10)]
                asks = [[str(float(asks[j][0]) + ask_change), str(0.5)] for j in range(10)]
                
                market_state.update_order_book(bids, asks)
                
                # Simulate time passing
                market_state.timestamp += 60000  # 1 minute
            
            # Check if timeframes were updated
            timeframes = ['1m', '5m', '15m', '1h']
            for tf in timeframes:
                self.test_results["market_state"][f"timeframe_{tf}"] = {
                    "data_points": len(market_state.price_history[tf]),
                    "has_indicators": len(market_state.indicators[tf]) > 0 if market_state.indicators[tf] else False
                }
                
                logger.info(f"Timeframe {tf}: {len(market_state.price_history[tf])} data points, indicators: {'Yes' if market_state.indicators[tf] else 'No'}")
            
            # Test slippage calculation
            self.test_results["market_state"]["slippage"] = {
                "calculated": market_state.estimated_slippage is not None,
                "value": market_state.estimated_slippage
            }
            
            logger.info(f"Slippage calculation: {'Success' if market_state.estimated_slippage is not None else 'Failed'}, value: {market_state.estimated_slippage}")
            
            logger.info("Enhanced market state testing completed")
            return True
        except Exception as e:
            logger.error(f"Error testing enhanced market state: {str(e)}")
            self.test_results["errors"].append({
                "phase": "test_market_state",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            })
            return False
    
    def test_signal_generation(self):
        """Test signal generation with real data"""
        try:
            logger.info("Testing signal generation with real data")
            
            if not self.api_client or not self.signal_generator:
                logger.error("API client or signal generator not initialized")
                return False
            
            # Test symbols
            symbols = ["BTCUSDC", "ETHUSDC"]
            
            # Start signal generator
            self.signal_generator.start(symbols)
            
            # Test signal generation for each symbol
            for symbol in symbols:
                start_time = time.time()
                signals = self.signal_generator.generate_signals(symbol)
                end_time = time.time()
                
                self.test_results["signals"][symbol] = {
                    "count": len(signals),
                    "execution_time": end_time - start_time,
                    "signals": signals[:5]  # Store first 5 signals for review
                }
                
                logger.info(f"Signal generation for {symbol}: {len(signals)} signals in {end_time - start_time:.6f} seconds")
                
                # Test trading decision
                start_time = time.time()
                decision = self.signal_generator.make_trading_decision(symbol)
                end_time = time.time()
                
                self.test_results["decisions"][symbol] = {
                    "has_decision": decision is not None,
                    "execution_time": end_time - start_time,
                    "decision": decision
                }
                
                logger.info(f"Trading decision for {symbol}: {'Decision made' if decision else 'No decision'} in {end_time - start_time:.6f} seconds")
            
            # Test dynamic thresholding
            start_time = time.time()
            self.signal_generator.update_dynamic_thresholds()
            end_time = time.time()
            
            self.test_results["signals"]["dynamic_thresholds"] = {
                "execution_time": end_time - start_time,
                "thresholds": self.signal_generator.dynamic_thresholds
            }
            
            logger.info(f"Dynamic thresholding: Updated in {end_time - start_time:.6f} seconds")
            
            # Stop signal generator
            self.signal_generator.stop()
            
            logger.info("Signal generation testing completed")
            return True
        except Exception as e:
            logger.error(f"Error testing signal generation: {str(e)}")
            self.test_results["errors"].append({
                "phase": "test_signal_generation",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            })
            
            # Ensure signal generator is stopped
            if self.signal_generator and self.signal_generator.running:
                self.signal_generator.stop()
                
            return False
    
    def test_performance(self):
        """Test performance of enhanced signal generation"""
        try:
            logger.info("Testing performance")
            
            if not self.api_client or not self.signal_generator:
                logger.error("API client or signal generator not initialized")
                return False
            
            # Test symbols
            symbols = ["BTCUSDC", "ETHUSDC"]
            
            # Start signal generator
            self.signal_generator.start(symbols)
            
            # Performance metrics
            iterations = 10
            signal_times = []
            decision_times = []
            
            # Run multiple iterations
            for i in range(iterations):
                for symbol in symbols:
                    # Measure signal generation time
                    start_time = time.time()
                    self.signal_generator.generate_signals(symbol)
                    signal_time = time.time() - start_time
                    signal_times.append(signal_time)
                    
                    # Measure decision making time
                    start_time = time.time()
                    self.signal_generator.make_trading_decision(symbol)
                    decision_time = time.time() - start_time
                    decision_times.append(decision_time)
            
            # Calculate statistics
            self.test_results["performance"]["signal_generation"] = {
                "min": min(signal_times),
                "max": max(signal_times),
                "avg": sum(signal_times) / len(signal_times),
                "iterations": len(signal_times)
            }
            
            self.test_results["performance"]["decision_making"] = {
                "min": min(decision_times),
                "max": max(decision_times),
                "avg": sum(decision_times) / len(decision_times),
                "iterations": len(decision_times)
            }
            
            logger.info(f"Signal generation performance: Avg {self.test_results['performance']['signal_generation']['avg']:.6f} seconds")
            logger.info(f"Decision making performance: Avg {self.test_results['performance']['decision_making']['avg']:.6f} seconds")
            
            # Stop signal generator
            self.signal_generator.stop()
            
            logger.info("Performance testing completed")
            return True
        except Exception as e:
            logger.error(f"Error testing performance: {str(e)}")
            self.test_results["errors"].append({
                "phase": "test_performance",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            })
            
            # Ensure signal generator is stopped
            if self.signal_generator and self.signal_generator.running:
                self.signal_generator.stop()
                
            return False
    
    def run_extended_test(self, duration=60):
        """Run extended test for a specified duration
        
        Args:
            duration: Test duration in seconds
        """
        try:
            logger.info(f"Running extended test for {duration} seconds")
            
            if not self.api_client or not self.signal_generator:
                logger.error("API client or signal generator not initialized")
                return False
            
            # Test symbols
            symbols = ["BTCUSDC", "ETHUSDC"]
            
            # Start signal generator
            self.signal_generator.start(symbols)
            
            # Initialize results
            self.test_results["extended_test"] = {
                "duration": duration,
                "signals": {symbol: [] for symbol in symbols},
                "decisions": {symbol: [] for symbol in symbols},
                "start_time": int(time.time() * 1000),
                "end_time": None
            }
            
            # Run for specified duration
            start_time = time.time()
            while time.time() - start_time < duration and not self.stop_event.is_set():
                for symbol in symbols:
                    # Generate signals
                    signals = self.signal_generator.generate_signals(symbol)
                    
                    # Make trading decision
                    decision = self.signal_generator.make_trading_decision(symbol)
                    
                    # Store results
                    if signals:
                        self.test_results["extended_test"]["signals"][symbol].extend(signals)
                    
                    if decision:
                        self.test_results["extended_test"]["decisions"][symbol].append(decision)
                
                # Sleep for a short interval
                time.sleep(1.0)
            
            # Record end time
            self.test_results["extended_test"]["end_time"] = int(time.time() * 1000)
            
            # Calculate statistics
            for symbol in symbols:
                signal_count = len(self.test_results["extended_test"]["signals"][symbol])
                decision_count = len(self.test_results["extended_test"]["decisions"][symbol])
                
                logger.info(f"Extended test for {symbol}: {signal_count} signals, {decision_count} decisions")
                
                # Limit stored signals and decisions to prevent excessive memory usage
                if signal_count > 100:
                    self.test_results["extended_test"]["signals"][symbol] = self.test_results["extended_test"]["signals"][symbol][-100:]
                
                if decision_count > 100:
                    self.test_results["extended_test"]["decisions"][symbol] = self.test_results["extended_test"]["decisions"][symbol][-100:]
            
            # Stop signal generator
            self.signal_generator.stop()
            
            logger.info("Extended test completed")
            return True
        except Exception as e:
            logger.error(f"Error in extended test: {str(e)}")
            self.test_results["errors"].append({
                "phase": "run_extended_test",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            })
            
            # Ensure signal generator is stopped
            if self.signal_generator and self.signal_generator.running:
                self.signal_generator.stop()
                
            return False
    
    def save_results(self, output_file="validation_test_results.json"):
        """Save test results to file
        
        Args:
            output_file: Output file path
        """
        try:
            logger.info(f"Saving test results to {output_file}")
            
            # Add timestamp
            self.test_results["timestamp"] = int(time.time() * 1000)
            
            # Save to file
            with open(output_file, "w") as f:
                json.dump(self.test_results, f, indent=2)
            
            logger.info("Test results saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving test results: {str(e)}")
            return False
    
    def run_all_tests(self, extended_duration=60):
        """Run all validation tests
        
        Args:
            extended_duration: Duration for extended test in seconds
        """
        try:
            logger.info("Running all validation tests")
            
            # Set up test environment
            if not self.setup():
                logger.error("Failed to set up test environment")
                return False
            
            # Run tests
            self.test_indicators()
            self.test_market_state()
            self.test_signal_generation()
            self.test_performance()
            self.run_extended_test(extended_duration)
            
            # Save results
            self.save_results()
            
            logger.info("All validation tests completed")
            return True
        except Exception as e:
            logger.error(f"Error running all tests: {str(e)}")
            return False

# Main function
def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validation and Testing for Enhanced Flash Trading Signals')
    parser.add_argument('--env', type=str, help='Path to .env file')
    parser.add_argument('--duration', type=int, default=60, help='Duration for extended test in seconds')
    parser.add_argument('--output', type=str, default='validation_test_results.json', help='Output file for test results')
    
    args = parser.parse_args()
    
    # Run validation tests
    validation = ValidationTest(env_path=args.env)
    validation.run_all_tests(extended_duration=args.duration)
    validation.save_results(args.output)

if __name__ == "__main__":
    main()
