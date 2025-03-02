#!/usr/bin/env python
"""
Rust Integration Tests

This module tests the integration between Python and Rust components,
verifying that the Rust implementations work correctly and produce
results consistent with the Python implementations.
"""

import unittest
import logging
import random
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pure Python implementations for comparison
def py_sma(values: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average in pure Python."""
    if len(values) < period:
        return []
    
    result = []
    for i in range(len(values) - period + 1):
        window = values[i:i+period]
        avg = sum(window) / period
        result.append(avg)
    
    return result

def py_ema(values: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average in pure Python."""
    if len(values) < period:
        return []
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Initialize with SMA
    result = [sum(values[:period]) / period]
    
    # Calculate EMA
    for i in range(period, len(values)):
        ema = (values[i] * multiplier) + (result[-1] * (1 - multiplier))
        result.append(ema)
    
    return result

# Generate test data
def generate_price_data(length: int, volatility: float = 0.01) -> List[float]:
    """Generate random price data for testing."""
    prices = [100.0]  # Start with $100
    
    for _ in range(length - 1):
        change_pct = random.normalvariate(0, volatility)
        new_price = prices[-1] * (1 + change_pct)
        prices.append(new_price)
    
    return prices


class TestRustIntegration(unittest.TestCase):
    """Test the integration with Rust components."""
    
    def setUp(self):
        """Set up the test environment."""
        self.prices = generate_price_data(1000)
        self.periods = [10, 20, 50, 100]
        
        # Try to import Rust components
        try:
            from src.rust_bridge import Technical, is_rust_available
            self.rust_available = is_rust_available()
            if self.rust_available:
                self.Technical = Technical
                logger.info("Rust components are available for testing")
            else:
                logger.warning("Rust components are not available, skipping Rust tests")
        except ImportError:
            self.rust_available = False
            logger.warning("Failed to import Rust components, skipping Rust tests")
    
    def test_sma_calculation(self):
        """Test SMA calculation with Rust vs Python."""
        if not self.rust_available:
            self.skipTest("Rust components not available")
        
        for period in self.periods:
            # Calculate using Python
            py_result = py_sma(self.prices, period)
            
            # Calculate using Rust
            rust_result = self.Technical.sma(self.prices, period)
            
            # Check that results are the same length
            self.assertEqual(len(py_result), len(rust_result),
                            f"SMA results for period {period} have different lengths")
            
            # Check that results are close (account for floating point differences)
            for i, (py_val, rust_val) in enumerate(zip(py_result, rust_result)):
                self.assertAlmostEqual(py_val, rust_val, places=6,
                                      msg=f"SMA values at index {i} for period {period} differ")
            
            logger.info(f"SMA with period {period}: Python and Rust results match")
    
    def test_ema_calculation(self):
        """Test EMA calculation with Rust vs Python."""
        if not self.rust_available:
            self.skipTest("Rust components not available")
        
        for period in self.periods:
            # Calculate using Python
            py_result = py_ema(self.prices, period)
            
            # Calculate using Rust
            rust_result = self.Technical.ema(self.prices, period)
            
            # Check that results are the same length
            self.assertEqual(len(py_result), len(rust_result),
                            f"EMA results for period {period} have different lengths")
            
            # Check that results are close (account for floating point differences)
            for i, (py_val, rust_val) in enumerate(zip(py_result, rust_result)):
                self.assertAlmostEqual(py_val, rust_val, places=6,
                                      msg=f"EMA values at index {i} for period {period} differ")
            
            logger.info(f"EMA with period {period}: Python and Rust results match")
    
    def test_streaming_sma(self):
        """Test streaming SMA calculation."""
        if not self.rust_available:
            self.skipTest("Rust components not available")
        
        from src.analysis_agents.technical.indicators_rust import SMA
        
        for period in self.periods:
            # Python implementation for streaming SMA
            values = []
            py_results = []
            
            # Rust implementation
            rust_sma = SMA(period)
            rust_results = []
            
            # Process each price
            for price in self.prices:
                # Python calculation
                values.append(price)
                if len(values) > period:
                    values.pop(0)
                
                if len(values) == period:
                    py_results.append(sum(values) / period)
                else:
                    py_results.append(None)
                
                # Rust calculation
                rust_val = rust_sma.update(price)
                rust_results.append(rust_val)
            
            # Compare results (skipping None values)
            for i, (py_val, rust_val) in enumerate(zip(py_results, rust_results)):
                if py_val is not None and rust_val is not None:
                    self.assertAlmostEqual(py_val, rust_val, places=6,
                                         msg=f"Streaming SMA values at index {i} differ")
            
            logger.info(f"Streaming SMA with period {period}: Python and Rust results match")
    
    def test_streaming_ema(self):
        """Test streaming EMA calculation."""
        if not self.rust_available:
            self.skipTest("Rust components not available")
        
        from src.analysis_agents.technical.indicators_rust import EMA
        
        for period in self.periods:
            # Python implementation
            values = []
            py_results = []
            current_ema = None
            multiplier = 2 / (period + 1)
            
            # Rust implementation
            rust_ema = EMA(period)
            rust_results = []
            
            # Process each price
            for price in self.prices:
                # Python calculation
                values.append(price)
                
                if len(values) < period:
                    py_results.append(None)
                elif len(values) == period:
                    # Initialize with SMA
                    current_ema = sum(values) / period
                    py_results.append(current_ema)
                else:
                    # Update EMA
                    current_ema = (price * multiplier) + (current_ema * (1 - multiplier))
                    py_results.append(current_ema)
                
                # Rust calculation
                rust_val = rust_ema.update(price)
                rust_results.append(rust_val)
            
            # Compare results (skipping None values)
            for i, (py_val, rust_val) in enumerate(zip(py_results, rust_results)):
                if py_val is not None and rust_val is not None:
                    self.assertAlmostEqual(py_val, rust_val, places=6,
                                         msg=f"Streaming EMA values at index {i} differ")
            
            logger.info(f"Streaming EMA with period {period}: Python and Rust results match")
    
    def test_crossover_detection(self):
        """Test MA crossover detection."""
        if not self.rust_available:
            self.skipTest("Rust components not available")
        
        # Generate fast and slow MAs
        fast_period = 10
        slow_period = 30
        
        # Calculate using Python
        fast_ma = py_ema(self.prices, fast_period)
        slow_ma = py_ema(self.prices, slow_period)
        
        # Detect crossovers in Python
        py_signals = []
        prev_diff = None
        
        for i in range(len(fast_ma)):
            if i >= len(slow_ma):
                break
                
            diff = fast_ma[i] - slow_ma[i]
            
            if prev_diff is not None:
                if prev_diff < 0 and diff >= 0:
                    py_signals.append("bullish_crossover")
                elif prev_diff > 0 and diff <= 0:
                    py_signals.append("bearish_crossover")
                else:
                    py_signals.append("no_signal")
            else:
                py_signals.append("no_signal")
                
            prev_diff = diff
        
        # Detect crossovers using Rust
        rust_signals = self.Technical.detect_crossover(fast_ma, slow_ma)
        
        # Check that results match
        self.assertEqual(len(py_signals), len(rust_signals),
                        "Crossover signal lists have different lengths")
        
        for i, (py_signal, rust_signal) in enumerate(zip(py_signals, rust_signals)):
            self.assertEqual(py_signal, rust_signal,
                            f"Crossover signals at index {i} differ")
        
        logger.info("MA Crossover detection: Python and Rust results match")
    
    def test_streaming_crossover(self):
        """Test streaming MA crossover detection."""
        if not self.rust_available:
            self.skipTest("Rust components not available")
        
        from src.analysis_agents.technical.indicators_rust import MACrossover
        
        # Initialize Python tracking variables
        fast_values = []
        slow_values = []
        fast_period = 10
        slow_period = 30
        fast_ema = None
        slow_ema = None
        fast_multiplier = 2 / (fast_period + 1)
        slow_multiplier = 2 / (slow_period + 1)
        prev_diff = None
        py_signals = []
        
        # Initialize Rust crossover detector
        rust_crossover = MACrossover(
            fast_period=fast_period,
            slow_period=slow_period,
            fast_type="exponential",
            slow_type="exponential"
        )
        rust_signals = []
        
        # Process each price
        for price in self.prices:
            # Python calculation
            fast_values.append(price)
            slow_values.append(price)
            
            # Update fast EMA
            if len(fast_values) == fast_period:
                fast_ema = sum(fast_values) / fast_period
            elif len(fast_values) > fast_period:
                fast_values.pop(0)
                fast_ema = (price * fast_multiplier) + (fast_ema * (1 - fast_multiplier))
            
            # Update slow EMA
            if len(slow_values) == slow_period:
                slow_ema = sum(slow_values) / slow_period
            elif len(slow_values) > slow_period:
                slow_values.pop(0)
                slow_ema = (price * slow_multiplier) + (slow_ema * (1 - slow_multiplier))
            
            # Detect crossover
            if fast_ema is not None and slow_ema is not None:
                diff = fast_ema - slow_ema
                
                if prev_diff is not None:
                    if prev_diff < 0 and diff >= 0:
                        py_signals.append("bullish_crossover")
                    elif prev_diff > 0 and diff <= 0:
                        py_signals.append("bearish_crossover")
                    else:
                        py_signals.append("no_signal")
                else:
                    py_signals.append("no_signal")
                    
                prev_diff = diff
            else:
                py_signals.append("no_signal")
            
            # Rust calculation
            rust_signal = rust_crossover.update(price)
            rust_signals.append(rust_signal)
        
        # We need to account for initial periods where signals might differ due to initialization
        # So we'll skip the first slow_period entries
        for i in range(slow_period, len(py_signals)):
            self.assertEqual(py_signals[i], rust_signals[i],
                            f"Streaming crossover signals at index {i} differ")
        
        logger.info("Streaming MA Crossover detection: Python and Rust results match")
    
    def test_rust_availability(self):
        """Test that we can detect if Rust is available."""
        try:
            from src.rust_bridge import is_rust_available, version
            
            available = is_rust_available()
            logger.info(f"Rust available: {available}")
            
            if available:
                ver = version()
                logger.info(f"Rust version: {ver}")
                self.assertIsNotNone(ver, "Version should not be None when Rust is available")
            
        except ImportError as e:
            logger.warning(f"Import error: {e}")
            self.skipTest("Rust bridge could not be imported")


if __name__ == "__main__":
    unittest.main() 