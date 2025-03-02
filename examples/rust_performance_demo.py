#!/usr/bin/env python
"""
Rust Performance Demonstration for AI Crypto Trading System

This script demonstrates the performance benefits of using Rust implementations
for technical indicators and market data processing compared to pure Python.
"""

import time
import random
import logging
from typing import List, Dict, Any, Callable, Tuple
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import Rust implementations
try:
    from src.rust_bridge import Technical, is_rust_available
    from src.analysis_agents.technical.indicators_rust import SMA as RustSMA
    from src.analysis_agents.technical.indicators_rust import EMA as RustEMA
    RUST_AVAILABLE = is_rust_available()
except ImportError:
    logger.warning("Rust implementations not available. Only testing Python.")
    RUST_AVAILABLE = False

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

class PySMA:
    """Pure Python implementation of SMA for streaming data."""
    
    def __init__(self, period: int):
        self.period = period
        self.values = []
    
    def update(self, value: float) -> float:
        """Add new value and return updated SMA."""
        self.values.append(value)
        if len(self.values) > self.period:
            self.values.pop(0)
        
        if len(self.values) < self.period:
            return None
        
        return sum(self.values) / self.period
    
    def current(self) -> float:
        """Get current SMA value."""
        if len(self.values) < self.period:
            return None
        
        return sum(self.values) / self.period
    
    def reset(self) -> None:
        """Reset the calculator."""
        self.values = []

class PyEMA:
    """Pure Python implementation of EMA for streaming data."""
    
    def __init__(self, period: int):
        self.period = period
        self.values = []
        self.current_ema = None
        self.multiplier = 2 / (period + 1)
    
    def update(self, value: float) -> float:
        """Add new value and return updated EMA."""
        self.values.append(value)
        
        if len(self.values) < self.period:
            return None
        
        if self.current_ema is None:
            # Initialize with SMA
            self.current_ema = sum(self.values[-self.period:]) / self.period
        else:
            # Update EMA
            self.current_ema = (value * self.multiplier) + (self.current_ema * (1 - self.multiplier))
        
        return self.current_ema
    
    def current(self) -> float:
        """Get current EMA value."""
        return self.current_ema
    
    def reset(self) -> None:
        """Reset the calculator."""
        self.values = []
        self.current_ema = None

def benchmark(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """
    Benchmark a function call.
    
    Args:
        func: Function to benchmark
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Tuple of (execution_time, result)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return (end_time - start_time), result

def generate_price_data(length: int, volatility: float = 0.01) -> List[float]:
    """
    Generate random price data for testing.
    
    Args:
        length: Number of price points to generate
        volatility: Price volatility factor
        
    Returns:
        List of price values
    """
    prices = [100.0]  # Start with $100
    
    for _ in range(length - 1):
        change_pct = random.normalvariate(0, volatility)
        new_price = prices[-1] * (1 + change_pct)
        prices.append(new_price)
    
    return prices

def run_batch_benchmark(
    prices: List[float], 
    periods: List[int] = [10, 20, 50, 100, 200]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run batch processing benchmark for SMA and EMA.
    
    Args:
        prices: List of price values
        periods: List of periods to test
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "sma": {"python": [], "rust": [], "speedup": []},
        "ema": {"python": [], "rust": [], "speedup": []}
    }
    
    for period in periods:
        logger.info(f"Benchmarking SMA with period {period}")
        
        # Python SMA
        py_time, py_result = benchmark(py_sma, prices, period)
        logger.info(f"Python SMA took {py_time:.6f} seconds")
        results["sma"]["python"].append(py_time)
        
        # Rust SMA
        if RUST_AVAILABLE:
            rust_time, rust_result = benchmark(Technical.sma, prices, period)
            logger.info(f"Rust SMA took {rust_time:.6f} seconds")
            results["sma"]["rust"].append(rust_time)
            
            # Calculate speedup
            speedup = py_time / rust_time if rust_time > 0 else float('inf')
            logger.info(f"Rust is {speedup:.2f}x faster than Python")
            results["sma"]["speedup"].append(speedup)
            
            # Verify results match
            if len(py_result) > 0 and len(rust_result) > 0:
                error = max(abs(p - r) for p, r in zip(py_result, rust_result))
                logger.info(f"Maximum error between implementations: {error}")
        else:
            results["sma"]["rust"].append(0)
            results["sma"]["speedup"].append(0)
        
        logger.info(f"Benchmarking EMA with period {period}")
        
        # Python EMA
        py_time, py_result = benchmark(py_ema, prices, period)
        logger.info(f"Python EMA took {py_time:.6f} seconds")
        results["ema"]["python"].append(py_time)
        
        # Rust EMA
        if RUST_AVAILABLE:
            rust_time, rust_result = benchmark(Technical.ema, prices, period)
            logger.info(f"Rust EMA took {rust_time:.6f} seconds")
            results["ema"]["rust"].append(rust_time)
            
            # Calculate speedup
            speedup = py_time / rust_time if rust_time > 0 else float('inf')
            logger.info(f"Rust is {speedup:.2f}x faster than Python")
            results["ema"]["speedup"].append(speedup)
            
            # Verify results match
            if len(py_result) > 0 and len(rust_result) > 0:
                error = max(abs(p - r) for p, r in zip(py_result, rust_result))
                logger.info(f"Maximum error between implementations: {error}")
        else:
            results["ema"]["rust"].append(0)
            results["ema"]["speedup"].append(0)
    
    return results

def run_streaming_benchmark(
    prices: List[float], 
    period: int = 20,
    iterations: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Run streaming benchmark for real-time SMA and EMA calculations.
    
    Args:
        prices: List of price values to use
        period: Period for moving averages
        iterations: Number of iterations to average results
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "sma": {"python": 0, "rust": 0, "speedup": 0},
        "ema": {"python": 0, "rust": 0, "speedup": 0}
    }
    
    logger.info(f"Running streaming benchmark with {len(prices)} prices, period {period}")
    
    # SMA benchmark
    for _ in range(iterations):
        # Python SMA
        py_sma_obj = PySMA(period)
        start_time = time.time()
        for price in prices:
            py_sma_obj.update(price)
        py_time = time.time() - start_time
        results["sma"]["python"] += py_time / iterations
        
        # Rust SMA
        if RUST_AVAILABLE:
            rust_sma_obj = RustSMA(period)
            start_time = time.time()
            for price in prices:
                rust_sma_obj.update(price)
            rust_time = time.time() - start_time
            results["sma"]["rust"] += rust_time / iterations
    
    # EMA benchmark
    for _ in range(iterations):
        # Python EMA
        py_ema_obj = PyEMA(period)
        start_time = time.time()
        for price in prices:
            py_ema_obj.update(price)
        py_time = time.time() - start_time
        results["ema"]["python"] += py_time / iterations
        
        # Rust EMA
        if RUST_AVAILABLE:
            rust_ema_obj = RustEMA(period)
            start_time = time.time()
            for price in prices:
                rust_ema_obj.update(price)
            rust_time = time.time() - start_time
            results["ema"]["rust"] += rust_time / iterations
    
    # Calculate speedups
    if RUST_AVAILABLE:
        results["sma"]["speedup"] = results["sma"]["python"] / results["sma"]["rust"] if results["sma"]["rust"] > 0 else float('inf')
        results["ema"]["speedup"] = results["ema"]["python"] / results["ema"]["rust"] if results["ema"]["rust"] > 0 else float('inf')
    
    logger.info(f"SMA streaming: Python={results['sma']['python']:.6f}s, Rust={results['sma']['rust']:.6f}s, Speedup={results['sma']['speedup']:.2f}x")
    logger.info(f"EMA streaming: Python={results['ema']['python']:.6f}s, Rust={results['ema']['rust']:.6f}s, Speedup={results['ema']['speedup']:.2f}x")
    
    return results

def plot_results(batch_results: Dict, streaming_results: Dict, periods: List[int]):
    """
    Plot benchmark results.
    
    Args:
        batch_results: Results from batch benchmark
        streaming_results: Results from streaming benchmark
        periods: List of periods tested
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: SMA Execution Time
    plt.subplot(2, 2, 1)
    plt.title("SMA Execution Time")
    plt.plot(periods, batch_results["sma"]["python"], 'o-', label="Python")
    if RUST_AVAILABLE:
        plt.plot(periods, batch_results["sma"]["rust"], 'o-', label="Rust")
    plt.xlabel("Period")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: EMA Execution Time
    plt.subplot(2, 2, 2)
    plt.title("EMA Execution Time")
    plt.plot(periods, batch_results["ema"]["python"], 'o-', label="Python")
    if RUST_AVAILABLE:
        plt.plot(periods, batch_results["ema"]["rust"], 'o-', label="Rust")
    plt.xlabel("Period")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Speedup
    plt.subplot(2, 2, 3)
    plt.title("Speedup (Python/Rust)")
    if RUST_AVAILABLE:
        plt.plot(periods, batch_results["sma"]["speedup"], 'o-', label="SMA")
        plt.plot(periods, batch_results["ema"]["speedup"], 'o-', label="EMA")
    plt.xlabel("Period")
    plt.ylabel("Speedup Factor")
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Streaming Performance
    plt.subplot(2, 2, 4)
    plt.title("Streaming Performance")
    indicators = ["SMA", "EMA"]
    python_times = [streaming_results["sma"]["python"], streaming_results["ema"]["python"]]
    
    if RUST_AVAILABLE:
        rust_times = [streaming_results["sma"]["rust"], streaming_results["ema"]["rust"]]
        speedups = [streaming_results["sma"]["speedup"], streaming_results["ema"]["speedup"]]
        
        x = range(len(indicators))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], python_times, width, label='Python')
        plt.bar([i + width/2 for i in x], rust_times, width, label='Rust')
        
        # Add speedup labels
        for i, speedup in enumerate(speedups):
            plt.text(i, max(python_times[i], rust_times[i]) + 0.005, 
                    f"{speedup:.1f}x", 
                    ha='center', va='bottom')
    else:
        plt.bar(indicators, python_times)
    
    plt.xlabel("Indicator")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("rust_performance_benchmark.png")
    plt.show()

def main():
    """Run the performance comparison demonstration."""
    logger.info("Starting Rust Performance Demonstration")
    logger.info(f"Rust available: {RUST_AVAILABLE}")
    
    # Generate test data
    data_size = 10000
    logger.info(f"Generating {data_size} price points")
    prices = generate_price_data(data_size)
    
    # Define periods to test
    periods = [10, 20, 50, 100, 200]
    
    # Run batch benchmarks
    batch_results = run_batch_benchmark(prices, periods)
    
    # Run streaming benchmark
    streaming_results = run_streaming_benchmark(prices, period=20, iterations=5)
    
    # Plot results
    try:
        plot_results(batch_results, streaming_results, periods)
    except Exception as e:
        logger.error(f"Error plotting results: {e}")
    
    # Print summary
    if RUST_AVAILABLE:
        logger.info("\nPerformance Summary:")
        logger.info("--------------------")
        logger.info("Batch Processing:")
        for indicator in ["sma", "ema"]:
            avg_speedup = sum(batch_results[indicator]["speedup"]) / len(batch_results[indicator]["speedup"])
            logger.info(f"  {indicator.upper()}: {avg_speedup:.2f}x faster with Rust")
        
        logger.info("\nStreaming Processing:")
        logger.info(f"  SMA: {streaming_results['sma']['speedup']:.2f}x faster with Rust")
        logger.info(f"  EMA: {streaming_results['ema']['speedup']:.2f}x faster with Rust")
    else:
        logger.info("\nPerformance Summary: Rust not available for comparison")

if __name__ == "__main__":
    main() 