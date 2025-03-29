"""
Benchmark Rust vs Python implementations.

This script tests the performance difference between Rust and
pure Python implementations of the same functions.
"""

import time
import random
import numpy as np
from typing import List, Callable, Tuple
import matplotlib.pyplot as plt

# Import our Rust bridge
from src.rust_bridge import Technical, is_rust_available

# Pure Python implementations for comparison
def py_sma(values: List[float], period: int) -> List[float]:
    """Pure Python implementation of Simple Moving Average."""
    if len(values) < period:
        return []
    
    result = []
    for i in range(len(values) - period + 1):
        window = values[i:i+period]
        avg = sum(window) / period
        result.append(avg)
    
    return result

def py_ema(values: List[float], period: int) -> List[float]:
    """Pure Python implementation of Exponential Moving Average."""
    if len(values) < period:
        return []
    
    result = []
    # Start with SMA
    sma = sum(values[:period]) / period
    result.append(sma)
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Calculate EMA for the rest
    for i in range(period, len(values)):
        ema = values[i] * multiplier + result[-1] * (1 - multiplier)
        result.append(ema)
    
    return result

def benchmark(func: Callable, *args, **kwargs) -> Tuple[float, any]:
    """Benchmark a function and return execution time and result."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result

def generate_random_prices(size: int) -> List[float]:
    """Generate random price data for testing."""
    # Start with a base price
    base_price = 10000.0
    
    # Generate random price movements
    movements = np.random.normal(0, 1, size)
    
    # Apply a random walk
    for i in range(1, len(movements)):
        movements[i] = movements[i-1] + movements[i]
    
    # Scale and add to base price
    prices = [base_price + m * 100 for m in movements]
    
    return prices

def run_sma_benchmark(sizes, periods):
    """Run SMA benchmark for different data sizes and periods."""
    results = {}
    
    for size in sizes:
        prices = generate_random_prices(size)
        
        for period in periods:
            if period >= size:
                continue
                
            key = f"SMA (size={size}, period={period})"
            results[key] = {}
            
            # Benchmark Python implementation
            py_time, py_result = benchmark(py_sma, prices, period)
            results[key]['Python'] = py_time
            
            # Benchmark Rust implementation
            rust_time, rust_result = benchmark(Technical.sma, prices, period)
            results[key]['Rust'] = rust_time
            
            # Verify results match (approximately)
            if len(py_result) > 0 and len(rust_result) > 0:
                error = max(abs(p - r) for p, r in zip(py_result, rust_result))
                results[key]['Error'] = error
            
            # Calculate speedup
            speedup = py_time / rust_time if rust_time > 0 else float('inf')
            results[key]['Speedup'] = speedup
            
            print(f"{key}: Python: {py_time:.6f}s, Rust: {rust_time:.6f}s, Speedup: {speedup:.2f}x")
    
    return results

def run_ema_benchmark(sizes, periods):
    """Run EMA benchmark for different data sizes and periods."""
    results = {}
    
    for size in sizes:
        prices = generate_random_prices(size)
        
        for period in periods:
            if period >= size:
                continue
                
            key = f"EMA (size={size}, period={period})"
            results[key] = {}
            
            # Benchmark Python implementation
            py_time, py_result = benchmark(py_ema, prices, period)
            results[key]['Python'] = py_time
            
            # Benchmark Rust implementation
            rust_time, rust_result = benchmark(Technical.ema, prices, period)
            results[key]['Rust'] = rust_time
            
            # Verify results match (approximately)
            if len(py_result) > 0 and len(rust_result) > 0:
                error = max(abs(p - r) for p, r in zip(py_result, rust_result))
                results[key]['Error'] = error
            
            # Calculate speedup
            speedup = py_time / rust_time if rust_time > 0 else float('inf')
            results[key]['Speedup'] = speedup
            
            print(f"{key}: Python: {py_time:.6f}s, Rust: {rust_time:.6f}s, Speedup: {speedup:.2f}x")
    
    return results

def run_streaming_benchmark(size, periods):
    """Run benchmark for streaming calculations (updating point by point)."""
    results = {}
    
    prices = generate_random_prices(size)
    
    for period in periods:
        if period >= size:
            continue
            
        key = f"Streaming (size={size}, period={period})"
        results[key] = {}
        
        # Python SMA
        start_time = time.time()
        py_sma_obj = Technical.SMA(period)
        py_sma_results = []
        for price in prices:
            result = py_sma_obj.update(price)
            if result is not None:
                py_sma_results.append(result)
        py_time = time.time() - start_time
        results[key]['Python SMA'] = py_time
        
        # Python EMA
        start_time = time.time()
        py_ema_obj = Technical.EMA(period)
        py_ema_results = []
        for price in prices:
            result = py_ema_obj.update(price)
            if result is not None:
                py_ema_results.append(result)
        py_ema_time = time.time() - start_time
        results[key]['Python EMA'] = py_ema_time
        
        # Rust is always available because we're using our wrapper that falls back to Python
        # The real performance difference would be visible if Rust is actually available
        rust_msg = "Rust" if is_rust_available() else "Python fallback"
        
        print(f"{key}: Python SMA: {py_time:.6f}s, Python EMA: {py_ema_time:.6f}s")
        print(f"Using {rust_msg} implementation internally.")
    
    return results

def plot_results(results, title):
    """Plot benchmark results."""
    labels = list(results.keys())
    python_times = [results[k]['Python'] for k in labels]
    rust_times = [results[k]['Rust'] for k in labels]
    speedups = [results[k]['Speedup'] for k in labels]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot execution times
    ax1.bar(x, python_times, width, label='Python', color='blue', alpha=0.7)
    ax1.bar([i + width for i in x], rust_times, width, label='Rust', color='orange', alpha=0.7)
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_xticks([i + width/2 for i in x])
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Create a second y-axis for speedup
    ax2 = ax1.twinx()
    ax2.plot([i + width/2 for i in x], speedups, 'r-', marker='o', label='Speedup')
    ax2.set_ylabel('Speedup Factor (Python/Rust)')
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

def main():
    """Run all benchmarks."""
    print("Running benchmarks...")
    print(f"Rust available: {is_rust_available()}")
    
    # Parameters for benchmarks
    sizes = [1000, 10000, 100000]
    periods = [5, 20, 50, 200]
    
    # Run benchmarks
    sma_results = run_sma_benchmark(sizes, periods)
    ema_results = run_ema_benchmark(sizes, periods)
    streaming_results = run_streaming_benchmark(100000, periods)
    
    # Plot results
    if sma_results:
        plot_results(sma_results, "SMA Benchmark")
    
    if ema_results:
        plot_results(ema_results, "EMA Benchmark")
    
    print("Benchmarks complete.")

if __name__ == "__main__":
    main() 