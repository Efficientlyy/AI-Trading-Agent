"""
Benchmark module for comparing the performance of Rust extensions vs. Python implementations.

This module provides functions to benchmark the performance of feature engineering
functions implemented in Rust vs. their Python equivalents.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any, Tuple
import sys
import os

# Add the parent directory to the path to import the Rust extensions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Rust extensions
try:
    from rust_extensions.ai_trading_agent_rs import (
        create_lag_features_rs,
        create_diff_features_rs,
        create_pct_change_features_rs,
        create_rolling_window_features_rs,
        create_ema_features_rs
    )
    RUST_AVAILABLE = True
except ImportError:
    print("Warning: Rust extensions not available. Only Python implementations will be benchmarked.")
    RUST_AVAILABLE = False

# Python implementations for comparison
def create_lag_features_py(series: List[float], lags: List[int]) -> List[List[float]]:
    """
    Create lag features from a time series using Python.
    
    Args:
        series: Input time series
        lags: List of lag periods
        
    Returns:
        List of lists: Each inner list represents a lag feature
    """
    n_samples = len(series)
    n_features = len(lags)
    result = [[float('nan')] * n_features for _ in range(n_samples)]
    
    for i, lag in enumerate(lags):
        if lag <= 0:
            raise ValueError(f"Lag periods must be positive integers, got {lag}")
        
        for j in range(lag, n_samples):
            result[j][i] = series[j - lag]
    
    return result

def create_diff_features_py(series: List[float], periods: List[int]) -> List[List[float]]:
    """
    Create difference features from a time series using Python.
    
    Args:
        series: Input time series
        periods: List of periods for calculating differences
        
    Returns:
        List of lists: Each inner list represents a difference feature
    """
    n_samples = len(series)
    n_features = len(periods)
    result = [[float('nan')] * n_features for _ in range(n_samples)]
    
    for i, period in enumerate(periods):
        if period <= 0:
            raise ValueError(f"Periods must be positive integers, got {period}")
        
        for j in range(period, n_samples):
            result[j][i] = series[j] - series[j - period]
    
    return result

def create_pct_change_features_py(series: List[float], periods: List[int]) -> List[List[float]]:
    """
    Create percentage change features from a time series using Python.
    
    Args:
        series: Input time series
        periods: List of periods for calculating percentage changes
        
    Returns:
        List of lists: Each inner list represents a percentage change feature
    """
    n_samples = len(series)
    n_features = len(periods)
    result = [[float('nan')] * n_features for _ in range(n_samples)]
    
    for i, period in enumerate(periods):
        if period <= 0:
            raise ValueError(f"Periods must be positive integers, got {period}")
        
        for j in range(period, n_samples):
            previous_value = series[j - period]
            if previous_value != 0.0:
                result[j][i] = (series[j] - previous_value) / previous_value
            # If previous value is zero, result remains NaN
    
    return result

def create_rolling_window_features_py(
    series: List[float], 
    windows: List[int], 
    feature_type: int
) -> List[List[float]]:
    """
    Create rolling window features from a time series using Python.
    
    Args:
        series: Input time series
        windows: List of window sizes
        feature_type: Type of feature to calculate (0: mean, 1: std, 2: min, 3: max, 4: sum)
        
    Returns:
        List of lists: Each inner list represents a rolling window feature
    """
    n_samples = len(series)
    n_features = len(windows)
    result = [[float('nan')] * n_features for _ in range(n_samples)]
    
    if feature_type < 0 or feature_type > 4:
        raise ValueError(f"feature_type must be between 0 and 4, got {feature_type}")
    
    for i, window in enumerate(windows):
        if window <= 0:
            raise ValueError(f"Window sizes must be positive integers, got {window}")
        
        for j in range(window - 1, n_samples):
            window_data = series[j - (window - 1):j + 1]
            
            if feature_type == 0:  # Mean
                result[j][i] = sum(window_data) / window
            elif feature_type == 1:  # Standard deviation
                mean = sum(window_data) / window
                variance = sum((x - mean) ** 2 for x in window_data) / window
                result[j][i] = variance ** 0.5
            elif feature_type == 2:  # Min
                result[j][i] = min(window_data)
            elif feature_type == 3:  # Max
                result[j][i] = max(window_data)
            elif feature_type == 4:  # Sum
                result[j][i] = sum(window_data)
    
    return result

def create_ema_features_py(
    series: List[float], 
    spans: List[int], 
    alpha: float = None
) -> List[List[float]]:
    """
    Create exponential moving average (EMA) features from a time series using Python.
    
    Args:
        series: Input time series
        spans: List of EMA spans (periods)
        alpha: Optional smoothing factor (if None, alpha = 2/(span+1))
        
    Returns:
        List of lists: Each inner list represents an EMA feature
    """
    n_samples = len(series)
    n_features = len(spans)
    result = [[float('nan')] * n_features for _ in range(n_samples)]
    
    for i, span in enumerate(spans):
        if span <= 0:
            raise ValueError(f"Span periods must be positive integers, got {span}")
        
        alpha_value = alpha if alpha is not None else 2.0 / (span + 1.0)
        
        if alpha_value <= 0.0 or alpha_value > 1.0:
            raise ValueError(f"Alpha must be in (0, 1], got {alpha_value}")
        
        # Initialize EMA with the first value
        result[0][i] = series[0]
        
        # Calculate EMA for each time point
        for j in range(1, n_samples):
            result[j][i] = alpha_value * series[j] + (1.0 - alpha_value) * result[j-1][i]
    
    return result

def benchmark_function(
    func_py: Callable, 
    func_rs: Callable, 
    input_data: List[float], 
    params: Dict[str, Any],
    n_runs: int = 10
) -> Tuple[float, float]:
    """
    Benchmark a Python function against its Rust equivalent.
    
    Args:
        func_py: Python function to benchmark
        func_rs: Rust function to benchmark
        input_data: Input data for the functions
        params: Parameters for the functions
        n_runs: Number of runs to average over
        
    Returns:
        Tuple of (python_time, rust_time) in seconds
    """
    # Benchmark Python implementation
    py_times = []
    for _ in range(n_runs):
        start_time = time.time()
        func_py(input_data, **params)
        end_time = time.time()
        py_times.append(end_time - start_time)
    py_time = sum(py_times) / n_runs
    
    # Benchmark Rust implementation if available
    rs_time = None
    if RUST_AVAILABLE and func_rs is not None:
        rs_times = []
        for _ in range(n_runs):
            start_time = time.time()
            func_rs(input_data, **params)
            end_time = time.time()
            rs_times.append(end_time - start_time)
        rs_time = sum(rs_times) / n_runs
    
    return py_time, rs_time

def run_benchmarks(data_sizes: List[int], n_runs: int = 5) -> Dict[str, Dict[str, List[float]]]:
    """
    Run benchmarks for all feature engineering functions.
    
    Args:
        data_sizes: List of data sizes to benchmark
        n_runs: Number of runs to average over
        
    Returns:
        Dictionary of benchmark results
    """
    results = {
        'lag_features': {'python': [], 'rust': [], 'speedup': []},
        'diff_features': {'python': [], 'rust': [], 'speedup': []},
        'pct_change_features': {'python': [], 'rust': [], 'speedup': []},
        'rolling_window_features': {'python': [], 'rust': [], 'speedup': []},
        'ema_features': {'python': [], 'rust': [], 'speedup': []}
    }
    
    for size in data_sizes:
        print(f"Benchmarking with data size: {size}")
        
        # Generate random data
        data = np.random.randn(size).tolist()
        
        # Benchmark lag features
        lags = [1, 5, 10, 20]
        py_time, rs_time = benchmark_function(
            create_lag_features_py,
            create_lag_features_rs if RUST_AVAILABLE else None,
            data,
            {'lags': lags},
            n_runs
        )
        results['lag_features']['python'].append(py_time)
        if rs_time is not None:
            results['lag_features']['rust'].append(rs_time)
            results['lag_features']['speedup'].append(py_time / rs_time if rs_time > 0 else float('inf'))
        
        # Benchmark diff features
        periods = [1, 5, 10, 20]
        py_time, rs_time = benchmark_function(
            create_diff_features_py,
            create_diff_features_rs if RUST_AVAILABLE else None,
            data,
            {'periods': periods},
            n_runs
        )
        results['diff_features']['python'].append(py_time)
        if rs_time is not None:
            results['diff_features']['rust'].append(rs_time)
            results['diff_features']['speedup'].append(py_time / rs_time if rs_time > 0 else float('inf'))
        
        # Benchmark pct_change features
        periods = [1, 5, 10, 20]
        py_time, rs_time = benchmark_function(
            create_pct_change_features_py,
            create_pct_change_features_rs if RUST_AVAILABLE else None,
            data,
            {'periods': periods},
            n_runs
        )
        results['pct_change_features']['python'].append(py_time)
        if rs_time is not None:
            results['pct_change_features']['rust'].append(rs_time)
            results['pct_change_features']['speedup'].append(py_time / rs_time if rs_time > 0 else float('inf'))
        
        # Benchmark rolling_window features
        windows = [5, 10, 20, 50]
        feature_type = 0  # Mean
        py_time, rs_time = benchmark_function(
            create_rolling_window_features_py,
            create_rolling_window_features_rs if RUST_AVAILABLE else None,
            data,
            {'windows': windows, 'feature_type': feature_type},
            n_runs
        )
        results['rolling_window_features']['python'].append(py_time)
        if rs_time is not None:
            results['rolling_window_features']['rust'].append(rs_time)
            results['rolling_window_features']['speedup'].append(py_time / rs_time if rs_time > 0 else float('inf'))
        
        # Benchmark EMA features
        spans = [5, 10, 20, 50]
        py_time, rs_time = benchmark_function(
            create_ema_features_py,
            create_ema_features_rs if RUST_AVAILABLE else None,
            data,
            {'spans': spans},
            n_runs
        )
        results['ema_features']['python'].append(py_time)
        if rs_time is not None:
            results['ema_features']['rust'].append(rs_time)
            results['ema_features']['speedup'].append(py_time / rs_time if rs_time > 0 else float('inf'))
    
    return results

def plot_benchmark_results(results: Dict[str, Dict[str, List[float]]], data_sizes: List[int]):
    """
    Plot benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        data_sizes: List of data sizes used in the benchmark
    """
    plt.figure(figsize=(15, 10))
    
    # Plot execution times
    plt.subplot(2, 1, 1)
    for feature_name, feature_results in results.items():
        if 'python' in feature_results and feature_results['python']:
            plt.plot(data_sizes, feature_results['python'], 'o-', label=f'{feature_name} (Python)')
        if 'rust' in feature_results and feature_results['rust']:
            plt.plot(data_sizes, feature_results['rust'], 's--', label=f'{feature_name} (Rust)')
    
    plt.xlabel('Data Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs. Data Size')
    plt.legend()
    plt.grid(True)
    
    # Plot speedup
    plt.subplot(2, 1, 2)
    for feature_name, feature_results in results.items():
        if 'speedup' in feature_results and feature_results['speedup']:
            plt.plot(data_sizes, feature_results['speedup'], 'o-', label=feature_name)
    
    plt.xlabel('Data Size')
    plt.ylabel('Speedup (Python Time / Rust Time)')
    plt.title('Speedup vs. Data Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

def main():
    """
    Run benchmarks and plot results.
    """
    data_sizes = [1000, 5000, 10000, 50000, 100000]
    results = run_benchmarks(data_sizes)
    
    # Print results
    print("\nBenchmark Results:")
    for feature_name, feature_results in results.items():
        print(f"\n{feature_name}:")
        print("Data Size | Python Time (s) | Rust Time (s) | Speedup")
        print("-" * 60)
        for i, size in enumerate(data_sizes):
            py_time = feature_results['python'][i] if i < len(feature_results['python']) else float('nan')
            rs_time = feature_results['rust'][i] if 'rust' in feature_results and i < len(feature_results['rust']) else float('nan')
            speedup = feature_results['speedup'][i] if 'speedup' in feature_results and i < len(feature_results['speedup']) else float('nan')
            print(f"{size:9d} | {py_time:14.6f} | {rs_time:12.6f} | {speedup:7.2f}")
    
    # Plot results
    plot_benchmark_results(results, data_sizes)
    print("\nBenchmark plots saved to 'benchmark_results.png'")

if __name__ == "__main__":
    main()
