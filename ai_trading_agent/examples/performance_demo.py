"""
Performance Optimization Demo Script

This script demonstrates the performance improvements of the Technical Analysis Agent,
comparing performance with and without optimizations.
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.data_sources.mock_data_generator import generate_mock_data_with_pattern
from ai_trading_agent.utils.performance_profiler import profiler, PerformanceProfiler

def generate_test_data(num_symbols: int = 5, patterns: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate test data for multiple symbols and timeframes.
    
    Args:
        num_symbols: Number of symbols to generate
        patterns: List of patterns to use (one per symbol)
        
    Returns:
        Dictionary of market data in the format {symbol: {timeframe: DataFrame}}
    """
    if patterns is None:
        patterns = ['head_and_shoulders', 'double_top', 'ascending_triangle', 
                   'descending_triangle', 'rising_wedge']
    
    # Ensure we have enough patterns
    if len(patterns) < num_symbols:
        patterns = patterns * (num_symbols // len(patterns) + 1)
        
    timeframes = ["1h", "4h", "1d"]
    market_data = {}
    
    for i in range(num_symbols):
        symbol = f"SYMBOL_{i+1}"
        pattern = patterns[i % len(patterns)]
        
        market_data[symbol] = {}
        
        # Generate data for each timeframe
        for tf in timeframes:
            # Adjust the amount of data based on timeframe
            if tf == "1h":
                days = 90
            elif tf == "4h":
                days = 60
            else:
                days = 30
                
            # Generate mock data with pattern
            df = generate_mock_data_with_pattern(
                pattern=pattern,
                days_before=days,
                days_after=10
            )
            
            # Add some random variation to each symbol
            price_factor = 0.8 + (np.random.random() * 0.4)  # 0.8 to 1.2
            volume_factor = 0.7 + (np.random.random() * 0.6)  # 0.7 to 1.3
            
            df['open'] = df['open'] * price_factor
            df['high'] = df['high'] * price_factor
            df['low'] = df['low'] * price_factor
            df['close'] = df['close'] * price_factor
            df['volume'] = df['volume'] * volume_factor
            
            market_data[symbol][tf] = df
            
    return market_data

def create_agent_config(enable_parallel: bool = True, enable_profiling: bool = True, 
                        enable_caching: bool = True) -> Dict[str, Any]:
    """
    Create a configuration for the Technical Analysis Agent.
    
    Args:
        enable_parallel: Whether to enable parallel processing
        enable_profiling: Whether to enable performance profiling
        enable_caching: Whether to enable indicator caching
        
    Returns:
        Agent configuration dictionary
    """
    # Define indicator configurations
    sma_config = {"type": "sma", "params": {"window": 20}}
    ema_config = {"type": "ema", "params": {"window": 9}}
    macd_config = {"type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}
    rsi_config = {"type": "rsi", "params": {"window": 14}}
    adx_config = {"type": "adx", "params": {"window": 14}}
    
    # Create full configuration
    config = {
        "strategies": [
            {
                "name": "MACD_RSI_Strategy",
                "type": "multi_timeframe",
                "timeframes": ["1h", "4h", "1d"],
                "parameters": {
                    "primary_timeframe": "4h",
                    "confirmation_timeframes": ["1h", "1d"],
                    "indicators": ["macd", "rsi", "sma"],
                    "signal_threshold": 0.6
                }
            },
            {
                "name": "ADX_Trend_Strategy",
                "type": "standard",
                "parameters": {
                    "indicators": ["adx", "sma", "ema"],
                    "adx_threshold": 25,
                    "trend_strength": "medium"
                }
            }
        ],
        "indicators": [
            sma_config,
            ema_config,
            macd_config,
            rsi_config,
            adx_config
        ],
        "timeframes": ["1h", "4h", "1d"],
        "indicator_config": {
            "cache_enabled": enable_caching,
            "cache_ttl": 3600,  # 1 hour in seconds
            "max_cache_size": 100  # Maximum items in cache
        },
        "parallel_processing": {
            "enabled": enable_parallel,
            "max_workers": 4,
            "min_symbols": 3,
            "min_timeframes": 2
        },
        "profiling": {
            "enabled": enable_profiling
        }
    }
    
    return config

def run_performance_test(num_symbols: List[int] = None, optimizations: List[str] = None):
    """
    Run performance tests with different numbers of symbols and optimization settings.
    
    Args:
        num_symbols: List of symbol counts to test
        optimizations: List of optimization combinations to test
    """
    if num_symbols is None:
        num_symbols = [1, 5, 10, 20]
        
    if optimizations is None:
        optimizations = [
            "none",                # No optimizations
            "parallel",            # Parallel processing only
            "caching",             # Caching only
            "parallel+caching"     # Both optimizations
        ]
    
    results = {
        "num_symbols": [],
        "optimization": [],
        "execution_time_ms": [],
        "memory_usage_mb": [],
        "signals_generated": []
    }
    
    for opt in optimizations:
        # Configure optimizations
        enable_parallel = "parallel" in opt
        enable_caching = "caching" in opt
        enable_profiling = True  # Always enable profiling for measurements
        
        for num_sym in num_symbols:
            print(f"Testing with {num_sym} symbols, optimizations: {opt}")
            
            # Create agent config
            config = create_agent_config(
                enable_parallel=enable_parallel,
                enable_caching=enable_caching,
                enable_profiling=enable_profiling
            )
            
            # Create agent
            agent = AdvancedTechnicalAnalysisAgent(config)
            
            # Generate test data
            market_data = generate_test_data(num_symbols=num_sym)
            
            # Reset profiler
            profiler.reset()
            
            # Process data
            start_time = time.time()
            signals = agent.analyze(market_data)
            execution_time = time.time() - start_time
            
            # Get metrics
            metrics = agent.get_metrics()
            memory_usage = metrics.get("profiling", {}).get("memory_usage", {}).get("rss_mb", 0)
            
            # Store results
            results["num_symbols"].append(num_sym)
            results["optimization"].append(opt)
            results["execution_time_ms"].append(execution_time * 1000)  # Convert to ms
            results["memory_usage_mb"].append(memory_usage)
            results["signals_generated"].append(metrics.get("signals_generated", 0))
            
            print(f"  Execution time: {execution_time * 1000:.2f}ms")
            print(f"  Memory usage: {memory_usage:.2f}MB")
            print(f"  Signals generated: {metrics.get('signals_generated', 0)}")
            print(f"  Signals validated: {metrics.get('signals_validated', 0)}")
            print()
            
    return pd.DataFrame(results)

def plot_results(results: pd.DataFrame):
    """
    Plot performance test results.
    
    Args:
        results: DataFrame with test results
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution time
    for opt in results["optimization"].unique():
        opt_data = results[results["optimization"] == opt]
        ax1.plot(opt_data["num_symbols"], opt_data["execution_time_ms"], marker='o', label=opt)
    
    ax1.set_title('Execution Time vs. Number of Symbols')
    ax1.set_xlabel('Number of Symbols')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot memory usage
    for opt in results["optimization"].unique():
        opt_data = results[results["optimization"] == opt]
        ax2.plot(opt_data["num_symbols"], opt_data["memory_usage_mb"], marker='o', label=opt)
    
    ax2.set_title('Memory Usage vs. Number of Symbols')
    ax2.set_xlabel('Number of Symbols')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()
    
    # Create bar chart for execution time comparison
    plt.figure(figsize=(12, 6))
    
    # Pivot data for grouped bar chart
    pivot_data = results.pivot(index='num_symbols', columns='optimization', values='execution_time_ms')
    
    # Plot
    pivot_data.plot(kind='bar', ax=plt.gca())
    plt.title('Execution Time Comparison by Optimization Method')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(title='Optimizations')
    plt.tight_layout()
    plt.savefig('optimization_comparison.png')
    
    # Print a summary
    print("\nPerformance Improvement Summary:")
    
    for num_sym in results["num_symbols"].unique():
        base_time = results[(results["num_symbols"] == num_sym) & 
                          (results["optimization"] == "none")]["execution_time_ms"].values[0]
        
        best_opt = results[results["num_symbols"] == num_sym].sort_values("execution_time_ms").iloc[0]
        
        improvement = (base_time - best_opt["execution_time_ms"]) / base_time * 100
        
        print(f"{num_sym} Symbols:")
        print(f"  Best method: {best_opt['optimization']}")
        print(f"  No optimization: {base_time:.2f}ms")
        print(f"  With best optimization: {best_opt['execution_time_ms']:.2f}ms")
        print(f"  Improvement: {improvement:.2f}%")
        print()

if __name__ == "__main__":
    print("Running Technical Analysis Agent Performance Tests")
    print("=" * 50)
    
    # Run tests with different numbers of symbols
    results = run_performance_test()
    
    # Plot results
    plot_results(results)
    
    print("Tests completed. Results saved to performance_comparison.png and optimization_comparison.png")
