#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark comparing the performance of Rust vs Python backtesting implementations.

This script:
1. Generates synthetic price data
2. Runs a simple moving average crossover strategy in both implementations
3. Measures and compares execution time
4. Plots the performance results
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import backtesting components
from src.backtesting import (
    BacktestEngine, 
    PyBacktestEngine,
    TimeFrame, 
    OrderType, 
    OrderSide
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_backtest")

def generate_sample_data(days: int = 180, trend_strength: float = 0.01, 
                        volatility: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for backtesting.
    
    Args:
        days: Number of days to generate data for
        trend_strength: Strength of the trend component
        volatility: Volatility of the price movements
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    np.random.seed(seed)
    
    # Generate hourly timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(24 * days)]
    
    # Generate price data with trend and noise
    price = 10000.0  # Starting price
    prices = []
    
    for i in range(len(timestamps)):
        # Add trend component (cycles up and down)
        cycle = np.sin(i / (24 * 30) * np.pi)  # Monthly cycle
        trend = trend_strength * cycle
        
        # Add random component
        noise = np.random.normal(0, volatility)
        
        # Update price
        price *= (1 + trend + noise)
        prices.append(price)
    
    # Convert to OHLCV format
    df = pd.DataFrame()
    df['timestamp'] = timestamps
    df['close'] = prices
    
    # Generate realistic OHLC data
    df['open'] = df['close'].shift(1)
    df.loc[0, 'open'] = df.loc[0, 'close'] * (1 - np.random.uniform(0, 0.01))
    
    daily_volatility = df['close'].pct_change().std()
    df['high'] = df.apply(
        lambda x: max(x['open'], x['close']) * (1 + np.random.uniform(0, daily_volatility * 2)), 
        axis=1
    )
    df['low'] = df.apply(
        lambda x: min(x['open'], x['close']) * (1 - np.random.uniform(0, daily_volatility * 2)), 
        axis=1
    )
    
    # Generate volume
    df['volume'] = np.random.lognormal(mean=np.log(100), sigma=1, size=len(df))
    df['volume'] = df['volume'] * (1 + 5 * np.abs(df['close'].pct_change()))
    
    return df

def run_python_backtest(data: pd.DataFrame, initial_balance: float = 10000.0,
                      fast_period: int = 10, slow_period: int = 30,
                      commission_rate: float = 0.001) -> Tuple[float, Dict, float]:
    """
    Run backtest using pure Python implementation.
    
    Args:
        data: OHLCV DataFrame
        initial_balance: Starting account balance
        fast_period: Fast SMA period
        slow_period: Slow SMA period
        commission_rate: Trading fee percentage
        
    Returns:
        Tuple of (execution_time, backtest_stats, final_balance)
    """
    logger.info("Running Python backtest...")
    
    # Create backtest engine
    engine = PyBacktestEngine(
        initial_balance=initial_balance,
        symbols=["BTCUSDT"],
        commission_rate=commission_rate
    )
    
    # Setup strategy parameters
    fast_sma = []
    slow_sma = []
    position_size = 0.0
    
    # Measure execution time
    start_time = time.time()
    
    # Run backtest
    for i, row in data.iterrows():
        # Calculate indicators
        close_price = row['close']
        fast_sma.append(close_price)
        slow_sma.append(close_price)
        
        if len(fast_sma) > fast_period:
            fast_sma.pop(0)
        if len(slow_sma) > slow_period:
            slow_sma.pop(0)
        
        fast_sma_value = sum(fast_sma) / len(fast_sma) if fast_sma else 0
        slow_sma_value = sum(slow_sma) / len(slow_sma) if slow_sma else 0
        
        # Process candle
        engine.process_candle(
            symbol="BTCUSDT",
            timestamp=row['timestamp'],
            open_price=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timeframe=TimeFrame.HOUR_1
        )
        
        # Skip until we have enough data for both SMAs
        if len(fast_sma) < fast_period or len(slow_sma) < slow_period:
            continue
            
        # Trading logic
        if fast_sma_value > slow_sma_value and position_size == 0:
            # Buy signal
            engine.submit_market_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                quantity=engine.get_balance() * 0.95 / close_price
            )
            position_size = engine.get_balance() * 0.95 / close_price
        elif fast_sma_value < slow_sma_value and position_size > 0:
            # Sell signal
            engine.submit_market_order(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                quantity=position_size
            )
            position_size = 0
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Get final statistics
    stats = engine.get_stats()
    
    return execution_time, stats, engine.get_balance()

def run_rust_backtest(data: pd.DataFrame, initial_balance: float = 10000.0,
                     fast_period: int = 10, slow_period: int = 30,
                     commission_rate: float = 0.001) -> Tuple[float, Dict, float]:
    """
    Run backtest using Rust implementation.
    
    Args:
        data: OHLCV DataFrame
        initial_balance: Starting account balance
        fast_period: Fast SMA period
        slow_period: Slow SMA period
        commission_rate: Trading fee percentage
        
    Returns:
        Tuple of (execution_time, backtest_stats, final_balance)
    """
    logger.info("Running Rust backtest...")
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_balance=initial_balance,
        symbols=["BTCUSDT"],
        commission_rate=commission_rate
    )
    
    # Setup strategy parameters
    fast_sma = []
    slow_sma = []
    position_size = 0.0
    
    # Measure execution time
    start_time = time.time()
    
    # Run backtest
    for i, row in data.iterrows():
        # Calculate indicators
        close_price = row['close']
        fast_sma.append(close_price)
        slow_sma.append(close_price)
        
        if len(fast_sma) > fast_period:
            fast_sma.pop(0)
        if len(slow_sma) > slow_period:
            slow_sma.pop(0)
        
        fast_sma_value = sum(fast_sma) / len(fast_sma) if fast_sma else 0
        slow_sma_value = sum(slow_sma) / len(slow_sma) if slow_sma else 0
        
        # Process candle
        engine.process_candle(
            symbol="BTCUSDT",
            timestamp=row['timestamp'],
            open_price=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timeframe=TimeFrame.HOUR_1
        )
        
        # Skip until we have enough data for both SMAs
        if len(fast_sma) < fast_period or len(slow_sma) < slow_period:
            continue
            
        # Trading logic
        if fast_sma_value > slow_sma_value and position_size == 0:
            # Buy signal
            engine.submit_market_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                quantity=engine.get_balance() * 0.95 / close_price
            )
            position_size = engine.get_balance() * 0.95 / close_price
        elif fast_sma_value < slow_sma_value and position_size > 0:
            # Sell signal
            engine.submit_market_order(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                quantity=position_size
            )
            position_size = 0
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Get final statistics
    stats = engine.get_stats()
    
    return execution_time, stats, engine.get_balance()

def run_benchmarks(num_days_list: List[int] = [30, 90, 180, 365, 730]) -> pd.DataFrame:
    """
    Run benchmarks for different data sizes and compare Python vs Rust implementations.
    
    Args:
        num_days_list: List of different data sizes to benchmark
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    for days in num_days_list:
        logger.info(f"Benchmarking with {days} days of data...")
        
        # Generate sample data
        data = generate_sample_data(days=days)
        num_candles = len(data)
        
        # Run Python backtest
        py_time, py_stats, py_balance = run_python_backtest(data)
        
        # Run Rust backtest
        rust_time, rust_stats, rust_balance = run_rust_backtest(data)
        
        # Calculate speedup
        speedup = py_time / rust_time if rust_time > 0 else float('inf')
        
        # Add to results
        results.append({
            'days': days,
            'candles': num_candles,
            'python_time': py_time,
            'rust_time': rust_time,
            'speedup': speedup,
            'python_balance': py_balance,
            'rust_balance': rust_balance
        })
        
        logger.info(f"Results for {days} days:")
        logger.info(f"  Python: {py_time:.4f}s, Final balance: ${py_balance:.2f}")
        logger.info(f"  Rust: {rust_time:.4f}s, Final balance: ${rust_balance:.2f}")
        logger.info(f"  Speedup: {speedup:.2f}x")
    
    return pd.DataFrame(results)

def plot_benchmark_results(results: pd.DataFrame) -> None:
    """
    Plot benchmark results comparing Python vs Rust implementations.
    
    Args:
        results: DataFrame with benchmark results
    """
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Execution time comparison
    plt.subplot(2, 1, 1)
    plt.plot(results['candles'], results['python_time'], 'b-o', label='Python')
    plt.plot(results['candles'], results['rust_time'], 'r-o', label='Rust')
    plt.xlabel('Number of Candles')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Backtest Execution Time: Python vs Rust')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Speedup factor
    plt.subplot(2, 1, 2)
    plt.bar(results['days'].astype(str), results['speedup'], color='green')
    plt.xlabel('Days of Data')
    plt.ylabel('Speedup Factor (x times)')
    plt.title('Rust vs Python Speedup Factor')
    plt.grid(True, axis='y')
    
    # Add speedup values on top of bars
    for i, v in enumerate(results['speedup']):
        plt.text(i, v + 0.5, f"{v:.1f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()

def print_benchmark_summary(results: pd.DataFrame) -> None:
    """
    Print a summary of benchmark results.
    
    Args:
        results: DataFrame with benchmark results
    """
    print("\nBACKTEST BENCHMARK SUMMARY")
    print("==========================")
    print(f"{'Days':>6} | {'Candles':>8} | {'Python (s)':>10} | {'Rust (s)':>10} | {'Speedup':>8} | {'Balance Match':>13}")
    print("-" * 70)
    
    for _, row in results.iterrows():
        balance_match = "✓" if abs(row['python_balance'] - row['rust_balance']) < 0.01 else "✗"
        print(f"{row['days']:6d} | {row['candles']:8d} | {row['python_time']:10.4f} | {row['rust_time']:10.4f} | {row['speedup']:8.2f}x | {balance_match:^13}")
    
    # Add average speedup
    avg_speedup = results['speedup'].mean()
    print("-" * 70)
    print(f"{'Average':>17} | {' ':>10} | {' ':>10} | {avg_speedup:8.2f}x |")

if __name__ == "__main__":
    try:
        # Run benchmarks
        results = run_benchmarks()
        
        # Print summary
        print_benchmark_summary(results)
        
        # Plot results
        plot_benchmark_results(results)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
        sys.exit(1) 