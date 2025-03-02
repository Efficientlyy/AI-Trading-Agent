"""
Data utility functions for the backtesting framework.

This module provides functions for generating sample data, loading historical data,
and preprocessing data for backtesting.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import csv

from .models import CandleData, TimeFrame

# Set up logging
logger = logging.getLogger("backtester.data")


def load_csv_data(file_path: str, symbol: str, timeframe: TimeFrame) -> List[CandleData]:
    """
    Load market data from CSV file.
    
    Expected CSV format:
    timestamp,open,high,low,close,volume
    
    Args:
        file_path: Path to the CSV file
        symbol: Trading symbol
        timeframe: TimeFrame enum value
        
    Returns:
        List of CandleData objects
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    candles = []
    
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")
        
        # Parse data
        for _, row in df.iterrows():
            timestamp = row['timestamp']
            
            # Handle timestamp parsing
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            
            # Create candle data
            candle = CandleData(
                symbol=symbol,
                timestamp=timestamp,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']) if 'volume' in row else 0.0,
                timeframe=timeframe
            )
            
            candles.append(candle)
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise
    
    logger.info(f"Loaded {len(candles)} candles from {file_path}")
    return candles


def save_to_csv(candles: List[CandleData], file_path: str) -> None:
    """
    Save candle data to CSV file.
    
    Args:
        candles: List of CandleData objects
        file_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe'])
        
        for candle in candles:
            writer.writerow([
                candle.timestamp.isoformat(),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.symbol,
                candle.timeframe
            ])
    
    logger.info(f"Saved {len(candles)} candles to {file_path}")


def generate_sample_data(
    symbol: str, 
    timeframe: TimeFrame,
    start_time: datetime,
    end_time: datetime,
    base_price: float = 10000.0,
    volatility: float = 0.015,
    trend_strength: float = 0.0001,
    with_cycles: bool = True,
    seed: Optional[int] = None
) -> List[CandleData]:
    """
    Generate synthetic market data with controllable patterns.
    
    Creates realistic market data with trends, cycles, and random component
    that can be used for strategy testing.
    
    Args:
        symbol: Trading symbol
        timeframe: TimeFrame enum value
        start_time: Start time for the data
        end_time: End time for the data
        base_price: Starting price point
        volatility: Daily volatility (higher = more noisy price action)
        trend_strength: Strength of the long-term trend (can be positive or negative)
        with_cycles: Whether to add cyclic patterns to the data
        seed: Random seed for reproducibility
        
    Returns:
        List of CandleData objects with synthetic price data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate time delta based on timeframe
    if timeframe == TimeFrame.MINUTE_1:
        delta = timedelta(minutes=1)
    elif timeframe == TimeFrame.MINUTE_5:
        delta = timedelta(minutes=5)
    elif timeframe == TimeFrame.MINUTE_15:
        delta = timedelta(minutes=15)
    elif timeframe == TimeFrame.MINUTE_30:
        delta = timedelta(minutes=30)
    elif timeframe == TimeFrame.HOUR_1:
        delta = timedelta(hours=1)
    elif timeframe == TimeFrame.HOUR_4:
        delta = timedelta(hours=4)
    elif timeframe == TimeFrame.DAY_1:
        delta = timedelta(days=1)
    elif timeframe == TimeFrame.WEEK_1:
        delta = timedelta(weeks=1)
    else:
        delta = timedelta(hours=1)
    
    # Generate timestamps
    current_time = start_time
    timestamps = []
    
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += delta
    
    num_candles = len(timestamps)
    
    # Generate price data with realistic patterns
    prices = np.zeros((num_candles, 4))  # [open, high, low, close]
    
    # Start with the base price
    close_prices = np.zeros(num_candles)
    close_prices[0] = base_price
    
    # Long-term trend component
    trend = np.linspace(0, trend_strength * num_candles, num_candles)
    
    # Cycle components (if enabled)
    if with_cycles:
        # Medium-term cycle (about 30 days)
        medium_cycle_length = int(24 * 30 / delta.total_seconds() * 3600)
        medium_cycle = np.sin(np.linspace(0, 6 * np.pi, num_candles)) * (base_price * 0.03)
        
        # Short-term cycle (about 7 days)
        short_cycle_length = int(24 * 7 / delta.total_seconds() * 3600)
        short_cycle = np.sin(np.linspace(0, 15 * np.pi, num_candles)) * (base_price * 0.01)
        
        # Day/night cycle for intraday data
        day_cycle = np.sin(np.linspace(0, 2 * np.pi * num_candles / 24, num_candles)) * (base_price * 0.005)
    else:
        medium_cycle = np.zeros(num_candles)
        short_cycle = np.zeros(num_candles)
        day_cycle = np.zeros(num_candles)
    
    # Random component (scaled by volatility)
    daily_candles = 24 / (delta.total_seconds() / 3600)  # Number of candles per day
    candle_volatility = volatility / np.sqrt(daily_candles)  # Scale volatility to timeframe
    
    # Generate random returns
    random_returns = np.random.normal(0, candle_volatility, num_candles)
    
    # Occasionally introduce trend changes
    if num_candles > 100:
        num_trend_changes = num_candles // 100
        trend_change_points = np.random.choice(range(1, num_candles), num_trend_changes, replace=False)
        
        for point in trend_change_points:
            # Strong move over the next 20-30 candles
            move_length = np.random.randint(20, 30)
            move_end = min(point + move_length, num_candles)
            
            # Random direction and magnitude
            direction = np.random.choice([-1, 1])
            magnitude = base_price * np.random.uniform(0.02, 0.05)  # 2-5% move
            
            # Apply the trend change
            for i in range(point, move_end):
                progress = (i - point) / move_length
                random_returns[i] += direction * magnitude * (1 - progress) / num_candles
    
    # Combine components for the close price
    for i in range(1, num_candles):
        close_prices[i] = close_prices[i-1] * (1 + random_returns[i])
        
        # Add trend and cycles
        close_prices[i] += trend[i]
        
        if with_cycles:
            close_prices[i] += medium_cycle[i] - medium_cycle[i-1]
            close_prices[i] += short_cycle[i] - short_cycle[i-1]
            close_prices[i] += day_cycle[i] - day_cycle[i-1]
    
    # Generate OHLC from close prices
    candles = []
    
    for i in range(num_candles):
        # Calculate typical OHLC relationships
        if i > 0:
            open_price = close_prices[i-1]  # Open at previous close
        else:
            open_price = base_price
        
        close_price = close_prices[i]
        
        # Generate realistic high/low
        price_range = abs(close_price - open_price)
        extra_range = max(price_range * 0.5, close_price * candle_volatility)
        
        if close_price >= open_price:
            # Bullish candle
            high_price = close_price + np.random.uniform(0, extra_range)
            low_price = open_price - np.random.uniform(0, extra_range)
        else:
            # Bearish candle
            high_price = open_price + np.random.uniform(0, extra_range)
            low_price = close_price - np.random.uniform(0, extra_range)
        
        # Ensure low <= open,close <= high
        low_price = min(low_price, open_price, close_price)
        high_price = max(high_price, open_price, close_price)
        
        # Generate volume (higher on larger price moves)
        volume = np.random.normal(100000, 50000) * (1 + 5 * abs(close_price - open_price) / close_price)
        
        # Create candle object
        candle = CandleData(
            symbol=symbol,
            timestamp=timestamps[i],
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=max(volume, 1000),  # Ensure positive volume
            timeframe=timeframe
        )
        
        candles.append(candle)
    
    logger.info(f"Generated {len(candles)} sample candles for {symbol} ({timeframe})")
    return candles


def resample_timeframe(candles: List[CandleData], target_timeframe: TimeFrame) -> List[CandleData]:
    """
    Resample candles to a higher timeframe.
    
    Args:
        candles: List of CandleData objects
        target_timeframe: Target timeframe to resample to
        
    Returns:
        List of resampled CandleData objects
    """
    if not candles:
        return []
    
    # Convert candles to DataFrame for easier resampling
    df = pd.DataFrame([
        {
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
            'symbol': c.symbol,
            'timeframe': c.timeframe
        }
        for c in candles
    ])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Determine pandas resampling frequency based on target timeframe
    if target_timeframe == TimeFrame.MINUTE_1:
        freq = '1min'
    elif target_timeframe == TimeFrame.MINUTE_5:
        freq = '5min'
    elif target_timeframe == TimeFrame.MINUTE_15:
        freq = '15min'
    elif target_timeframe == TimeFrame.MINUTE_30:
        freq = '30min'
    elif target_timeframe == TimeFrame.HOUR_1:
        freq = '1H'
    elif target_timeframe == TimeFrame.HOUR_4:
        freq = '4H'
    elif target_timeframe == TimeFrame.DAY_1:
        freq = '1D'
    elif target_timeframe == TimeFrame.WEEK_1:
        freq = '1W'
    else:
        logger.warning(f"Unsupported timeframe: {target_timeframe}, defaulting to 1 hour")
        freq = '1H'
    
    # Resample data
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'symbol': 'first'
    })
    
    # Drop rows with NaN values
    resampled = resampled.dropna()
    
    # Convert back to CandleData objects
    resampled_candles = []
    
    for timestamp, row in resampled.iterrows():
        candle = CandleData(
            symbol=row['symbol'],
            timestamp=timestamp.to_pydatetime(),
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timeframe=target_timeframe
        )
        
        resampled_candles.append(candle)
    
    logger.info(f"Resampled {len(candles)} candles to {len(resampled_candles)} {target_timeframe} candles")
    return resampled_candles


def add_noise_to_data(candles: List[CandleData], noise_level: float = 0.005) -> List[CandleData]:
    """
    Add random noise to the price data.
    
    Useful for testing strategy robustness against slight data variations.
    
    Args:
        candles: List of CandleData objects
        noise_level: Level of noise to add (as a fraction of price)
        
    Returns:
        List of CandleData objects with added noise
    """
    if noise_level <= 0:
        return candles
    
    noisy_candles = []
    
    for candle in candles:
        # Calculate noise amounts for each price
        open_noise = candle.open * np.random.uniform(-noise_level, noise_level)
        high_noise = candle.high * np.random.uniform(0, noise_level)  # High only goes higher
        low_noise = candle.low * np.random.uniform(-noise_level, 0)   # Low only goes lower
        close_noise = candle.close * np.random.uniform(-noise_level, noise_level)
        
        # Create new candle with noise
        noisy_candle = CandleData(
            symbol=candle.symbol,
            timestamp=candle.timestamp,
            open=candle.open + open_noise,
            high=max(candle.high + high_noise, candle.open + open_noise, candle.close + close_noise),
            low=min(candle.low + low_noise, candle.open + open_noise, candle.close + close_noise),
            close=candle.close + close_noise,
            volume=candle.volume,
            timeframe=candle.timeframe
        )
        
        noisy_candles.append(noisy_candle)
    
    return noisy_candles


def prepare_data_for_indicators(candles: List[CandleData]) -> Dict[str, np.ndarray]:
    """
    Convert candle data to dictionary of numpy arrays for indicator calculation.
    
    Args:
        candles: List of CandleData objects
        
    Returns:
        Dictionary with price arrays for indicator calculation
    """
    if not candles:
        return {
            'open': np.array([]),
            'high': np.array([]),
            'low': np.array([]),
            'close': np.array([]),
            'volume': np.array([])
        }
    
    # Extract price data
    timestamps = [c.timestamp for c in candles]
    opens = np.array([c.open for c in candles])
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])
    volumes = np.array([c.volume for c in candles])
    
    return {
        'timestamp': np.array(timestamps),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }


def calculate_returns(candles: List[CandleData]) -> np.ndarray:
    """
    Calculate percentage returns from candle data.
    
    Args:
        candles: List of CandleData objects
        
    Returns:
        NumPy array of percentage returns
    """
    if len(candles) < 2:
        return np.array([])
    
    closes = np.array([c.close for c in candles])
    returns = np.zeros(len(closes))
    
    for i in range(1, len(closes)):
        returns[i] = (closes[i] / closes[i-1]) - 1
    
    return returns


def add_gap_data(candles: List[CandleData], gap_probability: float = 0.05, 
                max_gap_size: float = 0.02) -> List[CandleData]:
    """
    Add random price gaps to the data to simulate market gaps.
    
    Args:
        candles: List of CandleData objects
        gap_probability: Probability of a gap occurring between candles
        max_gap_size: Maximum gap size as a fraction of price
        
    Returns:
        List of CandleData objects with added gaps
    """
    if not candles or gap_probability <= 0:
        return candles
    
    gapped_candles = candles.copy()
    
    for i in range(1, len(gapped_candles)):
        # Check if a gap should be added
        if np.random.random() < gap_probability:
            # Determine gap direction (up or down)
            direction = np.random.choice([-1, 1])
            
            # Determine gap size
            gap_size = np.random.uniform(0.005, max_gap_size)
            
            # Apply the gap to the open price
            prev_close = gapped_candles[i-1].close
            new_open = prev_close * (1 + direction * gap_size)
            
            # Update the candle with the gap
            gapped_candles[i] = CandleData(
                symbol=gapped_candles[i].symbol,
                timestamp=gapped_candles[i].timestamp,
                open=new_open,
                high=max(new_open, gapped_candles[i].high),
                low=min(new_open, gapped_candles[i].low),
                close=gapped_candles[i].close,
                volume=gapped_candles[i].volume,
                timeframe=gapped_candles[i].timeframe
            )
    
    return gapped_candles 