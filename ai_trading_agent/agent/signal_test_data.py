"""
Signal Test Data Generator - Creates specialized data to test strategy signal generation.

This module contains functions to generate specific price action patterns
that will trigger signals in our technical analysis strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

def generate_ma_crossover_data(
    symbol: str,
    periods: int = 100,
    fast_period: int = 10,
    slow_period: int = 30,
    signal_position: int = -5,
    signal_type: str = "buy"
) -> pd.DataFrame:
    """
    Generate data with a moving average crossover at a specific position.
    
    Args:
        symbol: Symbol name for the data
        periods: Total number of periods to generate
        fast_period: Period of the fast moving average
        slow_period: Period of the slow moving average
        signal_position: Position from the end where the signal should occur (-1 = most recent)
        signal_type: Type of signal to generate ("buy" or "sell")
        
    Returns:
        DataFrame with OHLCV data engineered to create a crossover signal
    """
    # Ensure we have enough data
    if periods < max(fast_period, slow_period) + 20:
        periods = max(fast_period, slow_period) + 20
    
    # Create date range
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(periods)]
    dates.reverse()  # Put in chronological order
    
    # Filter to business days only
    dates = [date for date in dates if date.weekday() < 5]
    
    # Basic price series - start with flat prices
    base_price = 100.0
    close_prices = np.ones(len(dates)) * base_price
    
    # For a BUY signal, we need fast MA to cross above slow MA
    # For a SELL signal, we need fast MA to cross below slow MA
    
    # Calculate MA positions in the data
    current_pos = len(dates) + signal_position  # Position for the signal
    setup_start = max(0, current_pos - slow_period * 2)  # Where to start modifying data
    
    # Create the price action to generate the desired MA crossover
    if signal_type.lower() == "buy":
        # For buy signal: create declining trend followed by a sharp uptrend
        for i in range(setup_start, current_pos):
            # First create a slight downtrend
            position = (i - setup_start) / (current_pos - setup_start)
            if position < 0.7:
                # Downtrend for first 70% of the setup period
                close_prices[i] = base_price * (1 - 0.1 * position)
            else:
                # Sharp uptrend for the last 30% to create the bullish crossover
                recovery_position = (position - 0.7) / 0.3
                close_prices[i] = base_price * (0.93 + 0.12 * recovery_position)
    else:
        # For sell signal: create rising trend followed by a sharp downtrend
        for i in range(setup_start, current_pos):
            # First create an uptrend
            position = (i - setup_start) / (current_pos - setup_start)
            if position < 0.7:
                # Uptrend for first 70% of the setup period
                close_prices[i] = base_price * (1 + 0.1 * position)
            else:
                # Sharp downtrend for the last 30% to create the bearish crossover
                decline_position = (position - 0.7) / 0.3
                close_prices[i] = base_price * (1.07 - 0.12 * decline_position)
    
    # Generate other OHLC data from close prices
    open_prices = np.zeros(len(dates))
    high_prices = np.zeros(len(dates))
    low_prices = np.zeros(len(dates))
    volumes = np.zeros(len(dates))
    
    # Generate realistic OHLC values based on close prices
    for i in range(len(dates)):
        if i == 0:
            open_prices[i] = close_prices[i] * 0.99
        else:
            open_prices[i] = close_prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        daily_range = close_prices[i] * 0.02  # 2% daily range
        high_prices[i] = max(close_prices[i], open_prices[i]) + np.random.uniform(0.001, 0.01) * close_prices[i]
        low_prices[i] = min(close_prices[i], open_prices[i]) - np.random.uniform(0.001, 0.01) * close_prices[i]
        
        # Higher volume on trend changes
        price_change = abs(close_prices[i] / close_prices[i-1] - 1) if i > 0 else 0
        volumes[i] = 100000 * (1 + 10 * price_change)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes.astype(int)
    })
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Add the symbol as a column (useful when combining multiple symbols)
    df['symbol'] = symbol
    
    # Ensure all columns are numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    return df

def generate_rsi_signal_data(
    symbol: str,
    periods: int = 100,
    rsi_period: int = 14,
    signal_position: int = -5,
    signal_type: str = "oversold"  # or "overbought"
) -> pd.DataFrame:
    """
    Generate data with an RSI crossing overbought/oversold levels.
    
    Args:
        symbol: Symbol name for the data
        periods: Total number of periods to generate
        rsi_period: Period for RSI calculation
        signal_position: Position from the end where the signal should occur (-1 = most recent)
        signal_type: Type of signal to generate ("oversold" for buy, "overbought" for sell)
        
    Returns:
        DataFrame with OHLCV data engineered to create an RSI signal
    """
    # Ensure we have enough data
    if periods < rsi_period + 20:
        periods = rsi_period + 20
    
    # Create date range
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(periods)]
    dates.reverse()  # Put in chronological order
    
    # Filter to business days only
    dates = [date for date in dates if date.weekday() < 5]
    
    # Basic price series - start with flat prices
    base_price = 100.0
    close_prices = np.ones(len(dates)) * base_price
    
    # Calculate positions in the data
    current_pos = len(dates) + signal_position  # Position for the signal
    setup_start = max(0, current_pos - rsi_period * 2)  # Where to start modifying data
    
    # Create the price action to generate the desired RSI condition
    if signal_type.lower() == "oversold":
        # For oversold condition (buy signal): create a sharp decline followed by a small bounce
        for i in range(setup_start, current_pos):
            position = (i - setup_start) / (current_pos - setup_start)
            if position < 0.8:
                # Sharp decline for first 80% of the setup period to push RSI down
                close_prices[i] = base_price * (1 - 0.25 * position)
            else:
                # Small bounce to cross back above oversold threshold
                bounce_position = (position - 0.8) / 0.2
                close_prices[i] = base_price * 0.8 * (1 + 0.05 * bounce_position)
    else:
        # For overbought condition (sell signal): create a sharp rise followed by a small drop
        for i in range(setup_start, current_pos):
            position = (i - setup_start) / (current_pos - setup_start)
            if position < 0.8:
                # Sharp rise for first 80% of the setup period to push RSI up
                close_prices[i] = base_price * (1 + 0.25 * position)
            else:
                # Small drop to cross back below overbought threshold
                drop_position = (position - 0.8) / 0.2
                close_prices[i] = base_price * 1.2 * (1 - 0.05 * drop_position)
    
    # Generate other OHLC data from close prices
    open_prices = np.zeros(len(dates))
    high_prices = np.zeros(len(dates))
    low_prices = np.zeros(len(dates))
    volumes = np.zeros(len(dates))
    
    # Generate realistic OHLC values based on close prices
    for i in range(len(dates)):
        if i == 0:
            open_prices[i] = close_prices[i] * 0.99
        else:
            open_prices[i] = close_prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        daily_range = close_prices[i] * 0.02  # 2% daily range
        high_prices[i] = max(close_prices[i], open_prices[i]) + np.random.uniform(0.001, 0.01) * close_prices[i]
        low_prices[i] = min(close_prices[i], open_prices[i]) - np.random.uniform(0.001, 0.01) * close_prices[i]
        
        # Higher volume on trend changes
        price_change = abs(close_prices[i] / close_prices[i-1] - 1) if i > 0 else 0
        volumes[i] = 100000 * (1 + 10 * price_change)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes.astype(int)
    })
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Add the symbol as a column (useful when combining multiple symbols)
    df['symbol'] = symbol
    
    # Ensure all columns are numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    return df

def generate_signal_test_data() -> Dict[str, pd.DataFrame]:
    """
    Generate a suite of test data for signal generation testing.
    
    Returns:
        Dictionary mapping symbols to DataFrame with price data designed
        to trigger specific signals
    """
    test_data = {}
    
    # Generate MA crossover buy signal data
    test_data["MA_CROSS_BUY"] = generate_ma_crossover_data(
        symbol="MA_CROSS_BUY",
        fast_period=10,
        slow_period=30,
        signal_position=-5,
        signal_type="buy"
    )
    
    # Generate MA crossover sell signal data
    test_data["MA_CROSS_SELL"] = generate_ma_crossover_data(
        symbol="MA_CROSS_SELL",
        fast_period=10,
        slow_period=30,
        signal_position=-5,
        signal_type="sell"
    )
    
    # Generate RSI oversold (buy) signal data
    test_data["RSI_OVERSOLD"] = generate_rsi_signal_data(
        symbol="RSI_OVERSOLD",
        rsi_period=14,
        signal_position=-5,
        signal_type="oversold"
    )
    
    # Generate RSI overbought (sell) signal data
    test_data["RSI_OVERBOUGHT"] = generate_rsi_signal_data(
        symbol="RSI_OVERBOUGHT",
        rsi_period=14,
        signal_position=-5,
        signal_type="overbought"
    )
    
    return test_data
