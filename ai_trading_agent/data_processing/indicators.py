"""
Functions for calculating common technical indicators.

This module provides implementations of various technical indicators used in trading strategies.
When available, it uses Rust-accelerated implementations for better performance.
"""

import pandas as pd
import numpy as np
from ..common import logger

# Try to import Rust-accelerated implementations
try:
    from src.rust_integration.indicators import calculate_sma as calculate_sma_rs
    from src.rust_integration.indicators import calculate_ema as calculate_ema_rs
    from src.rust_integration.indicators import calculate_macd as calculate_macd_rs
    from src.rust_integration.indicators import calculate_rsi as calculate_rsi_rs
    RUST_AVAILABLE = True
    logger.info("Rust-accelerated indicators available.")
except ImportError:
    RUST_AVAILABLE = False
    logger.info("Rust-accelerated indicators not available. Using Python implementations.")

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA).
    
    Uses Rust-accelerated implementation when available for better performance.

    Args:
        data: Pandas Series of prices (e.g., close prices).
        window: The rolling window size.

    Returns:
        Pandas Series with SMA values.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if window > len(data):
        logger.warning(f"SMA window ({window}) is larger than data length ({len(data)}). Returning NaNs.")
        return pd.Series(index=data.index, dtype=np.float64)

    if RUST_AVAILABLE:
        # Use Rust-accelerated implementation
        logger.debug(f"Using Rust-accelerated SMA calculation with window={window}")
        result_array = calculate_sma_rs(data.values, window)
        return pd.Series(result_array, index=data.index)
    else:
        # Use pandas implementation
        logger.debug(f"Using Python SMA calculation with window={window}")
        return data.rolling(window=window, min_periods=window).mean()

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).
    
    Uses Rust-accelerated implementation when available for better performance.

    Args:
        data: Pandas Series of prices (e.g., close prices).
        window: The smoothing window size (often referred to as span in pandas).

    Returns:
        Pandas Series with EMA values.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if window > len(data):
        logger.warning(f"EMA window ({window}) is larger than data length ({len(data)}). Returning NaNs.")
        return pd.Series(index=data.index, dtype=np.float64)

    if RUST_AVAILABLE:
        # Use Rust-accelerated implementation
        logger.debug(f"Using Rust-accelerated EMA calculation with window={window}")
        result_array = calculate_ema_rs(data.values, window)
        return pd.Series(result_array, index=data.index)
    else:
        # Use pandas implementation
        logger.debug(f"Using Python EMA calculation with window={window}")
        # span corresponds to the window for EMA calculation
        return data.ewm(span=window, adjust=False, min_periods=window).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).
    
    Uses a consistent implementation that matches pandas' ewm behavior.

    Args:
        data: Pandas Series of prices (e.g., close prices).
        window: The window period (default is 14).

    Returns:
        Pandas Series with RSI values.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series.")
    if window <= 0:
        raise ValueError("Window size must be positive.")
    if window >= len(data):
        logger.warning(f"RSI window ({window}) is larger than data length ({len(data)}). Returning NaNs.")
        return pd.Series(index=data.index, dtype=np.float64)

    # Use pandas implementation for consistent results
    logger.debug(f"Using RSI calculation with window={window}")
    
    delta = data.diff(1)
    delta = delta.dropna() # Remove first NaN

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use EMA for average gain/loss calculation as is common practice
    avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle potential division by zero if avg_loss is 0
    rsi = rsi.replace([np.inf, -np.inf], 100.0) # If avg_loss is 0, RSI is 100

    # Reindex to match original data length, filling initial period with NaN
    rsi_aligned = pd.Series(np.nan, index=data.index)
    rsi_aligned.loc[rsi.index] = rsi

    return rsi_aligned

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    
    Uses Rust-accelerated implementation when available for better performance.

    Args:
        data: Pandas Series of prices (e.g., close prices).
        fast_period: The window for the fast EMA (default is 12).
        slow_period: The window for the slow EMA (default is 26).
        signal_period: The window for the signal line EMA (default is 9).

    Returns:
        Pandas DataFrame with columns: 'MACD', 'Signal', 'Histogram'.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series.")
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("Window periods must be positive.")
    if fast_period >= slow_period:
        raise ValueError("Fast period must be smaller than slow period.")
    min_len = max(slow_period, signal_period)
    if min_len >= len(data):
         logger.warning(f"MACD periods require at least {min_len} data points, have {len(data)}. Returning NaNs.")
         return pd.DataFrame(index=data.index, columns=['MACD', 'Signal', 'Histogram'], dtype=np.float64)

    if RUST_AVAILABLE:
        # Use Rust-accelerated implementation
        logger.debug(f"Using Rust-accelerated MACD calculation with fast_period={fast_period}, slow_period={slow_period}, signal_period={signal_period}")
        macd_line, signal_line, histogram = calculate_macd_rs(data.values, fast_period, slow_period, signal_period)
        
        macd_df = pd.DataFrame({
            'MACD': pd.Series(macd_line, index=data.index),
            'Signal': pd.Series(signal_line, index=data.index),
            'Histogram': pd.Series(histogram, index=data.index)
        })
    else:
        # Use Python implementation
        logger.debug(f"Using Python MACD calculation with fast_period={fast_period}, slow_period={slow_period}, signal_period={signal_period}")
        
        ema_fast = calculate_ema(data, window=fast_period)
        ema_slow = calculate_ema(data, window=slow_period)

        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, window=signal_period)
        histogram = macd_line - signal_line

        macd_df = pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }, index=data.index)

    return macd_df

# --- Example Usage (for demonstration, not part of the library) ---
if __name__ == '__main__':
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close_prices = pd.Series(np.random.randn(100).cumsum() + 50, index=dates)
    ohlcv_data = pd.DataFrame({
        'open': close_prices - np.random.rand(100) * 0.5,
        'high': close_prices + np.random.rand(100) * 0.5,
        'low': close_prices - np.random.rand(100) * 0.5,
        'close': close_prices,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)

    print("--- Sample Close Prices ---")
    print(close_prices.head())

    # Calculate indicators
    sma_10 = calculate_sma(close_prices, 10)
    ema_10 = calculate_ema(close_prices, 10)
    rsi_14 = calculate_rsi(close_prices, 14)
    macd_data = calculate_macd(close_prices)

    print("\n--- SMA(10) ---")
    print(sma_10.tail())

    print("\n--- EMA(10) ---")
    print(ema_10.tail())

    print("\n--- RSI(14) ---")
    print(rsi_14.tail())

    print("\n--- MACD(12, 26, 9) ---")
    print(macd_data.tail())

    # Example of adding indicators to a DataFrame
    ohlcv_data['SMA_10'] = sma_10
    ohlcv_data['EMA_10'] = ema_10
    ohlcv_data['RSI_14'] = rsi_14
    ohlcv_data[['MACD', 'Signal', 'Histogram']] = macd_data

    print("\n--- DataFrame with Indicators ---")
    print(ohlcv_data.tail())
