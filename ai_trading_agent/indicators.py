"""
Technical indicators module for AI Trading Agent.

This module provides functions to calculate various technical indicators
used by the trading strategies.
"""

import numpy as np
import pandas as pd


def calculate_rsi(data, window=14, column='close'):
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        data: DataFrame containing price data
        window: RSI calculation window (default: 14)
        column: Column name to use for calculation (default: 'close')
        
    Returns:
        Series containing RSI values
    """
    # Make a copy of the data to avoid modifying the original
    price = data[column].copy()
    
    # Calculate price changes
    delta = price.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Align the RSI values with the original data index
    rsi_aligned = pd.Series(np.nan, index=data.index)
    rsi_aligned.loc[rsi.index] = rsi
    
    return rsi_aligned


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9, column='close'):
    """
    Calculate the Moving Average Convergence Divergence (MACD).
    
    Args:
        data: DataFrame containing price data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        column: Column name to use for calculation (default: 'close')
        
    Returns:
        DataFrame containing MACD, Signal, and Histogram values
    """
    # Make a copy of the data to avoid modifying the original
    price = data[column].copy()
    
    # Calculate EMAs
    ema_fast = price.ewm(span=fast_period, min_periods=fast_period).mean()
    ema_slow = price.ewm(span=slow_period, min_periods=slow_period).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Create a DataFrame with the results
    macd_df = pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }, index=data.index)
    
    return macd_df


def calculate_bollinger_bands(data, window=20, num_std=2, column='close'):
    """
    Calculate Bollinger Bands.
    
    Args:
        data: DataFrame containing price data
        window: Window for moving average calculation (default: 20)
        num_std: Number of standard deviations for bands (default: 2)
        column: Column name to use for calculation (default: 'close')
        
    Returns:
        DataFrame containing Middle Band, Upper Band, Lower Band, and Position
    """
    # Make a copy of the data to avoid modifying the original
    price = data[column].copy()
    
    # Calculate middle band (SMA)
    middle_band = price.rolling(window=window, min_periods=window).mean()
    
    # Calculate standard deviation
    std = price.rolling(window=window, min_periods=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # Calculate position within the bands (0 to 1, where 0 is at lower band and 1 is at upper band)
    band_range = upper_band - lower_band
    position = (price - lower_band) / band_range
    
    # Create a DataFrame with the results
    bb_df = pd.DataFrame({
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band,
        'position': position
    }, index=data.index)
    
    return bb_df


def calculate_atr(data, window=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        data: DataFrame containing OHLC price data
        window: Window for ATR calculation (default: 14)
        
    Returns:
        Series containing ATR values
    """
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=window, min_periods=window).mean()
    
    return atr


def calculate_ema(data, window=20, column='close'):
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        data: DataFrame containing price data
        window: Window for EMA calculation (default: 20)
        column: Column name to use for calculation (default: 'close')
        
    Returns:
        Series containing EMA values
    """
    # Make a copy of the data to avoid modifying the original
    price = data[column].copy()
    
    # Calculate EMA
    ema = price.ewm(span=window, min_periods=window).mean()
    
    return ema


def calculate_sma(data, window=20, column='close'):
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data: DataFrame containing price data
        window: Window for SMA calculation (default: 20)
        column: Column name to use for calculation (default: 'close')
        
    Returns:
        Series containing SMA values
    """
    # Make a copy of the data to avoid modifying the original
    price = data[column].copy()
    
    # Calculate SMA
    sma = price.rolling(window=window, min_periods=window).mean()
    
    return sma


def calculate_stochastic(data, k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data: DataFrame containing OHLC price data
        k_window: Window for %K calculation (default: 14)
        d_window: Window for %D calculation (default: 3)
        
    Returns:
        DataFrame containing %K and %D values
    """
    # Calculate %K
    low_min = data['low'].rolling(window=k_window, min_periods=k_window).min()
    high_max = data['high'].rolling(window=k_window, min_periods=k_window).max()
    
    k = 100 * ((data['close'] - low_min) / (high_max - low_min))
    
    # Calculate %D
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    
    # Create a DataFrame with the results
    stoch_df = pd.DataFrame({
        'k': k,
        'd': d
    }, index=data.index)
    
    return stoch_df


def calculate_adx(data, window=14):
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        data: DataFrame containing OHLC price data
        window: Window for ADX calculation (default: 14)
        
    Returns:
        DataFrame containing ADX, +DI, and -DI values
    """
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = data['high'] - data['high'].shift(1)
    down_move = data['low'].shift(1) - data['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)
    
    # Calculate Smoothed Directional Movement and True Range
    smoothed_plus_dm = plus_dm.rolling(window=window, min_periods=window).sum()
    smoothed_minus_dm = minus_dm.rolling(window=window, min_periods=window).sum()
    smoothed_tr = tr.rolling(window=window, min_periods=window).sum()
    
    # Calculate Directional Indicators
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
    
    # Calculate Directional Index
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx = dx.rolling(window=window, min_periods=window).mean()
    
    # Create a DataFrame with the results
    adx_df = pd.DataFrame({
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di
    }, index=data.index)
    
    return adx_df


def calculate_volatility(data, window=20, column='close'):
    """
    Calculate price volatility (standard deviation of returns).
    
    Args:
        data: DataFrame containing price data
        window: Window for volatility calculation (default: 20)
        column: Column name to use for calculation (default: 'close')
        
    Returns:
        Series containing volatility values
    """
    # Calculate returns
    returns = data[column].pct_change()
    
    # Calculate rolling standard deviation of returns
    volatility = returns.rolling(window=window, min_periods=window).std()
    
    return volatility


def calculate_trend_strength(data, window=20, column='close'):
    """
    Calculate trend strength using linear regression R-squared.
    
    Args:
        data: DataFrame containing price data
        window: Window for trend strength calculation (default: 20)
        column: Column name to use for calculation (default: 'close')
        
    Returns:
        Series containing trend strength values (R-squared)
    """
    # Make a copy of the data to avoid modifying the original
    price = data[column].copy()
    
    # Initialize trend strength series
    trend_strength = pd.Series(np.nan, index=data.index)
    
    # Calculate trend strength for each window
    for i in range(window, len(price) + 1):
        # Get window data
        window_data = price.iloc[i-window:i]
        
        # Create X values (0 to window-1)
        x = np.arange(window)
        
        # Calculate linear regression
        slope, intercept = np.polyfit(x, window_data, 1)
        
        # Calculate predicted values
        predicted = intercept + slope * x
        
        # Calculate R-squared
        ss_total = np.sum((window_data - window_data.mean()) ** 2)
        ss_residual = np.sum((window_data - predicted) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        # Store R-squared value
        trend_strength.iloc[i-1] = r_squared
    
    return trend_strength


def calculate_price_channels(data, window=20):
    """
    Calculate Price Channels (Donchian Channels).
    
    Args:
        data: DataFrame containing OHLC price data
        window: Window for channel calculation (default: 20)
        
    Returns:
        DataFrame containing Upper Channel, Lower Channel, and Middle Channel values
    """
    # Calculate upper and lower channels
    upper_channel = data['high'].rolling(window=window, min_periods=window).max()
    lower_channel = data['low'].rolling(window=window, min_periods=window).min()
    
    # Calculate middle channel
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Create a DataFrame with the results
    channels_df = pd.DataFrame({
        'upper': upper_channel,
        'lower': lower_channel,
        'middle': middle_channel
    }, index=data.index)
    
    return channels_df


def detect_breakouts(data, channel_window=20, threshold=1.0):
    """
    Detect price breakouts from channels.
    
    Args:
        data: DataFrame containing OHLC price data
        channel_window: Window for channel calculation (default: 20)
        threshold: Threshold for breakout detection as multiple of ATR (default: 1.0)
        
    Returns:
        DataFrame containing Breakout signals (1 for upward, -1 for downward, 0 for none)
    """
    # Calculate price channels
    channels = calculate_price_channels(data, window=channel_window)
    
    # Calculate ATR for threshold
    atr = calculate_atr(data, window=channel_window)
    
    # Detect breakouts
    upward_breakout = (data['close'] > channels['upper'] + (atr * threshold)).astype(int)
    downward_breakout = (data['close'] < channels['lower'] - (atr * threshold)).astype(int) * -1
    
    # Combine signals
    breakout = upward_breakout + downward_breakout
    
    return pd.DataFrame({'breakout': breakout}, index=data.index)
