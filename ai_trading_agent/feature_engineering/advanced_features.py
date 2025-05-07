"""
Advanced Feature Engineering Module

This module provides high-performance implementations of advanced technical indicators
using Rust extensions for maximum performance.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional
import logging

# Import Rust extension
try:
    from ai_trading_agent.rust_extensions import (
        create_bollinger_bands_rs,
        create_rsi_features_rs,
        create_macd_features_rs,
        calculate_fibonacci_retracement_rs,
        calculate_pivot_points_rs,
        calculate_volume_profile_rs,
        create_ichimoku_cloud_rs,
        create_adx_features_rs,
        create_obv_features_rs
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust extensions not available. Using slower Python implementations.")

logger = logging.getLogger(__name__)


def create_bollinger_bands(
    series: Union[pd.Series, np.ndarray, List[float]],
    windows: List[int],
    num_std: float = 2.0,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create Bollinger Bands features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        windows: List of window sizes for calculating Bollinger Bands
        num_std: Number of standard deviations for the bands
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with Bollinger Bands features as columns
    """
    # Input validation
    if not windows:
        raise ValueError("windows must be a non-empty list of integers")
    
    # Convert input to list if needed
    if isinstance(series, pd.Series):
        series_values = series.values.tolist()
        index = series.index
    elif isinstance(series, np.ndarray):
        series_values = series.tolist()
        index = pd.RangeIndex(len(series))
    else:
        series_values = series
        index = pd.RangeIndex(len(series))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_bollinger_bands_rs(series_values, windows, num_std)
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            for i, window in enumerate(windows):
                # Extract bands for this window
                middle_band = [row[i][0] for row in result]
                upper_band = [row[i][1] for row in result]
                lower_band = [row[i][2] for row in result]
                
                # Add to DataFrame
                df[f'bb_{window}_middle'] = middle_band
                df[f'bb_{window}_upper'] = upper_band
                df[f'bb_{window}_lower'] = lower_band
            
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        
        # Calculate Bollinger Bands
        for window in windows:
            if window <= 0:
                raise ValueError(f"Window sizes must be positive integers, got {window}")
            
            rolling = series.rolling(window=window)
            middle_band = rolling.mean()
            std_dev = rolling.std()
            
            df[f'bb_{window}_middle'] = middle_band
            df[f'bb_{window}_upper'] = middle_band + (std_dev * num_std)
            df[f'bb_{window}_lower'] = middle_band - (std_dev * num_std)
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create Bollinger Bands features")


def create_rsi_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    periods: List[int],
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create Relative Strength Index (RSI) features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        periods: List of periods for calculating RSI
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with RSI features as columns
    """
    # Input validation
    if not periods:
        raise ValueError("periods must be a non-empty list of integers")
    
    # Convert input to list if needed
    if isinstance(series, pd.Series):
        series_values = series.values.tolist()
        index = series.index
    elif isinstance(series, np.ndarray):
        series_values = series.tolist()
        index = pd.RangeIndex(len(series))
    else:
        series_values = series
        index = pd.RangeIndex(len(series))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_rsi_features_rs(series_values, periods)
            
            # Convert result to DataFrame
            df = pd.DataFrame(result, index=index)
            df.columns = [f'rsi_{period}' for period in periods]
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        
        # Calculate RSI for each period
        for period in periods:
            if period <= 1:
                raise ValueError(f"RSI periods must be greater than 1, got {period}")
            
            # Calculate price changes
            delta = series.diff()
            
            # Create gain and loss series
            gain = delta.copy()
            gain[gain < 0] = 0
            loss = -delta.copy()
            loss[loss < 0] = 0
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            df[f'rsi_{period}'] = rsi
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create RSI features")


def create_macd_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    fast_periods: List[int],
    slow_periods: List[int],
    signal_period: int = 9,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create Moving Average Convergence Divergence (MACD) features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        fast_periods: List of fast EMA periods
        slow_periods: List of slow EMA periods
        signal_period: Number of periods for the signal line
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with MACD features as columns
    """
    # Input validation
    if not fast_periods or not slow_periods:
        raise ValueError("fast_periods and slow_periods must be non-empty lists of integers")
    
    if signal_period <= 0:
        raise ValueError(f"signal_period must be a positive integer, got {signal_period}")
    
    # Convert input to list if needed
    if isinstance(series, pd.Series):
        series_values = series.values.tolist()
        index = series.index
    elif isinstance(series, np.ndarray):
        series_values = series.tolist()
        index = pd.RangeIndex(len(series))
    else:
        series_values = series
        index = pd.RangeIndex(len(series))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_macd_features_rs(series_values, fast_periods, slow_periods, signal_period)
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            # Flatten the combinations
            combination_idx = 0
            for fast_period in fast_periods:
                for slow_period in slow_periods:
                    # Extract MACD components for this combination
                    macd_line = [row[combination_idx][0] for row in result]
                    signal_line = [row[combination_idx][1] for row in result]
                    histogram = [row[combination_idx][2] for row in result]
                    
                    # Add to DataFrame
                    df[f'macd_{fast_period}_{slow_period}_line'] = macd_line
                    df[f'macd_{fast_period}_{slow_period}_signal'] = signal_line
                    df[f'macd_{fast_period}_{slow_period}_hist'] = histogram
                    
                    combination_idx += 1
            
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        
        # Calculate MACD for each combination of fast and slow periods
        for fast_period in fast_periods:
            if fast_period <= 0:
                raise ValueError(f"Fast periods must be positive integers, got {fast_period}")
            
            for slow_period in slow_periods:
                if slow_period <= 0:
                    raise ValueError(f"Slow periods must be positive integers, got {slow_period}")
                
                if fast_period >= slow_period:
                    raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
                
                # Calculate fast and slow EMAs
                fast_ema = series.ewm(span=fast_period, adjust=False).mean()
                slow_ema = series.ewm(span=slow_period, adjust=False).mean()
                
                # Calculate MACD line
                macd_line = fast_ema - slow_ema
                
                # Calculate signal line
                signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
                
                # Calculate histogram
                histogram = macd_line - signal_line
                
                # Add to DataFrame
                df[f'macd_{fast_period}_{slow_period}_line'] = macd_line
                df[f'macd_{fast_period}_{slow_period}_signal'] = signal_line
                df[f'macd_{fast_period}_{slow_period}_hist'] = histogram
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create MACD features")


def calculate_fibonacci_retracement(
    high_prices: Union[pd.Series, np.ndarray, List[float]],
    low_prices: Union[pd.Series, np.ndarray, List[float]],
    is_uptrend: bool = True,
    levels: Optional[List[float]] = None,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Calculate Fibonacci retracement levels from high and low points.
    
    Args:
        high_prices: High prices (pandas Series, numpy array, or list)
        low_prices: Low prices (pandas Series, numpy array, or list)
        is_uptrend: Whether the trend is up (True) or down (False)
        levels: List of Fibonacci levels to calculate (default: [0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with Fibonacci retracement levels as columns
    """
    # Default Fibonacci levels if not provided
    if levels is None:
        levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    # Convert input to list if needed
    if isinstance(high_prices, pd.Series):
        high_values = high_prices.values.tolist()
        low_values = low_prices.values.tolist() if isinstance(low_prices, pd.Series) else low_prices
        index = high_prices.index
    elif isinstance(high_prices, np.ndarray):
        high_values = high_prices.tolist()
        low_values = low_prices.tolist() if isinstance(low_prices, np.ndarray) else low_prices
        index = pd.RangeIndex(len(high_prices))
    else:
        high_values = high_prices
        low_values = low_prices
        index = pd.RangeIndex(len(high_prices))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = calculate_fibonacci_retracement_rs(high_values, low_values, is_uptrend, levels)
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            for i, level in enumerate(levels):
                # Extract retracement level
                retracement = [row[i] for row in result]
                
                # Add to DataFrame
                df[f'fib_{level:.3f}'] = retracement
            
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(high_prices, pd.Series):
            high_prices = pd.Series(high_prices, index=index)
        if not isinstance(low_prices, pd.Series):
            low_prices = pd.Series(low_prices, index=index)
        
        # Find the highest high and lowest low
        highest_high = high_prices.max()
        lowest_low = low_prices.min()
        
        # Calculate the range
        price_range = highest_high - lowest_low
        
        # Calculate retracement levels
        for level in levels:
            if is_uptrend:
                # For uptrend: high - (range * level)
                df[f'fib_{level:.3f}'] = highest_high - (price_range * level)
            else:
                # For downtrend: low + (range * level)
                df[f'fib_{level:.3f}'] = lowest_low + (price_range * level)
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to calculate Fibonacci retracement levels")


def calculate_pivot_points(
    high_prices: Union[pd.Series, np.ndarray, List[float]],
    low_prices: Union[pd.Series, np.ndarray, List[float]],
    close_prices: Union[pd.Series, np.ndarray, List[float]],
    pivot_type: str = "standard",
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Calculate pivot points for price data.
    
    Args:
        high_prices: High prices (pandas Series, numpy array, or list)
        low_prices: Low prices (pandas Series, numpy array, or list)
        close_prices: Close prices (pandas Series, numpy array, or list)
        pivot_type: Type of pivot points to calculate ("standard", "fibonacci", "camarilla", "woodie")
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with pivot points as columns
    """
    # Input validation
    valid_types = ["standard", "fibonacci", "camarilla", "woodie"]
    if pivot_type not in valid_types:
        raise ValueError(f"pivot_type must be one of {valid_types}, got {pivot_type}")
    
    # Convert input to list if needed
    if isinstance(high_prices, pd.Series):
        high_values = high_prices.values.tolist()
        low_values = low_prices.values.tolist() if isinstance(low_prices, pd.Series) else low_prices
        close_values = close_prices.values.tolist() if isinstance(close_prices, pd.Series) else close_prices
        index = high_prices.index
    elif isinstance(high_prices, np.ndarray):
        high_values = high_prices.tolist()
        low_values = low_prices.tolist() if isinstance(low_prices, np.ndarray) else low_prices
        close_values = close_prices.tolist() if isinstance(close_prices, np.ndarray) else close_prices
        index = pd.RangeIndex(len(high_prices))
    else:
        high_values = high_prices
        low_values = low_prices
        close_values = close_prices
        index = pd.RangeIndex(len(high_prices))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = calculate_pivot_points_rs(high_values, low_values, close_values, pivot_type)
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            # Define column names based on pivot type
            if pivot_type == "standard" or pivot_type == "woodie":
                columns = ["PP", "S1", "S2", "R1", "R2"]
            elif pivot_type == "fibonacci":
                columns = ["PP", "S1", "S2", "S3", "R1", "R2", "R3"]
            elif pivot_type == "camarilla":
                columns = ["PP", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4"]
            
            # Add pivot points to DataFrame
            for i, col in enumerate(columns):
                df[f'{pivot_type}_{col}'] = [row[i] if i < len(row) else np.nan for row in result]
            
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(high_prices, pd.Series):
            high_prices = pd.Series(high_prices, index=index)
        if not isinstance(low_prices, pd.Series):
            low_prices = pd.Series(low_prices, index=index)
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices, index=index)
        
        # Calculate pivot points
        high_shifted = high_prices.shift(1)
        low_shifted = low_prices.shift(1)
        close_shifted = close_prices.shift(1)
        
        if pivot_type == "standard":
            # Standard pivot points
            pp = (high_shifted + low_shifted + close_shifted) / 3
            r1 = (2 * pp) - low_shifted
            r2 = pp + (high_shifted - low_shifted)
            s1 = (2 * pp) - high_shifted
            s2 = pp - (high_shifted - low_shifted)
            
            df['standard_PP'] = pp
            df['standard_S1'] = s1
            df['standard_S2'] = s2
            df['standard_R1'] = r1
            df['standard_R2'] = r2
        
        elif pivot_type == "fibonacci":
            # Fibonacci pivot points
            pp = (high_shifted + low_shifted + close_shifted) / 3
            r1 = pp + 0.382 * (high_shifted - low_shifted)
            r2 = pp + 0.618 * (high_shifted - low_shifted)
            r3 = pp + 1.0 * (high_shifted - low_shifted)
            s1 = pp - 0.382 * (high_shifted - low_shifted)
            s2 = pp - 0.618 * (high_shifted - low_shifted)
            s3 = pp - 1.0 * (high_shifted - low_shifted)
            
            df['fibonacci_PP'] = pp
            df['fibonacci_S1'] = s1
            df['fibonacci_S2'] = s2
            df['fibonacci_S3'] = s3
            df['fibonacci_R1'] = r1
            df['fibonacci_R2'] = r2
            df['fibonacci_R3'] = r3
        
        elif pivot_type == "camarilla":
            # Camarilla pivot points
            pp = (high_shifted + low_shifted + close_shifted) / 3
            r1 = close_shifted + 1.1 / 12.0 * (high_shifted - low_shifted)
            r2 = close_shifted + 1.1 / 6.0 * (high_shifted - low_shifted)
            r3 = close_shifted + 1.1 / 4.0 * (high_shifted - low_shifted)
            r4 = close_shifted + 1.1 / 2.0 * (high_shifted - low_shifted)
            s1 = close_shifted - 1.1 / 12.0 * (high_shifted - low_shifted)
            s2 = close_shifted - 1.1 / 6.0 * (high_shifted - low_shifted)
            s3 = close_shifted - 1.1 / 4.0 * (high_shifted - low_shifted)
            s4 = close_shifted - 1.1 / 2.0 * (high_shifted - low_shifted)
            
            df['camarilla_PP'] = pp
            df['camarilla_S1'] = s1
            df['camarilla_S2'] = s2
            df['camarilla_S3'] = s3
            df['camarilla_S4'] = s4
            df['camarilla_R1'] = r1
            df['camarilla_R2'] = r2
            df['camarilla_R3'] = r3
            df['camarilla_R4'] = r4
        
        elif pivot_type == "woodie":
            # Woodie's pivot points
            pp = (high_shifted + low_shifted + (2 * close_shifted)) / 4
            r1 = (2 * pp) - low_shifted
            r2 = pp + (high_shifted - low_shifted)
            s1 = (2 * pp) - high_shifted
            s2 = pp - (high_shifted - low_shifted)
            
            df['woodie_PP'] = pp
            df['woodie_S1'] = s1
            df['woodie_S2'] = s2
            df['woodie_R1'] = r1
            df['woodie_R2'] = r2
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to calculate pivot points")


def calculate_volume_profile(
    prices: Union[pd.Series, np.ndarray, List[float]],
    volumes: Union[pd.Series, np.ndarray, List[float]],
    n_bins: int = 10,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Calculate volume profile for price data.
    
    Args:
        prices: Price values (pandas Series, numpy array, or list)
        volumes: Volume values (pandas Series, numpy array, or list)
        n_bins: Number of price bins to use
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with price levels and volume at each level
    """
    # Input validation
    if n_bins <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")
    
    # Convert input to list if needed
    if isinstance(prices, pd.Series):
        price_values = prices.values.tolist()
        volume_values = volumes.values.tolist() if isinstance(volumes, pd.Series) else volumes
        index = prices.index
    elif isinstance(prices, np.ndarray):
        price_values = prices.tolist()
        volume_values = volumes.tolist() if isinstance(volumes, np.ndarray) else volumes
        index = pd.RangeIndex(len(prices))
    else:
        price_values = prices
        volume_values = volumes
        index = pd.RangeIndex(len(prices))
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = calculate_volume_profile_rs(price_values, volume_values, n_bins)
            
            # Convert result to DataFrame
            price_levels = [row[0] for row in result]
            volume_at_levels = [row[1] for row in result]
            
            df = pd.DataFrame({
                'price_level': price_levels,
                'volume': volume_at_levels
            })
            
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Convert to numpy arrays for efficient calculation
        if not isinstance(prices, np.ndarray):
            prices = np.array(price_values)
        if not isinstance(volumes, np.ndarray):
            volumes = np.array(volume_values)
        
        # Find the min and max prices
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        # Calculate the bin size
        bin_size = (max_price - min_price) / n_bins
        
        # Create bins for the volume profile
        bins = np.zeros(n_bins)
        bin_prices = np.zeros(n_bins)
        
        # Calculate the center price for each bin
        for i in range(n_bins):
            bin_prices[i] = min_price + (i + 0.5) * bin_size
        
        # Assign volumes to bins
        for i in range(len(prices)):
            bin_index = int((prices[i] - min_price) / bin_size)
            if 0 <= bin_index < n_bins:
                bins[bin_index] += volumes[i]
        
        # Create DataFrame
        df = pd.DataFrame({
            'price_level': bin_prices,
            'volume': bins
        })
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to calculate volume profile")


def create_all_advanced_features(
    series: Union[pd.Series, np.ndarray, List[float]],
    high_prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    low_prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    volumes: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    bb_windows: Optional[List[int]] = None,
    bb_num_std: float = 2.0,
    rsi_periods: Optional[List[int]] = None,
    macd_fast_periods: Optional[List[int]] = None,
    macd_slow_periods: Optional[List[int]] = None,
    macd_signal_period: int = 9,
    include_ichimoku: bool = False,
    include_adx: bool = False,
    include_obv: bool = False,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create all types of advanced features from a time series.
    
    Args:
        series: Input time series (pandas Series, numpy array, or list)
        bb_windows: List of window sizes for Bollinger Bands
        bb_num_std: Number of standard deviations for Bollinger Bands
        rsi_periods: List of periods for RSI
        macd_fast_periods: List of fast EMA periods for MACD
        macd_slow_periods: List of slow EMA periods for MACD
        macd_signal_period: Number of periods for the MACD signal line
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with all advanced features as columns
    """
    # Convert input to pandas Series if needed
    if isinstance(series, np.ndarray) or isinstance(series, list):
        series = pd.Series(series)
    
    # Create DataFrame to store results
    result_df = pd.DataFrame(index=series.index)
    
    # Add original series as a column
    result_df['original'] = series
    
    # Create Bollinger Bands features
    if bb_windows:
        bb_df = create_bollinger_bands(series, bb_windows, bb_num_std, fallback_to_python)
        result_df = pd.concat([result_df, bb_df], axis=1)
    
    # Create RSI features
    if rsi_periods:
        rsi_df = create_rsi_features(series, rsi_periods, fallback_to_python)
        result_df = pd.concat([result_df, rsi_df], axis=1)
    
    # Create MACD features
    if macd_fast_periods and macd_slow_periods:
        macd_df = create_macd_features(
            series, macd_fast_periods, macd_slow_periods, macd_signal_period, fallback_to_python
        )
        result_df = pd.concat([result_df, macd_df], axis=1)
    
    # Create Ichimoku Cloud features if requested and high/low prices are provided
    if include_ichimoku and high_prices is not None and low_prices is not None:
        ichimoku_df = create_ichimoku_cloud(high_prices, low_prices, series, fallback_to_python=fallback_to_python)
        result_df = pd.concat([result_df, ichimoku_df], axis=1)
    
    # Create ADX features if requested and high/low prices are provided
    if include_adx and high_prices is not None and low_prices is not None:
        adx_df = create_adx_features(high_prices, low_prices, series, fallback_to_python=fallback_to_python)
        result_df = pd.concat([result_df, adx_df], axis=1)
    
    # Create OBV features if requested and volumes are provided
    if include_obv and volumes is not None:
        obv_df = create_obv_features(series, volumes, fallback_to_python=fallback_to_python)
        result_df = pd.concat([result_df, obv_df], axis=1)
    
    return result_df


def create_ichimoku_cloud(
    high_prices: Union[pd.Series, np.ndarray, List[float]],
    low_prices: Union[pd.Series, np.ndarray, List[float]],
    close_prices: Union[pd.Series, np.ndarray, List[float]],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create Ichimoku Cloud features from price data.
    
    Args:
        high_prices: High prices (pandas Series, numpy array, or list)
        low_prices: Low prices (pandas Series, numpy array, or list)
        close_prices: Close prices (pandas Series, numpy array, or list)
        tenkan_period: Period for Tenkan-sen (Conversion Line)
        kijun_period: Period for Kijun-sen (Base Line)
        senkou_span_b_period: Period for Senkou Span B (Leading Span B)
        displacement: Displacement period for Senkou Span A and B
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with Ichimoku Cloud features as columns
    """
    # Input validation
    if tenkan_period <= 0 or kijun_period <= 0 or senkou_span_b_period <= 0 or displacement <= 0:
        raise ValueError("All periods must be positive integers")
    
    # Convert input to list if needed
    if isinstance(high_prices, pd.Series):
        high_values = high_prices.values.tolist()
        index = high_prices.index
    elif isinstance(high_prices, np.ndarray):
        high_values = high_prices.tolist()
        index = pd.RangeIndex(len(high_prices))
    else:
        high_values = high_prices
        index = pd.RangeIndex(len(high_prices))
    
    if isinstance(low_prices, pd.Series):
        low_values = low_prices.values.tolist()
    elif isinstance(low_prices, np.ndarray):
        low_values = low_prices.tolist()
    else:
        low_values = low_prices
    
    if isinstance(close_prices, pd.Series):
        close_values = close_prices.values.tolist()
    elif isinstance(close_prices, np.ndarray):
        close_values = close_prices.tolist()
    else:
        close_values = close_prices
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_ichimoku_cloud_rs(
                high_values,
                low_values,
                close_values,
                tenkan_period,
                kijun_period,
                senkou_span_b_period,
                displacement
            )
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            # Extract Ichimoku Cloud components
            df['tenkan_sen'] = result['tenkan_sen']
            df['kijun_sen'] = result['kijun_sen']
            df['senkou_span_a'] = result['senkou_span_a']
            df['senkou_span_b'] = result['senkou_span_b']
            df['chikou_span'] = result['chikou_span']
            
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(high_prices, pd.Series):
            high_prices = pd.Series(high_prices, index=index)
        if not isinstance(low_prices, pd.Series):
            low_prices = pd.Series(low_prices, index=index)
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices, index=index)
        
        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
        tenkan_sen = (high_prices.rolling(window=tenkan_period).max() + 
                      low_prices.rolling(window=tenkan_period).min()) / 2
        
        # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
        kijun_sen = (high_prices.rolling(window=kijun_period).max() + 
                     low_prices.rolling(window=kijun_period).min()) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, displaced forward by displacement periods
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_span_b_period, displaced forward by displacement periods
        senkou_span_b = ((high_prices.rolling(window=senkou_span_b_period).max() + 
                          low_prices.rolling(window=senkou_span_b_period).min()) / 2).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span): Close price, displaced backwards by displacement periods
        chikou_span = close_prices.shift(-displacement)
        
        # Add to DataFrame
        df['tenkan_sen'] = tenkan_sen
        df['kijun_sen'] = kijun_sen
        df['senkou_span_a'] = senkou_span_a
        df['senkou_span_b'] = senkou_span_b
        df['chikou_span'] = chikou_span
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create Ichimoku Cloud features")


def create_adx_features(
    high_prices: Union[pd.Series, np.ndarray, List[float]],
    low_prices: Union[pd.Series, np.ndarray, List[float]],
    close_prices: Union[pd.Series, np.ndarray, List[float]],
    period: int = 14,
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create Average Directional Index (ADX) features from price data.
    
    Args:
        high_prices: High prices (pandas Series, numpy array, or list)
        low_prices: Low prices (pandas Series, numpy array, or list)
        close_prices: Close prices (pandas Series, numpy array, or list)
        period: Period for ADX calculation
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with ADX features as columns
    """
    # Input validation
    if period <= 0:
        raise ValueError("Period must be a positive integer")
    
    # Convert input to list if needed
    if isinstance(high_prices, pd.Series):
        high_values = high_prices.values.tolist()
        index = high_prices.index
    elif isinstance(high_prices, np.ndarray):
        high_values = high_prices.tolist()
        index = pd.RangeIndex(len(high_prices))
    else:
        high_values = high_prices
        index = pd.RangeIndex(len(high_prices))
    
    if isinstance(low_prices, pd.Series):
        low_values = low_prices.values.tolist()
    elif isinstance(low_prices, np.ndarray):
        low_values = low_prices.tolist()
    else:
        low_values = low_prices
    
    if isinstance(close_prices, pd.Series):
        close_values = close_prices.values.tolist()
    elif isinstance(close_prices, np.ndarray):
        close_values = close_prices.tolist()
    else:
        close_values = close_prices
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_adx_features_rs(
                high_values,
                low_values,
                close_values,
                period
            )
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            # Extract ADX components
            df['adx'] = result['adx']
            df['di_plus'] = result['di_plus']
            df['di_minus'] = result['di_minus']
            
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(high_prices, pd.Series):
            high_prices = pd.Series(high_prices, index=index)
        if not isinstance(low_prices, pd.Series):
            low_prices = pd.Series(low_prices, index=index)
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices, index=index)
        
        # Calculate True Range (TR)
        tr1 = high_prices - low_prices
        tr2 = (high_prices - close_prices.shift(1)).abs()
        tr3 = (low_prices - close_prices.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high_prices - high_prices.shift(1)
        down_move = low_prices.shift(1) - low_prices
        
        # Calculate Plus Directional Movement (+DM)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        plus_dm = pd.Series(plus_dm, index=index)
        
        # Calculate Minus Directional Movement (-DM)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        minus_dm = pd.Series(minus_dm, index=index)
        
        # Calculate Smoothed TR, +DM, and -DM
        smoothed_tr = tr.rolling(window=period).sum()
        smoothed_plus_dm = plus_dm.rolling(window=period).sum()
        smoothed_minus_dm = minus_dm.rolling(window=period).sum()
        
        # Calculate Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
        di_plus = 100 * (smoothed_plus_dm / smoothed_tr)
        di_minus = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # Calculate Directional Index (DX)
        dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
        
        # Calculate Average Directional Index (ADX)
        adx = dx.rolling(window=period).mean()
        
        # Add to DataFrame
        df['adx'] = adx
        df['di_plus'] = di_plus
        df['di_minus'] = di_minus
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create ADX features")


def create_obv_features(
    close_prices: Union[pd.Series, np.ndarray, List[float]],
    volumes: Union[pd.Series, np.ndarray, List[float]],
    fallback_to_python: bool = True
) -> pd.DataFrame:
    """
    Create On-Balance Volume (OBV) features from price and volume data.
    
    Args:
        close_prices: Close prices (pandas Series, numpy array, or list)
        volumes: Volume data (pandas Series, numpy array, or list)
        fallback_to_python: Whether to fall back to Python implementation if Rust is not available
        
    Returns:
        DataFrame with OBV features as columns
    """
    # Convert input to list if needed
    if isinstance(close_prices, pd.Series):
        close_values = close_prices.values.tolist()
        index = close_prices.index
    elif isinstance(close_prices, np.ndarray):
        close_values = close_prices.tolist()
        index = pd.RangeIndex(len(close_prices))
    else:
        close_values = close_prices
        index = pd.RangeIndex(len(close_prices))
    
    if isinstance(volumes, pd.Series):
        volume_values = volumes.values.tolist()
    elif isinstance(volumes, np.ndarray):
        volume_values = volumes.tolist()
    else:
        volume_values = volumes
    
    # Check if we can use Rust implementation
    if RUST_AVAILABLE:
        try:
            # Call Rust implementation
            result = create_obv_features_rs(
                close_values,
                volume_values
            )
            
            # Convert result to DataFrame
            df = pd.DataFrame(index=index)
            
            # Extract OBV and OBV EMA
            df['obv'] = result['obv']
            df['obv_ema'] = result['obv_ema']
            
            return df
        except Exception as e:
            if not fallback_to_python:
                raise
            logger.warning(f"Rust implementation failed: {e}. Falling back to Python.")
    
    # Fall back to Python implementation if Rust is not available or failed
    if not RUST_AVAILABLE or fallback_to_python:
        # Create DataFrame to store results
        df = pd.DataFrame(index=index)
        
        # Convert to pandas Series if not already
        if not isinstance(close_prices, pd.Series):
            close_prices = pd.Series(close_prices, index=index)
        if not isinstance(volumes, pd.Series):
            volumes = pd.Series(volumes, index=index)
        
        # Calculate price changes
        price_change = close_prices.diff()
        
        # Initialize OBV
        obv = pd.Series(0.0, index=index)
        
        # Calculate OBV
        for i in range(1, len(close_prices)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # Calculate OBV EMA (21-period)
        obv_ema = obv.ewm(span=21, adjust=False).mean()
        
        # Add to DataFrame
        df['obv'] = obv
        df['obv_ema'] = obv_ema
        
        return df
    
    # This should never happen, but just in case
    raise RuntimeError("Failed to create OBV features")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    import numpy as np
    
    # Create a sample price series
    np.random.seed(42)
    n_samples = 1000
    price = 100.0
    prices = [price]
    high_prices = [price * 1.01]  # 1% higher than close
    low_prices = [price * 0.99]   # 1% lower than close
    volumes = [1000]              # Sample volume
    
    for _ in range(1, n_samples):
        change_pct = np.random.normal(0, 0.01)
        price *= (1 + change_pct)
        prices.append(price)
        high_prices.append(price * (1 + abs(np.random.normal(0, 0.005))))
        low_prices.append(price * (1 - abs(np.random.normal(0, 0.005))))
        volumes.append(np.random.randint(500, 1500))
    
    # Convert to pandas Series
    price_series = pd.Series(prices)
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    volume_series = pd.Series(volumes)
    
    # Create advanced features
    bb_df = create_bollinger_bands(price_series, windows=[20, 50])
    rsi_df = create_rsi_features(price_series, periods=[14, 21])
    macd_df = create_macd_features(
        price_series, 
        fast_periods=[12], 
        slow_periods=[26], 
        signal_period=9
    )
    
    # Create Fibonacci retracement levels
    fib_df = calculate_fibonacci_retracement(high_series, low_series, is_uptrend=True)
    
    # Create pivot points
    pivot_df = calculate_pivot_points(high_series, low_series, price_series, pivot_type="standard")
    
    # Create volume profile
    volume_profile_df = calculate_volume_profile(price_series, volume_series, n_bins=10)
    
    # Create new advanced indicators
    ichimoku_df = create_ichimoku_cloud(high_series, low_series, price_series)
    adx_df = create_adx_features(high_series, low_series, price_series)
    obv_df = create_obv_features(price_series, volume_series)
    
    # Create all features at once
    all_features = create_all_advanced_features(
        price_series,
        bb_windows=[20, 50],
        rsi_periods=[14, 21],
        macd_fast_periods=[12],
        macd_slow_periods=[26]
    )
    
    # Print results
    print("Bollinger Bands features:")
    print(bb_df.head())
    
    print("\nRSI features:")
    print(rsi_df.head())
    
    print("\nMACD features:")
    print(macd_df.head())
    
    print("\nFibonacci retracement levels:")
    print(fib_df.head())
    
    print("\nPivot points:")
    print(pivot_df.head())
    
    print("\nVolume profile:")
    print(volume_profile_df.head())
    
    print("\nIchimoku Cloud:")
    print(ichimoku_df.head())
    
    print("\nADX features:")
    print(adx_df.head())
    
    print("\nOBV features:")
    print(obv_df.head())
    
    print("\nAll features:")
    print(all_features.head())
