"""
Rust-accelerated technical indicators module.

This module provides Python wrappers for technical indicators implemented in Rust.
"""
import numpy as np
from typing import Union, List, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Placeholders for imported Rust functions
_calculate_sma_rs_rust = None
_calculate_ema_rs_rust = None
_calculate_macd_rs_rust = None
_calculate_rsi_rs_rust = None
_calculate_bollinger_bands_rs_rust = None
_calculate_stochastic_rs_rust = None
_calculate_stochastic_oscillator_rs_rust = None
_calculate_atr_rs_rust = None

RUST_AVAILABLE = {
    "sma": False,
    "ema": False,
    "macd": False,
    "rsi": False,
    "bollinger_bands": False,
    "stochastic": False,
    "stochastic_oscillator": False,
    "atr": False,
}

# Attempt to import Rust functions and update RUST_AVAILABLE accordingly
try:
    from ai_trading_agent_rs import (
        calculate_sma_rust_direct,
        create_ema_features_rs,
        create_macd_features_rs,
        create_rsi_features_rs,
        create_bollinger_bands_rs,
        create_stochastic_oscillator_rs,
        # Placeholder for future stochastic, atr if they get similar direct names
    )

    _calculate_sma_rs_rust = calculate_sma_rust_direct
    RUST_AVAILABLE["sma"] = True
    logger.debug("Successfully imported 'calculate_sma_rust_direct' from Rust for SMA.")

    _calculate_ema_rs_rust = create_ema_features_rs
    RUST_AVAILABLE["ema"] = True
    logger.debug("Successfully imported 'create_ema_features_rs' from Rust for EMA.")

    _calculate_macd_rs_rust = create_macd_features_rs
    RUST_AVAILABLE["macd"] = True
    logger.debug("Successfully imported 'create_macd_features_rs' from Rust for MACD.")

    _calculate_rsi_rs_rust = create_rsi_features_rs
    RUST_AVAILABLE["rsi"] = True
    logger.debug("Successfully imported 'create_rsi_features_rs' from Rust for RSI.")

    _calculate_bollinger_bands_rs_rust = create_bollinger_bands_rs
    RUST_AVAILABLE["bollinger_bands"] = True
    logger.debug("Successfully imported 'create_bollinger_bands_rs' from Rust for Bollinger Bands.")

    _calculate_stochastic_oscillator_rs_rust = create_stochastic_oscillator_rs
    RUST_AVAILABLE["stochastic_oscillator"] = True
    RUST_AVAILABLE["stochastic"] = True
    logger.debug("Successfully imported 'create_stochastic_oscillator_rs' from Rust for Stochastic Oscillator.")

    # Attempt to import ATR (expected to fail for now if not implemented in Rust with this name)
    try:
        from ai_trading_agent_rs import calculate_atr_rs # Assuming this will be the name
        _calculate_atr_rs_rust = calculate_atr_rs
        RUST_AVAILABLE["atr"] = True
        logger.debug("Successfully imported 'calculate_atr_rs' from Rust for ATR.")
    except ImportError:
        logger.debug("Rust function 'calculate_atr_rs' not found. Python fallback for ATR.")

except ImportError as e:
    logger.warning(f"Could not import one or more primary Rust functions from ai_trading_agent_rs: {e}. Ensure the Rust library is compiled and installed correctly. All indicators will use Python fallbacks.")
    # RUST_AVAILABLE flags remain False as initialized

logger.info(f"Rust functions availability: {RUST_AVAILABLE}")

# This map helps the IndicatorEngine call the correct Rust function with the right parameters.
# It should be defined after the RUST_AVAILABLE checks and function assignments.

RUST_INDICATOR_MAP = {
    "sma": {
        "function": _calculate_sma_rs_rust,
        "params": ["series", "window"],
        "default_params": {"window": 20} # Example default
    },
    "ema": {
        "function": _calculate_ema_rs_rust,
        "params": ["series", "window"],
        "default_params": {"window": 20}
    },
    "macd": {
        "function": _calculate_macd_rs_rust,
        "params": ["series", "fast_period", "slow_period", "signal_period"],
        "default_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
    },
    "rsi": {
        "function": _calculate_rsi_rs_rust,
        "params": ["closes", "window"],
        "default_params": {"window": 14}
    },
    "bollinger_bands": {
        "function": _calculate_bollinger_bands_rs_rust,
        "params": ["series", "window", "num_std_dev"],
        "default_params": {"window": 20, "num_std_dev": 2}
    },
    "stochastic_oscillator": {
        "function": _calculate_stochastic_oscillator_rs_rust,
        "params": ["highs", "lows", "closes", "k_period", "d_period"],
        "default_params": {"k_period": 14, "d_period": 3}
    }
    # Add other Rust-accelerated indicators here as they are implemented
}

def calculate_sma(data: Union[List[float], np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA) using Rust implementation if available.
    
    Args:
        data: Price data as a list, numpy array, or pandas Series
        period: The window period for the SMA
        
    Returns:
        numpy.ndarray: Array of SMA values, with NaN values for the first (period-1) elements
        
    Raises:
        ValueError: If data is empty or period is <= 0
        TypeError: If data is not a list, numpy array, or pandas Series
    """
    # Handle pandas Series
    if isinstance(data, pd.Series):
        data = data.values
    
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("data must be a list or numpy array")
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
        
    if period <= 0:
        raise ValueError("period must be positive")
    
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    else:
        data_list = data
    
    # Special case for period = 1 (should return the original data)
    if period == 1:
        return np.array(data_list, dtype=float)
    
    if RUST_AVAILABLE["sma"] and callable(_calculate_sma_rs_rust):
        # Use Rust implementation
        result_with_nones = _calculate_sma_rs_rust(data_list, period)
        
        # Convert None values to np.nan
        result = np.array([np.nan if x is None else x for x in result_with_nones], dtype=float)
    else:
        # Fallback to Python implementation
        result = np.full(len(data), np.nan)
        
        # Calculate SMA
        for i in range(period - 1, len(data)):
            result[i] = sum(data_list[i - period + 1:i + 1]) / period
            
    return result


def calculate_ema(data: Union[List[float], np.ndarray], period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA) using Rust implementation if available.
    
    Args:
        data: Price data as a list or numpy array
        period: The window period for the EMA
        
    Returns:
        numpy.ndarray: Array of EMA values, with NaN values for the first (period-1) elements
        
    Raises:
        ValueError: If data is empty or period is <= 0
        TypeError: If data is not a list or numpy array
    """
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("data must be a list or numpy array")
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
        
    if period <= 0:
        raise ValueError("period must be positive")
    
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    else:
        data_list = data
    
    # Special case for period = 1 (should return the original data)
    if period == 1:
        return np.array(data_list, dtype=float)
    
    if RUST_AVAILABLE["ema"] and callable(_calculate_ema_rs_rust):
        # Use Rust implementation
        result_with_nones = _calculate_ema_rs_rust(data_list, period)
        
        # Convert None values to np.nan
        result = np.array([np.nan if x is None else x for x in result_with_nones], dtype=float)
    else:
        # Fallback to Python implementation
        result = np.full(len(data), np.nan)
        
        # Calculate EMA
        alpha = 2.0 / (period + 1)
        
        # Initialize EMA with SMA for the first period values
        if len(data_list) >= period:
            result[period-1] = sum(data_list[:period]) / period
            
            # Calculate EMA for the rest of the data
            for i in range(period, len(data_list)):
                result[i] = alpha * data_list[i] + (1 - alpha) * result[i-1]
            
    return result


def calculate_macd(
    data: Union[List[float], np.ndarray], 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD) using Rust implementation if available.
    
    Args:
        data: Price data as a list or numpy array
        fast_period: The fast EMA period (default: 12)
        slow_period: The slow EMA period (default: 26)
        signal_period: The signal EMA period (default: 9)
        
    Returns:
        Tuple of numpy.ndarray: (macd_line, signal_line, histogram), each with NaN values for padding
        
    Raises:
        ValueError: If data is empty, periods are invalid, or slow_period <= fast_period
        TypeError: If data is not a list or numpy array
    """
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("data must be a list or numpy array")
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
        
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("All periods must be positive")
        
    if slow_period <= fast_period:
        raise ValueError("slow_period must be greater than fast_period")
    
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    else:
        data_list = data
    
    # Check if we have enough data
    min_data_needed = slow_period + signal_period
    if len(data_list) < min_data_needed:
        # Return arrays of NaNs if not enough data
        nan_array = np.full(len(data_list), np.nan)
        return nan_array, nan_array, nan_array
    
    if RUST_AVAILABLE["macd"] and callable(_calculate_macd_rs_rust):
        # Use Rust implementation
        macd_line_nones, signal_line_nones, histogram_nones = _calculate_macd_rs_rust(
            data_list, fast_period, slow_period, signal_period
        )
        
        # Convert None values to np.nan
        macd_line = np.array([np.nan if x is None else x for x in macd_line_nones], dtype=float)
        signal_line = np.array([np.nan if x is None else x for x in signal_line_nones], dtype=float)
        histogram = np.array([np.nan if x is None else x for x in histogram_nones], dtype=float)
    else:
        # Fallback to Python implementation
        # Calculate fast and slow EMAs
        fast_ema = calculate_ema(data_list, fast_period)
        slow_ema = calculate_ema(data_list, slow_period)
        
        # Calculate MACD line (fast_ema - slow_ema)
        macd_line = np.full(len(data_list), np.nan)
        for i in range(slow_period - 1, len(data_list)):
            if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]):
                macd_line[i] = fast_ema[i] - slow_ema[i]
        
        # Calculate signal line (EMA of MACD line)
        # First, extract valid MACD values
        valid_macd_start = slow_period - 1
        valid_macd = macd_line[valid_macd_start:]
        
        # Calculate EMA of valid MACD values
        signal_ema = calculate_ema(valid_macd, signal_period)
        
        # Create signal line array
        signal_line = np.full(len(data_list), np.nan)
        signal_offset = valid_macd_start + signal_period - 1
        
        # Copy signal EMA values to the correct positions
        for i in range(len(signal_ema)):
            if i + valid_macd_start < len(signal_line) and not np.isnan(signal_ema[i]):
                signal_line[i + valid_macd_start] = signal_ema[i]
        
        # Calculate histogram (MACD line - signal line)
        histogram = np.full(len(data_list), np.nan)
        for i in range(len(data_list)):
            if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
                histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def calculate_rsi(data: Union[List[float], np.ndarray], period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI) using Rust implementation if available.
    
    Args:
        data: Price data as a list or numpy array
        period: The window period for the RSI (default: 14)
        
    Returns:
        numpy.ndarray: Array of RSI values, with NaN values for the first period elements
        
    Raises:
        ValueError: If data is empty or period is <= 0
        TypeError: If data is not a list or numpy array
    """
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("data must be a list or numpy array")
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
        
    if period <= 0:
        raise ValueError("period must be positive")
    
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    else:
        data_list = data
    
    # Check if we have enough data (need at least period+1 data points to calculate first RSI value)
    if len(data_list) <= period:
        # Return array of NaNs if not enough data
        return np.full(len(data_list), np.nan)
    
    # Always use the Python implementation for consistent results
    # This ensures compatibility with pandas' ewm implementation
    result = np.full(len(data_list), np.nan)
    
    # Calculate price changes
    changes = np.zeros(len(data_list))
    for i in range(1, len(data_list)):
        changes[i] = data_list[i] - data_list[i-1]
    
    # Separate gains and losses
    gains = np.maximum(changes, 0)
    losses = np.abs(np.minimum(changes, 0))
    
    # Calculate average gains and losses using EWM
    # This matches pandas' ewm(span=period, adjust=False) behavior
    alpha = 2.0 / (period + 1.0)
    
    # Initialize with SMA of first period values
    avg_gain = np.full(len(data_list), np.nan)
    avg_loss = np.full(len(data_list), np.nan)
    
    avg_gain[period] = np.sum(gains[1:period+1]) / period
    avg_loss[period] = np.sum(losses[1:period+1]) / period
    
    # Calculate smoothed averages for the rest of the data
    for i in range(period + 1, len(data_list)):
        avg_gain[i] = alpha * gains[i] + (1.0 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * losses[i] + (1.0 - alpha) * avg_loss[i-1]
    
    # Calculate RSI
    for i in range(period, len(data_list)):
        if avg_loss[i] == 0:
            # If no losses, RSI is 100
            result[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


def calculate_bollinger_bands(data: Union[List[float], np.ndarray, pd.Series], period: int = 20, deviations: float = 2.0) -> dict:
    """
    Calculate Bollinger Bands using Rust implementation if available.
    
    Args:
        data: Price data as a list, numpy array, or pandas Series
        period: The window period for the SMA (default: 20)
        deviations: Number of standard deviations for the bands (default: 2.0)
        
    Returns:
        dict: Dictionary with 'upper', 'middle', and 'lower' band arrays
        
    Raises:
        ValueError: If data is empty or period is <= 0
        TypeError: If data is not a list, numpy array, or pandas Series
    """
    # Handle pandas Series
    if isinstance(data, pd.Series):
        data = data.values
    
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("data must be a list or numpy array")
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
        
    if period <= 0:
        raise ValueError("period must be positive")
    
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    else:
        data_list = data
    
    # Special case for period = 1 (should use the original data as the middle band)
    if period == 1:
        data_array = np.array(data_list, dtype=float)
        return {
            'upper': data_array,
            'middle': data_array,
            'lower': data_array
        }
    
    if RUST_AVAILABLE["bollinger_bands"] and callable(_calculate_bollinger_bands_rs_rust):
        # Use Rust implementation if available
        upper, middle, lower = _calculate_bollinger_bands_rs_rust(data_list, period, deviations)
        
        # Convert None values to np.nan
        upper_array = np.array([np.nan if x is None else x for x in upper], dtype=float)
        middle_array = np.array([np.nan if x is None else x for x in middle], dtype=float)
        lower_array = np.array([np.nan if x is None else x for x in lower], dtype=float)
    else:
        # Fallback to Python implementation
        data_array = np.array(data_list, dtype=float)
        
        # Calculate SMA for middle band
        middle_band = calculate_sma(data_array, period)
        
        # Calculate standard deviation
        std_dev = np.full(len(data_array), np.nan)
        for i in range(period - 1, len(data_array)):
            std_dev[i] = np.std(data_array[i - period + 1:i + 1], ddof=0)
        
        # Calculate upper and lower bands
        upper_array = middle_band + (std_dev * deviations)
        lower_array = middle_band - (std_dev * deviations)
        middle_array = middle_band
    
    return {
        'upper': upper_array,
        'middle': middle_array,
        'lower': lower_array
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR) using Rust implementation if available.
    
    Args:
        df: DataFrame containing 'high', 'low', and 'close' columns
        period: The window period for the ATR (default: 14)
        
    Returns:
        numpy.ndarray: Array of ATR values, with NaN values for the first period elements
        
    Raises:
        ValueError: If DataFrame is empty, missing required columns, or period is <= 0
        TypeError: If df is not a pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    for col in ['high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column")
    
    if period <= 0:
        raise ValueError("period must be positive")
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    if RUST_AVAILABLE["atr"] and callable(_calculate_atr_rs_rust):
        # Use Rust implementation if available
        atr_with_nones = _calculate_atr_rs_rust(high.tolist(), low.tolist(), close.tolist(), period)
        
        # Convert None values to np.nan
        atr = np.array([np.nan if x is None else x for x in atr_with_nones], dtype=float)
    else:
        # Fallback to Python implementation
        n = len(high)
        atr = np.full(n, np.nan)
        
        # Calculate True Range first
        tr = np.zeros(n)
        
        # First TR value is just high - low
        tr[0] = high[0] - low[0]
        
        # Calculate subsequent TR values
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],              # Current high - current low
                abs(high[i] - close[i-1]),     # Current high - previous close
                abs(low[i] - close[i-1])       # Current low - previous close
            )
        
        # Initialize ATR with SMA of first 'period' TR values
        if n >= period:
            atr[period-1] = np.mean(tr[:period])
            
            # Calculate ATR using the smoothing formula
            for i in range(period, n):
                atr[i] = ((period - 1) * atr[i-1] + tr[i]) / period
    
    return atr