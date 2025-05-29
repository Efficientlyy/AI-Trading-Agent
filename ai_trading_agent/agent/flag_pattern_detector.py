"""
Flag Pattern Detection Module for the AI Trading Agent

This module provides specialized detection for flag patterns, including
bullish and bearish flags, which are continuation patterns that appear
after a strong price movement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal, stats
from datetime import datetime

from .pattern_types import PatternDetectionResult, PatternType


def detect_flag_patterns(
    df: pd.DataFrame, 
    symbol: str,
    peak_indices: np.ndarray,
    trough_indices: np.ndarray,
    params: Dict
) -> List[PatternDetectionResult]:
    """Detect flag patterns (bullish and bearish flags).
    
    Flag patterns are continuation patterns that form after a strong directional move.
    A bullish flag forms during an uptrend, while a bearish flag forms during a downtrend.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: The trading symbol
        peak_indices: Pre-computed indices of price peaks
        trough_indices: Pre-computed indices of price troughs
        params: Dictionary of parameters for pattern detection
        
    Returns:
        List of PatternDetectionResult objects
    """
    # Get prices for processing
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    
    # Find potential flag patterns
    bullish_flags = _detect_bullish_flags(df, high_prices, low_prices, close_prices, 
                                         peak_indices, trough_indices, symbol, params)
    bearish_flags = _detect_bearish_flags(df, high_prices, low_prices, close_prices, 
                                         peak_indices, trough_indices, symbol, params)
    
    return bullish_flags + bearish_flags


def _detect_bullish_flags(
    df: pd.DataFrame,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    peak_indices: np.ndarray,
    trough_indices: np.ndarray,
    symbol: str,
    params: Dict
) -> List[PatternDetectionResult]:
    """Detect bullish flag patterns.
    
    A bullish flag consists of a sharp upward move (the pole) followed by
    a consolidation period (the flag) that slopes slightly downward or sideways.
    
    Args:
        df: DataFrame with OHLCV data
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        peak_indices: Array of peak indices
        trough_indices: Array of trough indices
        symbol: The trading symbol
        params: Dictionary of parameters for pattern detection
        
    Returns:
        List of PatternDetectionResult objects
    """
    results = []
    
    # Minimum required data points
    min_points = 10
    if len(df) < min_points:
        return results
    
    # Threshold for flagpole height (as a percentage of price)
    flagpole_min_height_pct = params.get("flagpole_min_height_pct", 0.05)  # 5%
    
    # Flag body characteristics
    min_flag_bars = params.get("min_flag_bars", 5)  # Minimum bars in the flag body
    max_flag_bars = params.get("max_flag_bars", 20)  # Maximum bars in the flag body
    
    # Look for potential flagpole and flag formations
    for i in range(len(df) - min_points):
        # Check for a strong upward move (flagpole)
        start_idx = i
        pole_end_idx = i + min_points // 2  # Approx. half the window for the pole
        
        # Calculate the height of the potential flagpole
        pole_start_price = low_prices[start_idx]
        pole_end_price = high_prices[pole_end_idx]
        pole_height = pole_end_price - pole_start_price
        
        # Check if the pole is steep enough
        if pole_height / pole_start_price < flagpole_min_height_pct:
            continue
        
        # Look for consolidation period after the flagpole (the flag)
        flag_start_idx = pole_end_idx
        
        # Try different flag lengths within the valid range
        for flag_length in range(min_flag_bars, min(max_flag_bars, len(df) - flag_start_idx)):
            flag_end_idx = flag_start_idx + flag_length
            
            # Extract flag prices
            flag_high_prices = high_prices[flag_start_idx:flag_end_idx]
            flag_low_prices = low_prices[flag_start_idx:flag_end_idx]
            
            # Calculate upper and lower trendlines for the flag
            x_points = np.arange(flag_length)
            
            # Upper trendline
            upper_slope, upper_intercept, upper_r, _, _ = stats.linregress(x_points, flag_high_prices)
            
            # Lower trendline
            lower_slope, lower_intercept, lower_r, _, _ = stats.linregress(x_points, flag_low_prices)
            
            # For a bullish flag, both trendlines should be flat or slightly downward
            if upper_slope > 0.001 or lower_slope > 0.001:
                continue
            
            # Calculate how parallel the trendlines are
            slope_diff = abs(upper_slope - lower_slope)
            
            # Calculate flag quality metrics
            trendline_fit = (abs(upper_r) + abs(lower_r)) / 2  # Average R-value fit
            flag_height_ratio = (flag_high_prices.max() - flag_low_prices.min()) / pole_height
            
            # Flag should be smaller than the pole
            if flag_height_ratio > 0.7:
                continue
            
            # Calculate confidence
            confidence = (
                trendline_fit * 0.4 +  # How well prices fit the trendlines
                (1 - slope_diff) * 0.3 +  # How parallel the trendlines are
                (1 - flag_height_ratio) * 0.3  # Flag height compared to pole height
            ) * 100
            confidence = min(max(confidence, 0), 100)  # Clamp to 0-100
            
            # Calculate target price (typical target is the pole height projected from the flag breakout)
            target_price = flag_low_prices[-1] + pole_height
            
            # Create pattern result
            pattern_result = PatternDetectionResult(
                pattern_type=PatternType.FLAG_BULLISH,
                symbol=symbol,
                confidence=confidence,
                start_time=df.index[start_idx],
                end_time=df.index[flag_end_idx],
                target_price=target_price,
                price_level=flag_low_prices[-1],  # Breakout level is the bottom of the flag
                additional_info={
                    "pole_start_idx": int(start_idx),
                    "pole_end_idx": int(pole_end_idx),
                    "flag_start_idx": int(flag_start_idx),
                    "flag_end_idx": int(flag_end_idx),
                    "pole_height": float(pole_height),
                    "upper_slope": float(upper_slope),
                    "upper_intercept": float(upper_intercept),
                    "lower_slope": float(lower_slope),
                    "lower_intercept": float(lower_intercept),
                    "trendline_fit": float(trendline_fit)
                }
            )
            
            results.append(pattern_result)
    
    return results


def _detect_bearish_flags(
    df: pd.DataFrame,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    peak_indices: np.ndarray,
    trough_indices: np.ndarray,
    symbol: str,
    params: Dict
) -> List[PatternDetectionResult]:
    """Detect bearish flag patterns.
    
    A bearish flag consists of a sharp downward move (the pole) followed by
    a consolidation period (the flag) that slopes slightly upward or sideways.
    
    Args:
        df: DataFrame with OHLCV data
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of close prices
        peak_indices: Array of peak indices
        trough_indices: Array of trough indices
        symbol: The trading symbol
        params: Dictionary of parameters for pattern detection
        
    Returns:
        List of PatternDetectionResult objects
    """
    results = []
    
    # Minimum required data points
    min_points = 10
    if len(df) < min_points:
        return results
    
    # Threshold for flagpole height (as a percentage of price)
    flagpole_min_height_pct = params.get("flagpole_min_height_pct", 0.05)  # 5%
    
    # Flag body characteristics
    min_flag_bars = params.get("min_flag_bars", 5)  # Minimum bars in the flag body
    max_flag_bars = params.get("max_flag_bars", 20)  # Maximum bars in the flag body
    
    # Look for potential flagpole and flag formations
    for i in range(len(df) - min_points):
        # Check for a strong downward move (flagpole)
        start_idx = i
        pole_end_idx = i + min_points // 2  # Approx. half the window for the pole
        
        # Calculate the height of the potential flagpole
        pole_start_price = high_prices[start_idx]
        pole_end_price = low_prices[pole_end_idx]
        pole_height = pole_start_price - pole_end_price  # Note: this is positive for a drop
        
        # Check if the pole is steep enough
        if pole_height / pole_start_price < flagpole_min_height_pct:
            continue
        
        # Look for consolidation period after the flagpole (the flag)
        flag_start_idx = pole_end_idx
        
        # Try different flag lengths within the valid range
        for flag_length in range(min_flag_bars, min(max_flag_bars, len(df) - flag_start_idx)):
            flag_end_idx = flag_start_idx + flag_length
            
            # Extract flag prices
            flag_high_prices = high_prices[flag_start_idx:flag_end_idx]
            flag_low_prices = low_prices[flag_start_idx:flag_end_idx]
            
            # Calculate upper and lower trendlines for the flag
            x_points = np.arange(flag_length)
            
            # Upper trendline
            upper_slope, upper_intercept, upper_r, _, _ = stats.linregress(x_points, flag_high_prices)
            
            # Lower trendline
            lower_slope, lower_intercept, lower_r, _, _ = stats.linregress(x_points, flag_low_prices)
            
            # For a bearish flag, both trendlines should be flat or slightly upward
            if upper_slope < -0.001 or lower_slope < -0.001:
                continue
            
            # Calculate how parallel the trendlines are
            slope_diff = abs(upper_slope - lower_slope)
            
            # Calculate flag quality metrics
            trendline_fit = (abs(upper_r) + abs(lower_r)) / 2  # Average R-value fit
            flag_height_ratio = (flag_high_prices.max() - flag_low_prices.min()) / pole_height
            
            # Flag should be smaller than the pole
            if flag_height_ratio > 0.7:
                continue
            
            # Calculate confidence
            confidence = (
                trendline_fit * 0.4 +  # How well prices fit the trendlines
                (1 - slope_diff) * 0.3 +  # How parallel the trendlines are
                (1 - flag_height_ratio) * 0.3  # Flag height compared to pole height
            ) * 100
            confidence = min(max(confidence, 0), 100)  # Clamp to 0-100
            
            # Calculate target price (typical target is the pole height projected from the flag breakout)
            target_price = flag_high_prices[-1] - pole_height
            
            # Create pattern result
            pattern_result = PatternDetectionResult(
                pattern_type=PatternType.FLAG_BEARISH,
                symbol=symbol,
                confidence=confidence,
                start_time=df.index[start_idx],
                end_time=df.index[flag_end_idx],
                target_price=target_price,
                price_level=flag_high_prices[-1],  # Breakout level is the top of the flag
                additional_info={
                    "pole_start_idx": int(start_idx),
                    "pole_end_idx": int(pole_end_idx),
                    "flag_start_idx": int(flag_start_idx),
                    "flag_end_idx": int(flag_end_idx),
                    "pole_height": float(pole_height),
                    "upper_slope": float(upper_slope),
                    "upper_intercept": float(upper_intercept),
                    "lower_slope": float(lower_slope),
                    "lower_intercept": float(lower_intercept),
                    "trendline_fit": float(trendline_fit)
                }
            )
            
            results.append(pattern_result)
    
    return results
