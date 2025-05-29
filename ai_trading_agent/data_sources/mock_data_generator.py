"""
Mock Data Generator Module

This module provides functions for generating realistic mock market data
for testing and development purposes.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ..common.utils import get_logger


def generate_mock_data(
    days: int = 100,
    volatility: float = 0.02,
    trend_strength: float = 0.5,
    start_price: float = 50000.0,
    volume_base: float = 1000000.0,
    pattern_type: str = "random"
) -> pd.DataFrame:
    """
    Generate mock OHLCV market data for testing.
    
    Args:
        days: Number of days of data to generate
        volatility: Volatility of the price data (standard deviation of returns)
        trend_strength: Strength of the trend (0-1)
        start_price: Starting price for the data
        volume_base: Base volume amount
        pattern_type: Type of pattern to generate (random, trend, range, volatile)
        
    Returns:
        DataFrame with OHLCV data
    """
    logger = get_logger("MockDataGenerator")
    
    # Initialize data structures
    dates = pd.date_range(end=datetime.now(), periods=days)
    data = pd.DataFrame(index=dates)
    
    # Generate price data based on pattern type
    if pattern_type == "trend":
        # Generate trending market (gradual uptrend or downtrend)
        direction = np.random.choice([-1, 1])  # Random direction
        trend = np.linspace(0, trend_strength * direction, days)
        random_component = np.random.normal(0, volatility, days)
        returns = trend + random_component
        
    elif pattern_type == "range":
        # Generate ranging market (oscillating between support and resistance)
        cycles = days / (np.random.randint(10, 30))  # Random cycle length
        oscillation = np.sin(np.linspace(0, cycles * 2 * np.pi, days)) * trend_strength
        random_component = np.random.normal(0, volatility, days)
        returns = oscillation + random_component
        
    elif pattern_type == "volatile":
        # Generate volatile market (periods of high volatility)
        base_returns = np.random.normal(0, volatility, days)
        
        # Add volatility clusters
        volatility_multiplier = np.ones(days)
        num_clusters = max(1, int(days / 20))
        
        for _ in range(num_clusters):
            cluster_start = np.random.randint(0, days - 5)
            cluster_length = np.random.randint(3, 10)
            cluster_end = min(days, cluster_start + cluster_length)
            volatility_multiplier[cluster_start:cluster_end] = np.random.uniform(2.0, 4.0)
            
        returns = base_returns * volatility_multiplier
        
    else:  # random
        # Generate random market (random walk)
        returns = np.random.normal(0, volatility, days)
        
        # Add a slight bias based on trend_strength
        if trend_strength > 0:
            direction = np.random.choice([-1, 1])  # Random direction
            returns += trend_strength * 0.01 * direction
    
    # Calculate price from returns
    price = start_price * (1 + returns).cumprod()
    
    # Generate OHLCV data
    data['close'] = price
    
    # Generate open, high, and low based on close
    daily_volatility = volatility * price
    data['open'] = data['close'].shift(1)
    data.loc[data.index[0], 'open'] = start_price  # Set first open price
    
    # High is the max of open and close plus a random amount
    data['high'] = np.maximum(data['open'], data['close'])
    data['high'] += np.random.uniform(0, daily_volatility * 1.5)
    
    # Low is the min of open and close minus a random amount
    data['low'] = np.minimum(data['open'], data['close'])
    data['low'] -= np.random.uniform(0, daily_volatility * 1.5)
    
    # Ensure low <= open,close <= high
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    
    # Generate volume data
    volume_trend = np.ones(days)
    
    # Volume tends to be higher on big price moves
    volume_volatility = 1.0 + np.abs(returns) / volatility
    
    # Random volume component
    volume_random = np.random.uniform(0.5, 2.0, days)
    
    # Volume tends to be higher in certain patterns
    if pattern_type == "volatile":
        # Higher volume during volatile periods
        volume_pattern = volatility_multiplier
    elif pattern_type == "trend":
        # Volume increases as trend progresses
        volume_pattern = np.linspace(0.8, 1.5, days) if direction > 0 else np.linspace(1.5, 0.8, days)
    else:
        # Default pattern
        volume_pattern = np.ones(days)
    
    # Combine all volume factors
    data['volume'] = volume_base * volume_trend * volume_volatility * volume_random * volume_pattern
    
    # Add some volume spikes
    num_spikes = max(1, int(days / 20))
    spike_indices = np.random.choice(range(days), num_spikes, replace=False)
    data.iloc[spike_indices, data.columns.get_loc('volume')] *= np.random.uniform(2.0, 5.0, num_spikes)
    
    # Round values for realism
    data['open'] = np.round(data['open'], 2)
    data['high'] = np.round(data['high'], 2)
    data['low'] = np.round(data['low'], 2)
    data['close'] = np.round(data['close'], 2)
    data['volume'] = np.round(data['volume']).astype(int)
    
    logger.info(
        f"Generated {days} days of mock {pattern_type} data "
        f"with volatility={volatility:.4f}, trend_strength={trend_strength:.4f}"
    )
    
    return data


def generate_mock_data_with_pattern(
    pattern: str,
    days_before: int = 30,
    days_after: int = 10,
    volatility: float = 0.02,
    pattern_strength: float = 0.8,
    start_price: float = 50000.0
) -> pd.DataFrame:
    """
    Generate mock data with a specific technical pattern embedded.
    
    Args:
        pattern: Type of pattern to generate (double_top, head_shoulders, etc.)
        days_before: Number of days of data before the pattern
        days_after: Number of days after the pattern
        volatility: Volatility of the price data
        pattern_strength: How pronounced the pattern should be (0-1)
        start_price: Starting price for the data
        
    Returns:
        DataFrame with OHLCV data containing the specified pattern
    """
    logger = get_logger("MockDataGenerator")
    
    # Pattern-specific parameters
    pattern_length = 0
    
    if pattern == "double_top":
        pattern_length = 20
        
    elif pattern == "head_shoulders":
        pattern_length = 30
        
    elif pattern == "cup_handle":
        pattern_length = 40
        
    elif pattern == "triangle":
        pattern_length = 15
        
    else:
        logger.warning(f"Unknown pattern: {pattern}, using double_top")
        pattern = "double_top"
        pattern_length = 20
    
    # Generate base data
    total_days = days_before + pattern_length + days_after
    data = generate_mock_data(
        days=total_days,
        volatility=volatility * 0.5,  # Reduce randomness for clearer patterns
        trend_strength=0.2,  # Slight trend
        start_price=start_price
    )
    
    # Get the price at the start of the pattern
    pattern_start_idx = days_before
    pattern_end_idx = days_before + pattern_length
    pattern_start_price = data['close'].iloc[pattern_start_idx]
    
    # Generate the pattern
    if pattern == "double_top":
        # Double top pattern
        first_peak = int(pattern_length * 0.3)
        second_peak = int(pattern_length * 0.7)
        peak_height = pattern_start_price * (1 + pattern_strength * 0.1)
        
        # Create the pattern shape
        pattern_shape = np.ones(pattern_length) * pattern_start_price
        
        # First peak
        pattern_shape[first_peak] = peak_height
        
        # Valley between peaks
        valley = int((first_peak + second_peak) / 2)
        pattern_shape[valley] = pattern_start_price * (1 - pattern_strength * 0.03)
        
        # Second peak
        pattern_shape[second_peak] = peak_height * 0.98  # Slightly lower
        
        # Decline after second peak
        for i in range(second_peak + 1, pattern_length):
            drop_factor = (i - second_peak) / (pattern_length - second_peak)
            pattern_shape[i] = peak_height * (1 - drop_factor * pattern_strength * 0.15)
        
        # Interpolate between points
        for i in range(1, pattern_length):
            if i in [first_peak, valley, second_peak]:
                continue
            prev_anchor = max([p for p in [0, first_peak, valley, second_peak] if p < i])
            next_anchors = [p for p in [first_peak, valley, second_peak, pattern_length-1] if p > i]
            if next_anchors:  # Make sure we have at least one anchor point
                next_anchor = min(next_anchors)
            else:
                next_anchor = pattern_length-1  # Default to end of pattern if no anchors left
            weight = (i - prev_anchor) / (next_anchor - prev_anchor)
            pattern_shape[i] = pattern_shape[prev_anchor] * (1 - weight) + pattern_shape[next_anchor] * weight
        
    elif pattern == "head_shoulders":
        # Head and shoulders pattern
        left_shoulder = int(pattern_length * 0.2)
        head = int(pattern_length * 0.5)
        right_shoulder = int(pattern_length * 0.8)
        
        shoulder_height = pattern_start_price * (1 + pattern_strength * 0.06)
        head_height = pattern_start_price * (1 + pattern_strength * 0.12)
        
        # Create the pattern shape
        pattern_shape = np.ones(pattern_length) * pattern_start_price
        
        # Left shoulder
        pattern_shape[left_shoulder] = shoulder_height
        
        # Head
        pattern_shape[head] = head_height
        
        # Right shoulder
        pattern_shape[right_shoulder] = shoulder_height * 0.95  # Slightly lower
        
        # Valleys
        left_valley = int((left_shoulder + head) / 2)
        right_valley = int((head + right_shoulder) / 2)
        
        pattern_shape[left_valley] = pattern_start_price * (1 + pattern_strength * 0.01)
        pattern_shape[right_valley] = pattern_start_price * (1 + pattern_strength * 0.01)
        
        # Decline after right shoulder
        for i in range(right_shoulder + 1, pattern_length):
            drop_factor = (i - right_shoulder) / (pattern_length - right_shoulder)
            pattern_shape[i] = shoulder_height * (1 - drop_factor * pattern_strength * 0.2)
        
        # Interpolate between points
        anchors = [0, left_shoulder, left_valley, head, right_valley, right_shoulder, pattern_length-1]
        for i in range(1, pattern_length):
            if i in anchors:
                continue
            prev_anchor = max([p for p in anchors if p < i])
            next_anchor = min([p for p in anchors if p > i])
            weight = (i - prev_anchor) / (next_anchor - prev_anchor)
            pattern_shape[i] = pattern_shape[prev_anchor] * (1 - weight) + pattern_shape[next_anchor] * weight
    
    elif pattern == "cup_handle":
        # Cup and handle pattern
        cup_bottom = int(pattern_length * 0.5)
        handle_bottom = int(pattern_length * 0.8)
        
        cup_depth = pattern_start_price * (1 - pattern_strength * 0.1)
        handle_depth = pattern_start_price * (1 - pattern_strength * 0.05)
        
        # Create the pattern shape
        pattern_shape = np.ones(pattern_length) * pattern_start_price
        
        # Cup
        for i in range(cup_bottom + 1):
            factor = np.sin(np.pi * i / cup_bottom)
            pattern_shape[i] = pattern_start_price - (pattern_start_price - cup_depth) * factor
        
        # Right side of cup
        for i in range(cup_bottom, int(pattern_length * 0.7)):
            factor = np.sin(np.pi * (1 - (i - cup_bottom) / (int(pattern_length * 0.7) - cup_bottom)))
            pattern_shape[i] = pattern_start_price - (pattern_start_price - cup_depth) * factor
        
        # Handle
        for i in range(int(pattern_length * 0.7), handle_bottom):
            factor = (i - int(pattern_length * 0.7)) / (handle_bottom - int(pattern_length * 0.7))
            pattern_shape[i] = pattern_start_price - (pattern_start_price - handle_depth) * factor
        
        # Breakout after handle
        for i in range(handle_bottom, pattern_length):
            breakout_factor = (i - handle_bottom) / (pattern_length - handle_bottom)
            pattern_shape[i] = handle_depth + (pattern_start_price * (1 + pattern_strength * 0.05) - handle_depth) * breakout_factor
    
    elif pattern == "triangle":
        # Triangle pattern (symmetrical, ascending, or descending)
        triangle_type = np.random.choice(["symmetrical", "ascending", "descending"])
        
        if triangle_type == "symmetrical":
            # Symmetrical triangle - highs get lower, lows get higher
            upper_start = pattern_start_price * (1 + pattern_strength * 0.08)
            lower_start = pattern_start_price * (1 - pattern_strength * 0.08)
            
            for i in range(pattern_length):
                progress = i / pattern_length
                upper_line = upper_start - progress * (upper_start - pattern_start_price) * 0.9
                lower_line = lower_start + progress * (pattern_start_price - lower_start) * 0.9
                
                # Oscillate between the two lines, getting narrower
                cycle = np.sin(i / 2 * np.pi)
                amplitude = (upper_line - lower_line) / 2
                midpoint = (upper_line + lower_line) / 2
                pattern_shape[i] = midpoint + cycle * amplitude
                
        elif triangle_type == "ascending":
            # Ascending triangle - flat top, rising bottom
            upper_line = pattern_start_price * (1 + pattern_strength * 0.06)
            lower_start = pattern_start_price * (1 - pattern_strength * 0.06)
            
            for i in range(pattern_length):
                progress = i / pattern_length
                lower_line = lower_start + progress * (upper_line - lower_start) * 0.9
                
                # Oscillate between the two lines, getting narrower
                cycle = np.sin(i / 2 * np.pi)
                amplitude = (upper_line - lower_line) / 2
                midpoint = (upper_line + lower_line) / 2
                pattern_shape[i] = midpoint + cycle * amplitude
                
        else:  # descending
            # Descending triangle - flat bottom, falling top
            lower_line = pattern_start_price * (1 - pattern_strength * 0.06)
            upper_start = pattern_start_price * (1 + pattern_strength * 0.06)
            
            for i in range(pattern_length):
                progress = i / pattern_length
                upper_line = upper_start - progress * (upper_start - lower_line) * 0.9
                
                # Oscillate between the two lines, getting narrower
                cycle = np.sin(i / 2 * np.pi)
                amplitude = (upper_line - lower_line) / 2
                midpoint = (upper_line + lower_line) / 2
                pattern_shape[i] = midpoint + cycle * amplitude
                
        # Add breakout at the end
        breakout_direction = np.random.choice([-1, 1])  # Random breakout direction
        for i in range(int(pattern_length * 0.8), pattern_length):
            breakout_factor = (i - int(pattern_length * 0.8)) / (pattern_length - int(pattern_length * 0.8))
            pattern_shape[i] += breakout_direction * breakout_factor * pattern_strength * 0.08 * pattern_start_price
    
    # Apply the pattern to the close prices
    data.iloc[pattern_start_idx:pattern_end_idx, data.columns.get_loc('close')] = pattern_shape
    
    # Regenerate open, high, and low based on the new close prices
    for i in range(pattern_start_idx, pattern_end_idx):
        prev_close = data['close'].iloc[i-1] if i > 0 else start_price
        
        # Open is based on previous close with some noise
        data.iloc[i, data.columns.get_loc('open')] = prev_close * (1 + np.random.normal(0, volatility * 0.3))
        
        # Daily volatility as a percentage of price
        daily_volatility = volatility * data['close'].iloc[i]
        
        # High is the max of open and close plus a random amount
        data.iloc[i, data.columns.get_loc('high')] = max(
            data['open'].iloc[i], 
            data['close'].iloc[i]
        ) + np.random.uniform(0, daily_volatility * 1.2)
        
        # Low is the min of open and close minus a random amount
        data.iloc[i, data.columns.get_loc('low')] = min(
            data['open'].iloc[i], 
            data['close'].iloc[i]
        ) - np.random.uniform(0, daily_volatility * 1.2)
    
    # Ensure low <= open,close <= high throughout the dataset
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    
    # Adjust volume based on pattern (higher volume at breakouts, lower in consolidation)
    for i in range(pattern_start_idx, pattern_end_idx):
        if pattern == "double_top":
            if i == pattern_start_idx + second_peak + 1:  # Breakout after second peak
                data.iloc[i, data.columns.get_loc('volume')] *= 2.5
            elif i in [pattern_start_idx + first_peak, pattern_start_idx + second_peak]:
                data.iloc[i, data.columns.get_loc('volume')] *= 1.8
                
        elif pattern == "head_shoulders":
            if i == pattern_start_idx + right_shoulder + 1:  # Breakout after right shoulder
                data.iloc[i, data.columns.get_loc('volume')] *= 2.2
            elif i in [pattern_start_idx + left_shoulder, pattern_start_idx + head, pattern_start_idx + right_shoulder]:
                data.iloc[i, data.columns.get_loc('volume')] *= 1.5
                
        elif pattern == "cup_handle":
            if i >= pattern_start_idx + handle_bottom:  # Breakout after handle
                data.iloc[i, data.columns.get_loc('volume')] *= 1.5 + (i - (pattern_start_idx + handle_bottom)) * 0.2
                
        elif pattern == "triangle":
            if i >= pattern_start_idx + int(pattern_length * 0.8):  # Breakout phase
                data.iloc[i, data.columns.get_loc('volume')] *= 1.7 + (i - (pattern_start_idx + int(pattern_length * 0.8))) * 0.3
            else:
                # Volume decreases as triangle progresses, then spikes on breakout
                vol_factor = 1.0 - (i - pattern_start_idx) / (pattern_length * 0.8) * 0.5
                data.iloc[i, data.columns.get_loc('volume')] *= max(0.5, vol_factor)
    
    # Round values for realism
    data['open'] = np.round(data['open'], 2)
    data['high'] = np.round(data['high'], 2)
    data['low'] = np.round(data['low'], 2)
    data['close'] = np.round(data['close'], 2)
    data['volume'] = np.round(data['volume']).astype(int)
    
    logger.info(
        f"Generated mock data with {pattern} pattern "
        f"(days_before={days_before}, pattern_length={pattern_length}, days_after={days_after})"
    )
    
    return data
