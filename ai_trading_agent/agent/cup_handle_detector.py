"""
Cup and Handle pattern detection module for the AI Trading Agent

This module provides specialized detection for cup and handle patterns,
which is a bullish continuation pattern resembling a cup with a handle.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats, signal
from datetime import datetime

from .pattern_types import PatternDetectionResult, PatternType


def detect_cup_and_handle(df: pd.DataFrame, symbol: str, peak_indices: np.ndarray, trough_indices: np.ndarray, params: Dict) -> List[PatternDetectionResult]:
    """
    Detect cup and handle patterns in the price data.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol
        peak_indices: Array of indices where peaks (local maxima) occur
        trough_indices: Array of indices where troughs (local minima) occur
        params: Dictionary of parameters for pattern detection
        
    Returns:
        List of detected cup and handle patterns
    """
    if len(peak_indices) < 2 or len(trough_indices) < 1:
        return []
    
    # Extract price data
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # Ultra-optimized parameters for maximum detection sensitivity
    cup_depth_threshold = params.get("cup_depth_threshold", 0.001)  # Cup depth at least 0.1% of price (extremely sensitive)
    cup_symmetry_threshold = params.get("cup_symmetry_threshold", 0.2)  # Cup should be 20% symmetrical (very lenient)
    min_handle_size = params.get("min_handle_size", 1)  # Minimum 1 period for handle (maximum sensitivity)
    max_handle_size = params.get("max_handle_size", 60)  # Maximum 60 periods for handle (very wide range)
    handle_retrace_threshold = params.get("handle_retrace_threshold", 0.7)  # Handle should retrace at most 70% of cup depth (extremely lenient)
    min_cup_duration = params.get("min_cup_duration", 3)  # Minimum cup duration (very short allowed)
    lookback_window = params.get("cup_lookback_window", 180)  # Look at most 180 bars back (maximum window)
    
    patterns = []
    
    # Process the data with the enhanced algorithm
    for i in range(len(peak_indices) - 1):
        left_rim_idx = peak_indices[i]
        
        # Skip if the left rim is too recent - need enough space for pattern to form
        if left_rim_idx > len(close_prices) - lookback_window:
            continue
        
        # Find a potential cup bottom (trough between two peaks)
        potential_bottoms = [t for t in trough_indices if t > left_rim_idx and t < peak_indices[i+1]]
        
        if not potential_bottoms:
            continue
        
        cup_bottom_idx = potential_bottoms[np.argmin(low_prices[potential_bottoms])]  # Lowest trough
        right_rim_idx = peak_indices[i+1]
        
        # Cup should have a minimum duration
        if right_rim_idx - left_rim_idx < min_cup_duration:
            continue
        
        # Check if cup has sufficient depth
        left_rim_price = high_prices[left_rim_idx]
        cup_bottom_price = low_prices[cup_bottom_idx]
        right_rim_price = high_prices[right_rim_idx]
        
        # Calculate cup depth as percentage of average rim height
        cup_depth = ((left_rim_price + right_rim_price) / 2 - cup_bottom_price) / ((left_rim_price + right_rim_price) / 2)
        
        if cup_depth < cup_depth_threshold:
            continue
        
        # Check cup symmetry (now more lenient)
        left_cup_length = cup_bottom_idx - left_rim_idx
        right_cup_length = right_rim_idx - cup_bottom_idx
        cup_symmetry = min(left_cup_length, right_cup_length) / max(left_cup_length, right_cup_length)
        
        if cup_symmetry < cup_symmetry_threshold:
            continue
        
        # Check if right rim is approximately at the level of left rim (looser constraint)
        rim_height_diff = abs(right_rim_price - left_rim_price) / left_rim_price
        if rim_height_diff > 0.10:  # Allow up to 10% difference between rims
            continue
            
        # Look for a rounded bottom (U-shape) rather than a V-shape
        # Extract a window around the cup bottom for shape analysis
        bottom_window_size = min(5, min(left_cup_length, right_cup_length) // 2)
        if bottom_window_size > 0:
            bottom_window = low_prices[cup_bottom_idx-bottom_window_size:cup_bottom_idx+bottom_window_size+1]
            if len(bottom_window) >= 3:
                # Check if the shape is more like a U (flat bottom) than a V (sharp bottom)
                u_shape_factor = np.std(bottom_window) / (np.max(bottom_window) - np.min(bottom_window) + 0.00001)
                if u_shape_factor > 0.5:  # More tolerant of V-shapes
                    continue
        
        # Look for handle formation after right rim
        handle_end_idx = min(len(close_prices) - 1, right_rim_idx + max_handle_size)
        
        # Handle should be at least min_handle_size periods
        if handle_end_idx - right_rim_idx < min_handle_size:
            continue
        
        # Find lowest point in the handle
        handle_range = np.arange(right_rim_idx + 1, handle_end_idx + 1)
        if len(handle_range) == 0:
            continue
            
        handle_low_idx = handle_range[np.argmin(low_prices[handle_range])]
        handle_low_price = low_prices[handle_low_idx]
        
        # Handle should have a small retrace (not too deep)
        handle_retrace = (right_rim_price - handle_low_price) / right_rim_price
        
        if handle_retrace > handle_retrace_threshold or handle_retrace < 0.01:
            continue
            
        # Check for potential breakout after handle
        breakout_confirmed = False
        for j in range(handle_low_idx + 1, min(len(close_prices), handle_low_idx + 10)):
            if close_prices[j] > right_rim_price * 1.01:  # 1% above right rim
                breakout_confirmed = True
                break
        
        # Calculate confidence score with improved weights and scaling for better detection
        # Factors:
        # 1. Cup depth (deeper is better)
        # 2. Cup symmetry (more symmetrical is better)
        # 3. Handle quality (not too deep, not too long)
        # 4. Handle position (close to right rim)
        # 5. Rim levelness (closer heights is better)

        # Normalize depth (much more sensitive to shallow cups)
        depth_score = min(1.0, cup_depth / 0.02)  # Normalize depth (cap at 2%)
        symmetry_score = cup_symmetry
        
        # Handle quality: combination of depth and length with relaxed constraints
        handle_depth_score = 1.0 - (handle_retrace / handle_retrace_threshold) 
        handle_length = (handle_end_idx - right_rim_idx)  # Calculate actual handle length
        handle_length_score = 1.0 - min(1.0, handle_length / max_handle_size)
        handle_quality = max(0.5, (handle_depth_score * 0.6 + handle_length_score * 0.4))  # Minimum 0.5 score
        
        # Rim levelness - more lenient
        rim_level_score = 1.0 - min(1.0, rim_height_diff / 0.1)  # Lower diff is better
        
        # Overall confidence with improved weighting
        confidence = (
            depth_score * 0.30 +
            symmetry_score * 0.20 +
            handle_quality * 0.25 +
            rim_level_score * 0.25
        )
        
        # Scale confidence from 0-1 to 0-100 for consistency with other patterns
        confidence = confidence * 100
        
        # Only include high confidence patterns
        if confidence < 60:
            continue
        
        # Ensure confidence is in valid range
        confidence = min(max(confidence, 0), 100)  # Clamp to 0-100
        
        # Calculate target price (cup depth projected from breakout)
        target_price = right_rim_price + cup_depth
        
        # Create result with the correct parameters
        try:
            result = PatternDetectionResult(
                pattern_type=PatternType.CUP_AND_HANDLE,
                symbol=symbol,
                confidence=confidence,
                start_idx=int(left_rim_idx),
                end_idx=int(handle_end_idx),
                additional_info={
                    "left_rim_idx": int(left_rim_idx),
                    "cup_bottom_idx": int(cup_bottom_idx),
                    "right_rim_idx": int(right_rim_idx),
                    "handle_low_idx": int(handle_low_idx),
                    "cup_depth": float(cup_depth),
                    "cup_symmetry": float(cup_symmetry),
                    "handle_retrace": float(handle_retrace),
                    "breakout_confirmed": breakout_confirmed
                }
            )
        except Exception as e:
            # Safely handle any errors that might occur during result creation
            print(f"Error creating cup and handle result: {e}")
            continue
        
        patterns.append(result)
    
    return patterns


def _calculate_cup_roundness(dates, high_prices, low_prices) -> float:
    """
    Calculate how round (U-shaped) the cup is.
    
    A perfect cup is U-shaped, not V-shaped.
    Returns a value between 0 (V-shaped) and 1 (perfectly U-shaped).
    """
    # Use the average of high and low for the cup shape
    avg_prices = (high_prices + low_prices) / 2
    
    # Normalize data for comparison
    x = np.linspace(0, 1, len(avg_prices))
    y_norm = (avg_prices - np.min(avg_prices)) / (np.max(avg_prices) - np.min(avg_prices))
    
    # Create ideal U-shape and V-shape for comparison
    u_shape = 1 - 4 * (x - 0.5) ** 2  # Parabola (U-shape)
    v_shape = 1 - 2 * np.abs(x - 0.5)  # V-shape
    
    # Calculate correlation with both shapes
    corr_u = np.corrcoef(y_norm, u_shape)[0, 1]
    corr_v = np.corrcoef(y_norm, v_shape)[0, 1]
    
    # Higher correlation with U-shape and lower with V-shape is better
    u_score = max(0, corr_u)
    v_penalty = max(0, corr_v) * 0.5  # Penalty for V-shape correlation
    
    roundness = max(0, min(1, u_score - v_penalty))
    return roundness


def _check_volume_pattern(volumes) -> float:
    """
    Check if volume pattern matches expectations for cup and handle.
    
    Ideally, volume should decrease during cup formation and 
    increase during handle breakout.
    
    Returns a score between 0 and 1.
    """
    if len(volumes) < 5:
        return 0.5  # Not enough data for reliable check
    
    # Split volume data for cup and handle
    cup_end_idx = len(volumes) * 2 // 3
    cup_volumes = volumes[:cup_end_idx]
    handle_volumes = volumes[cup_end_idx:]
    
    # Check for decreasing trend in cup
    cup_slope, _, _, _, _ = stats.linregress(range(len(cup_volumes)), cup_volumes)
    cup_score = 0.7 if cup_slope < 0 else 0.3
    
    # Check for increasing volume near the end (potential breakout)
    handle_trend = handle_volumes[-min(3, len(handle_volumes)):].mean() / handle_volumes.mean()
    handle_score = 0.7 if handle_trend > 1.1 else 0.3
    
    # Combine scores
    return (cup_score + handle_score) / 2