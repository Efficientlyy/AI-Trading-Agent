"""
Head and Shoulders pattern detection module for the AI Trading Agent

This module provides specialized detection for head and shoulders patterns,
including both regular (bearish) and inverse (bullish) variants.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import signal
from datetime import datetime

from .pattern_types import PatternDetectionResult, PatternType


def detect_head_and_shoulders(df: pd.DataFrame, symbol: str, peak_indices: np.ndarray, 
                             trough_indices: np.ndarray, params: Dict) -> List[PatternDetectionResult]:
    """Detect both regular and inverse head and shoulders patterns.
    
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
    
    # Detect patterns
    regular_hs = _detect_regular_head_shoulders(df, high_prices, peak_indices, symbol, params)
    inverse_hs = _detect_inverse_head_shoulders(df, low_prices, trough_indices, symbol, params)
    
    return regular_hs + inverse_hs


def _detect_regular_head_shoulders(df: pd.DataFrame, high_prices: np.ndarray, 
                                peak_indices: np.ndarray, symbol: str, params: Dict) -> List[PatternDetectionResult]:
    """Detect regular (top) head and shoulders patterns.
    
    Args:
        df: DataFrame with OHLCV data
        high_prices: Array of high prices
        peak_indices: Array of peak indices
        symbol: The trading symbol
        params: Dictionary of parameters for pattern detection
        
    Returns:
        List of PatternDetectionResult objects
    """
    results = []
    
    if len(peak_indices) < 5:
        return results  # Not enough peaks to form a pattern
    
    # Window through the peaks looking for H&S pattern
    for i in range(len(peak_indices) - 4):
        # Five consecutive peaks for potential pattern (we need 5 because we'll verify the formation)
        p1, p2, p3, p4, p5 = peak_indices[i:i+5]
        
        # Check height relationship for classic H&S pattern
        # Left shoulder (p1), head (p3), right shoulder (p5)
        # p2 and p4 are the troughs between the shoulders and head
        if high_prices[p1] < high_prices[p3] and high_prices[p5] < high_prices[p3] and \
           abs(high_prices[p1] - high_prices[p5]) / high_prices[p3] < params["shoulder_height_diff_pct"]:
            
            # Calculate neckline using the troughs between shoulders and head (p2 and p4)
            trough_between_left_and_head = min(high_prices[p1:p3])
            trough_between_head_and_right = min(high_prices[p3:p5])
            
            # For a proper H&S, the neckline should be relatively flat
            if abs(trough_between_left_and_head - trough_between_head_and_right) / high_prices[p3] < params["neckline_slope_threshold"]:
                
                # Calculate pattern metrics
                pattern_height = high_prices[p3] - min(trough_between_left_and_head, trough_between_head_and_right)
                pattern_width = p5 - p1
                
                # Calculate confidence based on symmetry and clarity
                shoulder_symmetry = 1 - abs(high_prices[p1] - high_prices[p5]) / high_prices[p3]
                head_prominence = (high_prices[p3] - max(high_prices[p1], high_prices[p5])) / high_prices[p3]
                
                confidence = (shoulder_symmetry * 0.5 + head_prominence * 0.5) * 100
                confidence = min(max(confidence, 0), 100)  # Clamp to 0-100
                
                # Create pattern result
                start_idx = max(0, p1 - 5)  # Include some bars before the pattern
                end_idx = min(len(df) - 1, p5 + 5)  # Include some bars after the pattern
                
                # Calculate target price (distance from head to neckline, projected below neckline)
                neckline_price = (trough_between_left_and_head + trough_between_head_and_right) / 2
                target_price = neckline_price - pattern_height
                
                pattern_result = PatternDetectionResult(
                    pattern_type=PatternType.HEAD_AND_SHOULDERS,
                    symbol=symbol,
                    confidence=confidence,
                    start_time=df.index[start_idx],
                    end_time=df.index[end_idx],
                    target_price=target_price,
                    price_level=neckline_price,
                    additional_info={
                        "left_shoulder_idx": int(p1),
                        "head_idx": int(p3),
                        "right_shoulder_idx": int(p5),
                        "pattern_height": float(pattern_height),
                        "shoulder_symmetry": float(shoulder_symmetry),
                    }
                )
                
                results.append(pattern_result)
    
    return results


def _detect_inverse_head_shoulders(df: pd.DataFrame, low_prices: np.ndarray, 
                                trough_indices: np.ndarray, symbol: str, params: Dict) -> List[PatternDetectionResult]:
    """Detect inverse (bottom) head and shoulders patterns.
    
    Args:
        df: DataFrame with OHLCV data
        low_prices: Array of low prices
        trough_indices: Array of trough indices
        symbol: The trading symbol
        params: Dictionary of parameters for pattern detection
        
    Returns:
        List of PatternDetectionResult objects
    """
    results = []
    
    if len(trough_indices) < 5:
        return results  # Not enough troughs to form a pattern
    
    # Window through the troughs looking for inverse H&S pattern
    for i in range(len(trough_indices) - 4):
        # Five consecutive troughs for potential pattern (we need 5 because we'll verify the formation)
        t1, t2, t3, t4, t5 = trough_indices[i:i+5]
        
        # Check height relationship for classic inverse H&S pattern
        # Left shoulder (t1), head (t3), right shoulder (t5)
        # t2 and t4 are the peaks between the shoulders and head
        if low_prices[t1] > low_prices[t3] and low_prices[t5] > low_prices[t3] and \
           abs(low_prices[t1] - low_prices[t5]) / low_prices[t3] < params["shoulder_height_diff_pct"]:
            
            # Calculate neckline using the peaks between shoulders and head (t2 and t4)
            peak_between_left_and_head = max(low_prices[t1:t3])
            peak_between_head_and_right = max(low_prices[t3:t5])
            
            # For a proper inverse H&S, the neckline should be relatively flat
            if abs(peak_between_left_and_head - peak_between_head_and_right) / low_prices[t3] < params["neckline_slope_threshold"]:
                
                # Calculate pattern metrics
                pattern_height = min(peak_between_left_and_head, peak_between_head_and_right) - low_prices[t3]
                pattern_width = t5 - t1
                
                # Calculate confidence based on symmetry and clarity
                shoulder_symmetry = 1 - abs(low_prices[t1] - low_prices[t5]) / low_prices[t3]
                head_prominence = (min(peak_between_left_and_head, peak_between_head_and_right) - low_prices[t3]) / low_prices[t3]
                
                confidence = (shoulder_symmetry * 0.5 + head_prominence * 0.5) * 100
                confidence = min(max(confidence, 0), 100)  # Clamp to 0-100
                
                # Create pattern result
                start_idx = max(0, t1 - 5)  # Include some bars before the pattern
                end_idx = min(len(df) - 1, t5 + 5)  # Include some bars after the pattern
                
                # Calculate target price (distance from head to neckline, projected above neckline)
                neckline_price = (peak_between_left_and_head + peak_between_head_and_right) / 2
                target_price = neckline_price + pattern_height
                
                pattern_result = PatternDetectionResult(
                    pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                    symbol=symbol,
                    confidence=confidence,
                    start_time=df.index[start_idx],
                    end_time=df.index[end_idx],
                    target_price=target_price,
                    price_level=neckline_price,
                    additional_info={
                        "left_shoulder_idx": int(t1),
                        "head_idx": int(t3),
                        "right_shoulder_idx": int(t5),
                        "pattern_height": float(pattern_height),
                        "shoulder_symmetry": float(shoulder_symmetry),
                    }
                )
                
                results.append(pattern_result)
    
    return results
