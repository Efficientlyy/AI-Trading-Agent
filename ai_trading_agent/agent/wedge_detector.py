"""
Wedge pattern detection module for the AI Trading Agent

This module provides specialized detection for wedge patterns, including
rising wedges (bearish) and falling wedges (bullish).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from datetime import datetime

from .pattern_types import PatternDetectionResult, PatternType


def detect_wedges(
    df: pd.DataFrame, 
    symbol: str,
    peak_indices: np.ndarray,
    trough_indices: np.ndarray,
    params: Dict
) -> List[PatternDetectionResult]:
    """
    Detect wedge patterns (rising and falling).
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol
        peak_indices: Indices of price peaks
        trough_indices: Indices of price troughs
        params: Detection parameters
        
    Returns:
        List of detected wedge patterns
    """
    # Extract price data
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    
    results = []
    
    # Detection parameters
    min_points_per_line = params.get("min_points_per_line", 3)
    min_pattern_size = params.get("min_pattern_size", 15)
    max_pattern_size = params.get("max_pattern_size", 100)
    min_line_quality = params.get("min_line_quality", 0.7)  # Minimum R² for trendlines
    convergence_min = params.get("convergence_min", 0.05)  # Minimum convergence rate
    
    # We need at least 3 points for each boundary line
    if len(peak_indices) >= min_points_per_line and len(trough_indices) >= min_points_per_line:
        # Find upper trendlines (using peaks)
        upper_trendlines = []
        for i in range(len(peak_indices) - min_points_per_line + 1):
            subset_indices = peak_indices[i:i+min_points_per_line]
            
            # Skip if pattern is too small
            if subset_indices[-1] - subset_indices[0] < min_pattern_size:
                continue
                
            # Skip if pattern is too large
            if subset_indices[-1] - subset_indices[0] > max_pattern_size:
                continue
                
            x_points = subset_indices
            y_points = high_prices[subset_indices]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)
            
            # Check line quality (R²)
            if r_value ** 2 < min_line_quality:
                continue
                
            # Store trendline parameters
            upper_trendlines.append({
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "start_idx": subset_indices[0],
                "end_idx": subset_indices[-1],
                "start_price": high_prices[subset_indices[0]],
                "end_price": high_prices[subset_indices[-1]]
            })
        
        # Find lower trendlines (using troughs)
        lower_trendlines = []
        for i in range(len(trough_indices) - min_points_per_line + 1):
            subset_indices = trough_indices[i:i+min_points_per_line]
            
            # Skip if pattern is too small
            if subset_indices[-1] - subset_indices[0] < min_pattern_size:
                continue
                
            # Skip if pattern is too large
            if subset_indices[-1] - subset_indices[0] > max_pattern_size:
                continue
                
            x_points = subset_indices
            y_points = low_prices[subset_indices]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)
            
            # Check line quality (R²)
            if r_value ** 2 < min_line_quality:
                continue
                
            # Store trendline parameters
            lower_trendlines.append({
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "start_idx": subset_indices[0],
                "end_idx": subset_indices[-1],
                "start_price": low_prices[subset_indices[0]],
                "end_price": low_prices[subset_indices[-1]]
            })
        
        # Find pairs of converging trendlines that form wedges
        for upper in upper_trendlines:
            for lower in lower_trendlines:
                # Determine the timespan of the potential wedge
                start_idx = max(upper["start_idx"], lower["start_idx"])
                end_idx = min(upper["end_idx"], lower["end_idx"])
                
                # Make sure there's overlap
                if start_idx < end_idx:
                    # Get slopes of upper and lower trendlines
                    upper_slope = upper["slope"]
                    lower_slope = lower["slope"]
                    
                    # Convergence check (both lines must be converging)
                    if (upper_slope - lower_slope) >= 0:  # Not converging
                        continue
                    
                    # Calculate start and end gaps between trendlines
                    start_upper = upper["intercept"] + upper["slope"] * start_idx
                    start_lower = lower["intercept"] + lower["slope"] * start_idx
                    start_gap = start_upper - start_lower
                    
                    end_upper = upper["intercept"] + upper["slope"] * end_idx
                    end_lower = lower["intercept"] + lower["slope"] * end_idx
                    end_gap = end_upper - end_lower
                    
                    # Check if gaps are positive (upper line should be above lower line)
                    if start_gap <= 0 or end_gap <= 0:
                        continue
                    
                    # Check for sufficient convergence
                    convergence_rate = 1 - (end_gap / start_gap)
                    if convergence_rate < convergence_min:
                        continue
                    
                    # Determine the wedge type
                    pattern_type = None
                    
                    # Rising wedge: both lines sloping upward, but converging
                    if upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope:
                        pattern_type = PatternType.WEDGE_RISING
                    
                    # Falling wedge: both lines sloping downward, but converging
                    elif upper_slope < 0 and lower_slope < 0 and upper_slope > lower_slope:
                        pattern_type = PatternType.WEDGE_FALLING
                    
                    if pattern_type:
                        # Calculate confidence
                        # 1. R-squared of trendlines
                        # 2. Pattern span
                        # 3. Convergence rate
                        # 4. Price containment
                        
                        r_squared_score = (upper["r_squared"] + lower["r_squared"]) / 2
                        span_score = min(1.0, (end_idx - start_idx) / (len(close_prices) * 0.2))  # Score of 1 for 20% of data
                        convergence_score = min(1.0, convergence_rate / 0.5)  # Score of 1 for 50% convergence
                        
                        # Check how well prices are contained within the wedge
                        contained_points = 0
                        for i in range(start_idx, end_idx + 1):
                            upper_bound = upper["intercept"] + upper["slope"] * i
                            lower_bound = lower["intercept"] + lower["slope"] * i
                            if lower_bound <= close_prices[i] <= upper_bound:
                                contained_points += 1
                        
                        containment_score = contained_points / (end_idx - start_idx + 1)
                        
                        confidence = (
                            r_squared_score * 0.25 + 
                            span_score * 0.15 + 
                            convergence_score * 0.3 + 
                            containment_score * 0.3
                        ) * 100
                        
                        # Find intersection point (apex of the wedge)
                        if upper_slope != lower_slope:
                            x_intersection = (lower["intercept"] - upper["intercept"]) / (upper_slope - lower_slope)
                            y_intersection = upper_slope * x_intersection + upper["intercept"]
                        else:
                            # Parallel lines, use end of pattern
                            x_intersection = end_idx + 20  # Extrapolate
                            y_intersection = upper["intercept"] + upper_slope * x_intersection
                        
                        # Determine target price based on wedge type
                        if pattern_type == PatternType.WEDGE_RISING:
                            # Bearish target: distance from breakout to upper line projected downward
                            target_distance = end_upper - end_lower
                            target_price = end_lower - target_distance
                        else:
                            # Bullish target: distance from breakout to lower line projected upward
                            target_distance = end_upper - end_lower
                            target_price = end_upper + target_distance
                        
                        # Create result
                        result = PatternDetectionResult(
                            pattern_type=pattern_type,
                            symbol=symbol,
                            confidence=confidence,
                            start_time=df.index[start_idx],
                            end_time=df.index[end_idx],
                            target_price=target_price,
                            price_level=end_lower if pattern_type == PatternType.WEDGE_RISING else end_upper,  # Breakout level
                            additional_info={
                                "upper_slope": float(upper_slope),
                                "upper_intercept": float(upper["intercept"]),
                                "lower_slope": float(lower_slope),
                                "lower_intercept": float(lower["intercept"]),
                                "apex_x": float(x_intersection),
                                "apex_y": float(y_intersection),
                                "start_gap": float(start_gap),
                                "end_gap": float(end_gap),
                                "convergence_rate": float(convergence_rate)
                            }
                        )
                        
                        results.append(result)
    
    return results