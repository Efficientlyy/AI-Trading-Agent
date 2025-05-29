"""
Triangle pattern detection module for the AI Trading Agent

This module provides specialized detection for triangle patterns, including
ascending, descending, and symmetrical triangles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from datetime import datetime

from .pattern_types import PatternDetectionResult, PatternType


def detect_triangles(
    df: pd.DataFrame, 
    symbol: str,
    peak_indices: np.ndarray,
    trough_indices: np.ndarray,
    params: Dict
) -> List[PatternDetectionResult]:
    """
    Detect triangle patterns (ascending, descending, symmetrical).
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol
        peak_indices: Indices of price peaks
        trough_indices: Indices of price troughs
        params: Detection parameters
        
    Returns:
        List of detected triangle patterns
    """
    # Extract price data
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    
    results = []
    min_points = params.get("trendline_min_points", 3)
    
    # We need at least 3 points for each boundary line
    if len(peak_indices) >= min_points and len(trough_indices) >= min_points:
        # Try to find pairs of converging trendlines
        # First, identify potential upper and lower trendlines
        
        # Find upper trendlines (using peaks)
        upper_trendlines = []
        for i in range(len(peak_indices) - min_points + 1):
            subset_indices = peak_indices[i:i+min_points]
            x_points = subset_indices
            y_points = high_prices[subset_indices]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)
            
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
        for i in range(len(trough_indices) - min_points + 1):
            subset_indices = trough_indices[i:i+min_points]
            x_points = subset_indices
            y_points = low_prices[subset_indices]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)
            
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
        
        # Find pairs of converging trendlines that form triangles
        for upper in upper_trendlines:
            for lower in lower_trendlines:
                # Determine the timespan of the potential triangle
                start_idx = max(upper["start_idx"], lower["start_idx"])
                end_idx = min(upper["end_idx"], lower["end_idx"])
                
                # Make sure there's overlap
                if start_idx < end_idx:
                    # Get slopes of upper and lower trendlines
                    upper_slope = upper["slope"]
                    lower_slope = lower["slope"]
                    
                    # Check if the trendlines are converging
                    # For ascending triangle: upper flat (slope ~ 0), lower positive
                    # For descending triangle: upper negative, lower flat (slope ~ 0)
                    # For symmetrical: upper negative, lower positive
                    
                    # Calculate the average price at the start of the pattern
                    start_upper_price = upper["intercept"] + upper["slope"] * start_idx
                    start_lower_price = lower["intercept"] + lower["slope"] * start_idx
                    
                    # Calculate the average price at the end of the pattern
                    end_upper_price = upper["intercept"] + upper["slope"] * end_idx
                    end_lower_price = lower["intercept"] + lower["slope"] * end_idx
                    
                    # Price gap should narrow from start to end
                    start_gap = start_upper_price - start_lower_price
                    end_gap = end_upper_price - end_lower_price
                    
                    # Make sure start_gap is positive (upper line above lower line)
                    if start_gap > 0 and end_gap < start_gap:
                        # Determine triangle type
                        pattern_type = None
                        
                        # Ascending triangle: flat top, rising bottom
                        if abs(upper_slope) < 0.001 and lower_slope > 0.001:
                            pattern_type = PatternType.TRIANGLE_ASCENDING
                        # Descending triangle: falling top, flat bottom
                        elif upper_slope < -0.001 and abs(lower_slope) < 0.001:
                            pattern_type = PatternType.TRIANGLE_DESCENDING
                        # Symmetrical triangle: falling top, rising bottom
                        elif upper_slope < -0.001 and lower_slope > 0.001:
                            pattern_type = PatternType.TRIANGLE_SYMMETRICAL
                        
                        if pattern_type:
                            # Calculate confidence
                            # 1. R-squared of trendlines
                            # 2. Pattern span
                            # 3. Convergence rate
                            # 4. Price containment
                            
                            r_squared_score = (upper["r_squared"] + lower["r_squared"]) / 2
                            span_score = min(1.0, (end_idx - start_idx) / (len(close_prices) * 0.2))  # Score of 1 for 20% of data
                            convergence_score = min(1.0, 1.0 - (end_gap / start_gap))  # Higher score for more convergence
                            
                            # Check how well prices are contained within the triangle
                            contained_points = 0
                            for i in range(start_idx, end_idx + 1):
                                upper_bound = upper["intercept"] + upper["slope"] * i
                                lower_bound = lower["intercept"] + lower["slope"] * i
                                if lower_bound <= close_prices[i] <= upper_bound:
                                    contained_points += 1
                            
                            containment_score = contained_points / (end_idx - start_idx + 1)
                            
                            confidence = (
                                r_squared_score * 0.2 + 
                                span_score * 0.2 + 
                                convergence_score * 0.3 + 
                                containment_score * 0.3
                            )
                            
                            # Find intersection point (apex of the triangle)
                            if upper_slope != lower_slope:
                                x_intersection = (lower["intercept"] - upper["intercept"]) / (upper_slope - lower_slope)
                                y_intersection = upper_slope * x_intersection + upper["intercept"]
                            else:
                                # Parallel lines, use end of pattern
                                x_intersection = end_idx
                                y_intersection = upper["intercept"] + upper_slope * end_idx
                            
                            # Create result
                            result = PatternDetectionResult(
                                pattern_type=pattern_type,
                                symbol=symbol,
                                confidence=confidence,
                                start_time=df.index[start_idx],
                                end_time=df.index[end_idx],
                                target_price=y_intersection,  # Target price is the apex of the triangle
                                price_level=(upper_slope * end_idx + upper["intercept"] + lower_slope * end_idx + lower["intercept"]) / 2,  # Average of upper and lower line at end
                                additional_info={
                                    "upper_slope": float(upper_slope),
                                    "upper_intercept": float(upper["intercept"]),
                                    "lower_slope": float(lower_slope),
                                    "lower_intercept": float(lower["intercept"]),
                                    "apex_x": float(x_intersection),
                                    "apex_y": float(y_intersection),
                                    "start_gap": float(start_gap),
                                    "end_gap": float(end_gap)
                                }
                            )
                            
                            results.append(result)
    
    return results
