"""
Pattern Detector module for the AI Trading Agent

This module provides chart pattern detection capabilities for technical analysis.
It identifies common chart patterns like support/resistance, trend lines, 
head and shoulders, double tops/bottoms, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import signal
from scipy import stats
import math
from datetime import datetime
import warnings

from ai_trading_agent.common.utils import get_logger
from .pattern_types import PatternType, PatternDetectionResult
from .head_shoulders_detector import detect_head_and_shoulders
from .triangle_detector import detect_triangles
from .flag_pattern_detector import detect_flag_patterns
from .cup_handle_detector import detect_cup_and_handle
from .wedge_detector import detect_wedges

# Try to import Rust functions, but gracefully handle if not available
try:
    import ai_trading_agent_rs
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    warnings.warn("Rust extensions not available. Using Python fallbacks for performance-critical functions.")


# PatternType is now imported from pattern_types.py


# PatternDetectionResult is now imported from pattern_types.py


class PatternDetector:
    """
    Class that detects chart patterns in market data.
    
    This detector implements various algorithms to identify common chart patterns
    used in technical analysis, such as support/resistance levels, trend lines,
    and complex patterns like head and shoulders.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pattern detector with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("pattern_detector")
        
        # Set default parameters
        self.default_params = {
            "peak_prominence": 0.02,       # Minimum prominence for peaks/troughs
            "peak_distance": 5,            # Minimum distance between peaks
            "support_resistance_touches": 3,  # Minimum touches for support/resistance
            "support_resistance_tolerance": 0.01,  # Tolerance for price levels
            "trendline_min_points": 3,     # Minimum points for trendline
            "trendline_tolerance": 0.02,   # Tolerance for trendline touches
            "pattern_lookback_periods": 100  # How far back to look for patterns
        }
        
        # Merge defaults with provided config
        self.params = {**self.default_params, **self.config.get("parameters", {})}
        
        # Initialize metrics
        self.metrics = {
            "patterns_detected": 0,
            "detection_time_ms": 0.0,
            "avg_confidence": 0.0
        }
    
    def detect_patterns(self, market_data: Dict[str, pd.DataFrame], symbols: List[str]) -> Dict[str, List[Dict]]:
        """
        Detect patterns in market data for specified symbols.
        
        Args:
            market_data: Dictionary mapping symbols to DataFrames
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary mapping symbols to lists of detected patterns
        """
        start_time = datetime.now()
        results = {}
        total_patterns = 0
        total_confidence = 0.0
        
        for symbol in symbols:
            if symbol not in market_data:
                self.logger.warning(f"No data for symbol {symbol}")
                continue
                
            df = market_data[symbol].copy()
            
            if len(df) < 20:  # Need minimum data for pattern detection
                self.logger.warning(f"Insufficient data for {symbol} (length={len(df)})")
                continue
            
            patterns = []
            
            # Find peaks and troughs for pattern detection
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            peak_prominence = np.mean(high_prices) * self.params["peak_prominence"]
            peak_indices, _ = signal.find_peaks(
                high_prices, 
                prominence=peak_prominence,
                distance=self.params["peak_distance"]
            )
            
            trough_prominence = np.mean(low_prices) * self.params["peak_prominence"]
            trough_indices, _ = signal.find_peaks(
                -low_prices,  # Invert to find troughs
                prominence=trough_prominence,
                distance=self.params["peak_distance"]
            )
            
            # Detect support and resistance levels
            patterns.extend(self.detect_support_levels(df, symbol))
            patterns.extend(self.detect_resistance_levels(df, symbol))
            
            # Detect trend lines
            patterns.extend(self.detect_trendlines(df, symbol))
            
            # Detect double tops and bottoms
            patterns.extend(self.detect_double_patterns(df, symbol))
            
            # Use specialized pattern detectors for more complex patterns
            # Head and shoulders patterns
            hs_patterns = detect_head_and_shoulders(
                df, symbol, peak_indices, trough_indices, self.params
            )
            patterns.extend(hs_patterns)
            
            # Triangle patterns
            triangle_patterns = detect_triangles(
                df, symbol, peak_indices, trough_indices, self.params
            )
            patterns.extend(triangle_patterns)
            
            # Flag patterns
            flag_patterns = detect_flag_patterns(
                df, symbol, peak_indices, trough_indices, self.params
            )
            patterns.extend(flag_patterns)
            
            # Cup and handle patterns
            cup_handle_patterns = detect_cup_and_handle(
                df, symbol, peak_indices, trough_indices, self.params
            )
            patterns.extend(cup_handle_patterns)
            
            # Wedge patterns (rising and falling)
            wedge_patterns = detect_wedges(
                df, symbol, peak_indices, trough_indices, self.params
            )
            patterns.extend(wedge_patterns)
            
            # Add each pattern to results dictionary
            if patterns:
                # Convert PatternDetectionResult objects to dictionaries
                pattern_dicts = []
                for pattern in patterns:
                    if isinstance(pattern, PatternDetectionResult):
                        pattern_dicts.append(pattern.to_dict())
                    else:
                        pattern_dicts.append(pattern)
                        
                results[symbol] = pattern_dicts
            else:
                results[symbol] = []
            
            # Update metrics
            total_patterns += len(patterns)
            if patterns:
                # Handle both PatternDetectionResult objects and dictionaries
                pattern_confidence_sum = 0
                for p in patterns:
                    if isinstance(p, PatternDetectionResult):
                        pattern_confidence_sum += p.confidence
                    elif isinstance(p, dict) and "confidence" in p:
                        pattern_confidence_sum += p["confidence"]
                total_confidence += pattern_confidence_sum
        
        # Update overall metrics
        self.metrics["patterns_detected"] = total_patterns
        self.metrics["detection_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        
        if total_patterns > 0:
            self.metrics["avg_confidence"] = total_confidence / total_patterns
        
        return results
    
    def detect_support_levels(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> List[PatternDetectionResult]:
        """
        Detect support levels in the price data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            List of detected support levels
        """
        self.logger.debug(f"Detecting support levels for {symbol}")
        
        # Python fallback implementation for support levels detection
        # Extract low prices
        low_prices = df['low'].values
        
        # Find troughs (local minima)
        trough_indices = signal.find_peaks(
            -low_prices,  # Negate to find local minima
            prominence=self.params["peak_prominence"] * np.mean(low_prices),
            distance=self.params["peak_distance"]
        )[0]
        
        if len(trough_indices) < 2:
            return []
        
        support_levels = []
        min_touches = self.params["support_resistance_touches"]
        tolerance = self.params["support_resistance_tolerance"]
        
        # Check each potential support level
        for i, idx in enumerate(trough_indices):
            level_price = low_prices[idx]
            
            # Count touches (prices within tolerance of the level)
            touches = np.sum(np.abs(low_prices - level_price) <= level_price * tolerance)
            
            if touches >= min_touches:
                # Calculate the strength/confidence based on:
                # 1. Number of touches
                # 2. Longevity (time span)
                # 3. How well prices respect the level
                
                # Find all indices where price comes close to the level
                touch_indices = np.where(np.abs(low_prices - level_price) <= level_price * tolerance)[0]
                
                if len(touch_indices) == 0:
                    continue
                    
                # Span of the support level
                span = max(touch_indices) - min(touch_indices)
                
                # Average deviation from the level
                avg_deviation = np.mean(np.abs(low_prices[touch_indices] - level_price) / level_price)
                
                # Calculate confidence score (0-1)
                touch_score = min(1.0, touches / (min_touches * 1.5))
                span_score = min(1.0, span / (len(low_prices) * 0.5))
                precision_score = 1.0 - min(1.0, avg_deviation / tolerance)
                
                confidence = (touch_score * 0.5 + span_score * 0.3 + precision_score * 0.2)
                
                # Create result object with necessary data
                try:
                    result = PatternDetectionResult(
                        pattern_type=PatternType.SUPPORT,
                        symbol=symbol,
                        confidence=confidence,
                        start_idx=int(min(touch_indices)),
                        end_idx=int(max(touch_indices)),
                        additional_info={
                            "price_level": float(level_price),
                            "touches": int(touches),
                            "span": int(span)
                        }
                    )
                    support_levels.append(result)
                except Exception as e:
                    self.logger.warning(f"Error creating support level result: {e}")
                    continue
        
        return support_levels
    
    def detect_resistance_levels(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> List[PatternDetectionResult]:
        """
        Detect resistance levels in the price data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            List of detected resistance levels
        """
        self.logger.debug(f"Detecting resistance levels for {symbol}")
        
        # Python fallback implementation for resistance levels detection
        # Extract high prices
        high_prices = df['high'].values
        
        # Find peaks (local maxima)
        peak_indices = signal.find_peaks(
            high_prices,
            prominence=self.params["peak_prominence"] * np.mean(high_prices),
            distance=self.params["peak_distance"]
        )[0]
        
        if len(peak_indices) < 2:
            return []
        
        resistance_levels = []
        min_touches = self.params["support_resistance_touches"]
        tolerance = self.params["support_resistance_tolerance"]
        
        # Check each potential resistance level
        for i, idx in enumerate(peak_indices):
            level_price = high_prices[idx]
            
            # Count touches (prices within tolerance of the level)
            touches = np.sum(np.abs(high_prices - level_price) <= level_price * tolerance)
            
            if touches >= min_touches:
                # Calculate the strength/confidence based on:
                # 1. Number of touches
                # 2. Longevity (time span)
                # 3. How well prices respect the level
                
                # Find all indices where price comes close to the level
                touch_indices = np.where(np.abs(high_prices - level_price) <= level_price * tolerance)[0]
                
                if len(touch_indices) == 0:
                    continue
                
                # Span of the resistance level
                span = max(touch_indices) - min(touch_indices)
                
                # Average deviation from the level
                avg_deviation = np.mean(np.abs(high_prices[touch_indices] - level_price) / level_price)
                
                # Calculate confidence score (0-1)
                touch_score = min(1.0, touches / (min_touches * 1.5))
                span_score = min(1.0, span / (len(high_prices) * 0.5))
                precision_score = 1.0 - min(1.0, avg_deviation / tolerance)
                
                confidence = (touch_score * 0.5 + span_score * 0.3 + precision_score * 0.2)
                
                # Create result object with necessary data
                try:
                    result = PatternDetectionResult(
                        pattern_type=PatternType.RESISTANCE,
                        symbol=symbol,
                        confidence=confidence,
                        start_idx=int(min(touch_indices)),
                        end_idx=int(max(touch_indices)),
                        additional_info={
                            "price_level": float(level_price),
                            "touches": int(touches),
                            "span": int(span)
                        }
                    )
                    resistance_levels.append(result)
                except Exception as e:
                    self.logger.warning(f"Error creating resistance level result: {e}")
                    continue
        
        return resistance_levels
        
    def detect_trendlines(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> List[PatternDetectionResult]:
        """
        Detect ascending and descending trendlines in the price data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            List of detected trendlines
        """
        self.logger.debug(f"Detecting trendlines for {symbol}")
        
        # Extract price data
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        # Find peaks and troughs
        peak_indices = signal.find_peaks(
            high_prices,
            prominence=self.params["peak_prominence"] * np.mean(high_prices),
            distance=self.params["peak_distance"]
        )[0]
        
        trough_indices = signal.find_peaks(
            -low_prices,  # Negate to find local minima
            prominence=self.params["peak_prominence"] * np.mean(low_prices),
            distance=self.params["peak_distance"]
        )[0]
        
        trendlines = []
        min_points = self.params["trendline_min_points"]
        tolerance = self.params["trendline_tolerance"]
        
        # Detect bullish (ascending) trendlines using lows
        if len(trough_indices) >= min_points:
            # Try to fit a line to different combinations of troughs
            for i in range(len(trough_indices) - min_points + 1):
                # Take a subset of troughs to check
                subset_indices = trough_indices[i:i+min_points]
                x_points = subset_indices
                y_points = low_prices[subset_indices]
                
                # Only look for ascending trendlines
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)
                
                if slope > 0:  # Ascending trendline
                    # Check how well other lows respect this trendline
                    # Calculate expected y-values for all x points
                    expected_y = slope * np.arange(len(low_prices)) + intercept
                    
                    # Count points that are close to but above the line
                    deviations = low_prices - expected_y
                    valid_points = np.where(
                        (deviations > -low_prices * tolerance) & 
                        (deviations < low_prices * tolerance * 0.5)
                    )[0]
                    
                    if len(valid_points) >= min_points:
                        # Calculate confidence based on:
                        # 1. Number of points that respect the line
                        # 2. R-squared of the linear regression
                        # 3. Span of the trendline
                        
                        points_score = min(1.0, len(valid_points) / (min_points * 2))
                        fit_score = max(0, r_value ** 2)  # R-squared
                        span_score = min(1.0, (max(valid_points) - min(valid_points)) / len(low_prices))
                        
                        confidence = (points_score * 0.4 + fit_score * 0.3 + span_score * 0.3)
                        
                        # Create result for ascending trendline
                        result = PatternDetectionResult(
                            pattern_type=PatternType.TRENDLINE_ASCENDING,
                            symbol=symbol,
                            confidence=confidence,
                            start_idx=min(valid_points),
                            end_idx=max(valid_points),
                            additional_info={
                                "slope": float(slope),
                                "intercept": float(intercept),
                                "r_squared": float(r_value ** 2),
                                "points": int(len(valid_points))
                            }
                        )
                        
                        trendlines.append(result)
        
        # Detect bearish (descending) trendlines using highs
        if len(peak_indices) >= min_points:
            # Try to fit a line to different combinations of peaks
            for i in range(len(peak_indices) - min_points + 1):
                # Take a subset of peaks to check
                subset_indices = peak_indices[i:i+min_points]
                x_points = subset_indices
                y_points = high_prices[subset_indices]
                
                # Only look for descending trendlines
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_points, y_points)
                
                if slope < 0:  # Descending trendline
                    # Check how well other highs respect this trendline
                    # Calculate expected y-values for all x points
                    expected_y = slope * np.arange(len(high_prices)) + intercept
                    
                    # Count points that are close to but below the line
                    deviations = expected_y - high_prices
                    valid_points = np.where(
                        (deviations > -high_prices * tolerance) & 
                        (deviations < high_prices * tolerance * 0.5)
                    )[0]
                    
                    if len(valid_points) >= min_points:
                        # Calculate confidence
                        points_score = min(1.0, len(valid_points) / (min_points * 2))
                        fit_score = max(0, r_value ** 2)  # R-squared
                        span_score = min(1.0, (max(valid_points) - min(valid_points)) / len(high_prices))
                        
                        confidence = (points_score * 0.4 + fit_score * 0.3 + span_score * 0.3)
                        
                        # Create result for descending trendline
                        result = PatternDetectionResult(
                            pattern_type=PatternType.TRENDLINE_DESCENDING,
                            symbol=symbol,
                            confidence=confidence,
                            start_idx=min(valid_points),
                            end_idx=max(valid_points),
                            additional_info={
                                "slope": float(slope),
                                "intercept": float(intercept),
                                "r_squared": float(r_value ** 2),
                                "points": int(len(valid_points))
                            }
                        )
                        
                        trendlines.append(result)
        
        return trendlines
    
    def detect_double_patterns(
        self, 
        df: pd.DataFrame, 
        symbol: str
    ) -> List[PatternDetectionResult]:
        """
        Detect double top and double bottom patterns.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            List of detected double top/bottom patterns
        """
        self.logger.debug(f"Detecting double patterns for {symbol}")
        
        # Extract price data
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        # Find peaks and troughs
        peak_indices, peak_properties = signal.find_peaks(
            high_prices,
            prominence=self.params["peak_prominence"] * np.mean(high_prices),
            distance=self.params["peak_distance"]
        )
        
        trough_indices, trough_properties = signal.find_peaks(
            -low_prices,  # Negate to find local minima
            prominence=self.params["peak_prominence"] * np.mean(low_prices),
            distance=self.params["peak_distance"]
        )
        
        results = []
        
        # Detect double tops
        if len(peak_indices) >= 2:
            for i in range(len(peak_indices) - 1):
                for j in range(i + 1, len(peak_indices)):
                    # Check if the two peaks are at similar levels
                    peak1_idx = peak_indices[i]
                    peak2_idx = peak_indices[j]
                    peak1_price = high_prices[peak1_idx]
                    peak2_price = high_prices[peak2_idx]
                    
                    # Peaks should be at similar levels
                    if abs(peak1_price - peak2_price) <= peak1_price * 0.03:
                        # There should be a significant trough between the peaks
                        between_troughs = [idx for idx in trough_indices if peak1_idx < idx < peak2_idx]
                        
                        if between_troughs:
                            middle_trough_idx = between_troughs[np.argmin(low_prices[between_troughs])]
                            middle_trough_price = low_prices[middle_trough_idx]
                            
                            # Calculate the depth of the trough relative to the peaks
                            avg_peak_price = (peak1_price + peak2_price) / 2
                            trough_depth = (avg_peak_price - middle_trough_price) / avg_peak_price
                            
                            # Trough should be deep enough
                            if trough_depth >= 0.03:
                                # Check for confirmation (close below the trough after the second peak)
                                confirmation = False
                                for k in range(peak2_idx + 1, len(close_prices)):
                                    if close_prices[k] < middle_trough_price:
                                        confirmation = True
                                        break
                                
                                # Calculate confidence based on:
                                # 1. Similarity of peak heights
                                # 2. Depth of the middle trough
                                # 3. Whether there's confirmation
                                # 4. Ideal spacing between peaks
                                
                                peak_similarity = 1.0 - abs(peak1_price - peak2_price) / peak1_price
                                depth_score = min(1.0, trough_depth / 0.1)  # Max score at 10% depth
                                
                                # Ideal spacing is 15-30% of the data length
                                spacing = (peak2_idx - peak1_idx) / len(high_prices)
                                spacing_score = 1.0 - min(1.0, abs(spacing - 0.2) / 0.15)
                                
                                confidence = (
                                    peak_similarity * 0.3 + 
                                    depth_score * 0.3 + 
                                    spacing_score * 0.2 + 
                                    (0.2 if confirmation else 0.0)
                                )
                                
                                # Create result
                                result = PatternDetectionResult(
                                    pattern_type=PatternType.DOUBLE_TOP,
                                    symbol=symbol,
                                    confidence=confidence,
                                    start_idx=peak1_idx,
                                    end_idx=peak2_idx,
                                    additional_info={
                                        "peak1_idx": int(peak1_idx),
                                        "peak2_idx": int(peak2_idx),
                                        "peak1_price": float(peak1_price),
                                        "peak2_price": float(peak2_price),
                                        "trough_idx": int(middle_trough_idx),
                                        "trough_price": float(middle_trough_price),
                                        "confirmed": confirmation
                                    }
                                )
                                
                                results.append(result)
        
        # Detect double bottoms - similar logic but with troughs
        if len(trough_indices) >= 2:
            for i in range(len(trough_indices) - 1):
                for j in range(i + 1, len(trough_indices)):
                    # Check if the two troughs are at similar levels
                    trough1_idx = trough_indices[i]
                    trough2_idx = trough_indices[j]
                    trough1_price = low_prices[trough1_idx]
                    trough2_price = low_prices[trough2_idx]
                    
                    # Troughs should be at similar levels
                    if abs(trough1_price - trough2_price) <= trough1_price * 0.03:
                        # There should be a significant peak between the troughs
                        between_peaks = [idx for idx in peak_indices if trough1_idx < idx < trough2_idx]
                        
                        if between_peaks:
                            middle_peak_idx = between_peaks[np.argmax(high_prices[between_peaks])]
                            middle_peak_price = high_prices[middle_peak_idx]
                            
                            # Calculate the height of the peak relative to the troughs
                            avg_trough_price = (trough1_price + trough2_price) / 2
                            peak_height = (middle_peak_price - avg_trough_price) / avg_trough_price
                            
                            # Peak should be high enough
                            if peak_height >= 0.03:
                                # Check for confirmation (close above the peak after the second trough)
                                confirmation = False
                                for k in range(trough2_idx + 1, len(close_prices)):
                                    if close_prices[k] > middle_peak_price:
                                        confirmation = True
                                        break
                                
                                # Calculate confidence similar to double top
                                trough_similarity = 1.0 - abs(trough1_price - trough2_price) / trough1_price
                                height_score = min(1.0, peak_height / 0.1)  # Max score at 10% height
                                
                                spacing = (trough2_idx - trough1_idx) / len(low_prices)
                                spacing_score = 1.0 - min(1.0, abs(spacing - 0.2) / 0.15)
                                
                                confidence = (
                                    trough_similarity * 0.3 + 
                                    height_score * 0.3 + 
                                    spacing_score * 0.2 + 
                                    (0.2 if confirmation else 0.0)
                                )
                                
                                # Create result
                                result = PatternDetectionResult(
                                    pattern_type=PatternType.DOUBLE_BOTTOM,
                                    symbol=symbol,
                                    confidence=confidence,
                                    start_idx=trough1_idx,
                                    end_idx=trough2_idx,
                                    additional_info={
                                        "trough1_idx": int(trough1_idx),
                                        "trough2_idx": int(trough2_idx),
                                        "trough1_price": float(trough1_price),
                                        "trough2_price": float(trough2_price),
                                        "peak_idx": int(middle_peak_idx),
                                        "peak_price": float(middle_peak_price),
                                        "confirmed": confirmation
                                    }
                                )
                                
                                results.append(result)
        
        return results
