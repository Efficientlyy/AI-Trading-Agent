"""
Three-Candle Pattern Detection Module

This module provides functions for detecting three-candle patterns in price data,
including Morning Star, Evening Star, Three White Soldiers, Three Black Crows, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from enum import Enum, auto

from ..common.utils import get_logger


class PatternType(Enum):
    """Enumeration of three-candle pattern types."""
    MORNING_STAR = auto()
    EVENING_STAR = auto()
    THREE_WHITE_SOLDIERS = auto()
    THREE_BLACK_CROWS = auto()
    THREE_INSIDE_UP = auto()
    THREE_INSIDE_DOWN = auto()
    THREE_OUTSIDE_UP = auto()
    THREE_OUTSIDE_DOWN = auto()
    ABANDONED_BABY_BULLISH = auto()
    ABANDONED_BABY_BEARISH = auto()


class ThreeCandlePatternDetector:
    """
    Detector for common three-candle patterns in price data.
    
    This class implements detection algorithms for various three-candle patterns
    with configurable parameters and confidence scoring.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the three-candle pattern detector.
        
        Args:
            config: Configuration dictionary with parameters
                - body_size_threshold: Minimum body size as a percentage of total range
                - doji_threshold: Maximum body size for a candle to be considered a doji
                - gap_threshold: Minimum gap size for patterns that require gaps
                - confidence_threshold: Minimum confidence score to report a pattern
        """
        self.logger = get_logger("ThreeCandlePatternDetector")
        
        # Default configuration
        self.config = {
            "body_size_threshold": 0.5,  # Body must be at least 50% of candle range
            "doji_threshold": 0.1,       # Body must be less than 10% to be a doji
            "gap_threshold": 0.2,        # Gap must be at least 20% of average range
            "confidence_threshold": 0.7,  # Confidence must be at least 70%
            "enable_confidence_scoring": True
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        self.logger.info("ThreeCandlePatternDetector initialized")
    
    def detect_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect all supported three-candle patterns in the data.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of dictionaries with pattern information
        """
        if len(data) < 3:
            return []
            
        patterns = []
        
        # Check for each pattern type
        morning_star = self.detect_morning_star(data)
        if morning_star:
            patterns.append(morning_star)
            
        evening_star = self.detect_evening_star(data)
        if evening_star:
            patterns.append(evening_star)
            
        three_white_soldiers = self.detect_three_white_soldiers(data)
        if three_white_soldiers:
            patterns.append(three_white_soldiers)
            
        three_black_crows = self.detect_three_black_crows(data)
        if three_black_crows:
            patterns.append(three_black_crows)
            
        three_inside_up = self.detect_three_inside_up(data)
        if three_inside_up:
            patterns.append(three_inside_up)
            
        three_inside_down = self.detect_three_inside_down(data)
        if three_inside_down:
            patterns.append(three_inside_down)
            
        three_outside_up = self.detect_three_outside_up(data)
        if three_outside_up:
            patterns.append(three_outside_up)
            
        three_outside_down = self.detect_three_outside_down(data)
        if three_outside_down:
            patterns.append(three_outside_down)
            
        abandoned_baby_bullish = self.detect_abandoned_baby_bullish(data)
        if abandoned_baby_bullish:
            patterns.append(abandoned_baby_bullish)
            
        abandoned_baby_bearish = self.detect_abandoned_baby_bearish(data)
        if abandoned_baby_bearish:
            patterns.append(abandoned_baby_bearish)
        
        return patterns
    
    def _calculate_candle_properties(self, data: pd.DataFrame) -> Dict:
        """Calculate properties for all candles in the data."""
        properties = {
            "body_size": abs(data["close"] - data["open"]),
            "candle_range": data["high"] - data["low"],
            "is_bullish": data["close"] > data["open"],
            "upper_shadow": data["high"] - data[["close", "open"]].max(axis=1),
            "lower_shadow": data[["close", "open"]].min(axis=1) - data["low"],
            "body_percent": abs(data["close"] - data["open"]) / (data["high"] - data["low"]),
            "midpoint": (data["close"] + data["open"]) / 2
        }
        
        # Add derived properties
        properties["is_doji"] = properties["body_percent"] < self.config["doji_threshold"]
        properties["is_long_body"] = properties["body_percent"] > self.config["body_size_threshold"]
        
        return properties
    
    def _check_gap_up(self, current_idx: int, props: Dict) -> bool:
        """Check if there is a gap up between candles."""
        prev_high = max(props["close"][current_idx-1], props["open"][current_idx-1])
        curr_low = min(props["close"][current_idx], props["open"][current_idx])
        
        return curr_low > prev_high
    
    def _check_gap_down(self, current_idx: int, props: Dict) -> bool:
        """Check if there is a gap down between candles."""
        prev_low = min(props["close"][current_idx-1], props["open"][current_idx-1])
        curr_high = max(props["close"][current_idx], props["open"][current_idx])
        
        return curr_high < prev_low
    
    def _calculate_confidence(self, pattern_type: PatternType, props: Dict, idx: int) -> float:
        """
        Calculate confidence score for a detected pattern.
        
        Args:
            pattern_type: Type of pattern detected
            props: Candle properties dictionary
            idx: Index of the last candle in the pattern
            
        Returns:
            Confidence score between 0 and 1
        """
        if not self.config["enable_confidence_scoring"]:
            return 1.0
            
        confidence = 0.7  # Base confidence
        
        if pattern_type == PatternType.MORNING_STAR:
            # First candle should be bearish with long body
            if props["is_long_body"][idx-2] and not props["is_bullish"][idx-2]:
                confidence += 0.1
                
            # Second candle should be a doji or small body
            if props["is_doji"][idx-1] or props["body_percent"][idx-1] < 0.3:
                confidence += 0.1
                
            # Third candle should be bullish with long body
            if props["is_long_body"][idx] and props["is_bullish"][idx]:
                confidence += 0.1
                
            # Check for gaps
            if self._check_gap_down(idx-1, props) and self._check_gap_up(idx, props):
                confidence += 0.2
                
            # Check if third candle closes above midpoint of first candle
            if props["close"][idx] > (props["open"][idx-2] + props["close"][idx-2]) / 2:
                confidence += 0.1
        
        elif pattern_type == PatternType.EVENING_STAR:
            # First candle should be bullish with long body
            if props["is_long_body"][idx-2] and props["is_bullish"][idx-2]:
                confidence += 0.1
                
            # Second candle should be a doji or small body
            if props["is_doji"][idx-1] or props["body_percent"][idx-1] < 0.3:
                confidence += 0.1
                
            # Third candle should be bearish with long body
            if props["is_long_body"][idx] and not props["is_bullish"][idx]:
                confidence += 0.1
                
            # Check for gaps
            if self._check_gap_up(idx-1, props) and self._check_gap_down(idx, props):
                confidence += 0.2
                
            # Check if third candle closes below midpoint of first candle
            if props["close"][idx] < (props["open"][idx-2] + props["close"][idx-2]) / 2:
                confidence += 0.1
        
        # Cap confidence at 1.0
        return min(confidence, 1.0)
    
    def detect_morning_star(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Morning Star pattern.
        
        A Morning Star is a bullish reversal pattern:
        1. First candle: Bearish with long body
        2. Second candle: Small body or doji, gapping down
        3. Third candle: Bullish with long body, gapping up
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check basic pattern criteria
        if (not props["is_bullish"][idx-2] and  # First candle is bearish
            props["is_doji"][idx-1] and  # Second candle is a doji
            props["is_bullish"][idx]):  # Third candle is bullish
            
            # Calculate confidence
            confidence = self._calculate_confidence(PatternType.MORNING_STAR, props, idx)
            
            # Return pattern if confidence meets threshold
            if confidence >= self.config["confidence_threshold"]:
                return {
                    "pattern": "morning_star",
                    "position": idx,
                    "direction": "bullish",
                    "confidence": confidence,
                    "candles": [idx-2, idx-1, idx]
                }
        
        return None
    
    def detect_evening_star(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Evening Star pattern.
        
        An Evening Star is a bearish reversal pattern:
        1. First candle: Bullish with long body
        2. Second candle: Small body or doji, gapping up
        3. Third candle: Bearish with long body, gapping down
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check basic pattern criteria
        if (props["is_bullish"][idx-2] and  # First candle is bullish
            props["is_doji"][idx-1] and  # Second candle is a doji
            not props["is_bullish"][idx]):  # Third candle is bearish
            
            # Calculate confidence
            confidence = self._calculate_confidence(PatternType.EVENING_STAR, props, idx)
            
            # Return pattern if confidence meets threshold
            if confidence >= self.config["confidence_threshold"]:
                return {
                    "pattern": "evening_star",
                    "position": idx,
                    "direction": "bearish",
                    "confidence": confidence,
                    "candles": [idx-2, idx-1, idx]
                }
        
        return None
    
    def detect_three_white_soldiers(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Three White Soldiers pattern.
        
        Three White Soldiers is a bullish reversal pattern:
        1. Three consecutive bullish candles
        2. Each candle opens within previous candle's body
        3. Each candle closes higher than the previous
        4. Each candle has small upper shadows
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check basic pattern criteria
        if (props["is_bullish"][idx-2] and  # First candle is bullish
            props["is_bullish"][idx-1] and  # Second candle is bullish
            props["is_bullish"][idx] and  # Third candle is bullish
            props["close"][idx-1] > props["close"][idx-2] and  # Each closes higher
            props["close"][idx] > props["close"][idx-1] and
            props["open"][idx-1] > props["open"][idx-2] and  # Each opens higher
            props["open"][idx] > props["open"][idx-1] and
            props["upper_shadow"][idx-2] < 0.3 * props["body_size"][idx-2] and  # Small upper shadows
            props["upper_shadow"][idx-1] < 0.3 * props["body_size"][idx-1] and
            props["upper_shadow"][idx] < 0.3 * props["body_size"][idx]):
            
            confidence = 0.8  # Base confidence for this pattern
            
            # Higher confidence if in a downtrend
            if idx > 5 and data["close"][idx-5:idx-2].mean() > data["close"][idx-2]:
                confidence += 0.1
                
            # Return pattern if confidence meets threshold
            if confidence >= self.config["confidence_threshold"]:
                return {
                    "pattern": "three_white_soldiers",
                    "position": idx,
                    "direction": "bullish",
                    "confidence": confidence,
                    "candles": [idx-2, idx-1, idx]
                }
        
        return None
    
    def detect_three_black_crows(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Three Black Crows pattern.
        
        Three Black Crows is a bearish reversal pattern:
        1. Three consecutive bearish candles
        2. Each candle opens within previous candle's body
        3. Each candle closes lower than the previous
        4. Each candle has small lower shadows
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check basic pattern criteria
        if (not props["is_bullish"][idx-2] and  # First candle is bearish
            not props["is_bullish"][idx-1] and  # Second candle is bearish
            not props["is_bullish"][idx] and  # Third candle is bearish
            props["close"][idx-1] < props["close"][idx-2] and  # Each closes lower
            props["close"][idx] < props["close"][idx-1] and
            props["open"][idx-1] < props["open"][idx-2] and  # Each opens lower
            props["open"][idx] < props["open"][idx-1] and
            props["lower_shadow"][idx-2] < 0.3 * props["body_size"][idx-2] and  # Small lower shadows
            props["lower_shadow"][idx-1] < 0.3 * props["body_size"][idx-1] and
            props["lower_shadow"][idx] < 0.3 * props["body_size"][idx]):
            
            confidence = 0.8  # Base confidence for this pattern
            
            # Higher confidence if in an uptrend
            if idx > 5 and data["close"][idx-5:idx-2].mean() < data["close"][idx-2]:
                confidence += 0.1
                
            # Return pattern if confidence meets threshold
            if confidence >= self.config["confidence_threshold"]:
                return {
                    "pattern": "three_black_crows",
                    "position": idx,
                    "direction": "bearish",
                    "confidence": confidence,
                    "candles": [idx-2, idx-1, idx]
                }
        
        return None
    
    def detect_three_inside_up(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Three Inside Up pattern.
        
        Three Inside Up is a bullish reversal pattern:
        1. First candle: Bearish with long body
        2. Second candle: Bullish, opens below first's close and closes above first's midpoint
        3. Third candle: Bullish, closes above second's close
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check basic pattern criteria
        if (not props["is_bullish"][idx-2] and  # First candle is bearish
            props["is_bullish"][idx-1] and  # Second candle is bullish
            props["is_bullish"][idx] and  # Third candle is bullish
            props["is_long_body"][idx-2] and  # First candle has a long body
            props["open"][idx-1] < props["close"][idx-2] and  # Second opens below first's close
            props["close"][idx-1] > (props["open"][idx-2] + props["close"][idx-2]) / 2 and  # Second closes above first's midpoint
            props["close"][idx] > props["close"][idx-1]):  # Third closes above second's close
            
            confidence = 0.75  # Base confidence for this pattern
            
            # Higher confidence if in a downtrend
            if idx > 5 and data["close"][idx-5:idx-2].mean() > data["close"][idx-2]:
                confidence += 0.1
                
            # Higher confidence if second candle closes in upper half of first
            midpoint = (props["open"][idx-2] + props["close"][idx-2]) / 2
            upper_half = (props["open"][idx-2] - midpoint) / 2
            if props["close"][idx-1] > midpoint + upper_half:
                confidence += 0.1
                
            # Return pattern if confidence meets threshold
            if confidence >= self.config["confidence_threshold"]:
                return {
                    "pattern": "three_inside_up",
                    "position": idx,
                    "direction": "bullish",
                    "confidence": confidence,
                    "candles": [idx-2, idx-1, idx]
                }
        
        return None
    
    def detect_three_inside_down(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Three Inside Down pattern.
        
        Three Inside Down is a bearish reversal pattern:
        1. First candle: Bullish with long body
        2. Second candle: Bearish, opens above first's close and closes below first's midpoint
        3. Third candle: Bearish, closes below second's close
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check basic pattern criteria
        if (props["is_bullish"][idx-2] and  # First candle is bullish
            not props["is_bullish"][idx-1] and  # Second candle is bearish
            not props["is_bullish"][idx] and  # Third candle is bearish
            props["is_long_body"][idx-2] and  # First candle has a long body
            props["open"][idx-1] > props["close"][idx-2] and  # Second opens above first's close
            props["close"][idx-1] < (props["open"][idx-2] + props["close"][idx-2]) / 2 and  # Second closes below first's midpoint
            props["close"][idx] < props["close"][idx-1]):  # Third closes below second's close
            
            confidence = 0.75  # Base confidence for this pattern
            
            # Higher confidence if in an uptrend
            if idx > 5 and data["close"][idx-5:idx-2].mean() < data["close"][idx-2]:
                confidence += 0.1
                
            # Higher confidence if second candle closes in lower half of first
            midpoint = (props["open"][idx-2] + props["close"][idx-2]) / 2
            lower_half = (midpoint - props["close"][idx-2]) / 2
            if props["close"][idx-1] < midpoint - lower_half:
                confidence += 0.1
                
            # Return pattern if confidence meets threshold
            if confidence >= self.config["confidence_threshold"]:
                return {
                    "pattern": "three_inside_down",
                    "position": idx,
                    "direction": "bearish",
                    "confidence": confidence,
                    "candles": [idx-2, idx-1, idx]
                }
        
        return None
    
    def detect_three_outside_up(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Three Outside Up pattern.
        
        Three Outside Up is a bullish reversal pattern:
        1. First candle: Bearish
        2. Second candle: Bullish, engulfs the first candle
        3. Third candle: Bullish, closes above second's close
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check if first candle is bearish
        if not props["is_bullish"][idx-2]:
            # Check if second candle is bullish and engulfs the first
            if (props["is_bullish"][idx-1] and
                props["open"][idx-1] < props["close"][idx-2] and  # Opens below first's close
                props["close"][idx-1] > props["open"][idx-2]):  # Closes above first's open
                
                # Check if third candle is bullish and closes above second's close
                if props["is_bullish"][idx] and props["close"][idx] > props["close"][idx-1]:
                    
                    confidence = 0.8  # Base confidence for this pattern
                    
                    # Higher confidence if in a downtrend
                    if idx > 5 and data["close"][idx-5:idx-2].mean() > data["close"][idx-2]:
                        confidence += 0.1
                    
                    # Higher confidence if second candle has strong body
                    if props["is_long_body"][idx-1]:
                        confidence += 0.05
                    
                    # Higher confidence if third candle has strong body
                    if props["is_long_body"][idx]:
                        confidence += 0.05
                    
                    # Return pattern if confidence meets threshold
                    if confidence >= self.config["confidence_threshold"]:
                        return {
                            "pattern": "three_outside_up",
                            "position": idx,
                            "direction": "bullish",
                            "confidence": confidence,
                            "candles": [idx-2, idx-1, idx]
                        }
        
        return None
    
    def detect_three_outside_down(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Three Outside Down pattern.
        
        Three Outside Down is a bearish reversal pattern:
        1. First candle: Bullish
        2. Second candle: Bearish, engulfs the first candle
        3. Third candle: Bearish, closes below second's close
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check if first candle is bullish
        if props["is_bullish"][idx-2]:
            # Check if second candle is bearish and engulfs the first
            if (not props["is_bullish"][idx-1] and
                props["open"][idx-1] > props["close"][idx-2] and  # Opens above first's close
                props["close"][idx-1] < props["open"][idx-2]):  # Closes below first's open
                
                # Check if third candle is bearish and closes below second's close
                if not props["is_bullish"][idx] and props["close"][idx] < props["close"][idx-1]:
                    
                    confidence = 0.8  # Base confidence for this pattern
                    
                    # Higher confidence if in an uptrend
                    if idx > 5 and data["close"][idx-5:idx-2].mean() < data["close"][idx-2]:
                        confidence += 0.1
                    
                    # Higher confidence if second candle has strong body
                    if props["is_long_body"][idx-1]:
                        confidence += 0.05
                    
                    # Higher confidence if third candle has strong body
                    if props["is_long_body"][idx]:
                        confidence += 0.05
                    
                    # Return pattern if confidence meets threshold
                    if confidence >= self.config["confidence_threshold"]:
                        return {
                            "pattern": "three_outside_down",
                            "position": idx,
                            "direction": "bearish",
                            "confidence": confidence,
                            "candles": [idx-2, idx-1, idx]
                        }
        
        return None
    
    def detect_abandoned_baby_bullish(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Abandoned Baby Bullish pattern.
        
        Similar to Morning Star but with gaps on both sides of the middle doji.
        1. First candle: Bearish with long body
        2. Second candle: Doji with gaps on both sides
        3. Third candle: Bullish with long body
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check basic pattern criteria
        if (not props["is_bullish"][idx-2] and  # First candle is bearish
            props["is_doji"][idx-1] and  # Second candle is a doji
            props["is_bullish"][idx]):  # Third candle is bullish
            
            # Check for gaps on both sides of the doji
            first_candle_high = props["high"][idx-2]
            first_candle_low = props["low"][idx-2]
            doji_high = props["high"][idx-1]
            doji_low = props["low"][idx-1]
            third_candle_high = props["high"][idx]
            third_candle_low = props["low"][idx]
            
            # There should be gaps between first and doji, and doji and third
            if (doji_high < first_candle_low and doji_high < third_candle_low):
                
                confidence = 0.9  # Base confidence for this strong pattern
                
                # Higher confidence if in a downtrend
                if idx > 5 and data["close"][idx-5:idx-2].mean() > data["close"][idx-2]:
                    confidence += 0.1
                
                # Return pattern if confidence meets threshold
                if confidence >= self.config["confidence_threshold"]:
                    return {
                        "pattern": "abandoned_baby_bullish",
                        "position": idx,
                        "direction": "bullish",
                        "confidence": confidence,
                        "candles": [idx-2, idx-1, idx]
                    }
        
        return None
    
    def detect_abandoned_baby_bearish(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Abandoned Baby Bearish pattern.
        
        Similar to Evening Star but with gaps on both sides of the middle doji.
        1. First candle: Bullish with long body
        2. Second candle: Doji with gaps on both sides
        3. Third candle: Bearish with long body
        """
        if len(data) < 3:
            return None
            
        idx = len(data) - 1  # Look at the most recent complete pattern
        props = self._calculate_candle_properties(data)
        
        # Check basic pattern criteria
        if (props["is_bullish"][idx-2] and  # First candle is bullish
            props["is_doji"][idx-1] and  # Second candle is a doji
            not props["is_bullish"][idx]):  # Third candle is bearish
            
            # Check for gaps on both sides of the doji
            first_candle_high = props["high"][idx-2]
            first_candle_low = props["low"][idx-2]
            doji_high = props["high"][idx-1]
            doji_low = props["low"][idx-1]
            third_candle_high = props["high"][idx]
            third_candle_low = props["low"][idx]
            
            # There should be gaps between first and doji, and doji and third
            if (doji_low > first_candle_high and doji_low > third_candle_high):
                
                confidence = 0.9  # Base confidence for this strong pattern
                
                # Higher confidence if in an uptrend
                if idx > 5 and data["close"][idx-5:idx-2].mean() < data["close"][idx-2]:
                    confidence += 0.1
                
                # Return pattern if confidence meets threshold
                if confidence >= self.config["confidence_threshold"]:
                    return {
                        "pattern": "abandoned_baby_bearish",
                        "position": idx,
                        "direction": "bearish",
                        "confidence": confidence,
                        "candles": [idx-2, idx-1, idx]
                    }
        
        return None
