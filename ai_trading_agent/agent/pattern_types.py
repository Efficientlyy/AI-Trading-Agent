"""
Pattern Types module for the AI Trading Agent

This module defines the common types used by pattern detection modules.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any


class PatternType(Enum):
    """Enumeration of supported chart patterns"""
    SUPPORT = auto()
    RESISTANCE = auto()
    TRENDLINE_ASCENDING = auto()
    TRENDLINE_DESCENDING = auto()
    HEAD_AND_SHOULDERS = auto()
    INVERSE_HEAD_AND_SHOULDERS = auto()
    DOUBLE_TOP = auto()
    DOUBLE_BOTTOM = auto()
    TRIANGLE_ASCENDING = auto()
    TRIANGLE_DESCENDING = auto()
    TRIANGLE_SYMMETRICAL = auto()
    FLAG_BULLISH = auto()
    FLAG_BEARISH = auto()
    WEDGE_RISING = auto()
    WEDGE_FALLING = auto()
    CHANNEL_UP = auto()
    CHANNEL_DOWN = auto()
    RECTANGLE = auto()
    CUP_AND_HANDLE = auto()


class PatternDetectionResult:
    """Class representing a detected pattern"""
    
    def __init__(
        self,
        pattern_type: PatternType,
        symbol: str,
        confidence: float,
        start_time=None,
        end_time=None,
        start_idx: int = -1,
        end_idx: int = -1,
        target_price: Optional[float] = None,
        price_level: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a pattern detection result.
        
        Args:
            pattern_type: Type of pattern detected
            symbol: Trading symbol
            confidence: Confidence score (0-100)
            start_time: Start timestamp of the pattern
            end_time: End timestamp of the pattern
            start_idx: Start index in the data array (optional)
            end_idx: End index in the data array (optional)
            target_price: Target price level (for projection)
            price_level: Significant price level (e.g., breakout level)
            additional_info: Additional pattern-specific metadata
        """
        self.pattern_type = pattern_type
        self.symbol = symbol
        self.confidence = confidence
        self.start_time = start_time
        self.end_time = end_time
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.target_price = target_price
        self.price_level = price_level
        self.additional_info = additional_info or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "pattern": self.pattern_type.name,
            "pattern_type": self.pattern_type.name,  # Add this for compatibility
            "symbol": self.symbol,
            "confidence": self.confidence,
        }
        
        if self.start_time:
            result["start_time"] = self.start_time
        if self.end_time:
            result["end_time"] = self.end_time
        if self.start_idx >= 0:
            result["start_idx"] = self.start_idx
        if self.end_idx >= 0:
            result["end_idx"] = self.end_idx
        if self.target_price is not None:
            result["target_price"] = self.target_price
        if self.price_level is not None:
            result["price_level"] = self.price_level
        
        # Add all additional info
        if self.additional_info:
            for key, value in self.additional_info.items():
                result[key] = value
            
        return result