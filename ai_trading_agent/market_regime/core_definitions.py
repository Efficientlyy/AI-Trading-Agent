"""
Market Regime Core Definitions

This module contains the core definitions and enums for market regime classification,
providing a shared vocabulary for different regime detection modules.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np


class MarketRegimeType(Enum):
    """Enum representing different types of market regimes."""
    UNKNOWN = "unknown"
    BULL = "bull"              # Trending upward with strong momentum
    BEAR = "bear"              # Trending downward with strong momentum
    SIDEWAYS = "sideways"      # Range-bound with low momentum
    VOLATILE = "volatile"      # High volatility regardless of direction
    CALM = "calm"              # Low volatility regardless of direction
    RECOVERY = "recovery"      # Bouncing back after a significant decline
    BREAKDOWN = "breakdown"    # Breaking down after a significant rise
    CHOPPY = "choppy"          # Volatile with no clear trend
    TRENDING = "trending"      # Strong trend regardless of direction


class VolatilityRegimeType(Enum):
    """Enum representing different volatility regimes."""
    UNKNOWN = "unknown"
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"
    CRISIS = "crisis"


class LiquidityRegimeType(Enum):
    """Enum representing different liquidity regimes."""
    UNKNOWN = "unknown"
    ABUNDANT = "abundant"
    NORMAL = "normal"
    REDUCED = "reduced"
    STRESSED = "stressed"
    CRISIS = "crisis"


class CorrelationRegimeType(Enum):
    """Enum representing different correlation regimes."""
    UNKNOWN = "unknown"
    DISPERSED = "dispersed"      # Low correlation across markets
    CONVERGENT = "convergent"    # High correlation across markets
    RISK_ON = "risk_on"          # Risk assets moving together
    RISK_OFF = "risk_off"        # Flight to safety
    SECTOR_ROTATION = "sector_rotation"  # Rotation between sectors


class RegimeChangeSignificance(Enum):
    """Enum representing the significance of regime change."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    MAJOR = "major"


class RegimeDetectionMethod(Enum):
    """Enum representing different methods for regime detection."""
    STATISTICAL = "statistical"
    INDICATOR_BASED = "indicator_based"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    MOMENTUM_FACTOR = "momentum_factor"
    CORRELATION_BASED = "correlation_based"
    LIQUIDITY_BASED = "liquidity_based"
    COMBINED = "combined"
    MACHINE_LEARNING = "machine_learning"


class MarketRegimeConfig:
    """Base configuration class for market regime detection."""
    def __init__(self, 
                 lookback_period: int = 60, 
                 short_lookback: int = 20,
                 medium_lookback: int = 60, 
                 long_lookback: int = 120,
                 volatility_window: int = 20,
                 regime_change_threshold: float = 0.15,
                 volatility_threshold: Dict[str, float] = None,
                 correlation_threshold: float = 0.6):
        """
        Initialize market regime configuration.
        
        Args:
            lookback_period: General lookback period for calculations
            short_lookback: Short-term lookback period for calculations
            medium_lookback: Medium-term lookback period for calculations
            long_lookback: Long-term lookback period for calculations
            volatility_window: Window for volatility calculations
            regime_change_threshold: Threshold for regime change detection
            volatility_threshold: Thresholds for different volatility regimes
            correlation_threshold: Threshold for correlation significance
        """
        self.lookback_period = lookback_period
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.long_lookback = long_lookback
        self.volatility_window = volatility_window
        self.regime_change_threshold = regime_change_threshold
        self.correlation_threshold = correlation_threshold
        
        # Default volatility thresholds if none provided
        self.volatility_threshold = volatility_threshold or {
            "very_low": 0.005,
            "low": 0.01,
            "moderate": 0.015,
            "high": 0.025,
            "very_high": 0.035,
            "extreme": 0.05,
            "crisis": 0.08
        }


class MarketRegimeInfo:
    """Class to store market regime information."""
    
    def __init__(self,
                regime_type: MarketRegimeType = MarketRegimeType.UNKNOWN,
                volatility_regime: VolatilityRegimeType = VolatilityRegimeType.UNKNOWN,
                liquidity_regime: LiquidityRegimeType = LiquidityRegimeType.UNKNOWN,
                correlation_regime: CorrelationRegimeType = CorrelationRegimeType.UNKNOWN,
                confidence: float = 0.0,
                detection_method: RegimeDetectionMethod = RegimeDetectionMethod.STATISTICAL,
                regime_change: Optional[RegimeChangeSignificance] = None,
                metrics: Dict[str, float] = None,
                timestamp: Optional[pd.Timestamp] = None):
        """
        Initialize market regime information.
        
        Args:
            regime_type: Type of market regime
            volatility_regime: Volatility regime classification
            liquidity_regime: Liquidity regime classification
            correlation_regime: Correlation regime classification
            confidence: Confidence level in the regime classification (0.0-1.0)
            detection_method: Method used for detection
            regime_change: Significance of regime change if detected
            metrics: Dictionary of supporting metrics
            timestamp: Timestamp of the regime assessment
        """
        self.regime_type = regime_type
        self.volatility_regime = volatility_regime
        self.liquidity_regime = liquidity_regime
        self.correlation_regime = correlation_regime
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        self.detection_method = detection_method
        self.regime_change = regime_change or RegimeChangeSignificance.NONE
        self.metrics = metrics or {}
        self.timestamp = timestamp or pd.Timestamp.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the regime information to a dictionary."""
        return {
            "regime_type": self.regime_type.value,
            "volatility_regime": self.volatility_regime.value,
            "liquidity_regime": self.liquidity_regime.value,
            "correlation_regime": self.correlation_regime.value,
            "confidence": self.confidence,
            "detection_method": self.detection_method.value,
            "regime_change": self.regime_change.value,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketRegimeInfo':
        """
        Create a MarketRegimeInfo instance from a dictionary.
        
        Args:
            data: Dictionary with regime information
            
        Returns:
            MarketRegimeInfo instance
        """
        return cls(
            regime_type=MarketRegimeType(data.get("regime_type", "unknown")),
            volatility_regime=VolatilityRegimeType(data.get("volatility_regime", "unknown")),
            liquidity_regime=LiquidityRegimeType(data.get("liquidity_regime", "unknown")),
            correlation_regime=CorrelationRegimeType(data.get("correlation_regime", "unknown")),
            confidence=data.get("confidence", 0.0),
            detection_method=RegimeDetectionMethod(data.get("detection_method", "statistical")),
            regime_change=RegimeChangeSignificance(data.get("regime_change", "none")),
            metrics=data.get("metrics", {}),
            timestamp=pd.Timestamp(data.get("timestamp")) if data.get("timestamp") else None
        )
