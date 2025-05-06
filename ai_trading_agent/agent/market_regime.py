"""
Market Regime Detection Strategy Module

This module implements various methods to detect and classify market regimes
such as trending, ranging, volatile, or calm periods. These classifications
can be used to dynamically adjust trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging

from .strategy import BaseStrategy, RichSignal, RichSignalsDict
from ..common import logger


class MarketRegimeType(Enum):
    """Enum representing different market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class MarketRegimeDetector:
    """
    Detects and classifies market regimes based on various indicators and metrics.
    
    This class uses multiple methods to identify the current market regime, including:
    - Volatility analysis
    - Trend strength indicators
    - Range identification
    - Breakout detection
    - Pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MarketRegimeDetector.
        
        Args:
            config: Configuration dictionary with parameters for regime detection.
                - lookback_window: Number of periods to analyze for regime detection
                - volatility_window: Window for volatility calculations
                - trend_threshold: Threshold for trend strength identification
                - range_threshold: Threshold for range identification
                - breakout_threshold: Threshold for breakout detection
                - indicators: List of indicators to use for regime detection
        """
        self.config = config
        self.lookback_window = config.get("lookback_window", 20)
        self.volatility_window = config.get("volatility_window", 10)
        self.trend_threshold = config.get("trend_threshold", 0.5)
        self.range_threshold = config.get("range_threshold", 0.3)
        self.breakout_threshold = config.get("breakout_threshold", 2.0)
        
        # Store historical regimes for reference
        self.regime_history = {}
        
        logger.info(f"MarketRegimeDetector initialized with lookback_window={self.lookback_window}")
    
    def detect_regime(self, data: pd.DataFrame, symbol: str) -> MarketRegimeType:
        """
        Detect the current market regime for a given symbol.
        
        Args:
            data: DataFrame with OHLCV data for the symbol
            symbol: The trading symbol to analyze
            
        Returns:
            MarketRegimeType indicating the detected market regime
        """
        if data is None or data.empty or len(data) < self.lookback_window:
            logger.warning(f"Insufficient data for regime detection for {symbol}")
            return MarketRegimeType.UNKNOWN
        
        # Get the most recent data for analysis
        recent_data = data.iloc[-self.lookback_window:]
        
        # Calculate key metrics
        volatility = self._calculate_volatility(recent_data)
        trend_strength = self._calculate_trend_strength(recent_data)
        is_ranging = self._is_ranging(recent_data)
        is_breakout = self._is_breakout(recent_data)
        
        # Determine regime based on metrics
        regime = self._classify_regime(
            volatility=volatility,
            trend_strength=trend_strength,
            is_ranging=is_ranging,
            is_breakout=is_breakout,
            recent_data=recent_data
        )
        
        # Store in history
        self.regime_history[symbol] = regime
        
        return regime
    
    def detect_all_regimes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, MarketRegimeType]:
        """
        Detect market regimes for multiple symbols.
        
        Args:
            data_dict: Dictionary mapping symbols to their OHLCV DataFrames
            
        Returns:
            Dictionary mapping symbols to their detected market regimes
        """
        regimes = {}
        for symbol, data in data_dict.items():
            regimes[symbol] = self.detect_regime(data, symbol)
        
        return regimes
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Calculate recent market volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Volatility measure (annualized standard deviation of returns)
        """
        # Calculate returns
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
        else:
            # Try to find a suitable price column
            price_cols = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price', 'adj'])]
            if price_cols:
                returns = data[price_cols[0]].pct_change().dropna()
            else:
                logger.warning("No suitable price column found for volatility calculation")
                return 0.0
        
        # Use the most recent window for volatility
        recent_returns = returns.iloc[-self.volatility_window:] if len(returns) > self.volatility_window else returns
        
        # Calculate annualized volatility (assuming daily data)
        volatility = recent_returns.std() * np.sqrt(252)
        return volatility
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate the strength and direction of the current trend.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Trend strength indicator (-1 to 1, where positive values indicate uptrend,
            negative values indicate downtrend, and values close to 0 indicate no trend)
        """
        if 'close' not in data.columns:
            price_cols = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price', 'adj'])]
            if not price_cols:
                logger.warning("No suitable price column found for trend calculation")
                return 0.0
            price_col = price_cols[0]
        else:
            price_col = 'close'
        
        # Simple trend calculation using linear regression slope
        prices = data[price_col].values
        x = np.arange(len(prices))
        
        # Calculate slope of the linear regression line
        if len(prices) < 2:
            return 0.0
            
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize the slope to a -1 to 1 range
        max_slope = np.std(prices) * 2  # Use 2 standard deviations as max slope
        normalized_slope = np.clip(slope / max_slope if max_slope != 0 else 0, -1, 1)
        
        return normalized_slope
    
    def _is_ranging(self, data: pd.DataFrame) -> bool:
        """
        Determine if the market is in a ranging (sideways) pattern.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Boolean indicating if the market is ranging
        """
        trend_strength = abs(self._calculate_trend_strength(data))
        return trend_strength < self.range_threshold
    
    def _is_breakout(self, data: pd.DataFrame) -> bool:
        """
        Detect if a recent breakout has occurred.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Boolean indicating if a breakout has occurred
        """
        if 'close' not in data.columns:
            price_cols = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price', 'adj'])]
            if not price_cols:
                logger.warning("No suitable price column found for breakout detection")
                return False
            price_col = price_cols[0]
        else:
            price_col = 'close'
            
        # Calculate recent price range
        lookback = min(self.lookback_window, len(data) - 1)
        if lookback < 5:  # Need at least 5 data points
            return False
            
        recent_prices = data[price_col].iloc[-lookback:-1]
        latest_price = data[price_col].iloc[-1]
        
        # Calculate the average true range as a volatility measure
        if 'high' in data.columns and 'low' in data.columns:
            high = data['high'].iloc[-lookback:-1]
            low = data['low'].iloc[-lookback:-1]
            prev_close = data[price_col].iloc[-lookback-1:-2]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.mean()
        else:
            # Approximate ATR using standard deviation
            atr = recent_prices.std() * 1.5
        
        # Check if the latest price is a breakout from the recent range
        recent_max = recent_prices.max()
        recent_min = recent_prices.min()
        
        # Breakout is detected if price moves significantly beyond the recent range
        breakout_up = latest_price > recent_max + (atr * self.breakout_threshold)
        breakout_down = latest_price < recent_min - (atr * self.breakout_threshold)
        
        return breakout_up or breakout_down
    
    def _classify_regime(self, volatility: float, trend_strength: float, 
                         is_ranging: bool, is_breakout: bool, 
                         recent_data: pd.DataFrame) -> MarketRegimeType:
        """
        Classify the market regime based on calculated metrics.
        
        Args:
            volatility: Calculated market volatility
            trend_strength: Strength and direction of the trend
            is_ranging: Whether the market is in a ranging pattern
            is_breakout: Whether a breakout has occurred
            recent_data: Recent OHLCV data
            
        Returns:
            MarketRegimeType classification
        """
        # Volatility threshold for distinguishing calm vs volatile markets
        volatility_threshold = self.config.get("volatility_threshold", 0.2)  # 20% annualized
        
        # Check for breakout first as it's a high priority signal
        if is_breakout:
            return MarketRegimeType.BREAKOUT
        
        # Check for trending markets
        if abs(trend_strength) > self.trend_threshold:
            if trend_strength > 0:
                return MarketRegimeType.TRENDING_UP
            else:
                return MarketRegimeType.TRENDING_DOWN
        
        # Check for ranging markets
        if is_ranging:
            return MarketRegimeType.RANGING
        
        # Check volatility level
        if volatility > volatility_threshold:
            return MarketRegimeType.VOLATILE
        else:
            return MarketRegimeType.CALM
        
        # Default case
        return MarketRegimeType.UNKNOWN


class MarketRegimeStrategy(BaseStrategy):
    """
    Strategy that detects market regimes and generates signals based on the current regime.
    
    This strategy can be used:
    1. As a standalone strategy that generates signals based on regime changes
    2. As a component in the IntegratedStrategyManager to inform the dynamic contextual weighting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MarketRegimeStrategy.
        
        Args:
            config: Configuration dictionary with parameters for the strategy.
                - name: Name of the strategy
                - regime_detector_config: Configuration for the MarketRegimeDetector
                - signal_mapping: How to map regimes to signals
        """
        super().__init__(config)
        self.name = config.get("name", "MarketRegimeStrategy")
        
        # Initialize the regime detector
        detector_config = config.get("regime_detector_config", {})
        self.regime_detector = MarketRegimeDetector(detector_config)
        
        # Signal mapping configuration
        self.signal_mapping = config.get("signal_mapping", {
            MarketRegimeType.TRENDING_UP.value: 0.8,    # Strong buy in uptrend
            MarketRegimeType.TRENDING_DOWN.value: -0.8, # Strong sell in downtrend
            MarketRegimeType.RANGING.value: 0.0,        # Neutral in ranging market
            MarketRegimeType.VOLATILE.value: -0.2,      # Slight sell in volatile markets (reduce risk)
            MarketRegimeType.CALM.value: 0.2,           # Slight buy in calm markets
            MarketRegimeType.BREAKOUT.value: 0.5,       # Moderate buy on breakouts
            MarketRegimeType.REVERSAL.value: -0.5,      # Moderate sell on reversals
            MarketRegimeType.UNKNOWN.value: 0.0         # Neutral when regime is unknown
        })
        
        # Confidence mapping configuration
        self.confidence_mapping = config.get("confidence_mapping", {
            MarketRegimeType.TRENDING_UP.value: 0.8,    # High confidence in uptrend
            MarketRegimeType.TRENDING_DOWN.value: 0.8,  # High confidence in downtrend
            MarketRegimeType.RANGING.value: 0.6,        # Moderate confidence in ranging market
            MarketRegimeType.VOLATILE.value: 0.5,       # Lower confidence in volatile markets
            MarketRegimeType.CALM.value: 0.7,           # Good confidence in calm markets
            MarketRegimeType.BREAKOUT.value: 0.7,       # Good confidence on breakouts
            MarketRegimeType.REVERSAL.value: 0.6,       # Moderate confidence on reversals
            MarketRegimeType.UNKNOWN.value: 0.3         # Low confidence when regime is unknown
        })
        
        logger.info(f"{self.name} initialized with {len(self.signal_mapping)} regime mappings")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> RichSignalsDict:
        """
        Generate trading signals based on detected market regimes.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            **kwargs: Additional keyword arguments
                - current_portfolio: Current portfolio state
                - timestamp: Current timestamp
                
        Returns:
            Dictionary mapping symbols to their signal dictionaries
        """
        if not data:
            logger.warning(f"{self.name}: No data provided for signal generation")
            return {}
        
        signals = {}
        timestamp = kwargs.get("timestamp", pd.Timestamp.now())
        
        # Detect regimes for all symbols
        regimes = self.regime_detector.detect_all_regimes(data)
        
        # Generate signals based on detected regimes
        for symbol, regime in regimes.items():
            regime_value = regime.value
            
            # Map regime to signal strength and confidence
            signal_strength = self.signal_mapping.get(regime_value, 0.0)
            confidence_score = self.confidence_mapping.get(regime_value, 0.5)
            
            # Create rich signal
            signals[symbol] = {
                "signal_strength": signal_strength,
                "confidence_score": confidence_score,
                "signal_type": self.name,
                "metadata": {
                    "regime": regime_value,
                    "timestamp": timestamp,
                    "lookback_window": self.regime_detector.lookback_window
                }
            }
            
        return signals
    
    def get_current_regimes(self, data: Dict[str, pd.DataFrame]) -> Dict[str, MarketRegimeType]:
        """
        Get the current market regimes for all symbols without generating signals.
        
        This method can be used by the IntegratedStrategyManager to inform
        the dynamic contextual weighting.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            
        Returns:
            Dictionary mapping symbols to their detected market regimes
        """
        return self.regime_detector.detect_all_regimes(data)
        
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the strategy's configuration parameters dynamically.
        
        Args:
            config_updates: A dictionary containing parameters to update.
        """
        # Update main config
        self.config.update(config_updates)
        
        # Update signal mapping if provided
        if "signal_mapping" in config_updates:
            self.signal_mapping.update(config_updates["signal_mapping"])
            
        # Update confidence mapping if provided
        if "confidence_mapping" in config_updates:
            self.confidence_mapping.update(config_updates["confidence_mapping"])
            
        # Update regime detector config if provided
        if "regime_detector_config" in config_updates:
            self.regime_detector.config.update(config_updates["regime_detector_config"])
            
            # Update specific detector parameters
            detector_config = config_updates["regime_detector_config"]
            if "lookback_window" in detector_config:
                self.regime_detector.lookback_window = detector_config["lookback_window"]
            if "volatility_window" in detector_config:
                self.regime_detector.volatility_window = detector_config["volatility_window"]
            if "trend_threshold" in detector_config:
                self.regime_detector.trend_threshold = detector_config["trend_threshold"]
            if "range_threshold" in detector_config:
                self.regime_detector.range_threshold = detector_config["range_threshold"]
            if "breakout_threshold" in detector_config:
                self.regime_detector.breakout_threshold = detector_config["breakout_threshold"]
                
        logger.info(f"{self.name} configuration updated")
