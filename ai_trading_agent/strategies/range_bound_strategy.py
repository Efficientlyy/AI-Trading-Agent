"""
Range Bound Market Strategy Module

This module implements a trading strategy specialized for range-bound (sideways) markets,
focusing on oscillator-based indicators and mean-reversion patterns.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .regime_strategies import BaseRegimeStrategy
from ..common.utils import get_logger

class RangeBoundStrategy(BaseRegimeStrategy):
    """
    Strategy optimized for sideways, consolidating markets.
    
    This strategy focuses on:
    - Oscillator extremes (overbought/oversold conditions)
    - Support and resistance levels
    - Mean reversion signals
    - Channel breakouts
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            config, 
            name="RangeBoundStrategy",
            description="Strategy optimized for sideways, consolidating markets"
        )
        
        # Range-bound market specific parameters
        self.range_filter_threshold = config.get("range_filter_threshold", 0.2)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.support_resistance_threshold = config.get("support_resistance_threshold", 0.02)
        self.channel_deviation = config.get("channel_deviation", 1.5)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals optimized for range-bound markets.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            **kwargs: Additional keyword arguments
                
        Returns:
            Dictionary mapping symbols to their signal dictionaries
        """
        if not data:
            self.logger.warning(f"{self.name}: No data provided for signal generation")
            return {}
        
        signals = {}
        timestamp = kwargs.get("timestamp", pd.Timestamp.now())
        
        for symbol, market_data in data.items():
            if market_data is None or market_data.empty or len(market_data) < self.lookback_window:
                continue
                
            # Get regime classification
            regime = self.regime_classifier.classify_regime(market_data)
            
            # Only generate signals for ranging regimes
            if regime != "ranging" and self.enable_filters:
                continue
                
            # Calculate indicators for this symbol
            indicators = self._calculate_common_indicators(symbol, market_data)
            if not indicators:
                continue
                
            recent_data = indicators["data"]
            last_close = indicators["last_close"]
            
            # Verify we're in a range-bound market by checking ADX
            adx_value = recent_data['adx'].iloc[-1] if 'adx' in recent_data else 0
            if adx_value > 25 and self.enable_filters:  # Strong trend detected
                continue
                
            # Calculate signal components
            oscillator_signal = self._calculate_oscillator_signal(recent_data)
            support_resistance_signal = self._calculate_support_resistance_signal(
                recent_data, indicators["pivots"], last_close
            )
            mean_reversion_signal = self._calculate_mean_reversion_signal(recent_data)
            
            # Combine signals with weights
            signal_strength = (
                0.4 * oscillator_signal +
                0.4 * support_resistance_signal +
                0.2 * mean_reversion_signal
            ) * self.sensitivity
            
            # Apply range filter
            bb_width = indicators["last_bb_width"]
            if bb_width > self.range_filter_threshold and self.enable_filters:
                # Reduce signal strength when Bollinger Band width is expanding
                signal_strength *= (self.range_filter_threshold / bb_width)
            
            # Get confirmation count
            confirmation_count = sum([
                abs(oscillator_signal) > 0.3,
                abs(support_resistance_signal) > 0.3,
                abs(mean_reversion_signal) > 0.3
            ])
            
            # Skip signals with insufficient confirmation
            if confirmation_count < self.confirmation_threshold and self.enable_filters:
                continue
            
            # Determine signal direction
            direction = "buy" if signal_strength > 0 else "sell"
            
            # Create signal with metadata
            signals[symbol] = {
                "signal": signal_strength,
                "direction": direction,
                "signal_type": self.name,
                "timestamp": timestamp,
                "metadata": {
                    "regime": regime,
                    "oscillator_component": oscillator_signal,
                    "support_resistance_component": support_resistance_signal,
                    "mean_reversion_component": mean_reversion_signal,
                    "confirmation_count": confirmation_count,
                    "bb_width": bb_width,
                    "near_support_resistance": abs(support_resistance_signal) > 0.5
                }
            }
        
        return signals
    
    def _calculate_oscillator_signal(self, data: pd.DataFrame) -> float:
        """Calculate oscillator-based signal component."""
        try:
            # Use RSI for oscillator signals
            rsi = data['rsi'].iloc[-1]
            
            # Calculate strength and direction
            if rsi < self.rsi_oversold:
                # Oversold - bullish signal
                signal_strength = (self.rsi_oversold - rsi) / self.rsi_oversold
                return min(1.0, signal_strength)
                
            elif rsi > self.rsi_overbought:
                # Overbought - bearish signal
                signal_strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                return -min(1.0, signal_strength)
                
            # Neutral zone - no signal
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating oscillator signal: {str(e)}")
            return 0.0
    
    def _calculate_support_resistance_signal(
        self, data: pd.DataFrame, pivots: Dict[str, List[float]], price: float
    ) -> float:
        """Calculate support/resistance based signal component."""
        try:
            support_levels = pivots.get("support", [])
            resistance_levels = pivots.get("resistance", [])
            
            # Find nearest support and resistance
            nearest_support = max(support_levels) if support_levels else 0
            nearest_resistance = min(resistance_levels) if resistance_levels else float('inf')
            
            for level in support_levels:
                if level < price and level > nearest_support:
                    nearest_support = level
                    
            for level in resistance_levels:
                if level > price and level < nearest_resistance:
                    nearest_resistance = level
            
            # Calculate distances to nearest levels
            if nearest_support == 0 or nearest_resistance == float('inf'):
                return 0.0
                
            distance_to_support = (price - nearest_support) / price
            distance_to_resistance = (nearest_resistance - price) / price
            
            # Check if price is near support or resistance
            if distance_to_support < self.support_resistance_threshold:
                # Near support - bullish signal
                signal_strength = 1.0 - (distance_to_support / self.support_resistance_threshold)
                return min(1.0, signal_strength)
                
            elif distance_to_resistance < self.support_resistance_threshold:
                # Near resistance - bearish signal
                signal_strength = 1.0 - (distance_to_resistance / self.support_resistance_threshold)
                return -min(1.0, signal_strength)
                
            # Not near any level - no signal
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance signal: {str(e)}")
            return 0.0
    
    def _calculate_mean_reversion_signal(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion signal component."""
        try:
            # Use Bollinger Bands for mean reversion
            close = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            
            # Price relative to Bollinger Bands
            if close < bb_lower:
                # Below lower band - bullish mean reversion signal
                deviation = (bb_lower - close) / (bb_middle * self.channel_deviation)
                return min(1.0, deviation)
                
            elif close > bb_upper:
                # Above upper band - bearish mean reversion signal
                deviation = (close - bb_upper) / (bb_middle * self.channel_deviation)
                return -min(1.0, deviation)
                
            # Within bands - no signal
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion signal: {str(e)}")
            return 0.0
