"""
Volatility Breakout Strategy Module

This module implements a trading strategy specialized for volatile markets
with potential breakouts, focusing on volatility expansion patterns.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .regime_strategies import BaseRegimeStrategy
from ..common.utils import get_logger

class VolatilityBreakoutStrategy(BaseRegimeStrategy):
    """
    Strategy optimized for volatile markets with potential breakouts.
    
    This strategy focuses on:
    - Volatility expansion patterns
    - Bollinger Band breakouts
    - High volume price movements
    - ATR-based breakout detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            config, 
            name="VolatilityBreakoutStrategy",
            description="Strategy optimized for volatile markets with breakouts"
        )
        
        # Volatility breakout specific parameters
        self.volatility_threshold = config.get("volatility_threshold", 0.02)
        self.volume_threshold = config.get("volume_threshold", 1.8)
        self.atr_multiple = config.get("atr_multiple", 1.5)
        self.bb_expansion_threshold = config.get("bb_expansion_threshold", 0.05)
        self.momentum_threshold = config.get("momentum_threshold", 0.015)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals optimized for volatile markets with breakouts.
        
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
            
            # Only generate signals for volatile or breakout regimes
            if regime not in ["volatile", "breakout"] and self.enable_filters:
                continue
                
            # Calculate indicators for this symbol
            indicators = self._calculate_common_indicators(symbol, market_data)
            if not indicators:
                continue
                
            recent_data = indicators["data"]
            
            # Calculate specialized volatility indicators
            volatility_indicators = self._calculate_volatility_indicators(recent_data)
            
            # Calculate signal components
            volatility_expansion_signal = self._calculate_volatility_expansion_signal(
                recent_data, volatility_indicators
            )
            bb_breakout_signal = self._calculate_bb_breakout_signal(recent_data, indicators)
            volume_price_signal = self._calculate_volume_price_signal(recent_data, indicators)
            
            # Combine signals with weights
            signal_strength = (
                0.4 * volatility_expansion_signal +
                0.3 * bb_breakout_signal +
                0.3 * volume_price_signal
            ) * self.sensitivity
            
            # Get confirmation count
            confirmation_count = sum([
                abs(volatility_expansion_signal) > 0.3,
                abs(bb_breakout_signal) > 0.3,
                abs(volume_price_signal) > 0.3
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
                    "volatility_expansion_component": volatility_expansion_signal,
                    "bb_breakout_component": bb_breakout_signal,
                    "volume_price_component": volume_price_signal,
                    "confirmation_count": confirmation_count,
                    "atr": volatility_indicators.get("atr", 0),
                    "volatility_percentile": volatility_indicators.get("volatility_percentile", 0),
                    "bb_width_change": volatility_indicators.get("bb_width_change", 0)
                }
            }
        
        return signals
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-specific indicators."""
        try:
            # ATR calculation
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Volatility percentile
            volatility_series = data['volatility'].dropna()
            if len(volatility_series) > 0:
                current_volatility = volatility_series.iloc[-1]
                volatility_percentile = sum(volatility_series < current_volatility) / len(volatility_series)
            else:
                volatility_percentile = 0.5
            
            # Bollinger Band width change
            bb_width = data['bb_width'].dropna()
            if len(bb_width) > 5:
                current_width = bb_width.iloc[-1]
                previous_width = bb_width.iloc[-5]
                bb_width_change = (current_width - previous_width) / previous_width
            else:
                bb_width_change = 0
            
            return {
                "atr": atr,
                "volatility_percentile": volatility_percentile,
                "bb_width_change": bb_width_change
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {str(e)}")
            return {"atr": 0, "volatility_percentile": 0, "bb_width_change": 0}
    
    def _calculate_volatility_expansion_signal(
        self, data: pd.DataFrame, volatility_indicators: Dict[str, float]
    ) -> float:
        """Calculate volatility expansion signal component."""
        try:
            # Get volatility metrics
            bb_width_change = volatility_indicators["bb_width_change"]
            volatility_percentile = volatility_indicators["volatility_percentile"]
            
            # Determine if volatility is expanding
            volatility_expanding = bb_width_change > self.bb_expansion_threshold
            
            if not volatility_expanding:
                return 0.0
            
            # Calculate price momentum
            close = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-5] if len(data) >= 5 else data['close'].iloc[0]
            price_change = (close - prev_close) / prev_close
            
            # Only generate signal if price change exceeds threshold
            if abs(price_change) < self.momentum_threshold:
                return 0.0
            
            # Signal direction follows price momentum
            signal_direction = np.sign(price_change)
            
            # Signal strength based on volatility percentile and BB width change
            signal_strength = (
                volatility_percentile * 0.5 + 
                min(1.0, bb_width_change / self.bb_expansion_threshold) * 0.5
            )
            
            return signal_direction * signal_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility expansion signal: {str(e)}")
            return 0.0
    
    def _calculate_bb_breakout_signal(self, data: pd.DataFrame, indicators: Dict) -> float:
        """Calculate Bollinger Band breakout signal component."""
        try:
            # Get Bollinger Band metrics
            close = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            
            # Check for price breakouts beyond Bollinger Bands
            upper_breakout = close > bb_upper
            lower_breakout = close < bb_lower
            
            if not (upper_breakout or lower_breakout):
                return 0.0
            
            # Check for volume confirmation
            high_volume = indicators.get("high_volume", False)
            
            # Signal direction
            signal_direction = 1 if upper_breakout else -1
            
            # Signal strength based on breakout magnitude
            if upper_breakout:
                breakout_pct = (close - bb_upper) / bb_upper
            else:
                breakout_pct = (bb_lower - close) / bb_lower
            
            signal_strength = min(1.0, breakout_pct * 20)  # Scale to [0, 1]
            
            # Adjust strength based on volume
            if high_volume:
                signal_strength *= 1.5
                signal_strength = min(1.0, signal_strength)
            
            return signal_direction * signal_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating BB breakout signal: {str(e)}")
            return 0.0
    
    def _calculate_volume_price_signal(self, data: pd.DataFrame, indicators: Dict) -> float:
        """Calculate volume-price action signal component."""
        try:
            # Get volume metrics
            volume_ratio = data['volume'].iloc[-1] / data['volume_sma20'].iloc[-1]
            
            # Only consider high volume situations
            if volume_ratio < self.volume_threshold:
                return 0.0
            
            # Get price change
            close = data['close'].iloc[-1]
            open_price = data['open'].iloc[-1]
            price_change = (close - open_price) / open_price
            
            # Only generate signal if price change exceeds threshold
            if abs(price_change) < self.volatility_threshold:
                return 0.0
            
            # Signal direction follows price change
            signal_direction = np.sign(price_change)
            
            # Signal strength based on volume ratio and price change
            signal_strength = (
                min(1.0, (volume_ratio - 1) / (self.volume_threshold - 1)) * 0.6 +
                min(1.0, abs(price_change) / self.volatility_threshold) * 0.4
            )
            
            return signal_direction * signal_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating volume-price signal: {str(e)}")
            return 0.0
