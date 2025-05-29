"""
Mean Reversion Strategy Module

This module implements a trading strategy specialized for overextended markets
likely to return to a mean, focusing on statistical indicators and deviations.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .regime_strategies import BaseRegimeStrategy
from ..common.utils import get_logger

class MeanReversionStrategy(BaseRegimeStrategy):
    """
    Strategy optimized for overextended markets returning to a mean.
    
    This strategy focuses on:
    - Statistical deviations from moving averages
    - Overbought/oversold conditions
    - Z-score extremes
    - Price channel reversions
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            config, 
            name="MeanReversionStrategy",
            description="Strategy optimized for overextended markets returning to a mean"
        )
        
        # Mean reversion specific parameters
        self.zscore_window = config.get("zscore_window", 20)
        self.zscore_threshold = config.get("zscore_threshold", 2.0)
        self.ma_deviation_threshold = config.get("ma_deviation_threshold", 0.05)
        self.rsi_extreme_threshold = config.get("rsi_extreme_threshold", 15)
        self.recovery_strength_threshold = config.get("recovery_strength_threshold", 0.3)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals optimized for mean reversion opportunities.
        
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
                
            # Calculate indicators for this symbol
            indicators = self._calculate_common_indicators(symbol, market_data)
            if not indicators:
                continue
                
            recent_data = indicators["data"]
            
            # Calculate specialized mean reversion indicators
            mr_indicators = self._calculate_mean_reversion_indicators(recent_data)
            
            # Calculate signal components
            zscore_signal = self._calculate_zscore_signal(mr_indicators)
            ma_deviation_signal = self._calculate_ma_deviation_signal(recent_data, indicators)
            rsi_extreme_signal = self._calculate_rsi_extreme_signal(recent_data, indicators)
            recovery_signal = self._calculate_recovery_signal(recent_data, mr_indicators)
            
            # Combine signals with weights
            signal_strength = (
                0.3 * zscore_signal +
                0.3 * ma_deviation_signal +
                0.2 * rsi_extreme_signal +
                0.2 * recovery_signal
            ) * self.sensitivity
            
            # Get confirmation count
            confirmation_count = sum([
                abs(zscore_signal) > 0.3,
                abs(ma_deviation_signal) > 0.3,
                abs(rsi_extreme_signal) > 0.3,
                abs(recovery_signal) > 0.3
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
                    "zscore_component": zscore_signal,
                    "ma_deviation_component": ma_deviation_signal,
                    "rsi_extreme_component": rsi_extreme_signal,
                    "recovery_component": recovery_signal,
                    "confirmation_count": confirmation_count,
                    "zscore": mr_indicators.get("zscore", 0),
                    "ma_deviation_pct": mr_indicators.get("ma_deviation_pct", 0),
                    "mean_distance": mr_indicators.get("mean_distance", 0)
                }
            }
        
        return signals
    
    def _calculate_mean_reversion_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate specialized mean reversion indicators."""
        try:
            results = {}
            
            # Calculate Z-score (standard deviations from the mean)
            close = data['close']
            returns = data['returns'].dropna()
            
            if len(returns) >= self.zscore_window:
                rolling_mean = returns.rolling(window=self.zscore_window).mean()
                rolling_std = returns.rolling(window=self.zscore_window).std()
                zscore = (returns - rolling_mean) / rolling_std
                results["zscore"] = zscore.iloc[-1]
            else:
                results["zscore"] = 0.0
            
            # Calculate deviation from moving average
            if 'sma50' in data.columns and not data['sma50'].isnull().all():
                current_close = close.iloc[-1]
                current_sma = data['sma50'].iloc[-1]
                ma_deviation_pct = (current_close - current_sma) / current_sma
                results["ma_deviation_pct"] = ma_deviation_pct
            else:
                results["ma_deviation_pct"] = 0.0
            
            # Calculate distance from recent mean (20-day)
            if len(close) >= 20:
                recent_mean = close.iloc[-20:].mean()
                current_close = close.iloc[-1]
                mean_distance = (current_close - recent_mean) / recent_mean
                results["mean_distance"] = mean_distance
            else:
                results["mean_distance"] = 0.0
            
            # Calculate recovery strength (for bounce detection)
            if len(close) >= 5:
                min_5d = close.iloc[-5:].min()
                max_5d = close.iloc[-5:].max()
                current_close = close.iloc[-1]
                
                # Measure recovery from recent low
                if current_close > min_5d:
                    recovery_strength = (current_close - min_5d) / (max_5d - min_5d) if (max_5d - min_5d) > 0 else 0
                    results["recovery_strength"] = recovery_strength
                else:
                    results["recovery_strength"] = 0.0
            else:
                results["recovery_strength"] = 0.0
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion indicators: {str(e)}")
            return {
                "zscore": 0.0,
                "ma_deviation_pct": 0.0,
                "mean_distance": 0.0,
                "recovery_strength": 0.0
            }
    
    def _calculate_zscore_signal(self, mr_indicators: Dict[str, float]) -> float:
        """Calculate Z-score based signal component."""
        try:
            zscore = mr_indicators.get("zscore", 0)
            
            # Skip if Z-score is within normal range
            if abs(zscore) < self.zscore_threshold:
                return 0.0
            
            # Calculate signal based on Z-score extremes
            if zscore <= -self.zscore_threshold:
                # Extremely negative Z-score (oversold) - bullish signal
                signal_strength = min(1.0, abs(zscore) / self.zscore_threshold)
                return signal_strength
                
            elif zscore >= self.zscore_threshold:
                # Extremely positive Z-score (overbought) - bearish signal
                signal_strength = min(1.0, abs(zscore) / self.zscore_threshold)
                return -signal_strength
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Z-score signal: {str(e)}")
            return 0.0
    
    def _calculate_ma_deviation_signal(self, data: pd.DataFrame, indicators: Dict) -> float:
        """Calculate moving average deviation signal component."""
        try:
            ma_deviation_pct = indicators.get("ma_deviation_pct", 0.0)
            
            # Skip if deviation is within normal range
            if abs(ma_deviation_pct) < self.ma_deviation_threshold:
                return 0.0
            
            # Calculate signal based on MA deviation extremes
            if ma_deviation_pct <= -self.ma_deviation_threshold:
                # Price significantly below MA - bullish signal
                signal_strength = min(1.0, abs(ma_deviation_pct) / self.ma_deviation_threshold)
                return signal_strength
                
            elif ma_deviation_pct >= self.ma_deviation_threshold:
                # Price significantly above MA - bearish signal
                signal_strength = min(1.0, abs(ma_deviation_pct) / self.ma_deviation_threshold)
                return -signal_strength
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating MA deviation signal: {str(e)}")
            return 0.0
    
    def _calculate_rsi_extreme_signal(self, data: pd.DataFrame, indicators: Dict) -> float:
        """Calculate RSI extreme signal component."""
        try:
            rsi = indicators.get("last_rsi", 50)
            
            # Check for extreme RSI readings
            if rsi <= (30 - self.rsi_extreme_threshold):
                # Extremely oversold - bullish signal
                signal_strength = min(1.0, (30 - rsi) / self.rsi_extreme_threshold)
                return signal_strength
                
            elif rsi >= (70 + self.rsi_extreme_threshold):
                # Extremely overbought - bearish signal
                signal_strength = min(1.0, (rsi - 70) / self.rsi_extreme_threshold)
                return -signal_strength
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI extreme signal: {str(e)}")
            return 0.0
    
    def _calculate_recovery_signal(self, data: pd.DataFrame, mr_indicators: Dict[str, float]) -> float:
        """Calculate recovery/bounce signal component."""
        try:
            recovery_strength = mr_indicators.get("recovery_strength", 0.0)
            mean_distance = mr_indicators.get("mean_distance", 0.0)
            
            # Skip if recovery is not strong enough
            if recovery_strength < self.recovery_strength_threshold:
                return 0.0
            
            # Determine direction based on mean distance
            if mean_distance < 0:
                # Price below mean, bouncing back - bullish signal
                signal_strength = min(1.0, recovery_strength)
                return signal_strength
                
            elif mean_distance > 0:
                # Price above mean, falling back - bearish signal
                signal_strength = min(1.0, recovery_strength)
                return -signal_strength
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating recovery signal: {str(e)}")
            return 0.0
