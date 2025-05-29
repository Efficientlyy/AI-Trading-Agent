"""
Regime Transition Strategy Module

This module implements a trading strategy specialized for periods of market regime change,
focusing on early detection of transitions between different market states.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .regime_strategies import BaseRegimeStrategy
from ..common.utils import get_logger

class RegimeTransitionStrategy(BaseRegimeStrategy):
    """
    Strategy optimized for periods of market regime change.
    
    This strategy focuses on:
    - Early detection of regime transitions
    - Volatility regime changes
    - Trend reversal signals
    - Correlation breakdowns
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            config, 
            name="RegimeTransitionStrategy",
            description="Strategy optimized for periods of market regime change"
        )
        
        # Regime transition specific parameters
        self.volatility_change_threshold = config.get("volatility_change_threshold", 0.5)
        self.trend_reversal_threshold = config.get("trend_reversal_threshold", 0.7)
        self.correlation_breakdown_threshold = config.get("correlation_breakdown_threshold", 0.5)
        self.regime_history_window = config.get("regime_history_window", 20)
        
        # Track regime history
        self.regime_history = {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals optimized for regime transition periods.
        
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
        
        # Get market regimes for all symbols
        current_regimes = {}
        for symbol, market_data in data.items():
            if market_data is None or market_data.empty or len(market_data) < self.lookback_window:
                continue
                
            current_regimes[symbol] = self.regime_classifier.classify_regime(market_data)
            
            # Update regime history
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
                
            self.regime_history[symbol].append(current_regimes[symbol])
            
            # Limit history size
            if len(self.regime_history[symbol]) > self.regime_history_window:
                self.regime_history[symbol] = self.regime_history[symbol][-self.regime_history_window:]
        
        for symbol, market_data in data.items():
            if symbol not in current_regimes:
                continue
                
            # Calculate indicators for this symbol
            indicators = self._calculate_common_indicators(symbol, market_data)
            if not indicators:
                continue
                
            recent_data = indicators["data"]
            
            # Check if this symbol is in a regime transition
            is_transition, transition_type = self._detect_regime_transition(symbol, current_regimes[symbol])
            
            if not is_transition and self.enable_filters:
                continue
                
            # Calculate signal components
            volatility_change_signal = self._calculate_volatility_change_signal(recent_data)
            trend_reversal_signal = self._calculate_trend_reversal_signal(recent_data, indicators)
            correlation_breakdown_signal = self._calculate_correlation_breakdown_signal(
                symbol, data, current_regimes
            )
            
            # Combine signals with weights
            signal_strength = (
                0.4 * volatility_change_signal +
                0.4 * trend_reversal_signal +
                0.2 * correlation_breakdown_signal
            ) * self.sensitivity
            
            # Weight signal strength by transition confidence
            transition_confidence = self._calculate_transition_confidence(symbol)
            signal_strength *= transition_confidence
            
            # Get confirmation count
            confirmation_count = sum([
                abs(volatility_change_signal) > 0.3,
                abs(trend_reversal_signal) > 0.3,
                abs(correlation_breakdown_signal) > 0.3
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
                    "current_regime": current_regimes[symbol],
                    "transition_type": transition_type,
                    "transition_confidence": transition_confidence,
                    "volatility_change_component": volatility_change_signal,
                    "trend_reversal_component": trend_reversal_signal,
                    "correlation_breakdown_component": correlation_breakdown_signal,
                    "confirmation_count": confirmation_count
                }
            }
        
        return signals
    
    def _detect_regime_transition(self, symbol: str, current_regime: str) -> Tuple[bool, str]:
        """
        Detect if a symbol is currently in a regime transition.
        
        Args:
            symbol: The symbol to check
            current_regime: The current regime classification
            
        Returns:
            Tuple of (is_transition, transition_type)
        """
        if symbol not in self.regime_history or len(self.regime_history[symbol]) < 3:
            return False, "none"
            
        # Get the previous regime (excluding current)
        previous_regimes = self.regime_history[symbol][:-1]
        most_common_prev_regime = max(set(previous_regimes), key=previous_regimes.count)
        
        # Check if the current regime is different from the previous common regime
        if current_regime != most_common_prev_regime:
            # Identify transition type
            transition_type = f"{most_common_prev_regime}_to_{current_regime}"
            return True, transition_type
            
        # Check for potential transition by analyzing regime stability
        if len(previous_regimes) >= 5:
            # Check last 5 regimes for instability
            last_5_regimes = previous_regimes[-5:]
            unique_regimes = set(last_5_regimes)
            
            # If there are 3 or more different regimes in the last 5 periods,
            # the market is unstable and might be transitioning
            if len(unique_regimes) >= 3:
                return True, "unstable"
        
        return False, "none"
    
    def _calculate_transition_confidence(self, symbol: str) -> float:
        """Calculate confidence in the regime transition detection."""
        if symbol not in self.regime_history or len(self.regime_history[symbol]) < 5:
            return 0.5
            
        # Get regime history
        regimes = self.regime_history[symbol]
        
        # Calculate the percentage of the dominant regime in recent history
        dominant_regime = max(set(regimes), key=regimes.count)
        dominant_pct = regimes.count(dominant_regime) / len(regimes)
        
        # Check for recent regime changes
        recent_changes = 0
        for i in range(1, min(5, len(regimes))):
            if regimes[-i] != regimes[-i-1]:
                recent_changes += 1
        
        # Calculate confidence based on regime stability and recent changes
        stability_factor = 1.0 - dominant_pct  # Lower stability = higher transition likelihood
        change_factor = recent_changes / 4  # More recent changes = higher transition likelihood
        
        # Combine factors with weights
        confidence = 0.6 * stability_factor + 0.4 * change_factor
        
        return min(1.0, max(0.1, confidence))
    
    def _calculate_volatility_change_signal(self, data: pd.DataFrame) -> float:
        """Calculate volatility change signal component."""
        try:
            # Need at least 40 data points for meaningful calculation
            if len(data) < 40:
                return 0.0
                
            # Calculate volatility in recent and prior periods
            recent_volatility = data['volatility'].iloc[-20:].mean()
            prior_volatility = data['volatility'].iloc[-40:-20].mean()
            
            if prior_volatility == 0:
                return 0.0
                
            # Calculate volatility change percentage
            volatility_change_pct = (recent_volatility - prior_volatility) / prior_volatility
            
            # Skip if change is below threshold
            if abs(volatility_change_pct) < self.volatility_change_threshold:
                return 0.0
            
            # Determine signal direction
            if volatility_change_pct > 0:
                # Increasing volatility - look at price direction
                recent_returns = data['returns'].iloc[-5:].mean()
                signal_direction = np.sign(recent_returns) if recent_returns != 0 else 0
            else:
                # Decreasing volatility - typically bullish for risk assets
                signal_direction = 1
            
            # Calculate signal strength
            signal_strength = min(1.0, abs(volatility_change_pct) / self.volatility_change_threshold)
            
            return signal_direction * signal_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility change signal: {str(e)}")
            return 0.0
    
    def _calculate_trend_reversal_signal(self, data: pd.DataFrame, indicators: Dict) -> float:
        """Calculate trend reversal signal component."""
        try:
            # Check for trend reversal conditions
            # 1. Moving average crossover
            sma_alignment = indicators.get("sma_alignment", False)
            
            # 2. MACD crossover
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            macd_prev = data['macd'].iloc[-2] if len(data) > 1 else 0
            macd_signal_prev = data['macd_signal'].iloc[-2] if len(data) > 1 else 0
            
            macd_cross_up = macd_prev < macd_signal_prev and macd > macd_signal
            macd_cross_down = macd_prev > macd_signal_prev and macd < macd_signal
            
            # 3. Price reversal pattern
            close = data['close'].iloc[-1]
            high = data['high'].iloc[-5:].max()
            low = data['low'].iloc[-5:].min()
            
            close_to_high = (high - close) / high < 0.01
            close_to_low = (close - low) / low < 0.01
            
            # Determine reversal conditions
            bullish_reversal = macd_cross_up or (not sma_alignment and close_to_low)
            bearish_reversal = macd_cross_down or (sma_alignment and close_to_high)
            
            # Skip if no reversal detected
            if not (bullish_reversal or bearish_reversal):
                return 0.0
            
            # Calculate signal direction and strength
            if bullish_reversal:
                # Confirmation of bullish reversal
                rsi = data['rsi'].iloc[-1]
                strength_factor = (40 - rsi) / 30 if rsi < 40 else 0.1
                signal_strength = min(1.0, strength_factor * self.trend_reversal_threshold)
                return signal_strength
                
            elif bearish_reversal:
                # Confirmation of bearish reversal
                rsi = data['rsi'].iloc[-1]
                strength_factor = (rsi - 60) / 30 if rsi > 60 else 0.1
                signal_strength = min(1.0, strength_factor * self.trend_reversal_threshold)
                return -signal_strength
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating trend reversal signal: {str(e)}")
            return 0.0
    
    def _calculate_correlation_breakdown_signal(
        self, symbol: str, data: Dict[str, pd.DataFrame], regimes: Dict[str, str]
    ) -> float:
        """Calculate correlation breakdown signal component."""
        try:
            # Need at least 3 symbols for correlation analysis
            if len(data) < 3:
                return 0.0
                
            # Get returns for all symbols
            symbol_returns = {}
            for sym, prices in data.items():
                if prices is None or len(prices) < 40:
                    continue
                    
                # Calculate returns
                returns = prices['close'].pct_change().dropna()
                if len(returns) >= 40:
                    symbol_returns[sym] = returns
            
            # Skip if not enough data
            if len(symbol_returns) < 3 or symbol not in symbol_returns:
                return 0.0
                
            # Calculate correlation matrix for different periods
            recent_corrs = {}
            prior_corrs = {}
            
            target_returns = symbol_returns[symbol]
            
            for sym, returns in symbol_returns.items():
                if sym == symbol:
                    continue
                    
                # Ensure returns are aligned
                aligned_data = pd.DataFrame({
                    'target': target_returns,
                    'other': returns
                }).dropna()
                
                if len(aligned_data) < 40:
                    continue
                    
                # Recent correlation (last 20 periods)
                recent_corr = aligned_data.iloc[-20:]['target'].corr(aligned_data.iloc[-20:]['other'])
                
                # Prior correlation (20 periods before that)
                prior_corr = aligned_data.iloc[-40:-20]['target'].corr(aligned_data.iloc[-40:-20]['other'])
                
                if not np.isnan(recent_corr) and not np.isnan(prior_corr):
                    recent_corrs[sym] = recent_corr
                    prior_corrs[sym] = prior_corr
            
            # Skip if not enough correlation data
            if not recent_corrs:
                return 0.0
                
            # Calculate average correlation change
            correlation_changes = []
            
            for sym in recent_corrs:
                if sym in prior_corrs:
                    change = abs(recent_corrs[sym] - prior_corrs[sym])
                    correlation_changes.append(change)
            
            avg_correlation_change = np.mean(correlation_changes) if correlation_changes else 0
            
            # Skip if change is below threshold
            if avg_correlation_change < self.correlation_breakdown_threshold:
                return 0.0
            
            # For correlation breakdowns, the signal direction depends on recent returns
            recent_return = target_returns.iloc[-5:].mean() if len(target_returns) >= 5 else 0
            signal_direction = np.sign(recent_return) if recent_return != 0 else 0
            
            # Calculate signal strength
            signal_strength = min(1.0, avg_correlation_change / self.correlation_breakdown_threshold)
            
            return signal_direction * signal_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation breakdown signal: {str(e)}")
            return 0.0
