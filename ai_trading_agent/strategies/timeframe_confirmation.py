"""
Advanced Multi-Timeframe Confirmation Module

This module provides sophisticated rules for confirming trading signals
across multiple timeframes with advanced filtering and validation logic.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from ..common.utils import get_logger

class TimeframeConfirmation:
    """
    Advanced rules for confirming signals across multiple timeframes.
    
    Features:
    - Hierarchical confirmation (higher timeframes confirm lower)
    - Momentum cascade detection
    - Divergence identification
    - Confluence scoring
    - Timeframe alignment algorithms
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the timeframe confirmation system.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.logger = get_logger("TimeframeConfirmation")
        self.config = config or {}
        
        # Extract configuration
        self.min_confirmation_count = self.config.get("min_confirmation_count", 2)
        self.timeframe_weights = self.config.get("timeframe_weights", {
            "1m": 0.1,
            "5m": 0.2,
            "15m": 0.3,
            "30m": 0.4,
            "1h": 0.6,
            "4h": 0.8,
            "1d": 1.0,
            "1w": 1.2,
        })
        self.default_weight = self.config.get("default_weight", 0.5)
        self.confirmation_threshold = self.config.get("confirmation_threshold", 0.7)
        self.divergence_threshold = self.config.get("divergence_threshold", 0.3)
        self.enable_momentum_cascade = self.config.get("enable_momentum_cascade", True)
        self.enable_divergence_detection = self.config.get("enable_divergence_detection", True)
        
        # Define timeframe hierarchy
        self.timeframe_hierarchy = self.config.get("timeframe_hierarchy", [
            "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"
        ])
        
        # Metrics tracking
        self.metrics = {
            "confirmations_processed": 0,
            "signals_confirmed": 0,
            "signals_rejected": 0,
            "avg_confirmation_score": 0.0,
            "divergences_detected": 0,
            "momentum_cascades_detected": 0,
        }
        
    def confirm_signal(self, signal: Dict[str, Any], 
                     market_data: Dict[str, Dict[str, pd.DataFrame]],
                     indicators: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Apply multi-timeframe confirmation rules to a trading signal.
        
        Args:
            signal: Trading signal dictionary
            market_data: Dictionary of market data by symbol and timeframe
            indicators: Dictionary of indicators by symbol, timeframe, and name
            
        Returns:
            Dictionary with confirmation results
        """
        self.metrics["confirmations_processed"] += 1
        
        # Extract signal information
        symbol = signal.get("symbol")
        direction = signal.get("direction")  # "buy" or "sell"
        source_timeframe = signal.get("timeframe")
        
        # Skip if we don't have the necessary data
        if not symbol or not direction or symbol not in market_data:
            return {
                "confirmed": False,
                "score": 0.0,
                "reason": "Missing signal information or market data",
                "confirmations": [],
                "divergences": [],
                "momentum_cascade": False,
            }
            
        # Get all available timeframes for this symbol
        available_timeframes = list(market_data[symbol].keys())
        
        # Sort timeframes by hierarchy
        sorted_timeframes = self._sort_timeframes_by_hierarchy(available_timeframes)
        
        # Apply hierarchical confirmation
        confirmation_results = self._apply_hierarchical_confirmation(
            symbol, direction, source_timeframe, sorted_timeframes, 
            market_data, indicators
        )
        
        # Check for divergences if enabled
        divergences = []
        if self.enable_divergence_detection:
            divergences = self._detect_divergences(
                symbol, direction, sorted_timeframes, 
                market_data, indicators
            )
            
        # Check for momentum cascade if enabled
        momentum_cascade = False
        if self.enable_momentum_cascade:
            momentum_cascade = self._detect_momentum_cascade(
                symbol, direction, sorted_timeframes, 
                market_data, indicators
            )
            
        # Calculate overall confirmation score
        confirmation_score = self._calculate_confirmation_score(
            confirmation_results, divergences, momentum_cascade
        )
        
        # Determine if signal is confirmed
        is_confirmed = confirmation_score >= self.confirmation_threshold
        
        # Update metrics
        if is_confirmed:
            self.metrics["signals_confirmed"] += 1
        else:
            self.metrics["signals_rejected"] += 1
            
        # Update average confirmation score
        n = self.metrics["confirmations_processed"]
        self.metrics["avg_confirmation_score"] = (
            (self.metrics["avg_confirmation_score"] * (n - 1) + confirmation_score) / n
        )
        
        # Add divergence and momentum cascade metrics
        if divergences:
            self.metrics["divergences_detected"] += 1
            
        if momentum_cascade:
            self.metrics["momentum_cascades_detected"] += 1
            
        # Prepare result
        confirmation_result = {
            "confirmed": is_confirmed,
            "score": confirmation_score,
            "reason": self._get_confirmation_reason(is_confirmed, confirmation_score, confirmation_results),
            "confirmations": confirmation_results,
            "divergences": divergences,
            "momentum_cascade": momentum_cascade,
        }
        
        return confirmation_result
        
    def _sort_timeframes_by_hierarchy(self, timeframes: List[str]) -> List[str]:
        """
        Sort timeframes according to the hierarchy (lowest to highest).
        
        Args:
            timeframes: List of timeframe strings
            
        Returns:
            Sorted list of timeframes
        """
        # Create a mapping of timeframe to hierarchy position
        hierarchy_map = {tf: i for i, tf in enumerate(self.timeframe_hierarchy)}
        
        # Sort timeframes by hierarchy position
        return sorted(timeframes, key=lambda tf: hierarchy_map.get(tf, 999))
        
    def _apply_hierarchical_confirmation(self, symbol: str, direction: str,
                                      source_timeframe: str, timeframes: List[str],
                                      market_data: Dict[str, Dict[str, pd.DataFrame]],
                                      indicators: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Apply hierarchical confirmation rules across timeframes.
        
        Args:
            symbol: Trading symbol
            direction: Signal direction ("buy" or "sell")
            source_timeframe: Original timeframe of the signal
            timeframes: Sorted list of available timeframes
            market_data: Market data dictionary
            indicators: Indicators dictionary
            
        Returns:
            List of confirmation results by timeframe
        """
        confirmation_results = []
        
        # Determine if this is a bullish or bearish signal
        is_bullish = direction.lower() in ["buy", "long"]
        
        # Get index of source timeframe
        try:
            source_index = timeframes.index(source_timeframe)
        except ValueError:
            source_index = -1
            
        # Check each timeframe
        for i, timeframe in enumerate(timeframes):
            # Skip the source timeframe
            if timeframe == source_timeframe:
                continue
                
            # Get timeframe weight
            weight = self.timeframe_weights.get(timeframe, self.default_weight)
            
            # Higher timeframes are more important for confirmation
            is_higher_timeframe = i > source_index
            
            # Get market data and indicators for this timeframe
            tf_market_data = market_data[symbol].get(timeframe)
            tf_indicators = indicators[symbol].get(timeframe, {})
            
            if tf_market_data is None or tf_market_data.empty:
                continue
                
            # Apply confirmation rules
            confirms_trend = self._confirm_trend_by_indicators(
                tf_market_data, tf_indicators, is_bullish
            )
            
            # Add to results
            confirmation_results.append({
                "timeframe": timeframe,
                "confirms": confirms_trend,
                "weight": weight,
                "is_higher_timeframe": is_higher_timeframe,
            })
            
        return confirmation_results
        
    def _confirm_trend_by_indicators(self, market_data: pd.DataFrame,
                                   indicators: Dict[str, Any],
                                   is_bullish: bool) -> bool:
        """
        Check if indicators in a timeframe confirm the trend direction.
        
        Args:
            market_data: Market data DataFrame
            indicators: Dictionary of indicators
            is_bullish: Whether the signal is bullish
            
        Returns:
            Boolean indicating confirmation
        """
        # Check if the DataFrame is empty
        if market_data.empty:
            return False
            
        # Initialize counters
        confirming_indicators = 0
        total_indicators = 0
        
        # Check trend with moving averages if available
        if "sma" in indicators:
            total_indicators += 1
            sma_values = indicators["sma"]
            
            # Get the latest close price
            latest_close = market_data["close"].iloc[-1]
            
            # Check if price is above SMA for bullish, below for bearish
            if is_bullish and latest_close > sma_values[-1]:
                confirming_indicators += 1
            elif not is_bullish and latest_close < sma_values[-1]:
                confirming_indicators += 1
                
        # Check MACD if available
        if "macd" in indicators:
            total_indicators += 1
            macd_values = indicators["macd"]
            
            # Check MACD line vs signal line
            if "macd_line" in macd_values and "signal_line" in macd_values:
                macd_line = macd_values["macd_line"][-1]
                signal_line = macd_values["signal_line"][-1]
                
                # Bullish: MACD above signal, Bearish: MACD below signal
                if is_bullish and macd_line > signal_line:
                    confirming_indicators += 1
                elif not is_bullish and macd_line < signal_line:
                    confirming_indicators += 1
                    
        # Check RSI if available
        if "rsi" in indicators:
            total_indicators += 1
            rsi_values = indicators["rsi"]
            latest_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50
            
            # Bullish: RSI above 50, Bearish: RSI below 50
            if is_bullish and latest_rsi > 50:
                confirming_indicators += 1
            elif not is_bullish and latest_rsi < 50:
                confirming_indicators += 1
                
        # Check ADX for trend strength if available
        if "adx" in indicators:
            total_indicators += 1
            adx_values = indicators["adx"]
            latest_adx = adx_values[-1] if len(adx_values) > 0 else 0
            
            # ADX above 25 indicates a strong trend
            if latest_adx > 25:
                confirming_indicators += 1
                
        # If we have no indicators to check, return False
        if total_indicators == 0:
            return False
            
        # Return True if majority of indicators confirm the trend
        return confirming_indicators / total_indicators >= 0.5
        
    def _detect_divergences(self, symbol: str, direction: str,
                         timeframes: List[str],
                         market_data: Dict[str, Dict[str, pd.DataFrame]],
                         indicators: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Detect divergences between price action and indicators across timeframes.
        
        Args:
            symbol: Trading symbol
            direction: Signal direction
            timeframes: List of timeframes
            market_data: Market data dictionary
            indicators: Indicators dictionary
            
        Returns:
            List of divergence dictionaries
        """
        if not self.enable_divergence_detection:
            return []
            
        divergences = []
        is_bullish = direction.lower() in ["buy", "long"]
        
        # For each timeframe, check for price/indicator divergences
        for timeframe in timeframes:
            tf_market_data = market_data[symbol].get(timeframe)
            tf_indicators = indicators[symbol].get(timeframe, {})
            
            if tf_market_data is None or tf_market_data.empty:
                continue
                
            # Check for RSI divergence
            rsi_divergence = self._check_rsi_divergence(
                tf_market_data, tf_indicators, is_bullish
            )
            
            if rsi_divergence:
                divergences.append({
                    "type": "rsi",
                    "timeframe": timeframe,
                    "bullish_divergence": rsi_divergence == "bullish",
                    "bearish_divergence": rsi_divergence == "bearish"
                })
                
            # Check for MACD divergence
            macd_divergence = self._check_macd_divergence(
                tf_market_data, tf_indicators, is_bullish
            )
            
            if macd_divergence:
                divergences.append({
                    "type": "macd",
                    "timeframe": timeframe,
                    "bullish_divergence": macd_divergence == "bullish",
                    "bearish_divergence": macd_divergence == "bearish"
                })
                
        return divergences
        
    def _check_rsi_divergence(self, market_data: pd.DataFrame,
                           indicators: Dict[str, Any],
                           is_bullish: bool) -> Optional[str]:
        """
        Check for RSI divergence.
        
        Args:
            market_data: Market data DataFrame
            indicators: Indicators dictionary
            is_bullish: Whether the signal is bullish
            
        Returns:
            String indicating divergence type or None
        """
        if "rsi" not in indicators or len(market_data) < 10:
            return None
            
        rsi_values = indicators["rsi"]
        if len(rsi_values) < 10:
            return None
            
        # Get the last 10 periods
        close_prices = market_data["close"].iloc[-10:].values
        rsi = rsi_values[-10:]
        
        # Find local extremes
        price_highs = []
        price_lows = []
        rsi_highs = []
        rsi_lows = []
        
        for i in range(1, len(close_prices) - 1):
            # Price highs and lows
            if close_prices[i] > close_prices[i-1] and close_prices[i] > close_prices[i+1]:
                price_highs.append((i, close_prices[i]))
            if close_prices[i] < close_prices[i-1] and close_prices[i] < close_prices[i+1]:
                price_lows.append((i, close_prices[i]))
                
            # RSI highs and lows
            if rsi[i] > rsi[i-1] and rsi[i] > rsi[i+1]:
                rsi_highs.append((i, rsi[i]))
            if rsi[i] < rsi[i-1] and rsi[i] < rsi[i+1]:
                rsi_lows.append((i, rsi[i]))
                
        # Need at least 2 highs or 2 lows to detect divergence
        if len(price_highs) < 2 or len(rsi_highs) < 2:
            if len(price_lows) < 2 or len(rsi_lows) < 2:
                return None
                
        # Check for bearish divergence: price makes higher highs, RSI makes lower highs
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
                return "bearish"
                
        # Check for bullish divergence: price makes lower lows, RSI makes higher lows
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
                return "bullish"
                
        return None
        
    def _check_macd_divergence(self, market_data: pd.DataFrame,
                            indicators: Dict[str, Any],
                            is_bullish: bool) -> Optional[str]:
        """
        Check for MACD divergence.
        
        Args:
            market_data: Market data DataFrame
            indicators: Indicators dictionary
            is_bullish: Whether the signal is bullish
            
        Returns:
            String indicating divergence type or None
        """
        if "macd" not in indicators or len(market_data) < 15:
            return None
            
        macd_values = indicators["macd"]
        if "macd_line" not in macd_values or "signal_line" not in macd_values:
            return None
            
        macd_line = macd_values["macd_line"]
        if len(macd_line) < 15:
            return None
            
        # Get the last 15 periods
        close_prices = market_data["close"].iloc[-15:].values
        macd = macd_line[-15:]
        
        # Find local extremes
        price_highs = []
        price_lows = []
        macd_highs = []
        macd_lows = []
        
        for i in range(1, len(close_prices) - 1):
            # Price highs and lows
            if close_prices[i] > close_prices[i-1] and close_prices[i] > close_prices[i+1]:
                price_highs.append((i, close_prices[i]))
            if close_prices[i] < close_prices[i-1] and close_prices[i] < close_prices[i+1]:
                price_lows.append((i, close_prices[i]))
                
            # MACD highs and lows
            if macd[i] > macd[i-1] and macd[i] > macd[i+1]:
                macd_highs.append((i, macd[i]))
            if macd[i] < macd[i-1] and macd[i] < macd[i+1]:
                macd_lows.append((i, macd[i]))
                
        # Need at least 2 highs or 2 lows to detect divergence
        if len(price_highs) < 2 or len(macd_highs) < 2:
            if len(price_lows) < 2 or len(macd_lows) < 2:
                return None
                
        # Check for bearish divergence: price makes higher highs, MACD makes lower highs
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and macd_highs[-1][1] < macd_highs[-2][1]:
                return "bearish"
                
        # Check for bullish divergence: price makes lower lows, MACD makes higher lows
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and macd_lows[-1][1] > macd_lows[-2][1]:
                return "bullish"
                
        return None
        
    def _detect_momentum_cascade(self, symbol: str, direction: str,
                              timeframes: List[str],
                              market_data: Dict[str, Dict[str, pd.DataFrame]],
                              indicators: Dict[str, Dict[str, Dict[str, Any]]]) -> bool:
        """
        Detect momentum cascading from higher timeframes to lower timeframes.
        
        Args:
            symbol: Trading symbol
            direction: Signal direction
            timeframes: List of timeframes
            market_data: Market data dictionary
            indicators: Indicators dictionary
            
        Returns:
            Boolean indicating if a momentum cascade is detected
        """
        if not self.enable_momentum_cascade or len(timeframes) < 2:
            return False
            
        # Check timeframes from highest to lowest
        reversed_timeframes = list(reversed(timeframes))
        is_bullish = direction.lower() in ["buy", "long"]
        
        # Track momentum in each timeframe
        momentum_scores = {}
        
        for timeframe in reversed_timeframes:
            tf_market_data = market_data[symbol].get(timeframe)
            tf_indicators = indicators[symbol].get(timeframe, {})
            
            if tf_market_data is None or tf_market_data.empty:
                continue
                
            # Calculate momentum score for this timeframe
            momentum_score = self._calculate_momentum_score(
                tf_market_data, tf_indicators, is_bullish
            )
            
            momentum_scores[timeframe] = momentum_score
            
        # Check if momentum cascades from higher to lower timeframes
        # (momentum should be stronger in higher timeframes and gradually decrease)
        cascade_detected = True
        prev_score = None
        prev_timeframe = None
        
        for timeframe in reversed_timeframes:
            if timeframe not in momentum_scores:
                continue
                
            current_score = momentum_scores[timeframe]
            
            if prev_score is not None:
                # For a cascade, higher timeframes should have stronger momentum
                if current_score > prev_score:
                    cascade_detected = False
                    break
                    
            prev_score = current_score
            prev_timeframe = timeframe
            
        return cascade_detected and len(momentum_scores) >= 2
        
    def _calculate_momentum_score(self, market_data: pd.DataFrame,
                               indicators: Dict[str, Any],
                               is_bullish: bool) -> float:
        """
        Calculate a momentum score for a timeframe.
        
        Args:
            market_data: Market data DataFrame
            indicators: Indicators dictionary
            is_bullish: Whether the signal is bullish
            
        Returns:
            Momentum score (0.0 to 1.0)
        """
        momentum_indicators = 0
        total_indicators = 0
        
        # Check trend with moving averages if available
        if "sma" in indicators:
            total_indicators += 1
            sma_values = indicators["sma"]
            
            # Get the latest close prices
            if len(market_data) >= 3:
                close_prices = market_data["close"].iloc[-3:].values
                
                # Calculate price slope
                price_slope = (close_prices[-1] - close_prices[0]) / 2
                
                # Check if price slope matches direction
                if (is_bullish and price_slope > 0) or (not is_bullish and price_slope < 0):
                    momentum_indicators += 1
                    
        # Check MACD if available
        if "macd" in indicators:
            total_indicators += 1
            macd_values = indicators["macd"]
            
            # Check MACD histogram
            if "histogram" in macd_values and len(macd_values["histogram"]) >= 3:
                histogram = macd_values["histogram"][-3:]
                
                # Calculate histogram slope
                hist_slope = (histogram[-1] - histogram[0]) / 2
                
                # Check if histogram slope matches direction
                if (is_bullish and hist_slope > 0) or (not is_bullish and hist_slope < 0):
                    momentum_indicators += 1
                    
        # Check RSI if available
        if "rsi" in indicators:
            total_indicators += 1
            rsi_values = indicators["rsi"]
            
            if len(rsi_values) >= 3:
                rsi_slope = (rsi_values[-1] - rsi_values[-3]) / 2
                
                # Check if RSI slope matches direction
                if (is_bullish and rsi_slope > 0) or (not is_bullish and rsi_slope < 0):
                    momentum_indicators += 1
                    
        # Check ADX for trend strength if available
        if "adx" in indicators:
            total_indicators += 1
            adx_values = indicators["adx"]
            
            if len(adx_values) >= 3:
                # Rising ADX indicates increasing trend strength
                adx_slope = (adx_values[-1] - adx_values[-3]) / 2
                
                if adx_slope > 0:
                    momentum_indicators += 1
                    
        # If we have no indicators to check, return 0
        if total_indicators == 0:
            return 0.0
            
        # Return momentum score (0.0 to 1.0)
        return momentum_indicators / total_indicators
        
    def _calculate_confirmation_score(self, confirmations: List[Dict[str, Any]],
                                    divergences: List[Dict[str, Any]],
                                    momentum_cascade: bool) -> float:
        """
        Calculate an overall confirmation score.
        
        Args:
            confirmations: List of confirmation results
            divergences: List of divergence results
            momentum_cascade: Whether momentum cascade is detected
            
        Returns:
            Confirmation score (0.0 to 1.0)
        """
        # Start with base score
        score = 0.0
        total_weight = 0.0
        
        # Add confirmation scores
        for confirmation in confirmations:
            weight = confirmation.get("weight", self.default_weight)
            
            # Higher timeframes get a bonus
            if confirmation.get("is_higher_timeframe", False):
                weight *= 1.5
                
            if confirmation.get("confirms", False):
                score += weight
                
            total_weight += weight
            
        # Calculate base score
        if total_weight > 0:
            score = score / total_weight
            
        # Apply divergence penalty
        bullish_divergences = [d for d in divergences if d.get("bullish_divergence", False)]
        bearish_divergences = [d for d in divergences if d.get("bearish_divergence", False)]
        
        # Bearish divergences reduce bullish score, bullish divergences reduce bearish score
        if bullish_divergences and score < 0:  # Bearish signal with bullish divergence
            penalty = len(bullish_divergences) * self.divergence_threshold
            score = max(-1.0, score + penalty)  # Reduce bearish conviction
            
        elif bearish_divergences and score > 0:  # Bullish signal with bearish divergence
            penalty = len(bearish_divergences) * self.divergence_threshold
            score = min(1.0, score - penalty)  # Reduce bullish conviction
            
        # Apply momentum cascade bonus
        if momentum_cascade:
            if score > 0:  # Bullish signal with bullish cascade
                score = min(1.0, score + 0.2)  # Increase bullish conviction
            elif score < 0:  # Bearish signal with bearish cascade
                score = max(-1.0, score - 0.2)  # Increase bearish conviction
                
        # Convert to 0-1 range
        return (score + 1) / 2
        
    def _get_confirmation_reason(self, is_confirmed: bool, score: float, 
                              confirmations: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable reason for the confirmation result.
        
        Args:
            is_confirmed: Whether the signal is confirmed
            score: Confirmation score
            confirmations: List of confirmation results
            
        Returns:
            Reason string
        """
        if is_confirmed:
            # Count confirmations by timeframe type
            higher_tf_confirms = sum(1 for c in confirmations if c.get("is_higher_timeframe", False) 
                                   and c.get("confirms", False))
            lower_tf_confirms = sum(1 for c in confirmations if not c.get("is_higher_timeframe", False) 
                                  and c.get("confirms", False))
            
            if higher_tf_confirms > 0 and lower_tf_confirms > 0:
                return f"Confirmed by {higher_tf_confirms} higher and {lower_tf_confirms} lower timeframes (score: {score:.2f})"
            elif higher_tf_confirms > 0:
                return f"Confirmed by {higher_tf_confirms} higher timeframes (score: {score:.2f})"
            elif lower_tf_confirms > 0:
                return f"Confirmed by {lower_tf_confirms} lower timeframes (score: {score:.2f})"
            else:
                return f"Confirmed with score {score:.2f}"
        else:
            # Count rejections
            higher_tf_rejects = sum(1 for c in confirmations if c.get("is_higher_timeframe", False) 
                                  and not c.get("confirms", False))
            
            if higher_tf_rejects > 0:
                return f"Rejected by {higher_tf_rejects} higher timeframes (score: {score:.2f})"
            elif len(confirmations) == 0:
                return "Insufficient data for confirmation"
            else:
                return f"Insufficient confirmation (score: {score:.2f})"
                
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
        
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            "confirmations_processed": 0,
            "signals_confirmed": 0,
            "signals_rejected": 0,
            "avg_confirmation_score": 0.0,
            "divergences_detected": 0,
            "momentum_cascades_detected": 0,
        }
