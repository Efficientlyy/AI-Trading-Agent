"""
Strategy Manager - Manages and executes trading strategies based on technical indicators.

This module provides functionality for registering, configuring, and executing
technical analysis trading strategies, utilizing both Python and Rust implementations
for optimal performance.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime
import json
import importlib
import logging
import os

from ..common.utils import get_logger
from .strategies.pattern_breakout_strategy import PatternBreakoutStrategy


class SignalDirection(Enum):
    """Possible directions for trading signals."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0


class SignalStrength(Enum):
    """Signal strength categories."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class Strategy:
    """
    Base class for all trading strategies.
    
    Strategies are responsible for analyzing technical indicators and generating
    trading signals based on predefined rules or algorithms.
    """
    
    def __init__(self, name: str, description: str, config: Dict[str, Any]):
        """
        Initialize a strategy with configuration parameters.
        
        Args:
            name: Unique name for the strategy
            description: Description of how the strategy works
            config: Configuration dictionary with strategy parameters
        """
        self.name = name
        self.description = description
        self.config = config
        self.logger = get_logger(f"Strategy_{name}")
        
        # Performance metrics
        self.metrics = {
            "signals_generated": 0,
            "avg_execution_time_ms": 0,
            "win_rate": None,
            "profit_factor": None
        }
    
    def generate_signals(
        self, 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict],
        symbols: List[str]
    ) -> List[Dict]:
        """
        Generate trading signals for the given symbols based on market data and indicators.
        
        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
            indicators: Dictionary with calculated indicator values
            symbols: List of symbols to generate signals for
            
        Returns:
            List of signal dictionaries
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement generate_signals()")
    
    def update_metrics(self, metrics_update: Dict[str, Any]) -> None:
        """Update strategy performance metrics."""
        self.metrics.update(metrics_update)
        
    def reset_metrics(self) -> None:
        """Reset strategy performance metrics."""
        self.metrics = {
            "signals_generated": 0,
            "avg_execution_time_ms": 0,
            "win_rate": None,
            "profit_factor": None
        }


# BaseStrategy serves as an alias for Strategy to maintain compatibility with new components
BaseStrategy = Strategy


class MovingAverageCrossStrategy(Strategy):
    """
    Strategy that generates signals based on moving average crossovers.
    
    Identifies potential buy/sell opportunities when a faster moving average
    crosses above/below a slower moving average.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Moving Average Cross strategy.
        
        Args:
            config: Dictionary with configuration parameters
                - fast_ma: Type and period for the fast moving average
                - slow_ma: Type and period for the slow moving average
                - signal_threshold: Minimum distance between MAs to generate a signal
                - min_lookback: Minimum number of data points required for signal generation
                - confirmation_period: Number of periods to confirm crossover (prevents false signals)
                - max_volatility: Maximum normalized ATR (% of price) to allow signals
                - volatility_exit_factor: Factor to scale down exit threshold during high volatility
        """
        name = "MA_Cross"
        description = "Moving Average Crossover Strategy"
        super().__init__(name, description, config)
        
        # Extract strategy parameters with defaults
        self.fast_ma = config.get("fast_ma", {"type": "ema", "period": 9})
        self.slow_ma = config.get("slow_ma", {"type": "ema", "period": 21})
        self.signal_threshold = config.get("signal_threshold", 0.001)
        self.min_lookback = config.get("min_lookback", 3)  # Minimum periods to look back for trend confirmation
        self.confirmation_period = config.get("confirmation_period", 1)  # Periods to confirm crossover
        self.max_volatility = config.get("max_volatility", 3.0)  # Max ATR as % of price
        self.volatility_exit_factor = config.get("volatility_exit_factor", 0.7)  # Scale down threshold in high volatility
        
        # Validate parameters
        if self.fast_ma.get("period", 0) >= self.slow_ma.get("period", 0):
            self.logger.warning(
                f"Fast MA period ({self.fast_ma.get('period')}) should be less than "
                f"Slow MA period ({self.slow_ma.get('period')}). Signal quality may be affected."
            )
        
        self.logger.info(
            f"Initialized MA Cross Strategy with fast_ma={self.fast_ma}, "
            f"slow_ma={self.slow_ma}, threshold={self.signal_threshold}, "
            f"min_lookback={self.min_lookback}, confirmation_period={self.confirmation_period}"
        )
    
    def generate_signals(
        self, 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict],
        symbols: List[str]
    ) -> List[Dict]:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
            indicators: Dictionary with calculated indicator values
            symbols: List of symbols to generate signals for
            
        Returns:
            List of signal dictionaries
        """
        start_time = datetime.now()
        signals = []
        
        for symbol in symbols:
            if symbol not in market_data or symbol not in indicators:
                continue
                
            df = market_data[symbol]
            if df.empty:
                continue
                
            self.logger.debug(f"[{self.name} - {symbol}] Processing with {len(df)} data points.")
            
            # Get the latest prices
            latest_close = df['close'].iloc[-1]
            
            # Get moving averages
            fast_type = self.fast_ma["type"]
            fast_period = str(self.fast_ma["period"])
            slow_type = self.slow_ma["type"]
            slow_period = str(self.slow_ma["period"])
            
            # Check if we have the required indicators
            if fast_type not in indicators[symbol]:
                self.logger.warning(f"Missing {fast_type} indicator for {symbol}")
                continue
                
            if slow_type not in indicators[symbol]:
                self.logger.warning(f"Missing {slow_type} indicator for {symbol}")
                continue
            
            # Check if we have the required periods
            if fast_period not in indicators[symbol][fast_type]:
                self.logger.warning(f"Missing {fast_type}({fast_period}) for {symbol}")
                continue
                
            if slow_period not in indicators[symbol][slow_type]:
                self.logger.warning(f"Missing {slow_type}({slow_period}) for {symbol}")
                continue
            
            # Get the MA values
            fast_ma_values = indicators[symbol][fast_type][fast_period]
            slow_ma_values = indicators[symbol][slow_type][slow_period]
            
            # Check for None values or insufficient data points
            if fast_ma_values is None or slow_ma_values is None:
                self.logger.warning(f"Null indicator values for {symbol}")
                continue
                
            # Check for at least 2 data points
            if len(fast_ma_values) < 2 or len(slow_ma_values) < 2:
                self.logger.warning(f"Insufficient data points for {symbol}")
                continue
                
            # Calculate the crossover signal - handle both pandas Series and numpy arrays
            if hasattr(fast_ma_values, 'iloc'):
                # It's a pandas Series
                # Get more data points for trend confirmation
                lookback = min(len(fast_ma_values), self.min_lookback + 2)  # +2 for current and previous points
                recent_fast = fast_ma_values.iloc[-lookback:]
                recent_slow = slow_ma_values.iloc[-lookback:]
                
                # Calculate differences for trend analysis
                diffs = recent_fast - recent_slow
                current_diff = diffs.iloc[-1]
                previous_diff = diffs.iloc[-2]
                
                self.logger.debug(f"[{self.name} - {symbol}] Latest Fast MA: {recent_fast.iloc[-1]}, Slow MA: {recent_slow.iloc[-1]}, Close: {latest_close}")
                
                # Check for confirmation over multiple periods
                is_confirmed = True
                if self.confirmation_period > 0:
                    # For buy signal (crossing above), check if the difference is increasing
                    if current_diff > 0 and previous_diff <= 0:  # Potential buy crossover
                        # Calculate the slope of recent differences to confirm trend
                        diff_changes = diffs.diff().iloc[-self.confirmation_period:]
                        is_confirmed = (diff_changes > 0).all()  # All recent changes should be positive
                        
                        self.logger.debug(f"[{self.name} - {symbol}] Confirmation check for BUY: {'PASSED' if is_confirmed else 'FAILED'}. Lookback confirmation values: {diff_changes.tolist()}")
                    
                    # For sell signal (crossing below), check if the difference is decreasing
                    elif current_diff < 0 and previous_diff >= 0:  # Potential sell crossover
                        # Calculate the slope of recent differences to confirm trend
                        diff_changes = diffs.diff().iloc[-self.confirmation_period:]
                        is_confirmed = (diff_changes < 0).all()  # All recent changes should be negative
                        
                        self.logger.debug(f"[{self.name} - {symbol}] Confirmation check for SELL: {'PASSED' if is_confirmed else 'FAILED'}. Lookback confirmation values: {diff_changes.tolist()}")
            else:
                # It's a numpy array
                # Get more data points for trend confirmation
                lookback = min(len(fast_ma_values), self.min_lookback + 2)  # +2 for current and previous points
                recent_fast = fast_ma_values[-lookback:]
                recent_slow = slow_ma_values[-lookback:]
                
                # Calculate differences for trend analysis
                diffs = recent_fast - recent_slow
                current_diff = diffs[-1]
                previous_diff = diffs[-2]
                
                self.logger.debug(f"[{self.name} - {symbol}] Latest Fast MA: {recent_fast[-1]}, Slow MA: {recent_slow[-1]}, Close: {latest_close}")
                
                # Check for confirmation over multiple periods
                is_confirmed = True
                if self.confirmation_period > 0 and len(diffs) > 2:
                    # For buy signal (crossing above), check if the difference is increasing
                    if current_diff > 0 and previous_diff <= 0:  # Potential buy crossover
                        # Calculate the changes in differences to confirm trend
                        diff_changes = np.diff(diffs)[-self.confirmation_period:]
                        is_confirmed = np.all(diff_changes > 0)  # All recent changes should be positive
                        
                        self.logger.debug(f"[{self.name} - {symbol}] Confirmation check for BUY: {'PASSED' if is_confirmed else 'FAILED'}. Lookback confirmation values: {diff_changes.tolist()}")
                    
                    # For sell signal (crossing below), check if the difference is decreasing
                    elif current_diff < 0 and previous_diff >= 0:  # Potential sell crossover
                        # Calculate the changes in differences to confirm trend
                        diff_changes = np.diff(diffs)[-self.confirmation_period:]
                        is_confirmed = np.all(diff_changes < 0)  # All recent changes should be negative
                        
                        self.logger.debug(f"[{self.name} - {symbol}] Confirmation check for SELL: {'PASSED' if is_confirmed else 'FAILED'}. Lookback confirmation values: {diff_changes.tolist()}")
        
            # Determine signal direction
            signal_direction = SignalDirection.NEUTRAL
            signal_strength = 0.0
            signal_metadata = {}
            
            # Check for crossovers with confirmation
            if previous_diff <= 0 and current_diff > 0 and is_confirmed:
                # Bullish crossover (fast MA crosses above slow MA)
                signal_direction = SignalDirection.BUY
                # Calculate signal strength based on the size of the crossover and the recent trend
                signal_strength = min(abs(current_diff) / latest_close * 1.5, 1.0)  # Amplify confirmed signals
                
                self.logger.debug(f"[{self.name} - {symbol}] BUY crossover detected. Fast MA ({recent_fast[-1]}) crossed above Slow MA ({recent_slow[-1]}). Diff: {current_diff}")
                
                signal_metadata["crossover_type"] = "bullish"
                signal_metadata["confirmation"] = "strong" if is_confirmed else "weak"
            elif previous_diff >= 0 and current_diff < 0 and is_confirmed:
                # Bearish crossover (fast MA crosses below slow MA)
                signal_direction = SignalDirection.SELL
                # Calculate signal strength based on the size of the crossover and the recent trend
                signal_strength = min(abs(current_diff) / latest_close * 1.5, 1.0)  # Amplify confirmed signals
                
                self.logger.debug(f"[{self.name} - {symbol}] SELL crossover detected. Fast MA ({recent_fast[-1]}) crossed below Slow MA ({recent_slow[-1]}). Diff: {current_diff}")
                
                signal_metadata["crossover_type"] = "bearish"
                signal_metadata["confirmation"] = "strong" if is_confirmed else "weak"
            
            # Check for excessive volatility using ATR if signal is detected
            if signal_direction != SignalDirection.NEUTRAL and "atr" in indicators[symbol]:
                # Get ATR value
                atr_value = indicators[symbol]["atr"]
                
                # Handle different data types (pandas Series vs numpy array)
                if hasattr(atr_value, 'iloc'):
                    current_atr = atr_value.iloc[-1]
                else:
                    current_atr = atr_value[-1]
                
                # Calculate normalized ATR as percentage of price
                atr_percent = current_atr / latest_close * 100
                
                # Add to metadata regardless of whether we filter out the signal
                signal_metadata["volatility_atr"] = float(current_atr)
                signal_metadata["volatility_percent"] = float(atr_percent)
                signal_metadata["volatility_threshold"] = float(self.max_volatility)
                
                # Check if volatility exceeds our threshold
                if atr_percent > self.max_volatility:
                    # Market is too volatile, adjust signal
                    signal_metadata["volatility_status"] = "excessive"
                    
                    # Adjust signal strength based on volatility
                    # Scale down signal strength proportionally to how much volatility exceeds threshold
                    excess_factor = min(2.0, atr_percent / self.max_volatility)  # Cap at 2x threshold
                    signal_strength = signal_strength / excess_factor
                    
                    # Increase signal threshold during high volatility
                    adjusted_threshold = self.signal_threshold / self.volatility_exit_factor
                    signal_metadata["adjusted_threshold"] = float(adjusted_threshold)
                    
                    self.logger.debug(f"[{self.name} - {symbol}] High volatility detected: ATR {atr_percent:.2f}% > {self.max_volatility:.2f}%, signal strength adjusted from {signal_strength * excess_factor:.4f} to {signal_strength:.4f}")
                else:
                    signal_metadata["volatility_status"] = "normal"
            
            # Apply volume confirmation if signal is detected
            if signal_direction != SignalDirection.NEUTRAL:
                # Check if volume data is available
                if 'volume' in df.columns:
                    # Get recent volume data
                    recent_volume = df['volume'].iloc[-lookback:]
                    # Calculate average volume over the lookback period
                    avg_volume = recent_volume.mean()
                    # Get current volume
                    current_volume = recent_volume.iloc[-1]
                    
                    # Volume confirmation (higher volume during crossover)
                    volume_confirmation = current_volume > avg_volume * 1.2  # 20% above average
                    
                    # Adjust signal strength based on volume
                    if volume_confirmation:
                        signal_strength *= 1.2  # Boost confidence for high volume signals
                        signal_metadata["volume_confirmation"] = "high"
                    else:
                        signal_strength *= 0.8  # Reduce confidence for low volume signals
                        signal_metadata["volume_confirmation"] = "low"
                    
                    # Add volume metrics to metadata
                    signal_metadata["volume_current"] = float(current_volume)
                    signal_metadata["volume_average"] = float(avg_volume)
                    signal_metadata["volume_ratio"] = float(current_volume / avg_volume)
                else:
                    # No volume data available
                    signal_metadata["volume_confirmation"] = "unavailable"
            
            # Create signal if not neutral and strength meets threshold
            if signal_direction != SignalDirection.NEUTRAL and signal_strength >= self.signal_threshold:
                # Create signal
                signal = {
                    "type": "technical_signal",
                    "payload": {
                        "symbol": symbol,
                        "signal": signal_direction.value * signal_strength,
                        "confidence": signal_strength,
                        "strategy": self.name,
                        "price_at_signal": float(latest_close),
                        "timestamp": df.index[-1].isoformat() if isinstance(df.index, pd.DatetimeIndex) else datetime.now().isoformat(),
                        "indicators_used": {
                            "fast_ma": {
                                "type": fast_type,
                                "period": fast_period,
                                "value": float(recent_fast[-1] if hasattr(recent_fast, 'iloc') else recent_fast[-1])
                            },
                            "slow_ma": {
                                "type": slow_type,
                                "period": slow_period,
                                "value": float(recent_slow[-1] if hasattr(recent_slow, 'iloc') else recent_slow[-1])
                            }
                        },
                        "metadata": {
                            "crossover_type": signal_metadata.get("crossover_type", "bullish" if signal_direction == SignalDirection.BUY else "bearish"),
                            "diff_percentage": float(abs(current_diff) / latest_close * 100),
                            "confirmation": signal_metadata.get("confirmation", "unknown"),
                            "lookback_periods": self.min_lookback,
                            "confirmation_periods": self.confirmation_period
                        }
                    }
                }
                signals.append(signal)
                
                self.logger.info(
                    f"Generated {signal_direction.name} signal for {symbol} with "
                    f"strength {signal_strength:.4f}"
                )
        
        # Update metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["signals_generated"] += len(signals)
        self.metrics["avg_execution_time_ms"] = (
            (self.metrics["avg_execution_time_ms"] * (self.metrics["signals_generated"] - len(signals)) + 
             execution_time * len(signals)) / 
            max(1, self.metrics["signals_generated"])
        )
        
        return signals


class RSIOverboughtOversoldStrategy(Strategy):
    """
    Strategy that generates signals based on RSI overbought/oversold conditions.
    
    Identifies potential buy opportunities when RSI drops below the oversold level
    and potential sell opportunities when it rises above the overbought level.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RSI Overbought/Oversold strategy.
        
        Args:
            config: Dictionary with configuration parameters
                - period: RSI calculation period
                - overbought: RSI level considered overbought (default: 70)
                - oversold: RSI level considered oversold (default: 30)
                - confirmation_periods: Number of periods to confirm signal
                - min_divergence: Minimum divergence from threshold to trigger signal
                - exit_buffer: Buffer for exit signals (ex: exit overbought at 65 instead of 70)
                - max_volatility: Maximum normalized ATR (% of price) to allow signals
                - volatility_adjustment: How much to adjust thresholds in high volatility
        """
        name = "RSI_OB_OS"
        description = "RSI Overbought/Oversold Strategy"
        super().__init__(name, description, config)
        
        # Extract strategy parameters with defaults
        self.period = config.get("period", 14)
        self.overbought = config.get("overbought", 70)
        self.oversold = config.get("oversold", 30)
        self.confirmation_periods = config.get("confirmation_periods", 1)
        self.min_divergence = config.get("min_divergence", 2.0)  # Min points beyond threshold
        self.exit_buffer = config.get("exit_buffer", 5.0)  # Points inside threshold to exit
        self.max_volatility = config.get("max_volatility", 3.0)  # Max ATR as % of price
        self.volatility_adjustment = config.get("volatility_adjustment", 5.0)  # Points to adjust thresholds
        
        # Validate parameters
        if self.overbought <= self.oversold:
            self.logger.warning(
                f"Overbought level ({self.overbought}) should be higher than "
                f"oversold level ({self.oversold}). Signal quality may be affected."
            )
        
        # Calculate exit levels
        self.overbought_exit = self.overbought - self.exit_buffer
        self.oversold_exit = self.oversold + self.exit_buffer
        
        self.logger.info(
            f"Initialized RSI OB/OS Strategy with period={self.period}, "
            f"overbought={self.overbought}, oversold={self.oversold}, "
            f"confirmation_periods={self.confirmation_periods}, min_divergence={self.min_divergence}, "
            f"exit_buffer={self.exit_buffer}"
        )
    
    def generate_signals(
        self, 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict],
        symbols: List[str]
    ) -> List[Dict]:
        """
        Generate trading signals based on RSI overbought/oversold levels.
        
        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
            indicators: Dictionary with calculated indicator values
            symbols: List of symbols to generate signals for
            
        Returns:
            List of signal dictionaries
        """
        start_time = datetime.now()
        signals = []
        
        for symbol in symbols:
            if symbol not in market_data or symbol not in indicators:
                continue
                
            df = market_data[symbol]
            if df.empty:
                continue
                
            # Get the latest prices
            latest_close = df['close'].iloc[-1]
            
            # Check if we have RSI indicator
            if "rsi" not in indicators[symbol]:
                self.logger.warning(f"Missing RSI indicator for {symbol}")
                continue
            
            # Get RSI values
            if "rsi" not in indicators[symbol]:
                self.logger.warning(f"Missing RSI indicator for {symbol}")
                continue
                
            rsi_values = indicators[symbol]["rsi"]
            
            # Check if RSI values is None or too short
            if rsi_values is None or len(rsi_values) < 2:
                self.logger.warning(f"Insufficient RSI data for {symbol}")
                continue
                
            # Get current and previous RSI - handle both pandas Series and numpy arrays
            if hasattr(rsi_values, 'iloc'):
                # It's a pandas Series
                current_rsi = rsi_values.iloc[-1]
                previous_rsi = rsi_values.iloc[-2]
            else:
                # It's a numpy array
                current_rsi = rsi_values[-1]
                previous_rsi = rsi_values[-2]
            
            # Determine signal direction
            signal_direction = SignalDirection.NEUTRAL
            signal_strength = 0.0
            
            # Get more data points for confirmation and trend analysis
            lookback = min(len(rsi_values), max(5, self.confirmation_periods + 2))  # +2 for current and previous
            
            signal_metadata = {} # Initialize for each symbol
            
            if hasattr(rsi_values, 'iloc'):
                # It's a pandas Series
                recent_rsi = rsi_values.iloc[-lookback:]
                
                # Determine effective thresholds based on volatility
                effective_oversold_threshold = signal_metadata.get("adjusted_oversold", self.oversold) if signal_metadata.get("volatility_status") == "excessive" else self.oversold
                effective_overbought_threshold = signal_metadata.get("adjusted_overbought", self.overbought) if signal_metadata.get("volatility_status") == "excessive" else self.overbought
                
                self.logger.debug(f"[{self.name} - {symbol}] Effective OS: {effective_oversold_threshold:.2f}, Effective OB: {effective_overbought_threshold:.2f}. Volatility status: {signal_metadata.get('volatility_status')}")
                
                # Oversold condition (crossing up from below) - buy signal
                if previous_rsi < effective_oversold_threshold and current_rsi >= self.oversold_exit:
                    self.logger.debug(f"[{self.name} - {symbol}] Raw OVERSOLD condition triggered: prev_rsi ({previous_rsi:.2f}) < eff_OS ({effective_oversold_threshold:.2f}) AND curr_rsi ({current_rsi:.2f}) >= OS_exit ({self.oversold_exit:.2f})")
                    # Calculate minimum oversold divergence (how far it went below threshold)
                    min_value = recent_rsi.min()
                    divergence = effective_oversold_threshold - min_value
                    self.logger.debug(f"[{self.name} - {symbol}] Oversold divergence: {divergence:.2f} (threshold: {self.min_divergence})")
                    
                    # Check for confirmation (consistently moving up after crossing below threshold)
                    is_confirmed = False
                    if self.confirmation_periods > 0:
                        # Check if RSI has been increasing since the minimum
                        min_idx = recent_rsi.idxmin()
                        if min_idx is not None:
                            since_min = recent_rsi.loc[min_idx:]
                            if len(since_min) >= self.confirmation_periods:
                                # Check if RSI has been trending up
                                is_confirmed = (since_min.diff().dropna() > 0).sum() >= self.confirmation_periods
                    
                    self.logger.debug(f"[{self.name} - {symbol}] Oversold confirmation: {'PASSED' if is_confirmed else 'FAILED'}. Lookback values: {since_min.tolist() if min_idx is not None else 'N/A'}")
                    
                    # Signal if divergence and confirmation criteria are met
                    if divergence >= self.min_divergence and (is_confirmed or self.confirmation_periods == 0):
                        signal_direction = SignalDirection.BUY
                        # Calculate signal strength based on divergence and recovery speed
                        # Find the position index of the minimum value
                        min_pos = recent_rsi.values.argmin()
                        # Calculate recovery speed based on position difference
                        recovery_speed = (current_rsi - min_value) / max(1, len(recent_rsi) - min_pos)
                        signal_strength = min(0.95, (divergence / 20.0) * (1 + recovery_speed / 5.0))
                        
                        self.logger.debug(f"[{self.name} - {symbol}] OVERSOLD signal potential. Divergence: {divergence:.2f}, Confirmed: {is_confirmed}, Recovery: {recovery_speed:.2f}, Strength: {signal_strength:.2f}")
                
                # Overbought condition (crossing down from above) - sell signal
                elif previous_rsi > effective_overbought_threshold and current_rsi <= self.overbought_exit:
                    self.logger.debug(f"[{self.name} - {symbol}] Raw OVERBOUGHT condition triggered: prev_rsi ({previous_rsi:.2f}) > eff_OB ({effective_overbought_threshold:.2f}) AND curr_rsi ({current_rsi:.2f}) <= OB_exit ({self.overbought_exit:.2f})")
                    # Calculate maximum overbought divergence (how far it went above threshold)
                    max_value = recent_rsi.max()
                    divergence = max_value - effective_overbought_threshold
                    self.logger.debug(f"[{self.name} - {symbol}] Overbought divergence: {divergence:.2f} (threshold: {self.min_divergence})")
                    
                    # Check for confirmation (consistently moving down after crossing above threshold)
                    is_confirmed = False
                    if self.confirmation_periods > 0:
                        # Check if RSI has been decreasing since the maximum
                        max_idx = recent_rsi.idxmax()
                        if max_idx is not None:
                            since_max = recent_rsi.loc[max_idx:]
                            if len(since_max) >= self.confirmation_periods:
                                # Check if RSI has been trending down
                                is_confirmed = (since_max.diff().dropna() < 0).sum() >= self.confirmation_periods
                    
                    self.logger.debug(f"[{self.name} - {symbol}] Overbought confirmation: {'PASSED' if is_confirmed else 'FAILED'}. Lookback values: {since_max.tolist() if max_idx is not None else 'N/A'}")
                    
                    # Signal if divergence and confirmation criteria are met
                    if divergence >= self.min_divergence and (is_confirmed or self.confirmation_periods == 0):
                        signal_direction = SignalDirection.SELL
                        # Calculate signal strength based on divergence and reversion speed
                        # Find the position index of the maximum value
                        max_pos = recent_rsi.values.argmax()
                        # Calculate reversion speed based on position difference
                        reversion_speed = (max_value - current_rsi) / max(1, len(recent_rsi) - max_pos)
                        signal_strength = min(0.95, (divergence / 20.0) * (1 + reversion_speed / 5.0))
                        
                        self.logger.debug(f"[{self.name} - {symbol}] OVERBOUGHT signal potential. Divergence: {divergence:.2f}, Confirmed: {is_confirmed}, Reversion: {reversion_speed:.2f}, Strength: {signal_strength:.2f}")
            else:
                # It's a numpy array
                recent_rsi = rsi_values[-lookback:]
                
                # Determine effective thresholds based on volatility for NumPy arrays
                effective_oversold_threshold_np = signal_metadata.get("adjusted_oversold", self.oversold) if signal_metadata.get("volatility_status") == "excessive" else self.oversold
                effective_overbought_threshold_np = signal_metadata.get("adjusted_overbought", self.overbought) if signal_metadata.get("volatility_status") == "excessive" else self.overbought
                
                self.logger.debug(f"[{self.name} - {symbol}] (NumPy) Effective OS: {effective_oversold_threshold_np:.2f}, Effective OB: {effective_overbought_threshold_np:.2f}. Volatility status: {signal_metadata.get('volatility_status')}")
                
                # Oversold condition (crossing up from below) - buy signal
                if previous_rsi < effective_oversold_threshold_np and current_rsi >= self.oversold_exit:
                    self.logger.debug(f"[{self.name} - {symbol}] (NumPy) Raw OVERSOLD condition triggered: prev_rsi ({previous_rsi:.2f}) < eff_OS ({effective_oversold_threshold_np:.2f}) AND curr_rsi ({current_rsi:.2f}) >= OS_exit ({self.oversold_exit:.2f})")
                    # Calculate minimum oversold divergence (how far it went below threshold)
                    min_value = np.min(recent_rsi)
                    divergence = effective_oversold_threshold_np - min_value
                    self.logger.debug(f"[{self.name} - {symbol}] (NumPy) Oversold divergence: {divergence:.2f} (threshold: {self.min_divergence})")
                    
                    # Check for confirmation (consistently moving up after crossing below threshold)
                    is_confirmed = False
                    if self.confirmation_periods > 0:
                        # Check if RSI has been increasing for the confirmation period
                        min_idx = np.argmin(recent_rsi)
                        if len(recent_rsi) - min_idx >= self.confirmation_periods:
                            # Check if RSI has been trending up since minimum
                            since_min = recent_rsi[min_idx:]
                            is_confirmed = np.all(np.diff(since_min[-self.confirmation_periods:]) > 0)
                    
                    self.logger.debug(f"[{self.name} - {symbol}] (NumPy) Oversold confirmation: {'PASSED' if is_confirmed else 'FAILED'}. Lookback values: {since_min.tolist() if min_idx is not None else 'N/A'}")
                    
                    # Signal if divergence and confirmation criteria are met
                    if divergence >= self.min_divergence and (is_confirmed or self.confirmation_periods == 0):
                        signal_direction = SignalDirection.BUY
                        # Calculate signal strength based on divergence and recovery speed
                        recovery_speed = (current_rsi - min_value) / max(1, (len(recent_rsi) - np.argmin(recent_rsi)))
                        signal_strength = min(0.95, (divergence / 20.0) * (1 + recovery_speed / 5.0))
                        
                        self.logger.debug(f"[{self.name} - {symbol}] (NumPy) OVERSOLD signal potential. Divergence: {divergence:.2f}, Confirmed: {is_confirmed}, Recovery: {recovery_speed:.2f}, Strength: {signal_strength:.2f}")
                
                # Overbought condition (crossing down from above) - sell signal
                elif previous_rsi > effective_overbought_threshold_np and current_rsi <= self.overbought_exit:
                    self.logger.debug(f"[{self.name} - {symbol}] (NumPy) Raw OVERBOUGHT condition triggered: prev_rsi ({previous_rsi:.2f}) > eff_OB ({effective_overbought_threshold_np:.2f}) AND curr_rsi ({current_rsi:.2f}) <= OB_exit ({self.overbought_exit:.2f})")
                    # Calculate maximum overbought divergence (how far it went above threshold)
                    max_value = np.max(recent_rsi)
                    divergence = max_value - effective_overbought_threshold_np
                    self.logger.debug(f"[{self.name} - {symbol}] (NumPy) Overbought divergence: {divergence:.2f} (threshold: {self.min_divergence})")
                    
                    # Check for confirmation (consistently moving down after crossing above threshold)
                    is_confirmed = False
                    if self.confirmation_periods > 0:
                        # Check if RSI has been decreasing for the confirmation period
                        max_idx = np.argmax(recent_rsi)
                        if len(recent_rsi) - max_idx >= self.confirmation_periods:
                            # Check if RSI has been trending down since maximum
                            since_max = recent_rsi[max_idx:]
                            is_confirmed = np.all(np.diff(since_max[-self.confirmation_periods:]) < 0)
                    
                    self.logger.debug(f"[{self.name} - {symbol}] (NumPy) Overbought confirmation: {'PASSED' if is_confirmed else 'FAILED'}. Lookback values: {since_max.tolist() if max_idx is not None else 'N/A'}")
                    
                    # Signal if divergence and confirmation criteria are met
                    if divergence >= self.min_divergence and (is_confirmed or self.confirmation_periods == 0):
                        signal_direction = SignalDirection.SELL
                        # Calculate signal strength based on divergence and reversion speed
                        reversion_speed = (max_value - current_rsi) / max(1, (len(recent_rsi) - np.argmax(recent_rsi)))
                        signal_strength = min(0.95, (divergence / 20.0) * (1 + reversion_speed / 5.0))
                        
                        self.logger.debug(f"[{self.name} - {symbol}] (NumPy) OVERBOUGHT signal potential. Divergence: {divergence:.2f}, Confirmed: {is_confirmed}, Reversion: {reversion_speed:.2f}, Strength: {signal_strength:.2f}")
            
            # Check for excessive volatility using ATR if signal is detected
            if signal_direction != SignalDirection.NEUTRAL and "atr" in indicators[symbol]:
                # Get ATR value
                atr_value = indicators[symbol]["atr"]
                
                # Handle different data types (pandas Series vs numpy array)
                if hasattr(atr_value, 'iloc'):
                    current_atr = atr_value.iloc[-1]
                else:
                    current_atr = atr_value[-1]
                
                # Calculate normalized ATR as percentage of price
                atr_percent = current_atr / latest_close * 100
                
                # Add to metadata regardless of whether we filter out the signal
                signal_metadata["volatility_atr"] = float(current_atr)
                signal_metadata["volatility_percent"] = float(atr_percent)
                signal_metadata["volatility_threshold"] = float(self.max_volatility)
                
                # Check if volatility exceeds our threshold
                if atr_percent > self.max_volatility:
                    # Market is too volatile, adjust RSI thresholds and signal strength
                    signal_metadata["volatility_status"] = "excessive"
                    
                    # Adjust thresholds based on volatility (make more extreme in high volatility)
                    vol_ratio = min(2.0, atr_percent / self.max_volatility)
                    
                    if signal_direction == SignalDirection.BUY:
                        # For buy signals: lower the oversold threshold in high volatility
                        adjusted_threshold = self.oversold - (self.volatility_adjustment * (vol_ratio - 1))
                        signal_metadata["adjusted_oversold"] = float(adjusted_threshold)
                    else:  # SELL signal
                        # For sell signals: raise the overbought threshold in high volatility
                        adjusted_threshold = self.overbought + (self.volatility_adjustment * (vol_ratio - 1))
                        signal_metadata["adjusted_overbought"] = float(adjusted_threshold)
                    
                    # Adjust signal strength based on volatility
                    original_strength = signal_strength
                    signal_strength = signal_strength / vol_ratio
                    
                    # Log the volatility filtering
                    self.logger.debug(
                        f"[{self.name} - {symbol}] High volatility detected: ATR {atr_percent:.2f}% > {self.max_volatility:.2f}%, "
                        f"signal strength adjusted from {original_strength:.4f} to {signal_strength:.4f}"
                    )
                else:
                    signal_metadata["volatility_status"] = "normal"
            
            # Apply volume confirmation if signal is detected
            if signal_direction != SignalDirection.NEUTRAL:
                # Check if volume data is available
                if 'volume' in df.columns:
                    # Get recent volume data for confirmation
                    recent_volume = df['volume'].iloc[-lookback:]
                    # Calculate average volume over the lookback period
                    avg_volume = recent_volume.mean()
                    # Get current volume
                    current_volume = recent_volume.iloc[-1]
                    
                    # Define expected volume pattern based on signal type
                    # For oversold (buy) signals, we expect rising volume as price rebounds
                    # For overbought (sell) signals, we expect rising volume as price falls
                    if signal_direction == SignalDirection.BUY:
                        # For buy signals, check if volume is increasing as RSI rises from bottom
                        min_idx = np.argmin(recent_rsi.values)
                        if min_idx < len(recent_volume) - 1:  # Ensure we have volume data at minimum RSI
                            vol_since_min = recent_volume.iloc[min_idx:]
                            vol_change = vol_since_min.iloc[-1] / vol_since_min.iloc[0] if len(vol_since_min) > 1 else 1.0
                            # Strong confirmation if volume increases during RSI recovery
                            volume_confirmation = vol_change > 1.2  # 20% volume increase during recovery
                        else:
                            volume_confirmation = False
                    else:  # SELL signal
                        # For sell signals, check if volume is increasing as RSI falls from top
                        max_idx = np.argmax(recent_rsi.values)
                        if max_idx < len(recent_volume) - 1:  # Ensure we have volume data at maximum RSI
                            vol_since_max = recent_volume.iloc[max_idx:]
                            vol_change = vol_since_max.iloc[-1] / vol_since_max.iloc[0] if len(vol_since_max) > 1 else 1.0
                            # Strong confirmation if volume increases during RSI decline
                            volume_confirmation = vol_change > 1.2  # 20% volume increase during decline
                        else:
                            volume_confirmation = False
                            
                    # Adjust signal strength based on volume confirmation
                    if volume_confirmation:
                        signal_strength *= 1.2  # Boost confidence for confirmed volume pattern
                        signal_strength = min(signal_strength, 0.95)  # Cap at 0.95
                    else:
                        signal_strength *= 0.8  # Reduce confidence without volume confirmation
                    
                    # Add volume metrics to existing metadata
                    signal_metadata["volume_confirmation"] = "high" if volume_confirmation else "low"
                    signal_metadata["volume_current"] = float(current_volume)
                    signal_metadata["volume_average"] = float(avg_volume)
                    signal_metadata["volume_ratio"] = float(current_volume / avg_volume)
                else:
                    # No volume data available
                    signal_metadata["volume_confirmation"] = "unavailable"
            
            # Create signal if not neutral and strength meets threshold
            self.logger.debug(f"[{self.name} - {symbol}] Final check: Direction={signal_direction.name}, Strength={signal_strength:.4f}, Threshold={self.signal_threshold:.2f}")
            if signal_direction != SignalDirection.NEUTRAL and signal_strength >= self.signal_threshold:
                self.logger.info(f"[{self.name} - {symbol}] Confirmed {signal_direction.name} signal. Strength: {signal_strength:.4f}")
                # Create signal
                signal = {
                    "type": "technical_signal",
                    "payload": {
                        "symbol": symbol,
                        "signal": signal_direction.value * signal_strength,
                        "confidence": signal_strength,
                        "strategy": self.name,
                        "price_at_signal": float(latest_close),
                        "timestamp": df.index[-1].isoformat() if isinstance(df.index, pd.DatetimeIndex) else datetime.now().isoformat(),
                        "indicators_used": {
                            "rsi": {
                                "period": str(self.period),
                                "value": float(current_rsi)
                            }
                        },
                        "metadata": {
                            "condition": "oversold" if signal_direction == SignalDirection.BUY else "overbought",
                            "threshold": self.oversold if signal_direction == SignalDirection.BUY else self.overbought,
                            "exit_level": self.oversold_exit if signal_direction == SignalDirection.BUY else self.overbought_exit,
                            "divergence": float(divergence),
                            "min_divergence_required": float(self.min_divergence),
                            "confirmation_periods": self.confirmation_periods,
                            "confirmation_status": "confirmed" if is_confirmed else "unconfirmed"
                        }
                    }
                }
                signals.append(signal)
            else:
                if signal_direction != SignalDirection.NEUTRAL:
                    self.logger.debug(f"[{self.name} - {symbol}] Signal direction was {signal_direction.name} with strength {signal_strength:.4f}, but did not meet threshold {self.signal_threshold:.2f}. No signal generated.")
                else:
                    self.logger.debug(f"[{self.name} - {symbol}] No conditions met for signal generation at timestamp {df.index[-1]}.")
        
        # Update metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["signals_generated"] += len(signals)
        self.metrics["avg_execution_time_ms"] = (
            (self.metrics["avg_execution_time_ms"] * (self.metrics["signals_generated"] - len(signals)) + 
             execution_time * len(signals)) / 
            max(1, self.metrics["signals_generated"])
        )
        
        return signals


class StrategyManager:
    """
    Manages technical analysis trading strategies.
    
    This class is responsible for registering, configuring, and executing
    trading strategies based on technical indicators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the StrategyManager with configuration.
        
        Args:
            config: Configuration dictionary with strategy settings
        """
        self.config = config
        self.logger = get_logger("StrategyManager")
        self.strategies = {}
        
        # Initialize metrics
        self.metrics = {
            "total_signals_generated": 0,
            "strategies_executed": 0,
            "avg_execution_time_ms": 0
        }
        
        # Load strategies from configuration
        self._load_strategies()
        
        self.logger.info(f"StrategyManager initialized with {len(self.strategies)} strategies")
    
    def _load_strategies(self) -> None:
        """Load and initialize strategies from configuration."""
        strategies_config = self.config.get("strategies", {})
        
        # Load Moving Average Cross strategy if enabled
        if strategies_config.get("ma_cross", {}).get("enabled", False):
            ma_config = strategies_config["ma_cross"]
            self.register_strategy(MovingAverageCrossStrategy(ma_config))
            self.logger.debug("Registered MA Cross strategy")
            
        # Load RSI Overbought/Oversold strategy if enabled
        if strategies_config.get("rsi_ob_os", {}).get("enabled", False):
            rsi_config = strategies_config["rsi_ob_os"]
            self.register_strategy(RSIOverboughtOversoldStrategy(rsi_config))
            self.logger.debug("Registered RSI OB/OS strategy")
            
        # Load Pattern Breakout strategy if enabled
        if strategies_config.get("pattern_breakout", {}).get("enabled", False):
            pattern_config = strategies_config["pattern_breakout"]
            self.register_strategy(PatternBreakoutStrategy(pattern_config))
            self.logger.debug("Registered Pattern Breakout strategy")
    
    def register_strategy(self, strategy: Strategy) -> None:
        """
        Register a new strategy with the manager.
        
        Args:
            strategy: Strategy instance to register
        """
        self.strategies[strategy.name] = strategy
        self.logger.info(f"Registered strategy: {strategy.name}")
    
    def unregister_strategy(self, strategy_name: str) -> None:
        """
        Unregister a strategy from the manager.
        
        Args:
            strategy_name: Name of the strategy to unregister
        """
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            self.logger.info(f"Unregistered strategy: {strategy_name}")
    
    def generate_signals(
        self, 
        market_data: Dict[str, pd.DataFrame], 
        indicators: Dict[str, Dict],
        symbols: List[str]
    ) -> List[Dict]:
        """
        Generate signals from all registered strategies.
        
        Args:
            market_data: Dictionary mapping symbols to market data DataFrames
            indicators: Dictionary with calculated indicator values
            symbols: List of symbols to generate signals for
            
        Returns:
            List of signal dictionaries from all strategies
        """
        if not self.strategies:
            self.logger.warning("No strategies registered")
            return []
            
        start_time = datetime.now()
        all_signals = []
        
        # Generate signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(market_data, indicators, symbols)
                all_signals.extend(signals)
                
                self.logger.debug(
                    f"Strategy {strategy_name} generated {len(signals)} signals"
                )
            except Exception as e:
                self.logger.error(
                    f"Error executing strategy {strategy_name}: {str(e)}",
                    exc_info=True
                )
        
        # Update metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["total_signals_generated"] += len(all_signals)
        self.metrics["strategies_executed"] += len(self.strategies)
        self.metrics["avg_execution_time_ms"] = (
            (self.metrics["avg_execution_time_ms"] * 
             (self.metrics["strategies_executed"] - len(self.strategies)) + 
             execution_time * len(self.strategies)) / 
            max(1, self.metrics["strategies_executed"])
        )
        
        self.logger.info(
            f"Generated {len(all_signals)} signals from {len(self.strategies)} strategies "
            f"in {execution_time:.2f}ms"
        )
        
        return all_signals
    
    def get_strategy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all registered strategies."""
        metrics = {}
        for name, strategy in self.strategies.items():
            metrics[name] = strategy.metrics
        return metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get overall strategy manager metrics."""
        return self.metrics
