"""
Market Regime-Specific Trading Strategies

This module provides specialized trading strategies optimized for different market regimes:
- TrendingMarketStrategy: For strong trending markets (up or down)
- RangeBoundStrategy: For sideways, consolidating markets
- VolatilityBreakoutStrategy: For volatile markets with potential breakouts
- MeanReversionStrategy: For overextended markets likely to return to a mean
- RegimeTransitionStrategy: For periods of regime change/transition
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from functools import lru_cache

from ..agent.strategy import BaseStrategy, StrategyType
from ..agent.market_regime import MarketRegimeClassifier
from ..common.utils import get_logger

class BaseRegimeStrategy(BaseStrategy):
    """Base class for all regime-specific strategies with common functionality."""
    
    def __init__(self, config: Dict[str, Any], name: str, description: str):
        """
        Initialize the base regime strategy.
        
        Args:
            config: Strategy configuration dictionary
            name: Strategy name
            description: Strategy description
        """
        super().__init__(config, name, description, StrategyType.TECHNICAL)
        self.logger = get_logger(f"RegimeStrategy-{name}")
        
        # Common regime strategy parameters
        self.lookback_window = config.get("lookback_window", 50)
        self.min_samples = config.get("min_samples", 20)
        self.sensitivity = config.get("sensitivity", 1.0)
        self.confirmation_threshold = config.get("confirmation_threshold", 2)
        self.enable_filters = config.get("enable_filters", True)
        
        # Initialize regime classifier if not provided
        regime_config = config.get("regime_config", {})
        self.regime_classifier = MarketRegimeClassifier(regime_config)
        
        # Signal strength mapping
        self.signal_strength_mapping = {
            "strong_buy": 1.0,
            "buy": 0.5,
            "neutral": 0.0,
            "sell": -0.5,
            "strong_sell": -1.0
        }
        
        # Track performance metrics
        self.performance = {
            "signals_generated": 0,
            "profitable_signals": 0,
            "accuracy": 0.0
        }
        
    def update_performance(self, was_profitable: bool):
        """Update performance tracking."""
        self.performance["signals_generated"] += 1
        if was_profitable:
            self.performance["profitable_signals"] += 1
        
        if self.performance["signals_generated"] > 0:
            self.performance["accuracy"] = (
                self.performance["profitable_signals"] / 
                self.performance["signals_generated"]
            )
    
    @lru_cache(maxsize=128)
    def _calculate_common_indicators(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate common technical indicators used by multiple regime strategies."""
        if data is None or data.empty or len(data) < self.lookback_window:
            return {}
            
        # Get recent data
        recent = data.iloc[-self.lookback_window:].copy()
        
        try:
            # Calculate basic price data
            recent['returns'] = recent['close'].pct_change()
            recent['log_returns'] = np.log(recent['close'] / recent['close'].shift(1))
            recent['volatility'] = recent['returns'].rolling(window=20).std()
            
            # Moving averages
            recent['sma20'] = recent['close'].rolling(window=20).mean()
            recent['sma50'] = recent['close'].rolling(window=50).mean()
            recent['sma200'] = recent['close'].rolling(window=200).mean()
            
            # Bollinger Bands
            bb_window = 20
            recent['bb_middle'] = recent['close'].rolling(window=bb_window).mean()
            recent['bb_std'] = recent['close'].rolling(window=bb_window).std()
            recent['bb_upper'] = recent['bb_middle'] + 2 * recent['bb_std']
            recent['bb_lower'] = recent['bb_middle'] - 2 * recent['bb_std']
            recent['bb_width'] = (recent['bb_upper'] - recent['bb_lower']) / recent['bb_middle']
            
            # RSI
            delta = recent['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            recent['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = recent['close'].ewm(span=12, adjust=False).mean()
            ema26 = recent['close'].ewm(span=26, adjust=False).mean()
            recent['macd'] = ema12 - ema26
            recent['macd_signal'] = recent['macd'].ewm(span=9, adjust=False).mean()
            recent['macd_hist'] = recent['macd'] - recent['macd_signal']
            
            # Volume indicators
            recent['volume_sma20'] = recent['volume'].rolling(window=20).mean()
            recent['volume_ratio'] = recent['volume'] / recent['volume_sma20']
            
            # Trend strength
            recent['adx'] = self._calculate_adx(recent)
            
            # Support and resistance
            pivots = self._identify_pivots(recent)
            
            return {
                "data": recent,
                "pivots": pivots,
                "last_close": recent['close'].iloc[-1],
                "last_volume": recent['volume'].iloc[-1],
                "last_rsi": recent['rsi'].iloc[-1],
                "last_macd": recent['macd'].iloc[-1],
                "last_macd_signal": recent['macd_signal'].iloc[-1],
                "last_bb_width": recent['bb_width'].iloc[-1],
                "sma_alignment": (
                    recent['sma20'].iloc[-1] > recent['sma50'].iloc[-1] > recent['sma200'].iloc[-1]
                ),
                "below_lower_bb": recent['close'].iloc[-1] < recent['bb_lower'].iloc[-1],
                "above_upper_bb": recent['close'].iloc[-1] > recent['bb_upper'].iloc[-1],
                "high_volume": recent['volume_ratio'].iloc[-1] > 1.5,
                "strong_trend": recent['adx'].iloc[-1] > 25 if 'adx' in recent else False
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
            return {}
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) for trend strength."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range (TR)
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Calculate Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            # Positive Directional Movement (+DM)
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            pos_dm = pd.Series(pos_dm, index=data.index)
            
            # Negative Directional Movement (-DM)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            neg_dm = pd.Series(neg_dm, index=data.index)
            
            # Smooth +DM and -DM using ATR
            pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
            neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
            
            # Calculate Directional Index (DX)
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            
            # Calculate ADX as smoothed DX
            adx = dx.rolling(window=period).mean()
            
            return adx
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            return pd.Series([0] * len(data), index=data.index)
    
    def _identify_pivots(self, data: pd.DataFrame, window: int = 5) -> Dict[str, List[float]]:
        """Identify support and resistance pivot points."""
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            resistance_levels = []
            support_levels = []
            
            # Find pivot highs (resistance)
            for i in range(window, len(highs) - window):
                if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] > highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append(highs[i])
            
            # Find pivot lows (support)
            for i in range(window, len(lows) - window):
                if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] < lows[i+j] for j in range(1, window+1)):
                    support_levels.append(lows[i])
            
            return {
                "resistance": resistance_levels,
                "support": support_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying pivots: {str(e)}")
            return {"resistance": [], "support": []}
    
    def _is_near_level(self, price: float, levels: List[float], threshold: float = 0.02) -> bool:
        """Check if price is near any support/resistance level."""
        if not levels:
            return False
            
        for level in levels:
            if abs(price - level) / price < threshold:
                return True
                
        return False
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Base implementation of signal generation for regime strategies.
        
        Args:
            data: Dictionary mapping symbols to their historical data
            **kwargs: Additional keyword arguments
                
        Returns:
            Dictionary mapping symbols to their signal dictionaries
        """
        # This should be overridden by concrete strategy implementations
        return {}


class TrendingMarketStrategy(BaseRegimeStrategy):
    """
    Strategy optimized for strong trending markets (both up and down).
    
    This strategy focuses on:
    - Momentum continuation
    - Pullback entries in the direction of the trend
    - Breakouts from consolidation within a trend
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            config, 
            name="TrendingMarketStrategy",
            description="Strategy optimized for strong trending markets"
        )
        
        # Trending market specific parameters
        self.trend_strength_threshold = config.get("trend_strength_threshold", 25)
        self.pullback_threshold = config.get("pullback_threshold", 0.4)
        self.breakout_threshold = config.get("breakout_threshold", 0.02)
        self.momentum_factor = config.get("momentum_factor", 1.2)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals optimized for trending markets.
        
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
            
            # Only generate signals for trending regimes
            if regime not in ["trending_up", "trending_down"]:
                continue
                
            # Calculate indicators for this symbol
            indicators = self._calculate_common_indicators(symbol, market_data)
            if not indicators:
                continue
                
            recent_data = indicators["data"]
            
            # Determine trend direction
            trend_direction = 1 if regime == "trending_up" else -1
            
            # Calculate signal components
            momentum_signal = self._calculate_momentum_signal(recent_data, trend_direction)
            pullback_signal = self._calculate_pullback_signal(recent_data, trend_direction)
            breakout_signal = self._calculate_breakout_signal(recent_data, indicators, trend_direction)
            
            # Combine signals with weights
            signal_strength = (
                0.5 * momentum_signal +
                0.3 * pullback_signal +
                0.2 * breakout_signal
            ) * trend_direction * self.sensitivity
            
            # Apply trend strength filter
            adx_value = recent_data['adx'].iloc[-1] if 'adx' in recent_data else 0
            if adx_value < self.trend_strength_threshold and self.enable_filters:
                signal_strength *= (adx_value / self.trend_strength_threshold)
            
            # Get confirmation count
            confirmation_count = sum([
                momentum_signal * trend_direction > 0.3,
                pullback_signal * trend_direction > 0.3,
                breakout_signal * trend_direction > 0.3
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
                    "adx": adx_value,
                    "momentum_component": momentum_signal,
                    "pullback_component": pullback_signal,
                    "breakout_component": breakout_signal,
                    "confirmation_count": confirmation_count,
                    "trend_direction": trend_direction
                }
            }
        
        return signals
    
    def _calculate_momentum_signal(self, data: pd.DataFrame, trend_direction: int) -> float:
        """Calculate momentum-based signal component."""
        try:
            # Use MACD for momentum
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            macd_hist = macd - macd_signal
            
            # ROC (Rate of Change)
            close_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-20] if len(data) >= 20 else data['close'].iloc[0]
            roc = (close_price - prev_price) / prev_price
            
            # Combine signals
            momentum = (0.6 * np.sign(macd_hist) * min(1.0, abs(macd_hist) * self.momentum_factor) + 
                        0.4 * np.sign(roc) * min(1.0, abs(roc) * 10))
                        
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum signal: {str(e)}")
            return 0.0
    
    def _calculate_pullback_signal(self, data: pd.DataFrame, trend_direction: int) -> float:
        """Calculate pullback entry signal component."""
        try:
            # Identify pullbacks against the trend
            rsi = data['rsi'].iloc[-1]
            
            # For uptrends, look for RSI pullbacks to oversold areas
            if trend_direction > 0:
                # RSI below 40 in uptrend suggests pullback
                pullback_strength = max(0, (40 - rsi) / 30) if rsi < 40 else 0
                
                # Check if price is near SMA20 (common pullback level)
                close = data['close'].iloc[-1]
                sma20 = data['sma20'].iloc[-1]
                near_sma = abs(close - sma20) / sma20 < 0.01
                
                if near_sma:
                    pullback_strength *= 1.5
            
            # For downtrends, look for RSI pullbacks to overbought areas
            else:
                # RSI above 60 in downtrend suggests pullback
                pullback_strength = max(0, (rsi - 60) / 30) if rsi > 60 else 0
                
                # Check if price is near SMA20 (common pullback level)
                close = data['close'].iloc[-1]
                sma20 = data['sma20'].iloc[-1]
                near_sma = abs(close - sma20) / sma20 < 0.01
                
                if near_sma:
                    pullback_strength *= 1.5
            
            return pullback_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating pullback signal: {str(e)}")
            return 0.0
    
    def _calculate_breakout_signal(self, data: pd.DataFrame, indicators: Dict, trend_direction: int) -> float:
        """Calculate breakout signal component."""
        try:
            # Look for volume-confirmed breakouts in trend direction
            high_volume = indicators.get("high_volume", False)
            
            # Get recent volatility
            volatility = data['volatility'].iloc[-1] if 'volatility' in data else 0.01
            
            # Get recent price action
            close = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-2] if len(data) >= 2 else close
            
            # Price movement percentage
            price_change_pct = (close - prev_close) / prev_close
            
            # Significant move relative to volatility
            significant_move = abs(price_change_pct) > (volatility * self.breakout_threshold)
            
            # Direction aligned with trend
            aligned_direction = np.sign(price_change_pct) == trend_direction
            
            if significant_move and aligned_direction:
                breakout_strength = min(1.0, abs(price_change_pct) / (volatility * self.breakout_threshold))
                
                # Volume confirmation increases signal strength
                if high_volume:
                    breakout_strength *= 1.5
                    
                return breakout_strength
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout signal: {str(e)}")
            return 0.0
