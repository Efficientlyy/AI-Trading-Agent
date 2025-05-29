"""
Momentum Factor Analysis Module

This module provides tools for analyzing momentum in financial markets,
which is a critical factor in determining market regimes.

Momentum analysis helps identify trending vs. mean-reverting regimes and
the strength and persistence of market trends.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
import pandas_ta as ta

from .core_definitions import (
    MarketRegimeType,
    MarketRegimeConfig
)

# Set up logger
logger = logging.getLogger(__name__)


class MomentumFactorAnalyzer:
    """
    Class for analyzing momentum factors in financial time series.
    
    Uses various technical indicators and statistical methods to quantify momentum
    and identify trending vs. mean-reverting market regimes.
    """
    
    def __init__(self, config: Optional[MarketRegimeConfig] = None):
        """
        Initialize the momentum factor analyzer.
        
        Args:
            config: Configuration for momentum analysis
        """
        self.config = config or MarketRegimeConfig()
        self.momentum_history = {}
        self.regime_history = []
    
    def analyze_momentum(self, 
                         prices: pd.Series,
                         volume: Optional[pd.Series] = None,
                         asset_id: str = "default") -> Dict[str, any]:
        """
        Analyze momentum in price series with optional volume data.
        
        Args:
            prices: Series of asset prices
            volume: Optional volume data
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary with momentum analysis metrics
        """
        if prices is None or len(prices) < max(self.config.long_lookback, 50):
            logger.warning(f"Insufficient data for momentum analysis: {len(prices) if prices is not None else 0} points")
            return {
                "market_regime": MarketRegimeType.UNKNOWN.value,
                "momentum_score": 0.0,
                "trend_strength": 0.0,
                "mean_reversion_score": 0.0,
                "momentum_indicators": {}
            }
        
        # Calculate momentum indicators
        indicators = self._calculate_momentum_indicators(prices, volume)
        
        # Determine trend strength based on ADX
        trend_strength = indicators.get('adx', 0.0) / 100.0
        
        # Calculate momentum score (range -1.0 to 1.0)
        momentum_score = self._calculate_composite_momentum_score(indicators)
        
        # Mean reversion score is inversely related to momentum strength
        mean_reversion_score = max(0.0, 1.0 - abs(momentum_score))
        
        # Determine market regime based on momentum and trend strength
        market_regime = self._classify_momentum_regime(momentum_score, trend_strength)
        
        # Create results dictionary
        results = {
            "market_regime": market_regime.value,
            "momentum_score": momentum_score,
            "trend_strength": trend_strength,
            "mean_reversion_score": mean_reversion_score,
            "momentum_indicators": indicators
        }
        
        # Track momentum history
        if asset_id not in self.momentum_history:
            self.momentum_history[asset_id] = []
            
        self.momentum_history[asset_id].append({
            "timestamp": prices.index[-1] if hasattr(prices.index[-1], 'timestamp') else pd.Timestamp.now(),
            "momentum_score": momentum_score,
            "trend_strength": trend_strength,
            "market_regime": market_regime.value
        })
        
        # Trim history if too long
        if len(self.momentum_history[asset_id]) > 1000:
            self.momentum_history[asset_id] = self.momentum_history[asset_id][-1000:]
        
        return results
    
    def _calculate_momentum_indicators(self, 
                                      prices: pd.Series, 
                                      volume: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate various momentum indicators.
        
        Args:
            prices: Series of asset prices
            volume: Optional volume data
            
        Returns:
            Dictionary with indicator values
        """
        # Convert to numpy for talib compatibility
        price_array = np.array(prices.values.astype(float))
        
        try:
            # Calculate RSI
            rsi = ta.RSI(price_array, timeperiod=14)[-1]
            
            # Calculate moving averages
            sma_fast = ta.SMA(price_array, timeperiod=self.config.short_lookback)[-1]
            sma_slow = ta.SMA(price_array, timeperiod=self.config.medium_lookback)[-1]
            sma_very_slow = ta.SMA(price_array, timeperiod=self.config.long_lookback)[-1]
            
            # Calculate MA crossover signals
            fast_vs_slow = sma_fast / sma_slow - 1.0 if sma_slow != 0 else 0
            fast_vs_very_slow = sma_fast / sma_very_slow - 1.0 if sma_very_slow != 0 else 0
            slow_vs_very_slow = sma_slow / sma_very_slow - 1.0 if sma_very_slow != 0 else 0
            
            # Calculate price momentum (rate of change)
            mom_short = ta.MOM(price_array, timeperiod=self.config.short_lookback)[-1] / price_array[-1] if price_array[-1] != 0 else 0
            mom_medium = ta.MOM(price_array, timeperiod=self.config.medium_lookback)[-1] / price_array[-1] if price_array[-1] != 0 else 0
            mom_long = ta.MOM(price_array, timeperiod=self.config.long_lookback)[-1] / price_array[-1] if price_array[-1] != 0 else 0
            
            # Calculate MACD
            macd, macd_signal, macd_hist = ta.MACD(price_array, fastperiod=12, slowperiod=26, signalperiod=9)
            macd_value = macd[-1]
            macd_signal = macd_signal[-1]
            macd_hist = macd_hist[-1]
            
            # Calculate ADX (Average Directional Index) for trend strength
            adx = ta.ADX(price_array, price_array, price_array, timeperiod=14)[-1]
            
            # Calculate CCI (Commodity Channel Index) for overbought/oversold
            cci = ta.CCI(price_array, price_array, price_array, timeperiod=20)[-1]
            
            # Calculate Aroon indicators for trend identification
            aroon_down, aroon_up = ta.AROON(price_array, price_array, timeperiod=25)
            aroon_oscillator = aroon_up[-1] - aroon_down[-1]
            
            # Calculate volume indicators if volume data is available
            if volume is not None and len(volume) >= len(prices):
                volume_array = np.array(volume.values.astype(float))
                obv = ta.OBV(price_array, volume_array)[-1]
                volume_sma = ta.SMA(volume_array, timeperiod=20)[-1]
                relative_volume = volume_array[-1] / volume_sma if volume_sma != 0 else 1.0
            else:
                obv = None
                relative_volume = None
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return {}
            
        # Compile indicators
        indicators = {
            "rsi": rsi if 'rsi' in locals() and rsi is not None and not np.isnan(rsi) else 50.0,
            "fast_vs_slow_ma": fast_vs_slow if 'fast_vs_slow' in locals() and fast_vs_slow is not None and not np.isnan(fast_vs_slow) else 0.0,
            "fast_vs_very_slow_ma": fast_vs_very_slow if 'fast_vs_very_slow' in locals() and fast_vs_very_slow is not None and not np.isnan(fast_vs_very_slow) else 0.0,
            "slow_vs_very_slow_ma": slow_vs_very_slow if 'slow_vs_very_slow' in locals() and slow_vs_very_slow is not None and not np.isnan(slow_vs_very_slow) else 0.0,
            "mom_short": mom_short if 'mom_short' in locals() and mom_short is not None and not np.isnan(mom_short) else 0.0,
            "mom_medium": mom_medium if 'mom_medium' in locals() and mom_medium is not None and not np.isnan(mom_medium) else 0.0,
            "mom_long": mom_long if 'mom_long' in locals() and mom_long is not None and not np.isnan(mom_long) else 0.0,
            "macd": macd_value if 'macd_value' in locals() and macd_value is not None and not np.isnan(macd_value) else 0.0,
            "macd_signal": macd_signal if 'macd_signal' in locals() and macd_signal is not None and not np.isnan(macd_signal) else 0.0,
            "macd_hist": macd_hist if 'macd_hist' in locals() and macd_hist is not None and not np.isnan(macd_hist) else 0.0,
            "adx": adx if 'adx' in locals() and adx is not None and not np.isnan(adx) else 0.0,
            "cci": cci if 'cci' in locals() and cci is not None and not np.isnan(cci) else 0.0,
            "aroon_oscillator": aroon_oscillator if 'aroon_oscillator' in locals() and aroon_oscillator is not None and not np.isnan(aroon_oscillator) else 0.0,
            "obv": obv if 'obv' in locals() and obv is not None and not np.isnan(obv) else None,
            "relative_volume": relative_volume if 'relative_volume' in locals() and relative_volume is not None and not np.isnan(relative_volume) else None
        }
        
        return indicators
    
    def _calculate_composite_momentum_score(self, indicators: Dict[str, float]) -> float:
        """
        Calculate a composite momentum score from indicators.
        
        Range is from -1.0 (strong downtrend) to 1.0 (strong uptrend).
        
        Args:
            indicators: Dictionary of indicator values
            
        Returns:
            Composite momentum score
        """
        # Extract indicators with defaults for missing values
        rsi = indicators.get('rsi', 50.0)
        fast_vs_slow_ma = indicators.get('fast_vs_slow_ma', 0.0)
        fast_vs_very_slow_ma = indicators.get('fast_vs_very_slow_ma', 0.0)
        mom_short = indicators.get('mom_short', 0.0)
        mom_medium = indicators.get('mom_medium', 0.0)
        macd = indicators.get('macd', 0.0)
        macd_hist = indicators.get('macd_hist', 0.0)
        aroon_oscillator = indicators.get('aroon_oscillator', 0.0)
        
        # Normalize RSI from 0-100 to -1 to 1 scale
        rsi_normalized = (rsi - 50) / 50
        
        # Normalize Aroon Oscillator from -100-100 to -1 to 1 scale
        aroon_normalized = aroon_oscillator / 100
        
        # Calculate weighted score
        score = (
            0.25 * rsi_normalized +
            0.15 * np.clip(fast_vs_slow_ma * 20, -1, 1) +
            0.15 * np.clip(fast_vs_very_slow_ma * 10, -1, 1) +
            0.10 * np.clip(mom_short * 10, -1, 1) +
            0.10 * np.clip(mom_medium * 5, -1, 1) +
            0.10 * np.sign(macd) * min(1.0, abs(macd) / 2.0) +
            0.10 * np.sign(macd_hist) * min(1.0, abs(macd_hist) / 0.5) +
            0.05 * aroon_normalized
        )
        
        # Ensure score is within -1 to 1 range
        return max(-1.0, min(1.0, score))
    
    def _classify_momentum_regime(self, 
                                 momentum_score: float, 
                                 trend_strength: float) -> MarketRegimeType:
        """
        Classify market regime based on momentum score and trend strength.
        
        Args:
            momentum_score: Composite momentum score (-1.0 to 1.0)
            trend_strength: Strength of trend (0.0 to 1.0)
            
        Returns:
            MarketRegimeType enum
        """
        abs_momentum = abs(momentum_score)
        
        if trend_strength < 0.15:
            # Low trend strength indicates sideways/choppy market
            if abs_momentum < 0.2:
                return MarketRegimeType.SIDEWAYS
            else:
                return MarketRegimeType.CHOPPY
        elif trend_strength >= 0.15 and trend_strength < 0.4:
            # Moderate trend strength
            if momentum_score > 0.3:
                return MarketRegimeType.BULL
            elif momentum_score < -0.3:
                return MarketRegimeType.BEAR
            else:
                return MarketRegimeType.TRENDING
        else:
            # Strong trend
            if momentum_score > 0.3:
                return MarketRegimeType.BULL
            elif momentum_score < -0.3:
                return MarketRegimeType.BEAR
            elif momentum_score > 0.1:
                return MarketRegimeType.RECOVERY
            elif momentum_score < -0.1:
                return MarketRegimeType.BREAKDOWN
            else:
                return MarketRegimeType.TRENDING
    
    def detect_regime_change(self,
                            current_score: float,
                            prev_score: float,
                            threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Detect if there has been a significant change in momentum regime.
        
        Args:
            current_score: Current momentum score
            prev_score: Previous momentum score
            threshold: Custom threshold for regime change detection
            
        Returns:
            Tuple of (has_changed, change_magnitude)
        """
        if current_score is None or prev_score is None:
            return False, 0.0
        
        # Use provided threshold or default from config
        change_threshold = threshold or self.config.regime_change_threshold
        
        # Calculate absolute change
        change_magnitude = abs(current_score - prev_score)
        has_changed = change_magnitude > change_threshold
        
        # Sign flip is always significant
        if current_score * prev_score < 0 and abs(current_score) > 0.2 and abs(prev_score) > 0.2:
            has_changed = True
        
        return has_changed, change_magnitude
    
    def get_momentum_statistics(self, 
                              asset_id: str = "default") -> Dict[str, any]:
        """
        Get summary statistics of momentum history.
        
        Args:
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary with momentum statistics
        """
        if asset_id not in self.momentum_history or len(self.momentum_history[asset_id]) == 0:
            return {
                "mean_momentum": None,
                "median_momentum": None,
                "mean_trend_strength": None,
                "regime_counts": {}
            }
        
        # Extract momentum scores and trend strengths
        momentum_scores = [entry["momentum_score"] for entry in self.momentum_history[asset_id]]
        trend_strengths = [entry["trend_strength"] for entry in self.momentum_history[asset_id]]
        regimes = [entry["market_regime"] for entry in self.momentum_history[asset_id]]
        
        # Calculate statistics
        stats_dict = {
            "mean_momentum": np.mean(momentum_scores),
            "median_momentum": np.median(momentum_scores),
            "mean_trend_strength": np.mean(trend_strengths),
            "momentum_std": np.std(momentum_scores)
        }
        
        # Count regime occurrences
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        stats_dict["regime_counts"] = regime_counts
        
        return stats_dict
