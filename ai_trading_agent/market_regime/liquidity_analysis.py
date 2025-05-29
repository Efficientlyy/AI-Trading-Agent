"""
Liquidity Analysis Module

This module provides tools for analyzing market liquidity conditions,
which is important for identifying stress periods, market dislocations,
and potential execution challenges.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats

from .core_definitions import (
    LiquidityRegimeType,
    MarketRegimeConfig
)

# Set up logger
logger = logging.getLogger(__name__)


class LiquidityAnalyzer:
    """
    Class for analyzing market liquidity conditions.
    
    Identifies different liquidity regimes from abundant to crisis levels.
    """
    
    def __init__(self, config: Optional[MarketRegimeConfig] = None):
        """
        Initialize the liquidity analyzer.
        
        Args:
            config: Configuration for liquidity analysis
        """
        self.config = config or MarketRegimeConfig()
        self.liquidity_history = {}
        self.regime_history = []
    
    def analyze_liquidity(self,
                         prices: pd.Series,
                         volume: pd.Series,
                         high_prices: Optional[pd.Series] = None,
                         low_prices: Optional[pd.Series] = None,
                         asset_id: str = "default") -> Dict[str, any]:
        """
        Analyze liquidity conditions based on price and volume data.
        
        Args:
            prices: Series of asset prices (closing prices)
            volume: Series of trading volumes
            high_prices: Optional series of high prices (for volatility calculation)
            low_prices: Optional series of low prices (for volatility calculation)
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary with liquidity analysis results
        """
        if prices is None or volume is None or len(prices) < 30 or len(volume) < 30:
            logger.warning(f"Insufficient data for liquidity analysis: {len(prices) if prices is not None else 0} price points, {len(volume) if volume is not None else 0} volume points")
            return {
                "liquidity_regime": LiquidityRegimeType.UNKNOWN.value,
                "liquidity_score": None,
                "volume_trend": None,
                "amihud_illiquidity": None,
                "volume_volatility_ratio": None,
                "bid_ask_proxy": None
            }
        
        # Ensure indexes match and data is aligned
        if hasattr(prices, 'index') and hasattr(volume, 'index'):
            common_idx = prices.index.intersection(volume.index)
            prices = prices.loc[common_idx]
            volume = volume.loc[common_idx]
            
            if high_prices is not None and low_prices is not None:
                high_prices = high_prices.loc[common_idx]
                low_prices = low_prices.loc[common_idx]
        
        # Calculate returns for liquidity metrics
        returns = prices.pct_change().dropna()
        abs_returns = returns.abs()
        
        # Calculate basic volume metrics
        volume_ma = volume.rolling(window=20).mean()
        current_volume = volume.iloc[-1]
        avg_volume = volume.iloc[-20:].mean() if len(volume) >= 20 else volume.mean()
        relative_volume = current_volume / avg_volume if avg_volume != 0 else 1.0
        
        # Calculate volume trend (slope of linear regression on recent volume)
        volume_trend = self._calculate_trend(volume, window=20)
        
        # Calculate Amihud illiquidity measure: |return| / (price * volume)
        try:
            # Use dollar volume (price * volume) in denominator
            dollar_volume = prices * volume
            amihud_illiquidity = abs_returns / dollar_volume
            avg_amihud = amihud_illiquidity.iloc[-20:].mean()
        except Exception as e:
            logger.warning(f"Error calculating Amihud illiquidity: {str(e)}")
            avg_amihud = None
        
        # Calculate volume/volatility ratio
        # Higher values suggest better liquidity relative to volatility
        try:
            if high_prices is not None and low_prices is not None:
                # Use high-low range as volatility proxy
                volatility = (high_prices / low_prices - 1).iloc[-20:].mean()
            else:
                # Use return standard deviation as volatility proxy
                volatility = returns.iloc[-20:].std()
                
            vol_vol_ratio = (avg_volume / volatility) if volatility != 0 else None
        except Exception as e:
            logger.warning(f"Error calculating volume/volatility ratio: {str(e)}")
            vol_vol_ratio = None
        
        # Estimate bid-ask spread proxy using price volatility
        # This is a rough proxy when actual bid-ask data is unavailable
        try:
            if high_prices is not None and low_prices is not None:
                # Use high-low range as spread proxy
                spread_proxy = (high_prices.iloc[-5:] / low_prices.iloc[-5:] - 1).mean()
            else:
                # Use price volatility as spread proxy
                spread_proxy = returns.iloc[-10:].std() * 2
        except Exception as e:
            logger.warning(f"Error calculating bid-ask proxy: {str(e)}")
            spread_proxy = None
        
        # Order book imbalance proxy (not possible with just OHLCV data)
        # In a real system, this would use actual order book data
        
        # Calculate composite liquidity score
        liquidity_score = self._calculate_liquidity_score(
            relative_volume=relative_volume,
            volume_trend=volume_trend,
            amihud_illiquidity=avg_amihud,
            spread_proxy=spread_proxy
        )
        
        # Determine liquidity regime
        liquidity_regime = self._classify_liquidity_regime(liquidity_score)
        
        # Create results dictionary
        results = {
            "liquidity_regime": liquidity_regime.value,
            "liquidity_score": liquidity_score,
            "relative_volume": relative_volume,
            "volume_trend": volume_trend,
            "amihud_illiquidity": avg_amihud,
            "volume_volatility_ratio": vol_vol_ratio,
            "bid_ask_proxy": spread_proxy
        }
        
        # Track liquidity history
        if asset_id not in self.liquidity_history:
            self.liquidity_history[asset_id] = []
            
        self.liquidity_history[asset_id].append({
            "timestamp": prices.index[-1] if hasattr(prices.index[-1], 'timestamp') else pd.Timestamp.now(),
            "liquidity_score": liquidity_score,
            "liquidity_regime": liquidity_regime.value
        })
        
        # Trim history if too long
        if len(self.liquidity_history[asset_id]) > 1000:
            self.liquidity_history[asset_id] = self.liquidity_history[asset_id][-1000:]
        
        # Track in regime history
        self.regime_history.append({
            "timestamp": prices.index[-1] if hasattr(prices.index[-1], 'timestamp') else pd.Timestamp.now(),
            "asset_id": asset_id,
            "liquidity_regime": liquidity_regime.value,
            "liquidity_score": liquidity_score
        })
        
        # Trim history if too long
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        return results
    
    def _calculate_trend(self, series: pd.Series, window: int = 20) -> float:
        """
        Calculate the trend slope using linear regression.
        
        Args:
            series: Data series to analyze
            window: Window length for trend calculation
            
        Returns:
            Slope of the linear trend
        """
        if len(series) < window:
            return 0.0
            
        try:
            # Use recent data points
            recent_data = series.iloc[-window:].values
            x = np.arange(len(recent_data))
            
            # Calculate slope using linear regression
            slope, _, _, _, _ = stats.linregress(x, recent_data)
            
            # Normalize by the average value to get relative trend
            avg_value = np.mean(recent_data)
            normalized_slope = slope / avg_value if avg_value != 0 else 0.0
            
            return normalized_slope
            
        except Exception as e:
            logger.warning(f"Error calculating trend: {str(e)}")
            return 0.0
    
    def _calculate_liquidity_score(self, 
                                  relative_volume: float,
                                  volume_trend: float,
                                  amihud_illiquidity: Optional[float],
                                  spread_proxy: Optional[float]) -> float:
        """
        Calculate a composite liquidity score.
        
        Score range is 0.0 (very illiquid) to 1.0 (very liquid).
        
        Args:
            relative_volume: Current volume relative to recent average
            volume_trend: Trend in volume
            amihud_illiquidity: Amihud illiquidity measure
            spread_proxy: Proxy for bid-ask spread
            
        Returns:
            Composite liquidity score
        """
        score_components = []
        
        # Volume component (higher is better)
        if relative_volume is not None:
            volume_score = min(1.0, relative_volume / 2.0)
            score_components.append(volume_score)
        
        # Volume trend component (positive trend is good)
        if volume_trend is not None:
            trend_score = 0.5 + min(0.5, max(-0.5, volume_trend * 10))
            score_components.append(trend_score)
        
        # Amihud illiquidity component (lower is better)
        if amihud_illiquidity is not None and not np.isnan(amihud_illiquidity):
            # Transform with negative exponential to get 0-1 score
            # Higher amihud = lower score
            illiq_score = np.exp(-20 * amihud_illiquidity)
            score_components.append(illiq_score)
        
        # Spread proxy component (lower is better)
        if spread_proxy is not None and not np.isnan(spread_proxy):
            # Transform to get 0-1 score
            # Higher spread = lower score
            spread_score = max(0.0, 1.0 - spread_proxy * 20)
            score_components.append(spread_score)
        
        # Calculate average score if components are available
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return 0.5  # Neutral score if no components available
    
    def _classify_liquidity_regime(self, liquidity_score: float) -> LiquidityRegimeType:
        """
        Classify liquidity regime based on liquidity score.
        
        Args:
            liquidity_score: Composite liquidity score (0.0-1.0)
            
        Returns:
            LiquidityRegimeType enum
        """
        if liquidity_score is None:
            return LiquidityRegimeType.UNKNOWN
        elif liquidity_score >= 0.85:
            return LiquidityRegimeType.ABUNDANT
        elif liquidity_score >= 0.65:
            return LiquidityRegimeType.NORMAL
        elif liquidity_score >= 0.45:
            return LiquidityRegimeType.REDUCED
        elif liquidity_score >= 0.25:
            return LiquidityRegimeType.STRESSED
        else:
            return LiquidityRegimeType.CRISIS
    
    def detect_liquidity_shock(self, 
                              current_score: float,
                              prev_score: float,
                              threshold: float = 0.15) -> Tuple[bool, float]:
        """
        Detect if there has been a sudden shock to market liquidity.
        
        Args:
            current_score: Current liquidity score
            prev_score: Previous liquidity score
            threshold: Threshold for shock detection
            
        Returns:
            Tuple of (is_shock, magnitude)
        """
        if current_score is None or prev_score is None:
            return False, 0.0
        
        # Calculate change magnitude
        change = prev_score - current_score  # Positive change = liquidity deterioration
        
        # Shock is a sudden large drop in liquidity
        is_shock = change > threshold
        
        return is_shock, change
    
    def get_liquidity_statistics(self, asset_id: str = "default") -> Dict[str, any]:
        """
        Get summary statistics of liquidity history.
        
        Args:
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary with liquidity statistics
        """
        if asset_id not in self.liquidity_history or not self.liquidity_history[asset_id]:
            return {
                "mean_liquidity": None,
                "min_liquidity": None,
                "max_liquidity": None,
                "liquidity_volatility": None,
                "regime_counts": {}
            }
        
        # Extract liquidity scores and regimes
        liquidity_scores = [entry["liquidity_score"] for entry in self.liquidity_history[asset_id]
                           if entry["liquidity_score"] is not None]
        regimes = [entry["liquidity_regime"] for entry in self.liquidity_history[asset_id]]
        
        if not liquidity_scores:
            return {
                "mean_liquidity": None,
                "min_liquidity": None,
                "max_liquidity": None,
                "liquidity_volatility": None,
                "regime_counts": {}
            }
        
        # Calculate statistics
        stats_dict = {
            "mean_liquidity": np.mean(liquidity_scores),
            "min_liquidity": min(liquidity_scores),
            "max_liquidity": max(liquidity_scores),
            "liquidity_volatility": np.std(liquidity_scores)
        }
        
        # Count regime occurrences
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        stats_dict["regime_counts"] = regime_counts
        
        return stats_dict
