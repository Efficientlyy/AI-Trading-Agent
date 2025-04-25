"""
signal_processing/regime.py

Provides market regime detection utilities for identifying trending, ranging, and volatile markets.
These regimes are important for adapting trading strategies and signal weights.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    UNKNOWN = 'unknown'


def volatility_regime(prices: pd.Series, window: int = 20, threshold: float = 0.02) -> pd.Series:
    """
    Simple volatility-based regime detector.
    Returns 'high_vol' or 'low_vol' for each timestamp.
    
    Args:
        prices: Price series (typically close prices)
        window: Rolling window size for volatility calculation
        threshold: Volatility threshold for regime classification
        
    Returns:
        Series with 'high_vol' or 'low_vol' labels
    """
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window=window).std()
    regime = np.where(rolling_vol > threshold, 'high_vol', 'low_vol')
    return pd.Series(regime, index=prices.index)


def rolling_kmeans_regime(prices: pd.Series, window: int = 60, n_clusters: int = 2) -> pd.Series:
    """
    Regime detection using rolling k-means clustering on returns volatility.
    Returns regime labels (0, 1, ...) for each timestamp.
    
    Args:
        prices: Price series (typically close prices)
        window: Rolling window size for volatility calculation
        n_clusters: Number of regime clusters to identify
        
    Returns:
        Series with integer regime labels
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.error("sklearn is required for rolling_kmeans_regime")
        return pd.Series(index=prices.index, dtype=int)
        
    returns = prices.pct_change().rolling(window=window).std().dropna()
    regimes = pd.Series(index=prices.index, dtype=int)
    if len(returns) < window:
        return regimes  # Not enough data
    X = returns.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    regimes.iloc[-len(labels):] = labels
    return regimes


class MarketRegimeDetector:
    """
    Detects market regimes (trending, ranging, volatile) based on price action.
    
    This class provides methods to analyze price data and determine the current
    market regime, which can be used to adapt trading strategies and signal weights.
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        volatility_threshold: float = 0.015,
        trend_threshold: float = 0.6,
        range_threshold: float = 0.3
    ):
        """
        Initialize the market regime detector.
        
        Args:
            volatility_window: Window size for volatility calculation
            trend_window: Window size for trend strength calculation
            volatility_threshold: Threshold for classifying as volatile
            trend_threshold: Threshold for classifying as trending
            range_threshold: Threshold for classifying as ranging
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
    
    def detect_regime(self, price_data: Union[pd.DataFrame, pd.Series]) -> MarketRegime:
        """
        Detect the current market regime based on price data.
        
        Args:
            price_data: DataFrame with OHLCV data or Series of close prices
            
        Returns:
            MarketRegime enum value
        """
        # Extract close prices if a DataFrame is provided
        if isinstance(price_data, pd.DataFrame):
            if 'close' in price_data.columns:
                close_prices = price_data['close']
            else:
                logger.warning("No 'close' column found in price_data DataFrame, using last column")
                close_prices = price_data.iloc[:, -1]
        else:
            close_prices = price_data
        
        # Check if we have enough data
        if len(close_prices) < max(self.volatility_window, self.trend_window):
            logger.warning(f"Not enough data for regime detection. Need at least {max(self.volatility_window, self.trend_window)} data points.")
            return MarketRegime.UNKNOWN
        
        # Calculate volatility (annualized standard deviation of returns)
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(window=self.volatility_window).std().iloc[-1] * np.sqrt(252)
        
        # Check if market is volatile
        if volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(close_prices)
        
        # Determine regime based on trend strength
        if trend_strength > self.trend_threshold:
            return MarketRegime.TRENDING
        elif trend_strength < self.range_threshold:
            return MarketRegime.RANGING
        else:
            # If in between thresholds, use additional metrics
            return self._determine_regime_with_additional_metrics(close_prices, returns)
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """
        Calculate the strength of the trend using R-squared of linear regression.
        
        Args:
            prices: Series of price data
            
        Returns:
            Trend strength as a value between 0 and 1
        """
        try:
            from scipy import stats
        except ImportError:
            logger.error("scipy is required for trend strength calculation")
            return 0.5
        
        # Get the last window of prices
        window_prices = prices.iloc[-self.trend_window:]
        
        # Create x values (0, 1, 2, ...)
        x = np.arange(len(window_prices))
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_prices.values)
        
        # R-squared value indicates trend strength
        r_squared = r_value ** 2
        
        return r_squared
    
    def _determine_regime_with_additional_metrics(self, prices: pd.Series, returns: pd.Series) -> MarketRegime:
        """
        Use additional metrics to determine the market regime when trend strength is inconclusive.
        
        Args:
            prices: Series of price data
            returns: Series of price returns
            
        Returns:
            MarketRegime enum value
        """
        # Calculate mean reversion tendency using autocorrelation
        autocorr = returns.autocorr(lag=1)
        
        # Negative autocorrelation suggests mean reversion (ranging)
        if autocorr < -0.2:
            return MarketRegime.RANGING
        
        # Positive autocorrelation suggests trend continuation
        elif autocorr > 0.2:
            return MarketRegime.TRENDING
        
        # Check for price compression (ranging indicator)
        recent_high = prices.rolling(window=self.volatility_window).max().iloc[-1]
        recent_low = prices.rolling(window=self.volatility_window).min().iloc[-1]
        price_range_pct = (recent_high - recent_low) / recent_low
        
        if price_range_pct < 0.05:  # Less than 5% range
            return MarketRegime.RANGING
        
        # Default to trending if no other conditions are met
        return MarketRegime.TRENDING
    
    def get_regime_history(self, price_data: Union[pd.DataFrame, pd.Series], window: int = 100) -> pd.Series:
        """
        Get historical market regimes over a rolling window.
        
        Args:
            price_data: DataFrame with OHLCV data or Series of close prices
            window: Minimum window size for regime calculation
            
        Returns:
            Series with MarketRegime values for each timestamp
        """
        # Extract close prices if a DataFrame is provided
        if isinstance(price_data, pd.DataFrame):
            if 'close' in price_data.columns:
                close_prices = price_data['close']
            else:
                close_prices = price_data.iloc[:, -1]
        else:
            close_prices = price_data
        
        # Initialize empty series for regimes
        regimes = pd.Series(index=close_prices.index, dtype='object')
        regimes[:] = MarketRegime.UNKNOWN
        
        # Need at least window data points to start
        if len(close_prices) < window:
            return regimes
        
        # Calculate regimes for each point with enough history
        for i in range(window, len(close_prices)):
            # Get historical data up to this point
            historical_data = close_prices.iloc[:i+1]
            
            # Create a temporary detector with the same parameters
            temp_detector = MarketRegimeDetector(
                volatility_window=min(self.volatility_window, len(historical_data) - 1),
                trend_window=min(self.trend_window, len(historical_data) - 1),
                volatility_threshold=self.volatility_threshold,
                trend_threshold=self.trend_threshold,
                range_threshold=self.range_threshold
            )
            
            # Detect regime
            regime = temp_detector.detect_regime(historical_data)
            regimes.iloc[i] = regime
        
        return regimes
    
    def get_regime_parameters(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get recommended parameters for a given market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Dictionary of recommended parameters for the regime
        """
        if regime == MarketRegime.TRENDING:
            return {
                'stop_loss_pct': 0.05,           # Wider stops in trending markets
                'take_profit_pct': 0.15,         # Larger targets in trending markets
                'position_size_pct': 1.0,        # Full position size in trending markets
                'sentiment_weight': 0.6,         # Medium-high sentiment weight
                'technical_weight': 0.8,         # High technical weight
                'trailing_stop': True            # Use trailing stops in trending markets
            }
        elif regime == MarketRegime.RANGING:
            return {
                'stop_loss_pct': 0.03,           # Tighter stops in ranging markets
                'take_profit_pct': 0.06,         # Smaller targets in ranging markets
                'position_size_pct': 0.7,        # Reduced position size in ranging markets
                'sentiment_weight': 0.3,         # Lower sentiment weight
                'technical_weight': 0.9,         # Higher technical weight
                'trailing_stop': False           # Fixed stops in ranging markets
            }
        elif regime == MarketRegime.VOLATILE:
            return {
                'stop_loss_pct': 0.08,           # Wider stops in volatile markets
                'take_profit_pct': 0.12,         # Medium targets in volatile markets
                'position_size_pct': 0.5,        # Reduced position size in volatile markets
                'sentiment_weight': 0.2,         # Low sentiment weight
                'technical_weight': 0.7,         # Medium technical weight
                'trailing_stop': True            # Use trailing stops in volatile markets
            }
        else:  # UNKNOWN
            return {
                'stop_loss_pct': 0.05,           # Default stop loss
                'take_profit_pct': 0.1,          # Default take profit
                'position_size_pct': 0.7,        # Default position size
                'sentiment_weight': 0.4,         # Default sentiment weight
                'technical_weight': 0.8,         # Default technical weight
                'trailing_stop': False           # Default stop type
            }
