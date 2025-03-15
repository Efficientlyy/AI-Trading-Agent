"""Trend-based market regime detection algorithm."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Sequence, cast
from sklearn.cluster import KMeans
import pandas as pd

from .base_detector import BaseRegimeDetector


class TrendRegimeDetector(BaseRegimeDetector):
    """
    Trend-based market regime detection.
    
    This detector identifies market regimes based on trend strength and direction
    using technical indicators such as moving averages, ADX (Average Directional Index),
    and linear regression slopes.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_window: int = 60,
        trend_method: str = "ma_crossover",
        fast_ma: int = 20,
        slow_ma: int = 50,
        adx_window: int = 14,
        adx_threshold: float = 25.0,
        slope_window: int = 20,
        use_log_prices: bool = False,
        **kwargs
    ):
        """
        Initialize the trend regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (default: 3)
            lookback_window: Window size for lookback period (default: 60)
            trend_method: Method for trend detection ('ma_crossover', 'adx', or 'slope') (default: 'ma_crossover')
            fast_ma: Fast moving average window (default: 20)
            slow_ma: Slow moving average window (default: 50)
            adx_window: Window for ADX calculation (default: 14)
            adx_threshold: Threshold for ADX to indicate trend (default: 25.0)
            slope_window: Window for slope calculation (default: 20)
            use_log_prices: Whether to use log prices for calculations (default: False)
            **kwargs: Additional parameters
        """
        super().__init__(n_regimes=n_regimes, lookback_window=lookback_window, **kwargs)
        self.trend_method = trend_method.lower()
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold
        self.slope_window = slope_window
        self.use_log_prices = use_log_prices
        
        # Set regime names based on number of regimes and trend method
        if self.trend_method == "ma_crossover":
            if n_regimes == 2:
                self.regime_names = ["Downtrend", "Uptrend"]
            elif n_regimes == 3:
                self.regime_names = ["Downtrend", "Sideways", "Uptrend"]
            elif n_regimes == 4:
                self.regime_names = ["Strong Downtrend", "Weak Downtrend", "Weak Uptrend", "Strong Uptrend"]
            else:
                self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        elif self.trend_method == "adx":
            if n_regimes == 2:
                self.regime_names = ["Ranging", "Trending"]
            elif n_regimes == 3:
                self.regime_names = ["Ranging", "Weak Trend", "Strong Trend"]
            elif n_regimes == 4:
                self.regime_names = ["Ranging", "Weak Downtrend", "Weak Uptrend", "Strong Trend"]
            else:
                self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        elif self.trend_method == "slope":
            if n_regimes == 2:
                self.regime_names = ["Negative Slope", "Positive Slope"]
            elif n_regimes == 3:
                self.regime_names = ["Negative Slope", "Flat", "Positive Slope"]
            elif n_regimes == 4:
                self.regime_names = ["Steep Negative", "Shallow Negative", "Shallow Positive", "Steep Positive"]
            else:
                self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        else:
            self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        
        # Initialize model and thresholds
        self.kmeans = None
        self.trend_thresholds: Optional[List[float]] = None
        self.trend_series: Optional[np.ndarray] = None
    
    def _calculate_moving_averages(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate fast and slow moving averages.
        
        Args:
            prices: Array of prices
            
        Returns:
            Tuple of (fast_ma, slow_ma)
        """
        # Convert to pandas Series for easier calculations
        price_series = pd.Series(prices)
        
        # Calculate moving averages
        fast_ma = price_series.rolling(window=self.fast_ma).mean()
        slow_ma = price_series.rolling(window=self.slow_ma).mean()
        
        # Fill NaN values at the beginning
        fast_ma = fast_ma.fillna(fast_ma.iloc[-1] if len(fast_ma) > 0 else 0)
        slow_ma = slow_ma.fillna(slow_ma.iloc[-1] if len(slow_ma) > 0 else 0)
        
        return np.array(fast_ma.values), np.array(slow_ma.values)
    
    def _calculate_ma_crossover(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate trend indicator based on moving average crossover.
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of trend indicator values
        """
        # Calculate moving averages
        fast_ma, slow_ma = self._calculate_moving_averages(prices)
        
        # Calculate MA difference (fast - slow)
        ma_diff = fast_ma - slow_ma
        
        # Calculate rate of change of MA difference
        ma_diff_roc = np.zeros_like(ma_diff)
        ma_diff_roc[1:] = ma_diff[1:] - ma_diff[:-1]
        
        # Combine MA difference and its rate of change
        # Positive values indicate uptrend, negative values indicate downtrend
        # The magnitude indicates trend strength
        trend_indicator = ma_diff + 0.5 * ma_diff_roc
        
        return trend_indicator
    
    def _calculate_adx(self, prices: np.ndarray, highs: Optional[np.ndarray] = None, lows: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate ADX (Average Directional Index) for trend strength.
        
        Args:
            prices: Array of closing prices
            highs: Array of high prices (optional)
            lows: Array of low prices (optional)
            
        Returns:
            Array of ADX values
        """
        # If highs and lows are not provided, estimate them from closing prices
        if highs is None or lows is None:
            # Estimate high and low prices with a simple approximation
            # In a real implementation, actual high and low prices should be used
            price_volatility = np.std(prices) * 0.5
            highs = prices + price_volatility
            lows = prices - price_volatility
        
        # Convert to pandas Series for easier calculations
        close_series = pd.Series(prices)
        high_series = pd.Series(highs)
        low_series = pd.Series(lows)
        
        # Calculate price changes
        price_diff = close_series.diff()
        
        # Calculate +DM and -DM
        up_move = high_series.diff()
        down_move = low_series.diff()
        
        # Calculate +DI and -DI
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0))
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0))
        
        # Calculate True Range
        tr1 = high_series - low_series
        tr2 = np.abs(high_series - close_series.shift(1))
        tr3 = np.abs(low_series - close_series.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate smoothed +DM, -DM, and TR
        window = self.adx_window
        smoothed_plus_dm = plus_dm.rolling(window=window).sum()
        smoothed_minus_dm = minus_dm.rolling(window=window).sum()
        smoothed_tr = tr.rolling(window=window).sum()
        
        # Calculate +DI and -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # Calculate DX
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
        
        # Calculate ADX - convert to pandas Series first
        dx_series = pd.Series(dx)
        adx = dx_series.rolling(window=window).mean()
        
        # Fill NaN values
        adx = adx.fillna(0)
        
        # Create trend indicator
        # Positive values indicate uptrend, negative values indicate downtrend
        # The magnitude indicates trend strength
        di_diff = plus_di - minus_di
        trend_indicator = di_diff * adx / 100
        
        return np.array(trend_indicator.values)
    
    def _calculate_slope(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate trend indicator based on linear regression slope.
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of slope values
        """
        # Convert to pandas Series for easier calculations
        price_series = pd.Series(prices)
        
        # Calculate rolling slope
        window = self.slope_window
        slopes = np.zeros_like(prices)
        
        for i in range(window, len(prices) + 1):
            y = price_series.iloc[i - window:i].values
            x = np.arange(window).astype(float)
            
            # Calculate slope using linear regression
            # Convert inputs to float arrays to satisfy polyfit requirements
            y_float = y.astype(float)
            slope, _ = np.polyfit(x, y_float, 1)
            slopes[i - 1] = slope
        
        # Fill initial values
        slopes[:window - 1] = slopes[window - 1]
        
        # Normalize slopes
        mean_price = np.mean(prices)
        slopes = slopes / mean_price * 100  # Express as percentage of mean price
        
        return slopes
    
    def _calculate_trend_indicator(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate trend indicator based on the selected method.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Array of trend indicator values
        """
        if 'prices' not in data:
            raise ValueError("Data must contain 'prices' for trend calculation")
        
        prices = np.array(data['prices'])
        
        # Apply log transformation if specified
        if self.use_log_prices:
            prices = np.log(prices)
        
        # Calculate trend indicator based on selected method
        if self.trend_method == "ma_crossover":
            return self._calculate_ma_crossover(prices)
        elif self.trend_method == "adx":
            # Use high and low prices if available
            highs = np.array(data['highs']) if 'highs' in data else None
            lows = np.array(data['lows']) if 'lows' in data else None
            return self._calculate_adx(prices, highs, lows)
        elif self.trend_method == "slope":
            return self._calculate_slope(prices)
        else:
            raise ValueError(f"Unknown trend method: {self.trend_method}")
    
    def _cluster_trend(self, trend: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster trend values into regimes.
        
        Args:
            trend: Array of trend indicator values
            
        Returns:
            Tuple of (labels, cluster_centers)
        """
        # Reshape for KMeans
        X = trend.reshape(-1, 1)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        kmeans.fit(X)
        
        # Get labels and centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()
        
        # Sort regimes by trend level (0 = lowest trend)
        sorted_indices = np.argsort(centers)
        mapping = {old: new for new, old in enumerate(sorted_indices)}
        
        # Remap labels
        remapped_labels = np.array([mapping[label] for label in labels])
        sorted_centers = centers[sorted_indices]
        
        return remapped_labels, sorted_centers
    
    def fit(self, data: Dict[str, Any]) -> None:
        """
        Fit the trend regime detector to the data.
        
        Args:
            data: Dictionary containing market data
        """
        # Calculate trend indicator
        trend = self._calculate_trend_indicator(data)
        self.trend_series = trend
        
        # Cluster trend
        _, centers = self._cluster_trend(trend)
        
        # Store thresholds between regimes
        thresholds: List[float] = []
        for i in range(len(centers) - 1):
            threshold = float((centers[i] + centers[i + 1]) / 2)
            thresholds.append(threshold)
        
        self.trend_thresholds = thresholds
        self.fitted = True
        
        # Store dates if available
        if 'dates' in data:
            self.dates = data['dates']
    
    def detect(self, data: Dict[str, Any]) -> List[int]:
        """
        Detect trend regimes in the data.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of regime labels
        """
        if not self.fitted and self.trend_thresholds is None:
            self.fit(data)
        
        # Calculate trend indicator
        trend = self._calculate_trend_indicator(data)
        
        # Assign regimes based on thresholds
        labels = np.zeros(len(trend), dtype=int)
        
        if self.trend_thresholds:
            for i, threshold in enumerate(self.trend_thresholds):
                labels[trend > threshold] = i + 1
        
        # Store results
        self.labels = [int(label) for label in labels]
        self.trend_series = trend
        
        # Calculate regime statistics
        self.calculate_regime_statistics(data, self.labels)
        
        return self.labels
    
    def get_trend_series(self) -> Optional[np.ndarray]:
        """
        Get the calculated trend series.
        
        Returns:
            Array of trend values or None if not calculated
        """
        return self.trend_series
    
    def get_regime_thresholds(self) -> Optional[List[float]]:
        """
        Get the trend thresholds between regimes.
        
        Returns:
            List of threshold values or None if not fitted
        """
        return self.trend_thresholds 