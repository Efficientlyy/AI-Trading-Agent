"""Momentum-based market regime detection algorithm."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Sequence, cast
from sklearn.cluster import KMeans
import pandas as pd

from .base_detector import BaseRegimeDetector


class MomentumRegimeDetector(BaseRegimeDetector):
    """
    Momentum-based market regime detection.
    
    This detector identifies market regimes based on momentum indicators,
    such as moving average convergence/divergence (MACD), rate of change (ROC),
    or relative strength index (RSI).
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_window: int = 60,
        momentum_type: str = "roc",
        fast_window: int = 12,
        slow_window: int = 26,
        signal_window: int = 9,
        roc_window: int = 20,
        rsi_window: int = 14,
        **kwargs
    ):
        """
        Initialize the momentum regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (default: 3)
            lookback_window: Window size for lookback period (default: 60)
            momentum_type: Type of momentum indicator to use ('macd', 'roc', or 'rsi') (default: 'roc')
            fast_window: Fast EMA window for MACD (default: 12)
            slow_window: Slow EMA window for MACD (default: 26)
            signal_window: Signal line window for MACD (default: 9)
            roc_window: Window for Rate of Change calculation (default: 20)
            rsi_window: Window for RSI calculation (default: 14)
            **kwargs: Additional parameters
        """
        super().__init__(n_regimes=n_regimes, lookback_window=lookback_window, **kwargs)
        self.momentum_type = momentum_type.lower()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_window = signal_window
        self.roc_window = roc_window
        self.rsi_window = rsi_window
        
        # Set regime names based on number of regimes and momentum type
        if self.momentum_type == "macd":
            if n_regimes == 2:
                self.regime_names = ["Bearish", "Bullish"]
            elif n_regimes == 3:
                self.regime_names = ["Bearish", "Neutral", "Bullish"]
            else:
                self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        elif self.momentum_type == "roc":
            if n_regimes == 2:
                self.regime_names = ["Negative Momentum", "Positive Momentum"]
            elif n_regimes == 3:
                self.regime_names = ["Negative Momentum", "Neutral", "Positive Momentum"]
            else:
                self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        elif self.momentum_type == "rsi":
            if n_regimes == 2:
                self.regime_names = ["Oversold", "Overbought"]
            elif n_regimes == 3:
                self.regime_names = ["Oversold", "Neutral", "Overbought"]
            else:
                self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        else:
            self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        
        # Initialize model and thresholds
        self.kmeans = None
        self.momentum_thresholds: Optional[List[float]] = None
        self.momentum_series: Optional[np.ndarray] = None
    
    def _calculate_macd(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate MACD (Moving Average Convergence/Divergence).
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of MACD values
        """
        # Convert to pandas Series for easier EMA calculations
        price_series = pd.Series(prices)
        
        # Calculate fast and slow EMAs
        fast_ema = price_series.ewm(span=self.fast_window, adjust=False).mean()
        slow_ema = price_series.ewm(span=self.slow_window, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_window, adjust=False).mean()
        
        # Calculate MACD histogram (MACD line - signal line)
        macd_histogram = macd_line - signal_line
        
        # Return MACD histogram as momentum indicator
        return np.array(macd_histogram.values)
    
    def _calculate_roc(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate ROC (Rate of Change).
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of ROC values
        """
        # Convert to pandas Series for easier calculations
        price_series = pd.Series(prices)
        
        # Calculate ROC
        roc = price_series.pct_change(periods=self.roc_window) * 100
        
        # Fill NaN values at the beginning
        roc = roc.fillna(0)
        
        return np.array(roc.values)
    
    def _calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of RSI values
        """
        # Convert to pandas Series for easier calculations
        price_series = pd.Series(prices)
        
        # Calculate price changes
        delta = price_series.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Fill NaN values at the beginning
        rsi = rsi.fillna(50)  # Neutral value
        
        return np.array(rsi.values)
    
    def _calculate_momentum(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate momentum indicator based on the selected type.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Array of momentum values
        """
        if 'prices' not in data:
            raise ValueError("Data must contain 'prices' for momentum calculation")
        
        prices = np.array(data['prices'])
        
        if self.momentum_type == "macd":
            return self._calculate_macd(prices)
        elif self.momentum_type == "roc":
            return self._calculate_roc(prices)
        elif self.momentum_type == "rsi":
            return self._calculate_rsi(prices)
        else:
            raise ValueError(f"Unknown momentum type: {self.momentum_type}")
    
    def _cluster_momentum(self, momentum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster momentum values into regimes.
        
        Args:
            momentum: Array of momentum values
            
        Returns:
            Tuple of (labels, cluster_centers)
        """
        # Reshape for KMeans
        X = momentum.reshape(-1, 1)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        kmeans.fit(X)
        
        # Get labels and centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()
        
        # Sort regimes by momentum level (0 = lowest momentum)
        sorted_indices = np.argsort(centers)
        mapping = {old: new for new, old in enumerate(sorted_indices)}
        
        # Remap labels
        remapped_labels = np.array([mapping[label] for label in labels])
        sorted_centers = centers[sorted_indices]
        
        return remapped_labels, sorted_centers
    
    def fit(self, data: Dict[str, Any]) -> None:
        """
        Fit the momentum regime detector to the data.
        
        Args:
            data: Dictionary containing market data
        """
        # Calculate momentum
        momentum = self._calculate_momentum(data)
        self.momentum_series = momentum
        
        # Cluster momentum
        _, centers = self._cluster_momentum(momentum)
        
        # Store thresholds between regimes
        thresholds: List[float] = []
        for i in range(len(centers) - 1):
            threshold = float((centers[i] + centers[i + 1]) / 2)
            thresholds.append(threshold)
        
        self.momentum_thresholds = thresholds
        self.fitted = True
        
        # Store dates if available
        if 'dates' in data:
            self.dates = data['dates']
    
    def detect(self, data: Dict[str, Any]) -> List[int]:
        """
        Detect momentum regimes in the data.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of regime labels
        """
        if not self.fitted and self.momentum_thresholds is None:
            self.fit(data)
        
        # Calculate momentum
        momentum = self._calculate_momentum(data)
        
        # Assign regimes based on thresholds
        labels = np.zeros(len(momentum), dtype=int)
        
        if self.momentum_thresholds:
            for i, threshold in enumerate(self.momentum_thresholds):
                labels[momentum > threshold] = i + 1
        
        # Store results
        self.labels = [int(label) for label in labels]
        self.momentum_series = momentum
        
        # Calculate regime statistics
        self.calculate_regime_statistics(data, self.labels)
        
        return self.labels
    
    def get_momentum_series(self) -> Optional[np.ndarray]:
        """
        Get the calculated momentum series.
        
        Returns:
            Array of momentum values or None if not calculated
        """
        return self.momentum_series
    
    def get_regime_thresholds(self) -> Optional[List[float]]:
        """
        Get the momentum thresholds between regimes.
        
        Returns:
            List of threshold values or None if not fitted
        """
        return self.momentum_thresholds 