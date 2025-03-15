"""Volatility-based market regime detection algorithm."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Sequence, cast
from sklearn.cluster import KMeans
import pandas as pd

from .base_detector import BaseRegimeDetector


class VolatilityRegimeDetector(BaseRegimeDetector):
    """
    Volatility-based market regime detection.
    
    This detector identifies market regimes based on volatility levels using
    rolling standard deviation of returns and clustering techniques.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_window: int = 60,
        vol_window: int = 21,
        use_log_returns: bool = True,
        use_ewm: bool = False,
        ewm_alpha: float = 0.1,
        **kwargs
    ):
        """
        Initialize the volatility regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (default: 3)
            lookback_window: Window size for lookback period (default: 60)
            vol_window: Window size for volatility calculation (default: 21)
            use_log_returns: Whether to use log returns for volatility calculation (default: True)
            use_ewm: Whether to use exponentially weighted moving average for volatility (default: False)
            ewm_alpha: Alpha parameter for EWM volatility (default: 0.1)
            **kwargs: Additional parameters
        """
        super().__init__(n_regimes=n_regimes, lookback_window=lookback_window, **kwargs)
        self.vol_window = vol_window
        self.use_log_returns = use_log_returns
        self.use_ewm = use_ewm
        self.ewm_alpha = ewm_alpha
        
        # Set regime names based on number of regimes
        if n_regimes == 2:
            self.regime_names = ["Low Volatility", "High Volatility"]
        elif n_regimes == 3:
            self.regime_names = ["Low Volatility", "Normal Volatility", "High Volatility"]
        elif n_regimes == 4:
            self.regime_names = ["Very Low Volatility", "Low Volatility", "High Volatility", "Very High Volatility"]
        else:
            self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        
        # Initialize model and thresholds
        self.kmeans = None
        self.vol_thresholds: Optional[List[float]] = None
        self.volatility_series: Optional[np.ndarray] = None
    
    def _calculate_volatility(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate rolling volatility from returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Array of volatility values
        """
        # Convert to pandas Series for easier rolling calculations
        returns_series = pd.Series(returns)
        
        # Calculate volatility
        if self.use_ewm:
            # Exponentially weighted volatility
            volatility = returns_series.ewm(alpha=self.ewm_alpha).std()
        else:
            # Simple rolling volatility
            volatility = returns_series.rolling(window=self.vol_window).std()
        
        # Fill NaN values at the beginning
        # Use pandas backfill method
        volatility = volatility.fillna(volatility.iloc[-1] if len(volatility) > 0 else 0)
        
        # Explicitly convert to numpy array
        return np.array(volatility.values)
    
    def _prepare_returns(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare returns data for volatility calculation.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Array of prepared returns
        """
        if 'returns' not in data:
            if 'prices' not in data:
                raise ValueError("Data must contain either 'returns' or 'prices'")
            
            # Calculate returns from prices
            prices = np.array(data['prices'])
            returns = np.diff(prices) / prices[:-1]
            returns = np.insert(returns, 0, 0)  # Add 0 at the beginning
        else:
            returns = np.array(data['returns'])
        
        # Use log returns if specified
        if self.use_log_returns:
            # Convert simple returns to log returns
            # log(1 + r) is approximately equal to r for small r
            returns = np.log1p(returns)
        
        return returns
    
    def _cluster_volatility(self, volatility: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster volatility values into regimes.
        
        Args:
            volatility: Array of volatility values
            
        Returns:
            Tuple of (labels, cluster_centers)
        """
        # Reshape for KMeans
        X = volatility.reshape(-1, 1)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        kmeans.fit(X)
        
        # Get labels and centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()
        
        # Sort regimes by volatility level (0 = lowest volatility)
        sorted_indices = np.argsort(centers)
        mapping = {old: new for new, old in enumerate(sorted_indices)}
        
        # Remap labels
        remapped_labels = np.array([mapping[label] for label in labels])
        sorted_centers = centers[sorted_indices]
        
        return remapped_labels, sorted_centers
    
    def fit(self, data: Dict[str, Any]) -> None:
        """
        Fit the volatility regime detector to the data.
        
        Args:
            data: Dictionary containing market data
        """
        # Prepare returns
        returns = self._prepare_returns(data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(returns)
        self.volatility_series = volatility
        
        # Cluster volatility
        _, centers = self._cluster_volatility(volatility)
        
        # Store thresholds between regimes
        thresholds: List[float] = []
        for i in range(len(centers) - 1):
            threshold = float((centers[i] + centers[i + 1]) / 2)
            thresholds.append(threshold)
        
        self.vol_thresholds = thresholds
        self.fitted = True
        
        # Store dates if available
        if 'dates' in data:
            self.dates = data['dates']
    
    def detect(self, data: Dict[str, Any]) -> List[int]:
        """
        Detect volatility regimes in the data.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of regime labels
        """
        if not self.fitted and self.vol_thresholds is None:
            self.fit(data)
        
        # Prepare returns
        returns = self._prepare_returns(data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(returns)
        
        # Assign regimes based on thresholds
        labels = np.zeros(len(volatility), dtype=int)
        
        if self.vol_thresholds:
            for i, threshold in enumerate(self.vol_thresholds):
                labels[volatility > threshold] = i + 1
        
        # Store results
        self.labels = [int(label) for label in labels]
        self.volatility_series = volatility
        
        # Calculate regime statistics
        self.calculate_regime_statistics(data, self.labels)
        
        return self.labels
    
    def get_volatility_series(self) -> Optional[np.ndarray]:
        """
        Get the calculated volatility series.
        
        Returns:
            Array of volatility values or None if not calculated
        """
        return self.volatility_series
    
    def get_regime_thresholds(self) -> Optional[List[float]]:
        """
        Get the volatility thresholds between regimes.
        
        Returns:
            List of threshold values or None if not fitted
        """
        return self.vol_thresholds 