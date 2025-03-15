"""Base class for market regime detection algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
import numpy as np
import pandas as pd
from datetime import datetime


class BaseRegimeDetector(ABC):
    """
    Abstract base class for all market regime detection algorithms.
    
    This class defines the interface that all regime detection algorithms must implement.
    It provides common functionality and ensures consistent behavior across different
    detection methods.
    """
    
    def __init__(self, n_regimes: int = 3, lookback_window: int = 60, **kwargs):
        """
        Initialize the regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (default: 3)
            lookback_window: Window size for lookback period (default: 60)
            **kwargs: Additional parameters specific to the detection method
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.fitted = False
        self.labels: List[int] = []
        self.dates: Optional[List[datetime]] = None
        self.regime_stats: Dict[Union[int, str], Dict[str, float]] = {}
        
        # Default regime names
        self.regime_names: List[str] = [f"Regime {i}" for i in range(n_regimes)]
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def detect(self, data: Dict[str, Any]) -> List[int]:
        """
        Detect market regimes in the provided data.
        
        Args:
            data: Dictionary containing market data with keys like 'dates', 'prices', 'returns', etc.
            
        Returns:
            List of integer labels representing detected regimes (0 to n_regimes-1)
        """
        pass
    
    @abstractmethod
    def fit(self, data: Dict[str, Any]) -> None:
        """
        Fit the detector to the provided data.
        
        Args:
            data: Dictionary containing market data with keys like 'dates', 'prices', 'returns', etc.
        """
        pass
    
    def predict(self, data: Dict[str, Any]) -> List[int]:
        """
        Predict regimes for new data based on previously fitted model.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of integer labels representing detected regimes
        """
        if not self.fitted:
            raise ValueError("Detector must be fitted before prediction")
        return self.detect(data)
    
    def fit_predict(self, data: Dict[str, Any]) -> List[int]:
        """
        Fit the detector and predict regimes in one step.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of integer labels representing detected regimes
        """
        self.fit(data)
        return self.detect(data)
    
    def calculate_regime_statistics(self, data: Dict[str, Any], labels: List[int]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each detected regime.
        
        Args:
            data: Dictionary containing market data
            labels: List of regime labels
            
        Returns:
            Dictionary with statistics for each regime
        """
        if 'returns' not in data:
            raise ValueError("Data must contain 'returns' for calculating statistics")
        
        returns = np.array(data['returns'])
        labels_array = np.array(labels)
        
        stats = {}
        for i in range(self.n_regimes):
            regime_returns = returns[labels_array == i]
            if len(regime_returns) > 0:
                stats[f"regime_{i}"] = {
                    "count": int(np.sum(labels_array == i)),
                    "percentage": float(np.sum(labels_array == i) / len(labels_array)),
                    "mean_return": float(np.mean(regime_returns)),
                    "std_return": float(np.std(regime_returns)),
                    "sharpe": float(np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0),
                    "min_return": float(np.min(regime_returns)),
                    "max_return": float(np.max(regime_returns))
                }
        
        # Calculate transitions
        transitions = int(np.sum(np.diff(labels_array) != 0))
        stats["transitions"] = transitions
        stats["transitions_per_period"] = float(transitions / len(labels_array))
        
        self.regime_stats = stats
        return stats
    
    def get_regime_at_date(self, date: datetime) -> Optional[int]:
        """
        Get the regime at a specific date.
        
        Args:
            date: The date to check
            
        Returns:
            Regime label or None if date not found
        """
        if self.labels is None or not hasattr(self, 'dates'):
            return None
        
        # Find the closest date
        dates = getattr(self, 'dates')
        if isinstance(dates[0], str):
            dates = [datetime.fromisoformat(d) for d in dates]
        
        for i, d in enumerate(dates):
            if d >= date:
                return self.labels[i]
        
        return None
    
    def get_regime_name(self, label: int) -> str:
        """
        Get the name of a regime.
        
        Args:
            label: Regime label
            
        Returns:
            Regime name
        """
        if not hasattr(self, 'regime_names') or self.regime_names is None:
            if 0 <= label < self.n_regimes:
                return f"Regime {label}"
            else:
                return f"Regime {label}"
        else:
            if 0 <= label < len(self.regime_names):
                return self.regime_names[label]
            else:
                return f"Regime {label}"
    
    def __str__(self) -> str:
        """Return string representation of the detector."""
        return f"{self.__class__.__name__}(n_regimes={self.n_regimes}, lookback_window={self.lookback_window})"
    
    def get_regime_thresholds(self) -> Optional[List[float]]:
        """
        Get the thresholds between regimes.
        
        Returns:
            List of threshold values or None if not fitted
        """
        return None
    
    def get_trend_series(self) -> Optional[np.ndarray]:
        """
        Get the calculated trend/indicator series.
        
        Returns:
            Array of values or None if not calculated
        """
        return None
    
    def get_ensemble_probas(self) -> Optional[np.ndarray]:
        """
        Get the ensemble probabilities for each regime.
        
        Returns:
            Array of shape (n_samples, n_regimes) or None if not calculated
        """
        return None
    
    def get_detector_weights(self) -> List[float]:
        """
        Get the weights for each detector.
        
        Returns:
            List of weights or empty list if not applicable
        """
        return []
    
    def get_detector_names(self) -> List[str]:
        """
        Get the names of the detectors.
        
        Returns:
            List of detector names or empty list if not applicable
        """
        return [] 