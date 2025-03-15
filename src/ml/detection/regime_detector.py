"""
Market regime detector implementation.

This module provides a comprehensive market regime detection system that combines
multiple detection methods to identify different market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
import logging

from src.ml.detection.base_detector import BaseRegimeDetector
from src.ml.detection.trend_detector import TrendRegimeDetector
from src.ml.detection.volatility_detector import VolatilityRegimeDetector
from src.ml.detection.momentum_detector import MomentumRegimeDetector
from src.ml.detection.ensemble_detector import EnsembleRegimeDetector

logger = logging.getLogger(__name__)

class MarketRegimeDetector(BaseRegimeDetector):
    """
    Market regime detector that combines multiple detection methods.
    
    This class implements a comprehensive market regime detection system that
    identifies different market regimes based on trend, volatility, and momentum
    indicators. It uses an ensemble approach to combine the results of multiple
    detection methods.
    
    Detected regimes:
    - 0: Trending Up (low volatility, positive trend)
    - 1: Ranging (low volatility, neutral trend)
    - 2: Trending Down (low volatility, negative trend)
    - 3: Volatile (high volatility)
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        lookback_window: int = 20,
        volatility_window: int = 10,
        trend_threshold: float = 0.01,
        volatility_threshold: float = 0.02,
        momentum_window: int = 14,
        ensemble_method: str = "voting",
        **kwargs
    ):
        """
        Initialize the market regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (default: 4)
            lookback_window: Window size for lookback period (default: 20)
            volatility_window: Window size for volatility calculation (default: 10)
            trend_threshold: Threshold for trend detection (default: 0.01)
            volatility_threshold: Threshold for volatility detection (default: 0.02)
            momentum_window: Window size for momentum calculation (default: 14)
            ensemble_method: Method for ensemble detection (default: "voting")
            **kwargs: Additional parameters
        """
        super().__init__(n_regimes=n_regimes, lookback_window=lookback_window, **kwargs)
        
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.momentum_window = momentum_window
        self.ensemble_method = ensemble_method
        
        # Define regime names
        self.regime_names = [
            "Trending Up",
            "Ranging",
            "Trending Down",
            "Volatile"
        ]
        
        # Initialize detectors
        self.trend_detector = TrendRegimeDetector(
            n_regimes=3,  # Up, Neutral, Down
            lookback_window=lookback_window,
            threshold=trend_threshold
        )
        
        self.volatility_detector = VolatilityRegimeDetector(
            n_regimes=2,  # Low, High
            lookback_window=volatility_window,
            threshold=volatility_threshold
        )
        
        self.momentum_detector = MomentumRegimeDetector(
            n_regimes=3,  # Positive, Neutral, Negative
            lookback_window=momentum_window
        )
        
        # Initialize ensemble detector
        self.ensemble_detector = EnsembleRegimeDetector(
            detectors=[self.trend_detector, self.volatility_detector, self.momentum_detector],
            n_regimes=n_regimes,
            method=ensemble_method
        )
    
    def preprocess_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Preprocess input data to ensure consistent format.
        
        Args:
            data: Input data as DataFrame or dictionary
            
        Returns:
            Dictionary with preprocessed data
        """
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to dictionary
            processed_data = {
                'dates': data.index.tolist(),
                'returns': data['returns'].values if 'returns' in data else data['close'].pct_change().fillna(0).values,
                'prices': data['close'].values,
                'high': data['high'].values if 'high' in data else None,
                'low': data['low'].values if 'low' in data else None,
                'volume': data['volume'].values if 'volume' in data else None
            }
        else:
            # Ensure dictionary has required keys
            processed_data = data.copy()
            if 'returns' not in processed_data and 'prices' in processed_data:
                prices = np.array(processed_data['prices'])
                processed_data['returns'] = np.diff(prices, prepend=prices[0]) / prices
        
        return processed_data
    
    def fit(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
        """
        Fit the detector to the provided data.
        
        Args:
            data: Market data as DataFrame or dictionary
        """
        processed_data = self.preprocess_data(data)
        
        # Fit individual detectors
        self.trend_detector.fit(processed_data)
        self.volatility_detector.fit(processed_data)
        self.momentum_detector.fit(processed_data)
        
        # Fit ensemble detector
        self.ensemble_detector.fit(processed_data)
        
        # Store dates
        self.dates = processed_data['dates']
        
        # Mark as fitted
        self.fitted = True
    
    def detect(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.Series:
        """
        Detect market regimes from price and feature data.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, Dict[str, Any]]
            Either DataFrame with price data or Dict with features
            
        Returns:
        --------
        pd.Series
            Series with regime labels/values
        """
        try:
            logger.info(f"Detecting regimes using MarketRegimeDetector with data type {type(data).__name__}")
            
            # Prepare data based on input type
            if isinstance(data, pd.DataFrame):
                # Start with DataFrame - extract features as needed
                logger.info(f"Input is DataFrame with shape {data.shape}")
                
                # Debug the data structure - what columns do we have?
                logger.info(f"Data columns: {data.columns.tolist()}")
                
                # Ensure we have a valid date index
                if not isinstance(data.index, pd.DatetimeIndex) and 'date' in data.columns:
                    logger.info("Converting 'date' column to index")
                    data = data.set_index('date')
                
                # Debug the data types for key columns
                for col in ['close', 'price', 'returns']:
                    if col in data.columns:
                        logger.info(f"Column '{col}' has dtype {data[col].dtype}")
                
                # Prepare features for the model
                prepared_data = self._prepare_features(data)
                
                # Extract features as a dictionary
                features = {
                    name: prepared_data[name].values 
                    for name in self.feature_names 
                    if name in prepared_data.columns
                }
                
                logger.info(f"Prepared {len(features)} features: {list(features.keys())}")
                
                # Store the original index for returning results
                original_index = prepared_data.index
            else:
                # Already have features dict
                logger.info(f"Input is Dict with {len(data)} features")
                features = data
                
                # We need to create a simple index since we don't have dates
                original_index = pd.RangeIndex(start=0, stop=len(next(iter(data.values()))), step=1)
                
                # Debug the feature names and dimensions
                logger.info(f"Feature names: {list(features.keys())}")
                for name, values in features.items():
                    logger.info(f"Feature '{name}' has shape {np.array(values).shape}")
                
            # Ensure all features have the same length
            feature_lengths = [len(values) for values in features.values()]
            if len(set(feature_lengths)) > 1:
                logger.error(f"Feature length mismatch: {feature_lengths}")
                # Try to fix by trimming to shortest feature
                min_length = min(feature_lengths)
                logger.warning(f"Trimming all features to length {min_length}")
                features = {
                    name: values[:min_length] for name, values in features.items()
                }
            
            # Get predictions from hmm_model
            if not hasattr(self, 'hmm_model') or self.hmm_model is None:
                logger.error("HMM model not initialized - run fit() first")
                # Handle missing model by returning dummy regime values
                logger.warning("Returning dummy regime values")
                regimes = pd.Series([1] * len(original_index), index=original_index, name='regime')
                return regimes
            
            # Get predictions from the model
            logger.info("Predicting regimes from HMM model")
            regime_values = self.hmm_model.predict(features)
            
            # Debug regime predictions
            logger.info(f"Regime prediction type: {type(regime_values).__name__}")
            logger.info(f"Regime prediction shape: {len(regime_values)}")
            logger.info(f"Unique regimes: {np.unique(regime_values, return_counts=True)}")
            
            # Map state numbers to meaningful regime names if available
            if hasattr(self, 'regime_names') and len(self.regime_names) > 0:
                logger.info(f"Mapping regime values to names: {self.regime_names}")
                
                # Ensure regime values are within valid range
                max_regime = max(regime_values) if len(regime_values) > 0 else 0
                if max_regime >= len(self.regime_names):
                    logger.warning(f"Regime value {max_regime} exceeds available names, adding generic names")
                    # Add generic names for any missing regimes
                    while len(self.regime_names) <= max_regime:
                        self.regime_names.append(f"Regime {len(self.regime_names)}")
            
            # Create a Series with the predictions and the original index
            if len(regime_values) != len(original_index):
                logger.warning(f"Length mismatch: regimes {len(regime_values)}, index {len(original_index)}")
                # Handle length mismatch
                if len(regime_values) < len(original_index):
                    # Pad with NaN values
                    padded_regimes = np.full(len(original_index), np.nan)
                    padded_regimes[:len(regime_values)] = regime_values
                    regime_values = padded_regimes
                else:
                    # Truncate
                    regime_values = regime_values[:len(original_index)]
            
            # Create Series with meaningful index
            regimes = pd.Series(regime_values, index=original_index, name='regime')
            
            # Ensure we don't have NaN values in the regime Series
            if regimes.isna().any():
                logger.warning(f"Found {regimes.isna().sum()} NaN values in regime Series, filling with forward fill")
                regimes = regimes.fillna(method='ffill').fillna(method='bfill')
                
                # If we still have NaN values (e.g., all values were NaN), fill with a default value
                if regimes.isna().any():
                    logger.warning("Still have NaN values after fill attempts, using default value 0")
                    regimes = regimes.fillna(0)
            
            # Debug final regimes
            logger.info(f"Final regime Series length: {len(regimes)}")
            logger.info(f"Regime value counts: {regimes.value_counts().to_dict()}")
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            logger.exception("Exception details:")
            
            # Return a dummy Series in case of error to avoid breaking dependent code
            if isinstance(data, pd.DataFrame):
                dummy_index = data.index
                dummy_length = len(data)
            else:
                dummy_length = len(next(iter(data.values())))
                dummy_index = range(dummy_length)
                
            logger.warning(f"Returning dummy regime values of length {dummy_length}")
            return pd.Series([0] * dummy_length, index=dummy_index, name='regime')
    
    def get_regime_probabilities(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Get the probability of each regime for the provided data.
        
        Args:
            data: Market data as DataFrame or dictionary
            
        Returns:
            Array of probabilities for each regime at each time point
        """
        if not self.fitted:
            self.fit(data)
        
        processed_data = self.preprocess_data(data)
        
        # Get probabilities from ensemble detector
        return self.ensemble_detector.get_ensemble_probas()
    
    def get_current_regime(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[int, str]:
        """
        Get the current market regime.
        
        Args:
            data: Market data as DataFrame or dictionary
            
        Returns:
            Tuple of (regime_label, regime_name)
        """
        labels = self.detect(data)
        current_label = labels.iloc[-1]
        current_name = self.get_regime_name(current_label)
        
        return current_label, current_name
    
    def get_regime_transitions(self) -> List[Tuple[datetime, int, int]]:
        """
        Get the list of regime transitions.
        
        Returns:
            List of tuples (date, from_regime, to_regime)
        """
        if not self.fitted or not self.labels or not self.dates:
            return []
        
        transitions = []
        for i in range(1, len(self.labels)):
            if self.labels[i] != self.labels[i-1]:
                transitions.append((
                    self.dates[i],
                    self.labels[i-1],
                    self.labels[i]
                ))
        
        return transitions
    
    def get_regime_duration_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate statistics about regime durations.
        
        Returns:
            Dictionary with duration statistics for each regime
        """
        if not self.fitted or not self.labels:
            return {}
        
        # Find continuous segments of each regime
        labels = np.array(self.labels)
        regime_durations = {i: [] for i in range(self.n_regimes)}
        
        current_regime = labels[0]
        current_duration = 1
        
        for i in range(1, len(labels)):
            if labels[i] == current_regime:
                current_duration += 1
            else:
                regime_durations[current_regime].append(current_duration)
                current_regime = labels[i]
                current_duration = 1
        
        # Add the last segment
        regime_durations[current_regime].append(current_duration)
        
        # Calculate statistics
        stats = {}
        for regime, durations in regime_durations.items():
            if durations:
                stats[regime] = {
                    "mean_duration": float(np.mean(durations)),
                    "median_duration": float(np.median(durations)),
                    "min_duration": float(np.min(durations)),
                    "max_duration": float(np.max(durations)),
                    "std_duration": float(np.std(durations)),
                    "count": len(durations)
                }
        
        return stats
    
    def analyze_regime_transition(self, historical_data: np.ndarray) -> Dict[str, List[float]]:
        """
        Analyze regime transitions in historical data.
        
        This method is used to study how market regimes evolve over time,
        helping to predict future regime changes and optimize trading strategies.
        
        Args:
            historical_data: Array of historical market data
            
        Returns:
            Dict mapping transition types to probability sequences
        """
        # TODO: Implement regime transition analysis
        pass


class RegimeDetectorFactory:
    """
    Factory class for creating different types of regime detectors.
    
    This class provides a unified interface for creating various types
    of market regime detectors based on specified parameters.
    """
    
    @staticmethod
    def create(detector_type: str = "ensemble", **kwargs) -> BaseRegimeDetector:
        """
        Create a regime detector of the specified type.
        
        Args:
            detector_type: Type of detector to create
                           ("trend", "volatility", "momentum", "ensemble", "hmm")
            **kwargs: Additional parameters to pass to the detector constructor
            
        Returns:
            BaseRegimeDetector: An instance of the requested detector type
            
        Raises:
            ValueError: If an invalid detector type is specified
        """
        detector_type = detector_type.lower()
        
        if detector_type == "trend":
            return TrendRegimeDetector(**kwargs)
        elif detector_type == "volatility":
            return VolatilityRegimeDetector(**kwargs)
        elif detector_type == "momentum":
            return MomentumRegimeDetector(**kwargs)
        elif detector_type == "ensemble":
            return EnsembleRegimeDetector(**kwargs)
        elif detector_type == "market":
            return MarketRegimeDetector(**kwargs)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
