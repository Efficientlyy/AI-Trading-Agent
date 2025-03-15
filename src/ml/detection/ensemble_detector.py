"""Ensemble-based market regime detection algorithm."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, cast
from collections import Counter
import pandas as pd

from .base_detector import BaseRegimeDetector
from .factory import RegimeDetectorFactory


class EnsembleRegimeDetector(BaseRegimeDetector):
    """
    Ensemble-based market regime detection.
    
    This detector combines multiple regime detection methods to produce more robust
    regime classifications. It supports different voting methods (majority, weighted)
    and ensemble techniques (bagging, boosting, stacking).
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_window: int = 60,
        methods: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        voting: str = 'soft',
        ensemble_type: str = 'bagging',
        normalize_outputs: bool = True,
        **kwargs
    ):
        """
        Initialize the ensemble regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (default: 3)
            lookback_window: Window size for lookback period (default: 60)
            methods: List of regime detection methods to use (default: ['volatility', 'momentum', 'trend', 'hmm'])
            weights: List of weights for each method (default: equal weights)
            voting: Voting method ('hard' or 'soft') (default: 'soft')
            ensemble_type: Ensemble technique ('bagging', 'boosting', 'stacking') (default: 'bagging')
            normalize_outputs: Whether to normalize detector outputs (default: True)
            **kwargs: Additional parameters to pass to the individual detectors
        """
        super().__init__(n_regimes=n_regimes, lookback_window=lookback_window, **kwargs)
        
        # Set default methods if not provided
        self.methods = methods or ['volatility', 'momentum', 'trend', 'hmm']
        
        # Set default weights if not provided
        if weights is None:
            self.weights = [1.0] * len(self.methods)
        else:
            if len(weights) != len(self.methods):
                raise ValueError("Number of weights must match number of methods")
            self.weights = weights
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
        
        self.voting = voting.lower()
        if self.voting not in ['hard', 'soft']:
            raise ValueError("Voting method must be 'hard' or 'soft'")
        
        self.ensemble_type = ensemble_type.lower()
        if self.ensemble_type not in ['bagging', 'boosting', 'stacking']:
            raise ValueError("Ensemble type must be 'bagging', 'boosting', or 'stacking'")
        
        self.normalize_outputs = normalize_outputs
        
        # Create detectors
        self.factory = RegimeDetectorFactory()
        self.detectors = []
        self.detector_kwargs = kwargs.copy()
        
        # Create and store detector instances
        self._create_detectors()
        
        # Initialize results storage
        self.individual_labels: List[List[int]] = []
        self.individual_probas: List[np.ndarray] = []
        self.ensemble_probas: Optional[np.ndarray] = None
        
        # Set regime names
        self.regime_names = [f"Ensemble Regime {i}" for i in range(n_regimes)]
    
    def _create_detectors(self) -> None:
        """Create detector instances for each method."""
        self.detectors = []
        
        for method in self.methods:
            # Create detector with common parameters
            detector_kwargs = self.detector_kwargs.copy()
            detector_kwargs['n_regimes'] = self.n_regimes
            detector_kwargs['lookback_window'] = self.lookback_window
            
            # Add method-specific parameters
            if method == 'volatility':
                detector_kwargs.setdefault('vol_window', 21)
            elif method == 'momentum':
                detector_kwargs.setdefault('momentum_type', 'roc')
            elif method == 'trend':
                detector_kwargs.setdefault('trend_method', 'ma_crossover')
            elif method == 'hmm':
                detector_kwargs.setdefault('hmm_type', 'gaussian')
            
            # Create detector
            detector = self.factory.create(method, **detector_kwargs)
            self.detectors.append(detector)
    
    def fit(self, data: Dict[str, Any]) -> None:
        """
        Fit the ensemble regime detector to the data.
        
        Args:
            data: Dictionary containing market data
        """
        # Fit each detector
        for detector in self.detectors:
            detector.fit(data)
        
        # Store dates if available
        if 'dates' in data:
            self.dates = data['dates']
        
        self.fitted = True
    
    def detect(self, data: Dict[str, Any]) -> List[int]:
        """
        Detect regimes using the ensemble method.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of regime labels
        """
        if not self.fitted:
            self.fit(data)
        
        # Get labels from each detector
        self.individual_labels = []
        for detector in self.detectors:
            labels = detector.detect(data)
            self.individual_labels.append(labels)
        
        # Calculate probabilities for each regime
        if self.voting == 'soft':
            self._calculate_soft_voting_probas()
        
        # Apply ensemble method
        if self.ensemble_type == 'bagging':
            self.labels = self._apply_bagging()
        elif self.ensemble_type == 'boosting':
            self.labels = self._apply_boosting(data)
        elif self.ensemble_type == 'stacking':
            self.labels = self._apply_stacking(data)
        else:
            # Default to bagging
            self.labels = self._apply_bagging()
        
        # Calculate regime statistics
        self.calculate_regime_statistics(data, self.labels)
        
        return self.labels
    
    def _calculate_soft_voting_probas(self) -> None:
        """
        Calculate soft voting probabilities for each regime.
        
        This converts the hard labels from each detector into probabilistic
        outputs that can be combined with weights.
        """
        n_samples = len(self.individual_labels[0])
        self.individual_probas = []
        
        for labels in self.individual_labels:
            # Convert labels to one-hot encoded probabilities
            probas = np.zeros((n_samples, self.n_regimes))
            for i, label in enumerate(labels):
                probas[i, label] = 1.0
            self.individual_probas.append(probas)
        
        # Initialize ensemble probabilities
        self.ensemble_probas = np.zeros((n_samples, self.n_regimes))
        
        # Combine probabilities with weights
        for i, probas in enumerate(self.individual_probas):
            weight = self.weights[i]
            self.ensemble_probas += weight * probas
        
        # Normalize if needed
        if self.normalize_outputs:
            row_sums = self.ensemble_probas.sum(axis=1)
            self.ensemble_probas = self.ensemble_probas / row_sums[:, np.newaxis]
    
    def _apply_bagging(self) -> List[int]:
        """
        Apply bagging ensemble method.
        
        Bagging (Bootstrap Aggregating) combines predictions by voting.
        
        Returns:
            List of regime labels
        """
        n_samples = len(self.individual_labels[0])
        labels = []
        
        if self.voting == 'hard':
            # Hard voting: majority vote
            for i in range(n_samples):
                votes = [labels[i] for labels in self.individual_labels]
                weighted_votes = []
                for vote, weight in zip(votes, self.weights):
                    weighted_votes.extend([vote] * int(weight * 100))
                
                # Count votes
                vote_counter = Counter(weighted_votes)
                most_common = vote_counter.most_common(1)
                most_common_label = most_common[0][0] if most_common else 0
                labels.append(most_common_label)
        else:
            # Soft voting: weighted probabilities
            if self.ensemble_probas is None:
                self._calculate_soft_voting_probas()
            
            # Get most probable regime for each sample
            if self.ensemble_probas is not None:
                labels = self.ensemble_probas.argmax(axis=1).tolist()
            else:
                # Fallback to hard voting if probas calculation failed
                return self._apply_hard_voting()
        
        return labels
    
    def _apply_hard_voting(self) -> List[int]:
        """
        Apply hard voting as a fallback method.
        
        Returns:
            List of regime labels
        """
        n_samples = len(self.individual_labels[0])
        labels = []
        
        for i in range(n_samples):
            votes = [method_labels[i] for method_labels in self.individual_labels]
            # Simple majority vote without weighting
            vote_counter = Counter(votes)
            most_common = vote_counter.most_common(1)
            most_common_label = most_common[0][0] if most_common else 0
            labels.append(most_common_label)
        
        return labels
    
    def _apply_boosting(self, data: Dict[str, Any]) -> List[int]:
        """
        Apply boosting ensemble method.
        
        Boosting focuses on samples that are misclassified by previous detectors.
        This is a simplified implementation that weights the predictions based
        on the performance of each detector.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of regime labels
        """
        # For simplicity, use the regime statistics to weight the detectors
        # Better detectors get higher weights
        n_samples = len(self.individual_labels[0])
        
        # Get returns from data
        if 'returns' in data:
            returns = data['returns']
        else:
            # Estimate returns from prices
            prices = np.array(data['prices'])
            returns = np.diff(np.log(prices))
            returns = np.insert(returns, 0, 0.0)
        
        # Calculate "performance" for each detector
        # by measuring how well its regimes correlate with returns
        performances = []
        for i, detector in enumerate(self.detectors):
            labels = self.individual_labels[i]
            regime_returns = {}
            
            # Calculate average return for each regime
            for regime in range(self.n_regimes):
                regime_indices = [j for j, label in enumerate(labels) if label == regime]
                if regime_indices:
                    regime_return = np.mean([returns[j] for j in regime_indices])
                    regime_returns[regime] = regime_return
            
            # Calculate variance of regime returns
            # Higher variance means better differentiation between regimes
            if regime_returns:
                performance = np.var(list(regime_returns.values()))
            else:
                performance = 0.01  # Small default value
            
            performances.append(performance)
        
        # Normalize performances to weights
        total_performance = sum(performances)
        if total_performance > 0:
            weights = [p / total_performance for p in performances]
        else:
            weights = [1.0 / len(performances)] * len(performances)
        
        # Update instance weights
        self.weights = weights
        
        # Apply soft voting with updated weights
        if self.voting == 'soft':
            self._calculate_soft_voting_probas()
            if self.ensemble_probas is not None:
                labels = self.ensemble_probas.argmax(axis=1).tolist()
            else:
                # Fallback to hard voting if probas calculation failed
                labels = self._apply_hard_voting()
        else:
            # Fall back to bagging with updated weights
            labels = self._apply_bagging()
        
        return labels
    
    def _apply_stacking(self, data: Dict[str, Any]) -> List[int]:
        """
        Apply stacking ensemble method.
        
        Stacking trains a meta-model on the outputs of the base detectors.
        This implementation uses a simple linear combination of outputs.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of regime labels
        """
        # Simple stacking: train a meta-model (linear) on detector outputs
        n_samples = len(self.individual_labels[0])
        
        # Create features matrix (detector outputs)
        features = np.zeros((n_samples, len(self.detectors) * self.n_regimes))
        
        for i, detector in enumerate(self.detectors):
            # Get detector probabilities
            if self.individual_probas:
                detector_probas = self.individual_probas[i]
            else:
                # Convert labels to one-hot
                detector_probas = np.zeros((n_samples, self.n_regimes))
                for j, label in enumerate(self.individual_labels[i]):
                    detector_probas[j, label] = 1.0
            
            # Add to features
            start_col = i * self.n_regimes
            end_col = start_col + self.n_regimes
            features[:, start_col:end_col] = detector_probas
        
        # Simple "meta-model": weighted average of features
        meta_weights = np.ones(features.shape[1]) / features.shape[1]
        ensemble_probas = features.dot(meta_weights).reshape(-1, self.n_regimes)
        
        # Normalize
        row_sums = ensemble_probas.sum(axis=1)
        ensemble_probas = ensemble_probas / row_sums[:, np.newaxis]
        
        # Get most probable regime
        labels = ensemble_probas.argmax(axis=1).tolist()
        
        # Store ensemble probabilities
        self.ensemble_probas = ensemble_probas
        
        return labels
    
    def get_ensemble_probas(self) -> Optional[np.ndarray]:
        """
        Get the ensemble probabilities for each regime.
        
        Returns:
            Array of shape (n_samples, n_regimes) or None if not calculated
        """
        return self.ensemble_probas
    
    def get_individual_labels(self) -> List[List[int]]:
        """
        Get the labels from each individual detector.
        
        Returns:
            List of label lists, one for each detector
        """
        return self.individual_labels
    
    def get_detector_weights(self) -> List[float]:
        """
        Get the weights for each detector.
        
        Returns:
            List of weights
        """
        return self.weights
    
    def set_detector_weights(self, weights: List[float]) -> None:
        """
        Set the weights for each detector.
        
        Args:
            weights: List of weights
        """
        if len(weights) != len(self.methods):
            raise ValueError("Number of weights must match number of methods")
        
        # Normalize weights
        weight_sum = sum(weights)
        self.weights = [w / weight_sum for w in weights]
    
    def get_detector_names(self) -> List[str]:
        """
        Get the names of the detectors.
        
        Returns:
            List of detector names
        """
        return self.methods 