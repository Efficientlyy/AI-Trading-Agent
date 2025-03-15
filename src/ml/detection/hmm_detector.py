"""Hidden Markov Model (HMM) based market regime detection algorithm."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Sequence, cast, Union
from hmmlearn import hmm
import pandas as pd
import warnings

from .base_detector import BaseRegimeDetector


class HMMRegimeDetector(BaseRegimeDetector):
    """
    Hidden Markov Model (HMM) based market regime detection.
    
    This detector identifies market regimes using Hidden Markov Models,
    which can capture the underlying hidden states (regimes) that generate
    the observed market data.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_window: int = 60,
        hmm_type: str = "gaussian",
        n_iter: int = 100,
        random_state: int = 42,
        use_returns: bool = True,
        use_log_returns: bool = True,
        **kwargs
    ):
        """
        Initialize the HMM regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (default: 3)
            lookback_window: Window size for lookback period (default: 60)
            hmm_type: Type of HMM to use ('gaussian', 'gmm', or 'multinomial') (default: 'gaussian')
            n_iter: Number of iterations for HMM training (default: 100)
            random_state: Random state for reproducibility (default: 42)
            use_returns: Whether to use returns instead of prices (default: True)
            use_log_returns: Whether to use log returns (default: True)
            **kwargs: Additional parameters
        """
        super().__init__(n_regimes=n_regimes, lookback_window=lookback_window, **kwargs)
        self.hmm_type = hmm_type.lower()
        self.n_iter = n_iter
        self.random_state = random_state
        self.use_returns = use_returns
        self.use_log_returns = use_log_returns
        
        # Set regime names
        self.regime_names = [f"Regime {i}" for i in range(n_regimes)]
        
        # Initialize model
        self.model: Optional[Union[hmm.GaussianHMM, hmm.GMMHMM]] = None
        self.means: Optional[np.ndarray] = None
        self.covars: Optional[np.ndarray] = None
        self.transmat: Optional[np.ndarray] = None
        self.labels: List[int] = []  # Initialize as empty list instead of None
    
    def _prepare_data(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare data for HMM training.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Array of prepared data
        """
        if self.use_returns:
            if 'returns' not in data:
                if 'prices' not in data:
                    raise ValueError("Data must contain either 'returns' or 'prices'")
                
                # Calculate returns from prices
                prices = np.array(data['prices'])
                returns = np.diff(prices) / prices[:-1]
                
                # Use log returns if specified
                if self.use_log_returns:
                    returns = np.log1p(returns)
                
                # Reshape for HMM
                X = returns.reshape(-1, 1)
            else:
                returns = np.array(data['returns'])
                
                # Use log returns if specified
                if self.use_log_returns and not getattr(self, '_log_returns_applied', False):
                    returns = np.log1p(returns)
                    self._log_returns_applied = True
                
                # Reshape for HMM
                X = returns.reshape(-1, 1)
        else:
            if 'prices' not in data:
                raise ValueError("Data must contain 'prices' when use_returns=False")
            
            # Use prices directly
            prices = np.array(data['prices'])
            
            # Reshape for HMM
            X = prices.reshape(-1, 1)
        
        return X
    
    def _create_hmm_model(self) -> Union[hmm.GaussianHMM, hmm.GMMHMM]:
        """
        Create an HMM model based on the specified type.
        
        Returns:
            HMM model
        """
        if self.hmm_type == "gaussian":
            model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state
            )
        elif self.hmm_type == "gmm":
            model = hmm.GMMHMM(
                n_components=self.n_regimes,
                n_mix=2,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported HMM type: {self.hmm_type}")
        
        return model
    
    def _sort_regimes(self, model: Union[hmm.GaussianHMM, hmm.GMMHMM], X: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Sort regimes by volatility (standard deviation) and remap labels.
        
        Args:
            model: Trained HMM model
            X: Input data
            
        Returns:
            Tuple of (remapped_labels, sorted_means)
        """
        # Get means and standard deviations for each regime
        means = model.means_.flatten()
        covars = np.array([np.sqrt(np.diag(c)) for c in model.covars_])
        stds = covars.flatten()
        
        # Get original labels
        labels = model.predict(X)
        
        # Sort regimes by volatility (standard deviation)
        sorted_indices = np.argsort(stds)
        mapping = {old: new for new, old in enumerate(sorted_indices)}
        
        # Remap labels
        remapped_labels = [mapping[label] for label in labels]
        sorted_means = means[sorted_indices]
        
        return remapped_labels, sorted_means
    
    def fit(self, data: Dict[str, Any]) -> None:
        """
        Fit the HMM regime detector to the data.
        
        Args:
            data: Dictionary containing market data
        """
        # Prepare data
        X = self._prepare_data(data)
        
        # Create and fit HMM model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._create_hmm_model()
            model.fit(X)
        
        # Store model parameters
        self.model = model
        self.means = model.means_
        self.covars = model.covars_
        self.transmat = model.transmat_
        
        # Sort regimes and get labels
        self.labels, _ = self._sort_regimes(model, X)
        
        self.fitted = True
        
        # Store dates if available
        if 'dates' in data:
            self.dates = data['dates']
    
    def detect(self, data: Dict[str, Any]) -> List[int]:
        """
        Detect regimes using the fitted HMM model.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            List of regime labels
        """
        if not self.fitted or self.model is None:
            self.fit(data)
            return self.labels
        
        # Prepare data
        X = self._prepare_data(data)
        
        # Predict regimes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = self.model.predict(X)
        
        # Sort regimes and remap labels
        remapped_labels, _ = self._sort_regimes(self.model, X)
        
        # Store results
        self.labels = remapped_labels
        
        # Calculate regime statistics
        self.calculate_regime_statistics(data, self.labels)
        
        return self.labels
    
    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """
        Get the transition probability matrix between regimes.
        
        Returns:
            Transition matrix or None if not fitted
        """
        return self.transmat
    
    def get_regime_means(self) -> Optional[np.ndarray]:
        """
        Get the mean values for each regime.
        
        Returns:
            Array of mean values or None if not fitted
        """
        return self.means
    
    def get_regime_covariances(self) -> Optional[np.ndarray]:
        """
        Get the covariance matrices for each regime.
        
        Returns:
            Array of covariance matrices or None if not fitted
        """
        return self.covars
    
    def predict_next_regime(self) -> Optional[int]:
        """
        Predict the next regime based on the current regime and transition matrix.
        
        Returns:
            Predicted next regime or None if not fitted
        """
        if not self.fitted or self.model is None or not self.labels or self.transmat is None:
            return None
        
        # Get current regime
        current_regime = self.labels[-1]
        
        # Get transition probabilities for current regime
        transition_probs = self.transmat[current_regime]
        
        # Return most likely next regime
        return int(np.argmax(transition_probs)) 