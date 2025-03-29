"""
Base Regime Detector

This module defines the base class for all market regime detection algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseRegimeDetector(ABC):
    """
    Abstract base class for all market regime detectors.
    
    This class defines the common interface that all regime detectors must implement.
    Specific implementations should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize the regime detector.
        
        Parameters:
        -----------
        name : str
            Name of the detector
        config : Dict, optional
            Configuration parameters for the detector
        """
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self.regimes = None
        self.regime_probabilities = None
        
        # Set up logger
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Initialize configuration with default values
        self._init_config()
    
    def _init_config(self):
        """Initialize configuration with default values."""
        default_config = {
            'n_regimes': 3,
            'window_size': 20,
            'min_samples': 100,
            'random_state': 42,
            'regime_names': ['bull', 'bear', 'neutral']
        }
        
        # Update default config with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseRegimeDetector':
        """
        Fit the regime detector to the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with at least columns for price and date/time
            
        Returns:
        --------
        self : BaseRegimeDetector
            Fitted detector
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes for the given data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with at least columns for price and date/time
            
        Returns:
        --------
        np.ndarray
            Array of regime labels
        """
        pass
    
    def fit_predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit the detector and predict regimes in one step.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with at least columns for price and date/time
            
        Returns:
        --------
        np.ndarray
            Array of regime labels
        """
        self.fit(data)
        return self.predict(data)
    
    @abstractmethod
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities for the given data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with at least columns for price and date/time
            
        Returns:
        --------
        np.ndarray
            Array of regime probabilities with shape (n_samples, n_regimes)
        """
        pass
    
    def get_regime_names(self) -> List[str]:
        """
        Get the names of the regimes.
        
        Returns:
        --------
        List[str]
            List of regime names
        """
        return self.config.get('regime_names', [f'regime_{i}' for i in range(self.config['n_regimes'])])
    
    def get_transition_matrix(self, regimes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate the transition matrix between regimes.
        
        Parameters:
        -----------
        regimes : np.ndarray, optional
            Array of regime labels. If None, use the regimes from the last fit.
            
        Returns:
        --------
        pd.DataFrame
            Transition matrix as a DataFrame
        """
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regimes available. Call fit() first.")
            regimes = self.regimes
        
        # Get unique regimes
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        
        # Initialize transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            
            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]
            
            transition_matrix[from_idx, to_idx] += 1
        
        # Normalize by row
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        # Create DataFrame
        regime_names = self.get_regime_names()
        if len(regime_names) < n_regimes:
            # Extend regime names if needed
            regime_names.extend([f'regime_{i}' for i in range(len(regime_names), n_regimes)])
        
        # Use only the regime names that appear in the data
        regime_names = [regime_names[i] for i in range(n_regimes)]
        
        return pd.DataFrame(
            transition_matrix,
            index=regime_names,
            columns=regime_names
        )
    
    def get_regime_stats(self, data: pd.DataFrame, regimes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate statistics for each regime.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with at least columns for price and returns
        regimes : np.ndarray, optional
            Array of regime labels. If None, use the regimes from the last fit.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with statistics for each regime
        """
        # NOTE: Temporarily commented out for testing notification thresholds
        # This should cause a significant coverage drop (>1%) for the detection component
        """
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regimes available. Call fit() first.")
            regimes = self.regimes
        
        if len(regimes) != len(data):
            raise ValueError("Length of regimes must match length of data.")
        
        # Ensure data has returns column
        if 'returns' not in data.columns:
            if 'close' in data.columns:
                data = data.copy()
                data["returns"] = data['close'].pct_change()
            else:
                raise ValueError("Data must have 'returns' or 'close' column.")
        
        # Get unique regimes
        unique_regimes = np.unique(regimes)
        
        # Calculate statistics for each regime
        stats = []
        
        for regime in unique_regimes:
            regime_data = data[regimes == regime]
            
            if len(regime_data) == 0:
                continue
            
            regime_returns = regime_data['returns'].dropna()
            
            if len(regime_returns) == 0:
                continue
            
            stats.append({
                'regime': regime,
                'count': len(regime_data),
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'min_return': regime_returns.min(),
                'max_return': regime_returns.max(),
                'positive_days': (regime_returns > 0).sum() / len(regime_returns),
                'negative_days': (regime_returns < 0).sum() / len(regime_returns)
            })
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Map regime numbers to names
        regime_names = self.get_regime_names()
        if len(regime_names) >= len(unique_regimes):
            regime_map = {i: regime_names[i] for i in range(len(unique_regimes))}
            stats_df["regime_name"] = stats_df['regime'].map(regime_map)
        
        return stats_df
        """
        # Placeholder implementation to avoid breaking code
        self.logger.warning("get_regime_stats is temporarily disabled for testing")
        return pd.DataFrame()
    
    def plot_regimes(self, data: pd.DataFrame, price_col: str = 'close', date_col: str = 'date',
                    regimes: Optional[np.ndarray] = None, ax=None):
        """
        Plot price with regime background colors.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with at least columns for price and date/time
        price_col : str, default='close'
            Column name for price data
        date_col : str, default='date'
            Column name for date/time data
        regimes : np.ndarray, optional
            Array of regime labels. If None, use the regimes from the last fit.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, create new figure and axes.
            
        Returns:
        --------
        matplotlib.axes.Axes
            Axes with the plot
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            self.logger.error("Matplotlib is required for plotting. Install it with 'pip install matplotlib'.")
            return None
        
        if regimes is None:
            if self.regimes is None:
                raise ValueError("No regimes available. Call fit() first.")
            regimes = self.regimes
        
        if len(regimes) != len(data):
            raise ValueError("Length of regimes must match length of data.")
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price
        ax.plot(data[date_col], data[price_col], color='black', lw=1)
        
        # Get unique regimes
        unique_regimes = np.unique(regimes)
        
        # Define colors for regimes
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
        
        # Plot regime backgrounds
        ymin, ymax = ax.get_ylim()
        height = ymax - ymin
        
        for i, regime in enumerate(unique_regimes):
            # Find continuous segments of the same regime
            regime_mask = (regimes == regime)
            change_points = np.where(np.diff(regime_mask.astype(int)) != 0)[0] + 1
            change_points = np.concatenate(([0], change_points, [len(regime_mask)]))
            
            for j in range(len(change_points) - 1):
                if regime_mask[change_points[j]]:
                    start_idx = change_points[j]
                    end_idx = change_points[j + 1] - 1
                    
                    if start_idx >= len(data) or end_idx >= len(data):
                        continue
                    
                    start_date = data[date_col].iloc[start_idx]
                    end_date = data[date_col].iloc[end_idx]
                    
                    rect = Rectangle(
                        (start_date, ymin),
                        end_date - start_date,
                        height,
                        facecolor=colors[i],
                        alpha=0.3,
                        edgecolor='none'
                    )
                    ax.add_patch(rect)
        
        # Add legend
        regime_names = self.get_regime_names()
        if len(regime_names) < len(unique_regimes):
            # Extend regime names if needed
            regime_names.extend([f'regime_{i}' for i in range(len(regime_names), len(unique_regimes))])
        
        # Use only the regime names that appear in the data
        regime_names = [regime_names[int(r)] for r in unique_regimes]
        
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.3) for i in range(len(unique_regimes))]
        ax.legend(handles, regime_names, loc='best')
        
        # Set labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Market Regimes - {self.name}')
        
        return ax
