"""
Volatility Clustering Detection Module

This module provides tools for detecting volatility clustering in financial time series data,
which is a key component of market regime classification.

Volatility clustering refers to the tendency of large price changes to be followed by large
price changes (of either sign) and small price changes to be followed by small price changes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from arch import arch_model
from scipy import stats
import logging

from .core_definitions import (
    VolatilityRegimeType,
    MarketRegimeConfig
)

# Set up logger
logger = logging.getLogger(__name__)


class VolatilityClusteringDetector:
    """
    Class for detecting volatility clustering and classifying volatility regimes.
    
    Uses GARCH models and statistical methods to identify volatility clustering
    in financial time series data.
    """
    
    def __init__(self, config: Optional[MarketRegimeConfig] = None):
        """
        Initialize the volatility clustering detector.
        
        Args:
            config: Configuration for volatility detection
        """
        self.config = config or MarketRegimeConfig()
        self.historical_volatility = {}
        self.garch_models = {}
        self.regime_history = []
    
    def detect_volatility_clustering(self, 
                                    prices: pd.Series,
                                    returns: Optional[pd.Series] = None,
                                    asset_id: str = "default") -> Dict[str, any]:
        """
        Detect volatility clustering in price or return series.
        
        Args:
            prices: Series of asset prices
            returns: Optional pre-calculated returns (will be calculated if not provided)
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary with volatility clustering metrics
        """
        # Calculate returns if not provided
        if returns is None and prices is not None:
            returns = prices.pct_change().dropna()
        
        if returns is None or len(returns) < 30:
            logger.warning(f"Insufficient data for volatility clustering detection: {len(returns) if returns is not None else 0} points")
            return {
                "volatility_regime": VolatilityRegimeType.UNKNOWN.value,
                "has_clustering": False,
                "clustering_score": 0.0,
                "garch_persistence": 0.0,
                "current_volatility": None,
                "historical_percentile": None
            }
        
        # Calculate basic volatility metrics
        current_volatility = returns.tail(self.config.volatility_window).std() * np.sqrt(252)
        historical_volatility = returns.rolling(window=self.config.volatility_window).std() * np.sqrt(252)
        
        # Store for later reference
        self.historical_volatility[asset_id] = historical_volatility
        
        # Calculate historical percentile of current volatility
        hist_vol_series = historical_volatility.dropna()
        if len(hist_vol_series) > 0:
            historical_percentile = stats.percentileofscore(hist_vol_series, current_volatility) / 100.0
        else:
            historical_percentile = None
        
        # Detect volatility clustering using ARCH/GARCH
        try:
            # Fit GARCH(1,1) model
            model = arch_model(returns * 100, vol='GARCH', p=1, q=1)
            model_fit = model.fit(disp='off')
            self.garch_models[asset_id] = model_fit
            
            # Extract parameters
            omega = model_fit.params['omega']
            alpha = model_fit.params['alpha[1]']
            beta = model_fit.params['beta[1]']
            
            # Calculate persistence (alpha + beta), high value indicates volatility clustering
            persistence = alpha + beta
            half_life = np.log(0.5) / np.log(persistence) if persistence < 1.0 else np.inf
            
            # Test for ARCH effects
            arch_effects = model_fit.arch_lm_test()
            has_clustering = arch_effects.pvalue <= 0.05
            
            # Calculate clustering score based on persistence and significance
            clustering_score = persistence * (1.0 - arch_effects.pvalue)
            
        except Exception as e:
            logger.error(f"Error in GARCH model fitting: {str(e)}")
            persistence = 0.0
            half_life = 0.0
            has_clustering = False
            clustering_score = 0.0
        
        # Determine volatility regime
        volatility_regime = self._classify_volatility_regime(current_volatility)
        
        # Create results dictionary
        results = {
            "volatility_regime": volatility_regime.value,
            "has_clustering": has_clustering,
            "clustering_score": clustering_score,
            "garch_persistence": persistence if 'persistence' in locals() else 0.0,
            "garch_half_life": half_life if 'half_life' in locals() else 0.0,
            "current_volatility": current_volatility,
            "historical_percentile": historical_percentile
        }
        
        # Track regime history
        self.regime_history.append({
            "timestamp": returns.index[-1] if hasattr(returns.index[-1], 'timestamp') else pd.Timestamp.now(),
            "asset_id": asset_id,
            "volatility_regime": volatility_regime.value,
            "current_volatility": current_volatility,
            "clustering_score": clustering_score
        })
        
        # Trim history if too long
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        return results
    
    def _classify_volatility_regime(self, volatility: float) -> VolatilityRegimeType:
        """
        Classify volatility into a regime.
        
        Args:
            volatility: Annualized volatility
            
        Returns:
            VolatilityRegimeType enum
        """
        thresholds = self.config.volatility_threshold
        
        if volatility is None:
            return VolatilityRegimeType.UNKNOWN
        elif volatility <= thresholds["very_low"]:
            return VolatilityRegimeType.VERY_LOW
        elif volatility <= thresholds["low"]:
            return VolatilityRegimeType.LOW
        elif volatility <= thresholds["moderate"]:
            return VolatilityRegimeType.MODERATE
        elif volatility <= thresholds["high"]:
            return VolatilityRegimeType.HIGH
        elif volatility <= thresholds["very_high"]:
            return VolatilityRegimeType.VERY_HIGH
        elif volatility <= thresholds["extreme"]:
            return VolatilityRegimeType.EXTREME
        else:
            return VolatilityRegimeType.CRISIS
    
    def get_volatility_forecast(self, 
                               asset_id: str = "default", 
                               horizon: int = 5) -> Optional[Dict[str, any]]:
        """
        Get volatility forecast from GARCH model.
        
        Args:
            asset_id: Identifier for the asset
            horizon: Forecast horizon in days
            
        Returns:
            Dictionary with volatility forecast or None if model not available
        """
        if asset_id not in self.garch_models:
            return None
        
        try:
            model_fit = self.garch_models[asset_id]
            forecast = model_fit.forecast(horizon=horizon)
            
            # Extract variance forecast and convert to volatility
            variance_forecast = forecast.variance.iloc[-1].values
            volatility_forecast = np.sqrt(variance_forecast) * np.sqrt(252) / 100.0
            
            # Create forecast dictionary
            result = {
                "forecast_horizons": list(range(1, horizon + 1)),
                "volatility_forecast": volatility_forecast.tolist(),
                "current_volatility": np.sqrt(model_fit.conditional_volatility[-1]**2) * np.sqrt(252) / 100.0,
                "predicted_regime": [self._classify_volatility_regime(vol).value for vol in volatility_forecast]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in volatility forecasting: {str(e)}")
            return None
    
    def detect_regime_change(self, 
                            current_volatility: float,
                            prev_volatility: float) -> Tuple[bool, float]:
        """
        Detect if there has been a significant change in volatility regime.
        
        Args:
            current_volatility: Current volatility value
            prev_volatility: Previous volatility value
            
        Returns:
            Tuple of (has_changed, change_magnitude)
        """
        if current_volatility is None or prev_volatility is None:
            return False, 0.0
        
        # Calculate relative change
        if prev_volatility > 0:
            rel_change = abs(current_volatility - prev_volatility) / prev_volatility
            has_changed = rel_change > self.config.regime_change_threshold
            return has_changed, rel_change
        else:
            return False, 0.0
    
    def get_volatility_statistics(self, 
                                asset_id: str = "default") -> Dict[str, any]:
        """
        Get summary statistics of volatility history.
        
        Args:
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary with volatility statistics
        """
        if asset_id not in self.historical_volatility or len(self.historical_volatility[asset_id]) == 0:
            return {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "std": None,
                "regime_counts": {}
            }
        
        vol_series = self.historical_volatility[asset_id].dropna()
        
        # Get basic statistics
        stats_dict = {
            "mean": vol_series.mean(),
            "median": vol_series.median(),
            "min": vol_series.min(),
            "max": vol_series.max(),
            "std": vol_series.std()
        }
        
        # Count regime occurrences
        regime_counts = {}
        for vol in vol_series:
            regime = self._classify_volatility_regime(vol).value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        stats_dict["regime_counts"] = regime_counts
        
        return stats_dict
