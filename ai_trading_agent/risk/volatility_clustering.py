"""
Volatility Clustering Module

This module implements advanced volatility modeling techniques including GARCH models
for volatility clustering detection and forecasting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from arch import arch_model
from scipy import stats

# Set up logger
logger = logging.getLogger(__name__)


class VolatilityClusteringModel:
    """
    Implements volatility clustering models to detect and forecast volatility regimes.
    
    Uses GARCH models to capture volatility clustering effects and provide more
    accurate volatility forecasts for risk management.
    """
    
    def __init__(
        self,
        lookback_window: int = 500,
        forecast_horizon: int = 10,
        use_cache: bool = True,
        max_cache_size: int = 100
    ):
        """
        Initialize the volatility clustering model.
        
        Args:
            lookback_window: Number of historical observations to use
            forecast_horizon: Number of days to forecast
            use_cache: Whether to cache model results for performance
            max_cache_size: Maximum number of cached models
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        
        # Cache for fitted models
        self.model_cache = {}
        self.cache_timestamps = {}
        
        logger.info("Volatility Clustering Model initialized")
    
    def fit_garch_model(
        self,
        returns: pd.Series,
        symbol: str,
        p: int = 1,
        q: int = 1,
        force_refit: bool = False
    ) -> Dict[str, Any]:
        """
        Fit a GARCH model to the returns series.
        
        Args:
            returns: Series of asset returns
            symbol: Asset symbol for caching
            p: GARCH lag order
            q: ARCH lag order
            force_refit: Whether to force refitting even if cached
            
        Returns:
            Dictionary with model results
        """
        # Check if we can use cached model (if available and not forced to refit)
        cache_key = f"{symbol}_p{p}_q{q}"
        if not force_refit and self.use_cache and cache_key in self.model_cache:
            # Update the cache timestamp
            self.cache_timestamps[cache_key] = pd.Timestamp.now()
            logger.debug(f"Using cached GARCH model for {symbol}")
            return self.model_cache[cache_key]
        
        # If series too short, return default values
        if len(returns) < 100:
            logger.warning(f"Not enough data to fit GARCH model for {symbol} (need 100, got {len(returns)})")
            return {
                "symbol": symbol,
                "volatility": returns.std() * np.sqrt(252),
                "forecast": [returns.std() * np.sqrt(252)] * self.forecast_horizon,
                "volatility_of_volatility": 0.0,
                "success": False,
                "model": None
            }
        
        try:
            # Prepare the data
            data = returns.dropna()
            if len(data) > self.lookback_window:
                data = data[-self.lookback_window:]
            
            # Fit GARCH model
            model = arch_model(
                data * 100,  # Scale returns to avoid numerical issues
                vol='Garch',
                p=p,
                q=q,
                mean='Zero',
                rescale=False
            )
            
            # Fit the model with a reasonable starting point for parameters
            fitted_model = model.fit(
                disp='off',
                show_warning=False,
                options={'maxiter': 500}
            )
            
            # Get conditional volatility series
            conditional_vol = fitted_model.conditional_volatility / 100  # Scale back
            
            # Forecast volatility
            forecast = fitted_model.forecast(horizon=self.forecast_horizon)
            forecasted_var = forecast.variance.iloc[-1].values / 10000  # Scale back and convert from variance
            forecasted_vol = np.sqrt(forecasted_var) * np.sqrt(252)  # Annualize
            
            # Calculate volatility of volatility
            vol_of_vol = conditional_vol.pct_change().dropna().std() * np.sqrt(252)
            
            # Create results dictionary
            results = {
                "symbol": symbol,
                "volatility": conditional_vol.iloc[-1] * np.sqrt(252),  # Annualized latest vol
                "forecast": forecasted_vol.tolist(),
                "volatility_of_volatility": vol_of_vol,
                "success": True,
                "model": fitted_model if self.use_cache else None
            }
            
            # Cache the results
            if self.use_cache:
                self.model_cache[cache_key] = results
                self.cache_timestamps[cache_key] = pd.Timestamp.now()
                
                # Manage cache size
                if len(self.model_cache) > self.max_cache_size:
                    self._prune_cache()
            
            logger.debug(f"Successfully fitted GARCH model for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model for {symbol}: {str(e)}")
            # Fall back to simple volatility estimate
            return {
                "symbol": symbol,
                "volatility": returns.std() * np.sqrt(252),
                "forecast": [returns.std() * np.sqrt(252)] * self.forecast_horizon,
                "volatility_of_volatility": 0.0,
                "success": False,
                "model": None
            }
    
    def detect_volatility_regime(
        self,
        returns: pd.Series,
        symbol: str,
        use_garch: bool = True
    ) -> Dict[str, Any]:
        """
        Detect the current volatility regime using GARCH models and statistical tests.
        
        Args:
            returns: Series of asset returns
            symbol: Asset symbol
            use_garch: Whether to use GARCH model or simple approach
            
        Returns:
            Dictionary with volatility regime information
        """
        try:
            # Simple statistical approach
            rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            
            # Volatility z-score
            if len(rolling_vol) > 100:
                vol_history = rolling_vol.dropna()
                recent_vol = vol_history.iloc[-1]
                vol_mean = vol_history.mean()
                vol_std = vol_history.std()
                vol_z_score = (recent_vol - vol_mean) / vol_std if vol_std > 0 else 0
                
                # Determine regime based on z-score
                if vol_z_score < -1.0:
                    regime = "low"
                elif vol_z_score < 0.5:
                    regime = "normal"
                elif vol_z_score < 2.0:
                    regime = "elevated"
                else:
                    regime = "high"
            else:
                regime = "normal"
                vol_z_score = 0
            
            # GARCH-based approach
            if use_garch:
                garch_results = self.fit_garch_model(returns, symbol)
                
                # If successful, use GARCH volatility
                if garch_results["success"]:
                    garch_vol = garch_results["volatility"]
                    garch_forecast = garch_results["forecast"]
                    vol_of_vol = garch_results["volatility_of_volatility"]
                    
                    # Check for volatility clustering
                    clustering_intensity = vol_of_vol / garch_vol if garch_vol > 0 else 0
                    
                    # Predict if volatility will increase
                    vol_increasing = garch_forecast[-1] > garch_vol
                    
                    # Use GARCH's more accurate volatility in the return value
                    historical_vol = garch_vol
                else:
                    clustering_intensity = 0
                    vol_increasing = False
            else:
                clustering_intensity = 0
                vol_increasing = False
                garch_forecast = [historical_vol] * self.forecast_horizon
            
            # Create results
            results = {
                "symbol": symbol,
                "current_volatility": historical_vol,
                "regime": regime,
                "z_score": vol_z_score,
                "clustering_intensity": clustering_intensity,
                "forecast": garch_forecast,
                "volatility_increasing": vol_increasing
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "current_volatility": returns.std() * np.sqrt(252),
                "regime": "normal",
                "z_score": 0,
                "clustering_intensity": 0,
                "forecast": [returns.std() * np.sqrt(252)] * self.forecast_horizon,
                "volatility_increasing": False
            }
    
    def analyze_multiple_assets(
        self,
        asset_returns: Dict[str, pd.Series],
        use_garch: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze volatility regimes for multiple assets.
        
        Args:
            asset_returns: Dictionary mapping symbols to return series
            use_garch: Whether to use GARCH models
            
        Returns:
            Dictionary with volatility analysis for each asset
        """
        results = {}
        
        for symbol, returns in asset_returns.items():
            results[symbol] = self.detect_volatility_regime(returns, symbol, use_garch)
        
        # Check for market-wide volatility regime
        if "SPY" in results:
            market_regime = results["SPY"]["regime"]
            logger.info(f"Market-wide volatility regime detected: {market_regime}")
            
            # Check if other assets have divergent regimes
            divergent = []
            for symbol, result in results.items():
                if symbol != "SPY" and result["regime"] != market_regime:
                    divergent.append(symbol)
            
            if divergent:
                logger.info(f"Assets with divergent volatility regimes: {divergent}")
                results["_market"] = {
                    "regime": market_regime,
                    "divergent_assets": divergent,
                    "market_increasing": results["SPY"]["volatility_increasing"]
                }
        
        return results
    
    def _prune_cache(self) -> None:
        """Remove oldest entries from the cache if it exceeds maximum size."""
        # Sort cache keys by timestamp
        sorted_keys = sorted(
            self.cache_timestamps.keys(),
            key=lambda k: self.cache_timestamps[k]
        )
        
        # Remove oldest entries
        num_to_remove = len(self.model_cache) - self.max_cache_size
        if num_to_remove > 0:
            for key in sorted_keys[:num_to_remove]:
                del self.model_cache[key]
                del self.cache_timestamps[key]
                
            logger.debug(f"Pruned {num_to_remove} entries from volatility model cache")
