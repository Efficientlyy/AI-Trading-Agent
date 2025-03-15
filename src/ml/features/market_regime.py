"""Market regime detection using statistical methods."""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import os
import joblib
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from hmmlearn import hmm
from scipy.signal import find_peaks

@dataclass
class RegimeDetectionParams:
    """Parameters for regime detection."""
    n_regimes: int = 3  # Number of market regimes
    lookback_window: int = 252  # One year of daily data
    min_samples: int = 63  # Minimum samples (3 months)
    volatility_window: int = 21  # One month
    correlation_window: int = 63  # Three months
    zscore_window: int = 21  # One month
    hmm_n_iter: int = 100  # HMM training iterations
    model_dir: str = "models"  # Directory to save trained models

class MarketRegimeDetector:
    """Detect market regimes using various statistical methods."""
    
    def __init__(self, params: Optional[RegimeDetectionParams] = None):
        """Initialize detector with parameters."""
        self.params = params or RegimeDetectionParams()
        self.scaler = StandardScaler()
        self.hmm_model: Optional[hmm.GaussianHMM] = None
        self.gmm_model: Optional[GaussianMixture] = None
        self.logger = logging.getLogger(__name__)
        
        # Create model directory if it doesn't exist
        os.makedirs(self.params.model_dir, exist_ok=True)
    
    def _validate_input(
        self,
        data: NDArray[np.float64],
        name: str = "data"
    ) -> None:
        """Validate input data.
        
        Args:
            data: Input array to validate
            name: Name of the data for error messages
        
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        
        if data.size == 0:
            raise ValueError(f"{name} is empty")
        
        if not np.isfinite(data).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        
        if len(data) < self.params.min_samples:
            raise ValueError(
                f"Insufficient data for regime detection. "
                f"Got {len(data)} samples, need at least "
                f"{self.params.min_samples}"
            )
    
    def detect_regimes_hmm(
        self,
        returns: NDArray[np.float64],
        volumes: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Detect market regimes using Hidden Markov Model.
        
        Args:
            returns: Array of price returns
            volumes: Array of trading volumes
        
        Returns:
            Tuple of (regime labels, regime probabilities)
        
        Raises:
            ValueError: If input data is invalid or model is not initialized
        """
        self._validate_input(returns, "returns")
        self._validate_input(volumes, "volumes")
        
        if len(returns) != len(volumes):
            raise ValueError(
                f"Length mismatch: returns ({len(returns)}) != "
                f"volumes ({len(volumes)})"
            )
        
        try:
            # Prepare features
            features = np.column_stack([
                returns,
                self._calculate_volatility(returns),
                self._normalize_volume(volumes)
            ])
            
            # Check for zero variance
            if np.any(np.var(features, axis=0) == 0):
                raise ValueError("Features contain constant values")
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Train HMM
            if self.hmm_model is None:
                self.hmm_model = hmm.GaussianHMM(
                    n_components=self.params.n_regimes,
                    covariance_type="full",
                    n_iter=self.params.hmm_n_iter,
                    random_state=42
                )
            
            # Fit model
            self.hmm_model.fit(scaled_features)
            
            # Check convergence
            try:
                if not self.hmm_model.monitor_.converged_:
                    self.logger.warning(
                        "HMM did not converge. Consider increasing n_iter"
                    )
            except AttributeError:
                self.logger.debug("Could not check HMM convergence status")
            
            # Get regime labels and probabilities
            if self.hmm_model is None:
                raise ValueError("HMM model not initialized")
                
            labels = self.hmm_model.predict(scaled_features)
            probs = self.hmm_model.predict_proba(scaled_features)
            
            return labels, probs
            
        except Exception as e:
            self.logger.error(f"Error in HMM regime detection: {str(e)}")
            raise
    
    def detect_regimes_gmm(
        self,
        returns: NDArray[np.float64],
        volumes: NDArray[np.float64]
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Detect market regimes using Gaussian Mixture Model.
        
        Args:
            returns: Array of price returns
            volumes: Array of trading volumes
        
        Returns:
            Tuple of (regime labels, regime probabilities)
        """
        if len(returns) < self.params.min_samples:
            raise ValueError("Insufficient data for regime detection")
        
        # Prepare features
        features = np.column_stack([
            returns,
            self._calculate_volatility(returns),
            self._normalize_volume(volumes)
        ])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train GMM
        self.gmm_model = GaussianMixture(
            n_components=self.params.n_regimes,
            covariance_type="full",
            random_state=42
        )
        
        self.gmm_model.fit(scaled_features)
        
        # Get regime labels and probabilities
        labels = self.gmm_model.predict(scaled_features)
        probs = self.gmm_model.predict_proba(scaled_features)
        
        return labels, probs
    
    def detect_volatility_regimes(
        self,
        returns: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Detect volatility regimes using threshold method.
        
        Args:
            returns: Array of price returns
        
        Returns:
            Array of regime labels (0=low, 1=normal, 2=high)
        """
        if len(returns) < self.params.min_samples:
            raise ValueError("Insufficient data for regime detection")
        
        # Calculate rolling volatility
        vol = self._calculate_volatility(returns)
        
        # Calculate volatility Z-scores
        vol_mean = np.mean(vol)
        vol_std = np.std(vol)
        z_scores = (vol - vol_mean) / vol_std
        
        # Classify regimes
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[z_scores > 1.0] = 2  # High volatility
        labels[(z_scores >= -1.0) & (z_scores <= 1.0)] = 1  # Normal
        labels[z_scores < -1.0] = 0  # Low volatility
        
        return labels
    
    def detect_trend_regimes(
        self,
        prices: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Detect trend regimes using moving averages.
        
        Args:
            prices: Array of prices
        
        Returns:
            Array of regime labels (0=downtrend, 1=sideways, 2=uptrend)
        """
        if len(prices) < self.params.min_samples:
            raise ValueError("Insufficient data for regime detection")
        
        # Calculate short and long moving averages
        short_ma = self._calculate_ema(prices, 20)  # 1 month
        long_ma = self._calculate_ema(prices, 100)  # 5 months
        
        # Calculate moving average crossovers
        ma_diff = short_ma - long_ma
        ma_diff_zscore = self._calculate_zscore(ma_diff)
        
        # Classify regimes
        labels = np.zeros(len(prices), dtype=np.int64)
        labels[ma_diff_zscore > 1.0] = 2  # Uptrend
        labels[(ma_diff_zscore >= -1.0) & (ma_diff_zscore <= 1.0)] = 1  # Sideways
        labels[ma_diff_zscore < -1.0] = 0  # Downtrend
        
        return labels
    
    def detect_correlation_regimes(
        self,
        returns_matrix: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Detect correlation regimes in a multi-asset system.
        
        Args:
            returns_matrix: Matrix of returns (assets in columns)
        
        Returns:
            Array of regime labels (0=low, 1=normal, 2=high correlation)
        """
        if len(returns_matrix) < self.params.min_samples:
            raise ValueError("Insufficient data for regime detection")
        
        # Calculate rolling correlations
        n_periods = len(returns_matrix) - self.params.correlation_window + 1
        avg_corr = np.zeros(len(returns_matrix))
        
        for i in range(self.params.correlation_window - 1, len(returns_matrix)):
            window = returns_matrix[i-self.params.correlation_window+1:i+1]
            corr_matrix = np.corrcoef(window.T)
            avg_corr[i] = (np.sum(corr_matrix) - len(corr_matrix)) / (
                len(corr_matrix) * (len(corr_matrix) - 1)
            )
        
        # Calculate correlation Z-scores
        corr_zscore = self._calculate_zscore(avg_corr)
        
        # Classify regimes
        labels = np.zeros(len(returns_matrix), dtype=np.int64)
        labels[corr_zscore > 1.0] = 2  # High correlation
        labels[(corr_zscore >= -1.0) & (corr_zscore <= 1.0)] = 1  # Normal
        labels[corr_zscore < -1.0] = 0  # Low correlation
        
        return labels
    
    def get_regime_statistics(
        self,
        returns: NDArray[np.float64],
        labels: NDArray[np.int64]
    ) -> Dict[int, Dict[str, float]]:
        """Calculate statistics for each regime.
        
        Args:
            returns: Array of returns
            labels: Array of regime labels
        
        Returns:
            Dictionary of regime statistics
        """
        stats = {}
        
        for regime in range(self.params.n_regimes):
            regime_returns = returns[labels == regime]
            
            if len(regime_returns) > 0:
                stats[regime] = {
                    "mean_return": float(np.mean(regime_returns)),
                    "volatility": float(np.std(regime_returns)),
                    "sharpe_ratio": float(
                        np.mean(regime_returns) / np.std(regime_returns)
                        if np.std(regime_returns) > 0 else 0
                    ),
                    "skewness": float(self._calculate_skewness(regime_returns)),
                    "kurtosis": float(self._calculate_kurtosis(regime_returns)),
                    "var_95": float(np.percentile(regime_returns, 5)),
                    "frequency": float(len(regime_returns) / len(returns))
                }
        
        return stats
    
    def _calculate_volatility(
        self,
        returns: NDArray[np.float64],
        window: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Calculate rolling volatility.
        
        Args:
            returns: Array of returns
            window: Optional window size (default: self.params.volatility_window)
        
        Returns:
            Array of volatility values
        """
        if window is None:
            window = self.params.volatility_window
            
        vol = np.zeros_like(returns)
        for i in range(window - 1, len(returns)):
            vol[i] = np.std(returns[i-window+1:i+1])
        return vol
    
    def _normalize_volume(
        self,
        volumes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Normalize volume using Z-score method.
        
        Args:
            volumes: Array of volumes
        
        Returns:
            Array of normalized volumes
        """
        log_vol = np.log1p(volumes)
        return self._calculate_zscore(log_vol)
    
    def _calculate_zscore(
        self,
        data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate rolling Z-scores.
        
        Args:
            data: Input data array
        
        Returns:
            Array of Z-scores
        """
        zscore = np.zeros_like(data)
        for i in range(self.params.zscore_window - 1, len(data)):
            window = data[i-self.params.zscore_window+1:i+1]
            zscore[i] = (data[i] - np.mean(window)) / np.std(window)
        return zscore
    
    def _calculate_ema(
        self,
        data: NDArray[np.float64],
        period: int
    ) -> NDArray[np.float64]:
        """Calculate Exponential Moving Average.
        
        Args:
            data: Input data array
            period: Moving average period
        
        Returns:
            Array of EMA values
        """
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        
        # Initialize EMA with SMA
        ema[period-1] = np.mean(data[:period])
        
        # Calculate EMA
        for i in range(period, len(data)):
            ema[i] = ((data[i] - ema[i-1]) * multiplier) + ema[i-1]
        
        return ema
    
    def _calculate_skewness(
        self,
        data: NDArray[np.float64]
    ) -> float:
        """Calculate skewness of data.
        
        Args:
            data: Input data array
        
        Returns:
            Skewness value
        """
        n = len(data)
        if n < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        
        return (
            np.sum((data - mean) ** 3) / ((n - 1) * std ** 3)
        )
    
    def _calculate_kurtosis(
        self,
        data: NDArray[np.float64]
    ) -> float:
        """Calculate excess kurtosis of data.
        
        Args:
            data: Input data array
        
        Returns:
            Excess kurtosis value
        """
        n = len(data)
        if n < 2:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        
        return (
            np.sum((data - mean) ** 4) / ((n - 1) * std ** 4)
        ) - 3  # Subtract 3 for excess kurtosis 

    def save_models(self, prefix: str = "") -> None:
        """Save trained models to disk.
        
        Args:
            prefix: Optional prefix for model filenames
        """
        try:
            if self.hmm_model is not None:
                hmm_path = os.path.join(
                    self.params.model_dir,
                    f"{prefix}hmm_model.joblib"
                )
                joblib.dump(self.hmm_model, hmm_path)
            
            if self.gmm_model is not None:
                gmm_path = os.path.join(
                    self.params.model_dir,
                    f"{prefix}gmm_model.joblib"
                )
                joblib.dump(self.gmm_model, gmm_path)
            
            scaler_path = os.path.join(
                self.params.model_dir,
                f"{prefix}scaler.joblib"
            )
            joblib.dump(self.scaler, scaler_path)
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, prefix: str = "") -> None:
        """Load trained models from disk.
        
        Args:
            prefix: Optional prefix for model filenames
        """
        try:
            hmm_path = os.path.join(
                self.params.model_dir,
                f"{prefix}hmm_model.joblib"
            )
            if os.path.exists(hmm_path):
                self.hmm_model = joblib.load(hmm_path)
            
            gmm_path = os.path.join(
                self.params.model_dir,
                f"{prefix}gmm_model.joblib"
            )
            if os.path.exists(gmm_path):
                self.gmm_model = joblib.load(gmm_path)
            
            scaler_path = os.path.join(
                self.params.model_dir,
                f"{prefix}scaler.joblib"
            )
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def detect_momentum_regimes(
        self,
        returns: NDArray[np.float64],
        lookback_periods: List[int] = [21, 63, 252]
    ) -> NDArray[np.int64]:
        """Detect momentum regimes using multiple lookback periods.
        
        Args:
            returns: Array of price returns
            lookback_periods: List of lookback periods for momentum calculation
        
        Returns:
            Array of regime labels (0=negative, 1=neutral, 2=positive)
        """
        self._validate_input(returns, "returns")
        
        # Calculate momentum scores for each period
        momentum_scores = np.zeros((len(returns), len(lookback_periods)))
        
        for i, period in enumerate(lookback_periods):
            # Calculate cumulative returns
            cum_rets = np.zeros_like(returns)
            for j in range(period - 1, len(returns)):
                cum_rets[j] = np.prod(1 + returns[j-period+1:j+1]) - 1
            
            # Convert to Z-scores
            momentum_scores[:, i] = self._calculate_zscore(cum_rets)
        
        # Average momentum scores
        avg_score = np.mean(momentum_scores, axis=1)
        
        # Classify regimes
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[avg_score > 1.0] = 2  # Strong momentum
        labels[(avg_score >= -1.0) & (avg_score <= 1.0)] = 1  # Neutral
        labels[avg_score < -1.0] = 0  # Weak momentum
        
        return labels
    
    def detect_liquidity_regimes(
        self,
        volumes: NDArray[np.float64],
        spreads: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Detect liquidity regimes using volume and spread data.
        
        Args:
            volumes: Array of trading volumes
            spreads: Array of bid-ask spreads
        
        Returns:
            Array of regime labels (0=low, 1=normal, 2=high liquidity)
        """
        self._validate_input(volumes, "volumes")
        self._validate_input(spreads, "spreads")
        
        if len(volumes) != len(spreads):
            raise ValueError(
                f"Length mismatch: volumes ({len(volumes)}) != "
                f"spreads ({len(spreads)})"
            )
        
        # Normalize volumes and spreads
        norm_vol = self._normalize_volume(volumes)
        norm_spread = self._calculate_zscore(spreads)
        
        # Combine into liquidity score (high volume + low spread = high liquidity)
        liquidity_score = norm_vol - norm_spread
        
        # Classify regimes
        labels = np.zeros(len(volumes), dtype=np.int64)
        labels[liquidity_score > 1.0] = 2  # High liquidity
        labels[(liquidity_score >= -1.0) & (liquidity_score <= 1.0)] = 1  # Normal
        labels[liquidity_score < -1.0] = 0  # Low liquidity
        
        return labels
    
    def detect_sentiment_regimes(
        self,
        returns: NDArray[np.float64],
        volumes: NDArray[np.float64],
        window: int = 21
    ) -> NDArray[np.int64]:
        """Detect sentiment regimes using price-volume relationship.
        
        Args:
            returns: Array of price returns
            volumes: Array of trading volumes
            window: Rolling window for analysis
        
        Returns:
            Array of regime labels (0=bearish, 1=neutral, 2=bullish)
        """
        self._validate_input(returns, "returns")
        self._validate_input(volumes, "volumes")
        
        if len(returns) != len(volumes):
            raise ValueError(
                f"Length mismatch: returns ({len(returns)}) != "
                f"volumes ({len(volumes)})"
            )
        
        # Calculate rolling correlation between returns and volume
        sentiment = np.zeros(len(returns))
        
        for i in range(window - 1, len(returns)):
            ret_window = returns[i-window+1:i+1]
            vol_window = volumes[i-window+1:i+1]
            sentiment[i] = np.corrcoef(ret_window, vol_window)[0, 1]
        
        # Calculate volume-weighted returns
        vol_weight = volumes / np.mean(volumes)
        weighted_returns = returns * vol_weight
        weighted_score = self._calculate_zscore(weighted_returns)
        
        # Combine signals
        combined_score = (
            self._calculate_zscore(sentiment) +
            weighted_score
        ) / 2
        
        # Classify regimes
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[combined_score > 1.0] = 2  # Bullish
        labels[(combined_score >= -1.0) & (combined_score <= 1.0)] = 1  # Neutral
        labels[combined_score < -1.0] = 0  # Bearish
        
        return labels
    
    def detect_volatility_structure(
        self,
        returns: NDArray[np.float64],
        windows: List[int] = [5, 21, 63]
    ) -> NDArray[np.int64]:
        """Detect volatility structure regimes using term structure.
        
        Args:
            returns: Array of price returns
            windows: List of windows for volatility calculation
        
        Returns:
            Array of regime labels (0=contango, 1=flat, 2=backwardation)
        """
        self._validate_input(returns, "returns")
        
        # Calculate volatilities for different windows
        vols = np.zeros((len(returns), len(windows)))
        
        for i, window in enumerate(windows):
            vols[:, i] = self._calculate_volatility(returns, window)
        
        # Calculate volatility ratios
        vol_ratios = np.zeros(len(returns))
        for i in range(len(returns)):
            if np.all(vols[i, :] > 0):
                # Fit exponential curve to vol term structure
                log_vols = np.log(vols[i, :])
                x = np.array(windows)
                slope = np.polyfit(x, log_vols, 1)[0]
                vol_ratios[i] = slope
        
        # Normalize ratios
        norm_ratios = self._calculate_zscore(vol_ratios)
        
        # Classify regimes
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[norm_ratios > 1.0] = 2  # Backwardation (short-term > long-term)
        labels[(norm_ratios >= -1.0) & (norm_ratios <= 1.0)] = 1  # Flat
        labels[norm_ratios < -1.0] = 0  # Contango (short-term < long-term)
        
        return labels
    
    def detect_seasonality_regimes(
        self,
        returns: NDArray[np.float64],
        period: int = 252,
        min_periods: int = 3
    ) -> NDArray[np.int64]:
        """Detect seasonality regimes using Fourier analysis.
        
        Args:
            returns: Array of returns
            period: Expected seasonality period (e.g., 252 for annual)
            min_periods: Minimum number of periods for detection
        
        Returns:
            Array of regime labels (0=weak, 1=neutral, 2=strong seasonality)
        """
        self._validate_input(returns, "returns")
        
        if len(returns) < period * min_periods:
            raise ValueError(
                f"Insufficient data for seasonality detection. "
                f"Need at least {period * min_periods} samples."
            )
        
        # Calculate periodogram
        freqs = np.fft.fftfreq(len(returns))
        power = np.abs(np.fft.fft(returns))**2
        
        # Find peaks in power spectrum
        peaks, _ = find_peaks(power[:len(freqs)//2])
        peak_powers = power[peaks]
        
        # Calculate seasonality strength
        if len(peaks) > 0:
            # Find peaks near expected frequency
            expected_freq = 1/period
            freq_diff = np.abs(freqs[peaks] - expected_freq)
            seasonal_idx = np.argmin(freq_diff)
            
            # Calculate seasonality score
            total_power = np.sum(power[:len(freqs)//2])
            seasonal_power = peak_powers[seasonal_idx]
            seasonality_score = seasonal_power / total_power
            
            # Convert to Z-score
            seasonality_zscore = self._calculate_zscore(
                np.full_like(returns, seasonality_score)
            )
        else:
            seasonality_zscore = np.zeros_like(returns)
        
        # Classify regimes
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[seasonality_zscore > 1.0] = 2  # Strong seasonality
        labels[(seasonality_zscore >= -1.0) & (seasonality_zscore <= 1.0)] = 1  # Neutral
        labels[seasonality_zscore < -1.0] = 0  # Weak seasonality
        
        return labels
    
    def detect_mean_reversion_regimes(
        self,
        prices: NDArray[np.float64],
        window: int = 21
    ) -> NDArray[np.int64]:
        """Detect mean reversion regimes using Hurst exponent.
        
        Args:
            prices: Array of prices
            window: Rolling window for Hurst calculation
        
        Returns:
            Array of regime labels (0=trending, 1=random, 2=mean-reverting)
        """
        self._validate_input(prices, "prices")
        
        def calculate_hurst(data: NDArray[np.float64]) -> float:
            """Calculate Hurst exponent using R/S analysis."""
            lags = range(2, min(len(data)//2, 21))
            tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]
        
        # Calculate rolling Hurst exponent
        hurst = np.zeros_like(prices)
        for i in range(window, len(prices)):
            hurst[i] = calculate_hurst(prices[i-window:i])
        
        # Fill initial values
        hurst[:window] = hurst[window]
        
        # Classify regimes based on Hurst exponent
        labels = np.zeros(len(prices), dtype=np.int64)
        labels[hurst < 0.4] = 2      # Mean-reverting (H < 0.4)
        labels[hurst > 0.6] = 0      # Trending (H > 0.6)
        labels[(hurst >= 0.4) & (hurst <= 0.6)] = 1  # Random walk
        
        return labels
    
    def detect_market_stress_regimes(
        self,
        returns: NDArray[np.float64],
        volumes: NDArray[np.float64],
        window: int = 21
    ) -> NDArray[np.int64]:
        """Detect market stress regimes using multiple indicators.
        
        Args:
            returns: Array of returns
            volumes: Array of volumes
            window: Rolling window for calculations
        
        Returns:
            Array of regime labels (0=low, 1=normal, 2=high stress)
        """
        self._validate_input(returns, "returns")
        self._validate_input(volumes, "volumes")
        
        if len(returns) != len(volumes):
            raise ValueError(
                f"Length mismatch: returns ({len(returns)}) != "
                f"volumes ({len(volumes)})"
            )
        
        # Calculate stress indicators
        vol = self._calculate_volatility(returns, window)
        skew = np.zeros_like(returns)
        kurt = np.zeros_like(returns)
        vol_of_vol = np.zeros_like(returns)
        
        for i in range(window, len(returns)):
            r = returns[i-window:i]
            skew[i] = self._calculate_skewness(r)
            kurt[i] = self._calculate_kurtosis(r)
            vol_of_vol[i] = np.std(vol[i-window:i])
        
        # Fill initial values
        skew[:window] = skew[window]
        kurt[:window] = kurt[window]
        vol_of_vol[:window] = vol_of_vol[window]
        
        # Normalize indicators
        norm_vol = self._calculate_zscore(vol)
        norm_skew = self._calculate_zscore(np.abs(skew))  # Use absolute skewness
        norm_kurt = self._calculate_zscore(kurt)
        norm_vol_vol = self._calculate_zscore(vol_of_vol)
        
        # Combine indicators into stress score
        stress_score = (
            norm_vol +
            norm_skew +
            norm_kurt +
            norm_vol_vol
        ) / 4
        
        # Classify regimes
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[stress_score > 1.0] = 2  # High stress
        labels[(stress_score >= -1.0) & (stress_score <= 1.0)] = 1  # Normal
        labels[stress_score < -1.0] = 0  # Low stress
        
        return labels
    
    def detect_microstructure_regimes(
        self,
        returns: NDArray[np.float64],
        volumes: NDArray[np.float64],
        spreads: NDArray[np.float64],
        window: int = 21
    ) -> NDArray[np.int64]:
        """Detect market microstructure regimes.
        
        Args:
            returns: Array of returns
            volumes: Array of volumes
            spreads: Array of bid-ask spreads
            window: Rolling window for calculations
        
        Returns:
            Array of regime labels (0=illiquid, 1=normal, 2=liquid)
        """
        self._validate_input(returns, "returns")
        self._validate_input(volumes, "volumes")
        self._validate_input(spreads, "spreads")
        
        # Calculate microstructure metrics
        vol_impact = np.zeros_like(returns)
        spread_impact = np.zeros_like(returns)
        
        for i in range(window, len(returns)):
            # Volume impact (Kyle's lambda)
            vol_window = volumes[i-window:i]
            ret_window = returns[i-window:i]
            vol_impact[i] = np.abs(np.corrcoef(ret_window, vol_window)[0, 1])
            
            # Spread impact (Amihud illiquidity)
            spread_impact[i] = np.mean(np.abs(ret_window) / vol_window)
        
        # Fill initial values
        vol_impact[:window] = vol_impact[window]
        spread_impact[:window] = spread_impact[window]
        
        # Combine metrics
        illiquidity_score = (
            self._calculate_zscore(vol_impact) +
            self._calculate_zscore(spread_impact)
        ) / 2
        
        # Classify regimes
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[illiquidity_score > 1.0] = 0  # Illiquid
        labels[(illiquidity_score >= -1.0) & (illiquidity_score <= 1.0)] = 1  # Normal
        labels[illiquidity_score < -1.0] = 2  # Liquid
        
        return labels
    
    def detect_tail_risk_regimes(
        self,
        returns: NDArray[np.float64],
        window: int = 63,
        quantile: float = 0.05
    ) -> NDArray[np.int64]:
        """Detect tail risk regimes using extreme value theory.
        
        Args:
            returns: Array of returns
            window: Rolling window for calculations
            quantile: Quantile for tail risk calculation
        
        Returns:
            Array of regime labels (0=low, 1=normal, 2=high tail risk)
        """
        self._validate_input(returns, "returns")
        
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be between 0 and 1")
        
        # Calculate tail risk metrics
        tail_risk = np.zeros_like(returns)
        
        for i in range(window, len(returns)):
            window_rets = returns[i-window:i]
            left_tail = np.percentile(window_rets, quantile * 100)
            right_tail = np.percentile(window_rets, (1 - quantile) * 100)
            tail_risk[i] = np.abs(left_tail) + np.abs(right_tail)
        
        # Fill initial values
        tail_risk[:window] = tail_risk[window]
        
        # Normalize tail risk
        tail_risk_score = self._calculate_zscore(tail_risk)
        
        # Classify regimes
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[tail_risk_score > 1.0] = 2  # High tail risk
        labels[(tail_risk_score >= -1.0) & (tail_risk_score <= 1.0)] = 1  # Normal
        labels[tail_risk_score < -1.0] = 0  # Low tail risk
        
        return labels
    
    def detect_dispersion_regimes(
        self,
        returns_matrix: NDArray[np.float64],
        window: int = 21
    ) -> NDArray[np.int64]:
        """Detect cross-sectional dispersion regimes.
        
        Args:
            returns_matrix: Matrix of returns (assets in columns)
            window: Rolling window for calculations
        
        Returns:
            Array of regime labels (0=low, 1=normal, 2=high dispersion)
        """
        self._validate_input(returns_matrix, "returns_matrix")
        
        # Calculate cross-sectional dispersion
        dispersion = np.zeros(len(returns_matrix))
        
        for i in range(window, len(returns_matrix)):
            window_rets = returns_matrix[i-window:i]
            # Calculate cross-sectional standard deviation
            dispersion[i] = np.std(window_rets, axis=1).mean()
        
        # Fill initial values
        dispersion[:window] = dispersion[window]
        
        # Normalize dispersion
        dispersion_score = self._calculate_zscore(dispersion)
        
        # Classify regimes
        labels = np.zeros(len(returns_matrix), dtype=np.int64)
        labels[dispersion_score > 1.0] = 2  # High dispersion
        labels[(dispersion_score >= -1.0) & (dispersion_score <= 1.0)] = 1  # Normal
        labels[dispersion_score < -1.0] = 0  # Low dispersion
        
        return labels 