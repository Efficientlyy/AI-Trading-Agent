"""Unit tests for market regime detection."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..src.ml.features.market_regime import MarketRegimeDetector, RegimeDetectionParams

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 252  # One year of daily data
    
    # Generate synthetic returns with regime shifts
    regimes = np.array([0] * 84 + [1] * 84 + [2] * 84)
    returns = np.zeros(n_samples)
    
    # Low volatility regime
    returns[:84] = np.random.normal(0.0001, 0.005, 84)
    
    # Normal regime
    returns[84:168] = np.random.normal(0.0005, 0.01, 84)
    
    # High volatility regime
    returns[168:] = np.random.normal(-0.001, 0.02, 84)
    
    # Generate synthetic volumes
    volumes = np.exp(np.random.normal(10, 1, n_samples))
    volumes[168:] *= 2  # Higher volumes in high volatility regime
    
    # Generate synthetic prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate synthetic multi-asset returns for correlation testing
    n_assets = 5
    asset_returns = np.random.normal(0, 0.01, (n_samples, n_assets))
    
    # Induce correlation in high volatility regime
    correlation_matrix = np.array([
        [1.0, 0.7, 0.7, 0.7, 0.7],
        [0.7, 1.0, 0.7, 0.7, 0.7],
        [0.7, 0.7, 1.0, 0.7, 0.7],
        [0.7, 0.7, 0.7, 1.0, 0.7],
        [0.7, 0.7, 0.7, 0.7, 1.0]
    ])
    L = np.linalg.cholesky(correlation_matrix)
    asset_returns[168:] = np.dot(
        asset_returns[168:],
        L.T
    )
    
    return {
        'returns': returns,
        'volumes': volumes,
        'prices': prices,
        'asset_returns': asset_returns,
        'true_regimes': regimes
    }

def test_regime_detection_params():
    """Test RegimeDetectionParams initialization."""
    params = RegimeDetectionParams()
    
    assert params.n_regimes == 3
    assert params.lookback_window == 252
    assert params.min_samples == 63
    assert params.volatility_window == 21
    assert params.correlation_window == 63
    assert params.zscore_window == 21
    assert params.hmm_n_iter == 100
    assert params.model_dir == "models"

def test_market_regime_detector_init():
    """Test MarketRegimeDetector initialization."""
    detector = MarketRegimeDetector()
    
    assert detector.params is not None
    assert detector.hmm_model is None
    assert detector.gmm_model is None
    assert detector.scaler is not None

def test_input_validation(sample_data):
    """Test input validation."""
    detector = MarketRegimeDetector()
    
    # Test empty array
    with pytest.raises(ValueError, match="empty"):
        detector._validate_input(np.array([], dtype=np.float64), "empty_data")
    
    # Test non-numpy array
    with pytest.raises(ValueError, match="must be a numpy array"):
        detector._validate_input(np.array([1.0, 2.0, 3.0], dtype=np.float64), "list_data")
    
    # Test NaN values
    data_with_nan = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN or infinite"):
        detector._validate_input(data_with_nan, "nan_data")
    
    # Test insufficient samples
    short_data = np.random.randn(10).astype(np.float64)
    with pytest.raises(ValueError, match="Insufficient data"):
        detector._validate_input(short_data, "short_data")

def test_hmm_regime_detection(sample_data):
    """Test HMM-based regime detection."""
    detector = MarketRegimeDetector()
    
    # Test regime detection
    labels, probs = detector.detect_regimes_hmm(
        sample_data['returns'],
        sample_data['volumes']
    )
    
    assert len(labels) == len(sample_data['returns'])
    assert probs.shape == (len(sample_data['returns']), 3)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    
    # Verify that similar regimes are clustered together
    regime_changes = np.diff(labels)
    assert len(np.where(regime_changes != 0)[0]) < 20  # Few regime changes

def test_gmm_regime_detection(sample_data):
    """Test GMM-based regime detection."""
    detector = MarketRegimeDetector()
    
    # Test regime detection
    labels, probs = detector.detect_regimes_gmm(
        sample_data['returns'],
        sample_data['volumes']
    )
    
    assert len(labels) == len(sample_data['returns'])
    assert probs.shape == (len(sample_data['returns']), 3)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_volatility_regime_detection(sample_data):
    """Test volatility regime detection."""
    detector = MarketRegimeDetector()
    
    # Test regime detection
    labels = detector.detect_volatility_regimes(sample_data['returns'])
    
    assert len(labels) == len(sample_data['returns'])
    assert set(labels) <= {0, 1, 2}  # Only allowed values
    
    # Verify that high volatility regime is detected
    high_vol_labels = labels[168:]  # Last third of data
    assert np.mean(high_vol_labels == 2) > 0.5  # Mostly high volatility

def test_trend_regime_detection(sample_data):
    """Test trend regime detection."""
    detector = MarketRegimeDetector()
    
    # Test regime detection
    labels = detector.detect_trend_regimes(sample_data['prices'])
    
    assert len(labels) == len(sample_data['prices'])
    assert set(labels) <= {0, 1, 2}  # Only allowed values

def test_correlation_regime_detection(sample_data):
    """Test correlation regime detection."""
    detector = MarketRegimeDetector()
    
    # Test regime detection
    labels = detector.detect_correlation_regimes(
        sample_data['asset_returns']
    )
    
    assert len(labels) == len(sample_data['asset_returns'])
    assert set(labels) <= {0, 1, 2}  # Only allowed values
    
    # Verify that high correlation regime is detected
    high_corr_labels = labels[168:]  # Last third of data
    assert np.mean(high_corr_labels == 2) > 0.4  # Often high correlation

def test_regime_statistics(sample_data):
    """Test regime statistics calculation."""
    detector = MarketRegimeDetector()
    
    # Get regime labels
    labels, _ = detector.detect_regimes_hmm(
        sample_data['returns'],
        sample_data['volumes']
    )
    
    # Calculate statistics
    stats = detector.get_regime_statistics(
        sample_data['returns'],
        labels
    )
    
    assert len(stats) <= 3  # At most 3 regimes
    
    for regime_stats in stats.values():
        assert "mean_return" in regime_stats
        assert "volatility" in regime_stats
        assert "sharpe_ratio" in regime_stats
        assert "skewness" in regime_stats
        assert "kurtosis" in regime_stats
        assert "var_95" in regime_stats
        assert "frequency" in regime_stats
        
        assert regime_stats["frequency"] > 0
        assert regime_stats["frequency"] <= 1

def test_model_persistence(sample_data, tmp_path):
    """Test model saving and loading."""
    # Create detector with custom model directory
    params = RegimeDetectionParams(model_dir=str(tmp_path))
    detector = MarketRegimeDetector(params)
    
    # Train models
    labels_hmm, _ = detector.detect_regimes_hmm(
        sample_data['returns'],
        sample_data['volumes']
    )
    
    labels_gmm, _ = detector.detect_regimes_gmm(
        sample_data['returns'],
        sample_data['volumes']
    )
    
    # Save models
    detector.save_models()
    
    # Create new detector and load models
    new_detector = MarketRegimeDetector(params)
    new_detector.load_models()
    
    # Test loaded models
    new_labels_hmm, _ = new_detector.detect_regimes_hmm(
        sample_data['returns'],
        sample_data['volumes']
    )
    
    new_labels_gmm, _ = new_detector.detect_regimes_gmm(
        sample_data['returns'],
        sample_data['volumes']
    )
    
    # Compare results
    assert_array_equal(labels_hmm, new_labels_hmm)
    assert_array_equal(labels_gmm, new_labels_gmm)

def test_helper_functions(sample_data):
    """Test helper functions."""
    detector = MarketRegimeDetector()
    
    # Test volatility calculation
    vol = detector._calculate_volatility(sample_data['returns'])
    assert len(vol) == len(sample_data['returns'])
    assert np.all(vol >= 0)  # Volatility is non-negative
    
    # Test volume normalization
    norm_vol = detector._normalize_volume(sample_data['volumes'])
    assert len(norm_vol) == len(sample_data['volumes'])
    assert abs(np.mean(norm_vol[detector.params.zscore_window:])) < 0.1
    
    # Test Z-score calculation
    zscore = detector._calculate_zscore(sample_data['returns'])
    assert len(zscore) == len(sample_data['returns'])
    assert abs(np.mean(zscore[detector.params.zscore_window:])) < 0.1
    
    # Test EMA calculation
    ema = detector._calculate_ema(sample_data['prices'], 20)
    assert len(ema) == len(sample_data['prices'])
    
    # Test skewness calculation
    skew = detector._calculate_skewness(sample_data['returns'])
    assert isinstance(skew, float)
    
    # Test kurtosis calculation
    kurt = detector._calculate_kurtosis(sample_data['returns'])
    assert isinstance(kurt, float) 