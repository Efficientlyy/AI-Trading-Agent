"""Unit tests for new market regime detection methods."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..src.ml.features.market_regime import MarketRegimeDetector

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 252  # One year of daily data
    
    # Generate synthetic returns
    returns = np.zeros(n_samples)
    
    # Create momentum regimes
    returns[:84] = np.random.normal(0.001, 0.005, 84)  # Positive momentum
    returns[84:168] = np.random.normal(0, 0.01, 84)    # Neutral momentum
    returns[168:] = np.random.normal(-0.001, 0.015, 84)  # Negative momentum
    
    # Generate synthetic volumes and spreads
    volumes = np.exp(np.random.normal(10, 1, n_samples))
    spreads = np.random.gamma(2, 0.002, n_samples)
    
    # Induce regime patterns
    volumes[168:] *= 2  # Higher volumes in last period
    spreads[168:] *= 0.5  # Lower spreads in last period
    
    return {
        'returns': returns,
        'volumes': volumes,
        'spreads': spreads
    }

def test_momentum_regimes(sample_data):
    """Test momentum regime detection."""
    detector = MarketRegimeDetector()
    
    # Test with default parameters
    labels = detector.detect_momentum_regimes(sample_data['returns'])
    
    assert len(labels) == len(sample_data['returns'])
    assert set(labels) <= {0, 1, 2}
    
    # Verify regime patterns
    early_labels = labels[:84]
    assert np.mean(early_labels == 2) > 0.4  # Mostly positive momentum
    
    late_labels = labels[168:]
    assert np.mean(late_labels == 0) > 0.4  # Mostly negative momentum

def test_liquidity_regimes(sample_data):
    """Test liquidity regime detection."""
    detector = MarketRegimeDetector()
    
    # Test regime detection
    labels = detector.detect_liquidity_regimes(
        sample_data['volumes'],
        sample_data['spreads']
    )
    
    assert len(labels) == len(sample_data['volumes'])
    assert set(labels) <= {0, 1, 2}
    
    # Verify high liquidity in last period (high volume, low spreads)
    late_labels = labels[168:]
    assert np.mean(late_labels == 2) > 0.4

def test_sentiment_regimes(sample_data):
    """Test sentiment regime detection."""
    detector = MarketRegimeDetector()
    
    # Test regime detection
    labels = detector.detect_sentiment_regimes(
        sample_data['returns'],
        sample_data['volumes']
    )
    
    assert len(labels) == len(sample_data['returns'])
    assert set(labels) <= {0, 1, 2}
    
    # Verify regime transitions
    assert len(np.unique(labels)) >= 2  # At least two different regimes

def test_volatility_structure(sample_data):
    """Test volatility structure detection."""
    detector = MarketRegimeDetector()
    
    # Test regime detection
    labels = detector.detect_volatility_structure(sample_data['returns'])
    
    assert len(labels) == len(sample_data['returns'])
    assert set(labels) <= {0, 1, 2}
    
    # Test with custom windows
    custom_labels = detector.detect_volatility_structure(
        sample_data['returns'],
        windows=[10, 30, 90]
    )
    
    assert len(custom_labels) == len(sample_data['returns'])
    assert set(custom_labels) <= {0, 1, 2}

def test_input_validation():
    """Test input validation for new methods."""
    detector = MarketRegimeDetector()
    
    # Test empty arrays
    with pytest.raises(ValueError, match="empty"):
        detector.detect_momentum_regimes(np.array([]))
    
    # Test arrays with NaN
    bad_data = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        detector.detect_momentum_regimes(bad_data)
    
    # Test length mismatch
    returns = np.random.randn(100)
    volumes = np.random.randn(50)
    with pytest.raises(ValueError, match="mismatch"):
        detector.detect_sentiment_regimes(returns, volumes)

def test_edge_cases(sample_data):
    """Test edge cases and boundary conditions."""
    detector = MarketRegimeDetector()
    
    # Test constant returns
    const_returns = np.zeros(252)
    labels = detector.detect_momentum_regimes(const_returns)
    assert len(labels) == len(const_returns)
    
    # Test extreme values
    extreme_returns = np.random.normal(0, 10, 252)  # Very high volatility
    labels = detector.detect_volatility_structure(extreme_returns)
    assert len(labels) == len(extreme_returns)
    
    # Test minimum required samples
    min_samples = detector.params.min_samples
    short_returns = np.random.randn(min_samples - 1)
    with pytest.raises(ValueError, match="Insufficient"):
        detector.detect_momentum_regimes(short_returns)

def test_seasonality_regimes(sample_data):
    """Test seasonality regime detection."""
    detector = MarketRegimeDetector()
    
    # Generate seasonal data
    t = np.linspace(0, 4*np.pi, 252)  # One year
    seasonal = 0.1 * np.sin(t) + 0.05 * np.random.randn(len(t))
    
    # Test regime detection
    labels = detector.detect_seasonality_regimes(seasonal)
    
    assert len(labels) == len(seasonal)
    assert set(labels) <= {0, 1, 2}
    
    # Test insufficient data
    short_data = np.random.randn(100)
    with pytest.raises(ValueError, match="Insufficient data"):
        detector.detect_seasonality_regimes(short_data)
    
    # Test with custom period
    custom_labels = detector.detect_seasonality_regimes(
        seasonal,
        period=126,  # Semi-annual
        min_periods=2
    )
    assert len(custom_labels) == len(seasonal)

def test_mean_reversion_regimes(sample_data):
    """Test mean reversion regime detection."""
    detector = MarketRegimeDetector()
    
    # Generate mean-reverting data
    n = 252
    prices = np.zeros(n)
    prices[0] = 100
    mean = 100
    
    for i in range(1, n):
        # Strong mean reversion
        prices[i] = prices[i-1] + 0.3 * (mean - prices[i-1]) + np.random.randn()
    
    # Test regime detection
    labels = detector.detect_mean_reversion_regimes(prices)
    
    assert len(labels) == len(prices)
    assert set(labels) <= {0, 1, 2}
    
    # Verify mean-reverting regime detection
    mean_rev_count = np.sum(labels == 2)
    assert mean_rev_count > n * 0.3  # At least 30% mean-reverting
    
    # Test with custom window
    custom_labels = detector.detect_mean_reversion_regimes(
        prices,
        window=42
    )
    assert len(custom_labels) == len(prices)

def test_market_stress_regimes(sample_data):
    """Test market stress regime detection."""
    detector = MarketRegimeDetector()
    
    # Generate stress periods
    n = 252
    returns = np.random.normal(0, 0.01, n)  # Normal period
    volumes = np.random.normal(1000, 100, n)
    
    # Add stress period
    stress_idx = slice(100, 150)
    returns[stress_idx] = np.random.normal(-0.02, 0.03, 50)  # Higher vol, negative returns
    volumes[stress_idx] = np.random.normal(2000, 300, 50)  # Higher volumes
    
    # Test regime detection
    labels = detector.detect_market_stress_regimes(returns, volumes)
    
    assert len(labels) == len(returns)
    assert set(labels) <= {0, 1, 2}
    
    # Verify stress regime detection
    stress_period_labels = labels[stress_idx]
    assert np.mean(stress_period_labels == 2) > 0.4  # High stress detected
    
    # Test with custom window
    custom_labels = detector.detect_market_stress_regimes(
        returns,
        volumes,
        window=42
    )
    assert len(custom_labels) == len(returns)
    
    # Test length mismatch
    with pytest.raises(ValueError, match="mismatch"):
        detector.detect_market_stress_regimes(
            returns,
            volumes[:100]
        )

def test_combined_regime_analysis():
    """Test combining multiple regime detection methods."""
    detector = MarketRegimeDetector()
    
    # Generate test data
    n = 252
    prices = np.zeros(n)
    prices[0] = 100
    returns = np.zeros(n)
    volumes = np.random.normal(1000, 100, n)
    
    # Create different regime periods
    # Trending up with low volatility
    prices[50:100] = np.linspace(100, 120, 50) + np.random.randn(50)
    
    # Mean-reverting with high volatility
    for i in range(100, 150):
        prices[i] = prices[i-1] + 0.5 * (110 - prices[i-1]) + 2 * np.random.randn()
    
    # Calculate returns
    returns[1:] = np.diff(prices) / prices[:-1]
    
    # Detect different regimes
    trend_labels = detector.detect_mean_reversion_regimes(prices)
    stress_labels = detector.detect_market_stress_regimes(returns, volumes)
    
    # Test regime overlap
    trend_up_stress_low = np.where(
        (trend_labels == 0) &  # Trending
        (stress_labels == 0)   # Low stress
    )[0]
    
    assert len(trend_up_stress_low) > 0  # Should find some overlap
    
    # Test regime transitions
    transitions = np.diff(trend_labels)
    regime_changes = np.where(transitions != 0)[0]
    assert len(regime_changes) > 0  # Should detect regime changes 