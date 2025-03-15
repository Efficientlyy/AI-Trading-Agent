# Market Regime Detection

This module provides algorithms for detecting market regimes in financial time series data. Market regimes are distinct periods in financial markets characterized by specific patterns of volatility, returns, and other market behaviors.

## Features

- Multiple regime detection methods:
  - **Volatility-based**: Detects regimes based on volatility levels
  - **Momentum-based**: Detects regimes based on momentum indicators (MACD, ROC, RSI)
  - **Hidden Markov Model (HMM)**: Detects regimes using unsupervised learning
  - **Trend-based**: Detects regimes based on trend strength and direction
  - **Ensemble**: Combines multiple detection methods for more robust results

- Factory pattern for easy creation and usage of detectors
- Comprehensive statistics for each detected regime
- Visualization tools for regime analysis

## Usage

### Basic Usage

```python
from src.ml.detection import RegimeDetectorFactory

# Create a detector
factory = RegimeDetectorFactory()
detector = factory.create('volatility', n_regimes=3)

# Prepare market data
data = {
    'dates': [...],  # List of dates
    'prices': [...],  # List of prices
    'returns': [...],  # List of returns
    'volumes': [...]   # List of volumes
}

# Detect regimes
labels = detector.fit_predict(data)

# Get regime statistics
stats = detector.regime_stats
print(stats)
```

### Available Methods

- `volatility`: Volatility-based regime detection
- `momentum`: Momentum-based regime detection
- `hmm`: Hidden Markov Model based regime detection
- `trend`: Trend-based regime detection
- `ensemble`: Ensemble-based regime detection (combines multiple methods)

### Running the Example

The module includes an example script that demonstrates how to use the regime detection algorithms with real market data:

```bash
# Run with default settings (SPY, 5 years)
python src/ml/detection/run_example.py

# Run with custom settings
python src/ml/detection/run_example.py --symbol AAPL --period 3y --output-dir ./plots
```

## Extending

### Adding a New Detector

To add a new regime detection method:

1. Create a new class that extends `BaseRegimeDetector`
2. Implement the required methods: `fit()` and `detect()`
3. Register the new detector with the factory:

```python
from src.ml.detection import RegimeDetectorFactory
from .my_detector import MyDetector

RegimeDetectorFactory.register('my_method', MyDetector)
```

## API Reference

### BaseRegimeDetector

Abstract base class for all regime detection algorithms.

- `__init__(n_regimes=3, lookback_window=60, **kwargs)`: Initialize the detector
- `fit(data)`: Fit the detector to the data
- `detect(data)`: Detect regimes in the data
- `fit_predict(data)`: Fit and detect in one step
- `calculate_regime_statistics(data, labels)`: Calculate statistics for each regime

### VolatilityRegimeDetector

Detects regimes based on volatility levels.

- `__init__(n_regimes=3, lookback_window=60, vol_window=21, use_log_returns=True, use_ewm=False, ewm_alpha=0.1, **kwargs)`: Initialize the detector
- `get_volatility_series()`: Get the calculated volatility series
- `get_regime_thresholds()`: Get the volatility thresholds between regimes

### MomentumRegimeDetector

Detects regimes based on momentum indicators.

- `__init__(n_regimes=3, lookback_window=60, momentum_type='roc', fast_window=12, slow_window=26, signal_window=9, roc_window=20, rsi_window=14, **kwargs)`: Initialize the detector
- `get_momentum_series()`: Get the calculated momentum series
- `get_regime_thresholds()`: Get the momentum thresholds between regimes

### HMMRegimeDetector

Detects regimes using Hidden Markov Models.

- `__init__(n_regimes=3, lookback_window=60, hmm_type='gaussian', n_iter=100, random_state=42, use_returns=True, use_log_returns=True, **kwargs)`: Initialize the detector
- `get_transition_matrix()`: Get the transition probability matrix between regimes
- `get_regime_means()`: Get the mean values for each regime
- `get_regime_covariances()`: Get the covariance matrices for each regime
- `predict_next_regime()`: Predict the next regime based on the current regime

### TrendRegimeDetector

Detects regimes based on trend strength and direction.

- `__init__(n_regimes=3, lookback_window=60, trend_method='ma_crossover', fast_ma=20, slow_ma=50, adx_window=14, adx_threshold=25.0, slope_window=20, use_log_prices=False, **kwargs)`: Initialize the detector
- `get_trend_series()`: Get the calculated trend indicator series
- `get_regime_thresholds()`: Get the trend thresholds between regimes

### EnsembleRegimeDetector

Combines multiple regime detection methods for more robust results.

- `__init__(n_regimes=3, lookback_window=60, methods=None, weights=None, voting='soft', ensemble_type='bagging', normalize_outputs=True, **kwargs)`: Initialize the detector
- `get_ensemble_probas()`: Get the ensemble probabilities for each regime
- `get_individual_labels()`: Get the labels from each individual detector
- `get_detector_weights()`: Get the weights for each detector
- `set_detector_weights(weights)`: Set the weights for each detector
- `get_detector_names()`: Get the names of the detectors

The ensemble detector supports three ensemble techniques:
- **Bagging**: Combines detector outputs through voting (hard or soft)
- **Boosting**: Weights detectors based on their performance
- **Stacking**: Uses a meta-model to combine detector outputs

### RegimeDetectorFactory

Factory for creating regime detection algorithms.

- `create(method, **kwargs)`: Create a detector instance
- `register(name, detector_class)`: Register a new detector class
- `get_available_methods()`: Get a list of available detector methods
- `create_all(methods=None, **kwargs)`: Create multiple detector instances 