# Market Regime Detection

This module provides tools for detecting and analyzing market regimes using various statistical methods. Market regimes are distinct states or conditions in financial markets that exhibit different characteristics in terms of returns, volatility, trends, and correlations.

## Features

- Multiple regime detection methods:
  - Hidden Markov Models (HMM)
  - Gaussian Mixture Models (GMM)
  - Volatility-based regimes
  - Trend-based regimes
  - Correlation-based regimes
- Comprehensive regime statistics
- Model persistence capabilities
- Input validation and error handling

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy scikit-learn hmmlearn
```

## Usage

### Basic Usage

```python
import numpy as np
from ml.features.market_regime import MarketRegimeDetector

# Initialize detector
detector = MarketRegimeDetector()

# Prepare your data
returns = np.array([...])  # Daily returns
volumes = np.array([...])  # Daily volumes

# Detect regimes using HMM
labels, probabilities = detector.detect_regimes_hmm(returns, volumes)

# Get regime statistics
stats = detector.get_regime_statistics(returns, labels)
```

### Configuration

You can customize the detector's parameters:

```python
from ml.features.market_regime import RegimeDetectionParams

params = RegimeDetectionParams(
    n_regimes=3,              # Number of regimes to detect
    lookback_window=252,      # One year of daily data
    min_samples=63,           # Minimum samples required (3 months)
    volatility_window=21,     # Window for volatility calculation
    correlation_window=63,    # Window for correlation calculation
    zscore_window=21,         # Window for Z-score calculation
    hmm_n_iter=100,          # HMM training iterations
    model_dir="models"        # Directory to save trained models
)

detector = MarketRegimeDetector(params)
```

### Regime Detection Methods

#### 1. Hidden Markov Model (HMM)

Uses a Hidden Markov Model to detect regimes based on returns, volatility, and volume:

```python
labels, probabilities = detector.detect_regimes_hmm(returns, volumes)
```

The HMM method:
- Considers the temporal dependence between regimes
- Provides regime probabilities
- Works well for capturing regime persistence

#### 2. Gaussian Mixture Model (GMM)

Uses a Gaussian Mixture Model for regime detection:

```python
labels, probabilities = detector.detect_regimes_gmm(returns, volumes)
```

The GMM method:
- Assumes regimes are independent
- Provides regime probabilities
- Useful for identifying distinct market conditions

#### 3. Volatility Regimes

Detects regimes based on volatility levels:

```python
labels = detector.detect_volatility_regimes(returns)
```

Returns labels:
- 0: Low volatility
- 1: Normal volatility
- 2: High volatility

#### 4. Trend Regimes

Detects trend regimes using moving averages:

```python
labels = detector.detect_trend_regimes(prices)
```

Returns labels:
- 0: Downtrend
- 1: Sideways
- 2: Uptrend

#### 5. Correlation Regimes

Detects correlation regimes in multi-asset systems:

```python
# asset_returns shape: (n_samples, n_assets)
labels = detector.detect_correlation_regimes(asset_returns)
```

Returns labels:
- 0: Low correlation
- 1: Normal correlation
- 2: High correlation

### Advanced Regime Detection Methods

#### 1. Momentum Regimes

Detect market momentum using multiple lookback periods:

```python
# Detect momentum regimes
labels = detector.detect_momentum_regimes(returns)

# Custom lookback periods
labels = detector.detect_momentum_regimes(
    returns,
    lookback_periods=[10, 30, 90]  # Short, medium, long-term momentum
)
```

Returns labels:
- 0: Negative momentum
- 1: Neutral momentum
- 2: Positive momentum

#### 2. Liquidity Regimes

Detect market liquidity using volume and spread data:

```python
# Detect liquidity regimes
labels = detector.detect_liquidity_regimes(volumes, spreads)
```

Returns labels:
- 0: Low liquidity (low volume, high spreads)
- 1: Normal liquidity
- 2: High liquidity (high volume, low spreads)

#### 3. Sentiment Regimes

Detect market sentiment using price-volume relationships:

```python
# Detect sentiment regimes
labels = detector.detect_sentiment_regimes(
    returns,
    volumes,
    window=21  # Optional window size
)
```

Returns labels:
- 0: Bearish sentiment
- 1: Neutral sentiment
- 2: Bullish sentiment

#### 4. Volatility Structure

Analyze volatility term structure:

```python
# Detect volatility structure regimes
labels = detector.detect_volatility_structure(
    returns,
    windows=[5, 21, 63]  # Short to long-term volatility windows
)
```

Returns labels:
- 0: Contango (short-term vol < long-term vol)
- 1: Flat term structure
- 2: Backwardation (short-term vol > long-term vol)

### Example: Combined Regime Analysis

```python
import numpy as np
from ml.features.market_regime import MarketRegimeDetector

# Initialize detector
detector = MarketRegimeDetector()

# Prepare data
returns = np.array([...])  # Daily returns
volumes = np.array([...])  # Daily volumes
spreads = np.array([...])  # Daily bid-ask spreads

# Detect different types of regimes
momentum_labels = detector.detect_momentum_regimes(returns)
liquidity_labels = detector.detect_liquidity_regimes(volumes, spreads)
sentiment_labels = detector.detect_sentiment_regimes(returns, volumes)
vol_struct_labels = detector.detect_volatility_structure(returns)

# Analyze regime overlaps
optimal_conditions = np.where(
    (momentum_labels == 2) &    # Strong momentum
    (liquidity_labels == 2) &   # High liquidity
    (sentiment_labels == 2) &   # Bullish sentiment
    (vol_struct_labels == 1)    # Stable volatility
)[0]

print(f"Found {len(optimal_conditions)} periods with optimal trading conditions")
```

### Best Practices for Advanced Regimes

1. **Momentum Analysis**
   - Use multiple lookback periods to capture different time horizons
   - Consider using exponentially weighted returns for recent emphasis
   - Validate momentum signals against volume patterns

2. **Liquidity Analysis**
   - Normalize volumes across different market regimes
   - Consider intraday liquidity patterns if available
   - Account for seasonal variations in liquidity

3. **Sentiment Analysis**
   - Combine price-volume analysis with other sentiment indicators
   - Consider market-specific sentiment patterns
   - Account for news and event impacts

4. **Volatility Structure**
   - Monitor regime transitions for early warning signals
   - Consider cross-asset volatility relationships
   - Use term structure for risk management decisions

### Performance Optimization

1. **Computational Efficiency**
   - Cache intermediate calculations (e.g., volatility)
   - Use vectorized operations where possible
   - Consider parallel processing for multiple assets

2. **Memory Management**
   - Use appropriate data types (np.float32 vs np.float64)
   - Clear cache for long-running applications
   - Implement efficient data storage for historical regimes

3. **Real-time Processing**
   - Implement streaming calculations where possible
   - Use incremental updates for rolling windows
   - Optimize critical paths for minimum latency

### Regime Statistics

Get detailed statistics for each regime:

```python
stats = detector.get_regime_statistics(returns, labels)
```

Statistics include:
- Mean return
- Volatility
- Sharpe ratio
- Skewness
- Kurtosis
- Value at Risk (95%)
- Regime frequency

### Model Persistence

Save trained models for later use:

```python
# Save models
detector.save_models(prefix="my_models_")

# Load models in a new session
new_detector = MarketRegimeDetector()
new_detector.load_models(prefix="my_models_")
```

## Best Practices

1. **Data Preparation**
   - Use sufficient historical data (at least 3 months)
   - Handle missing values and outliers
   - Ensure data is properly aligned

2. **Regime Detection**
   - Use multiple methods for robustness
   - Consider the temporal aspect of regimes
   - Validate regime changes against market events

3. **Parameter Tuning**
   - Adjust window sizes based on your trading frequency
   - Consider the trade-off between sensitivity and stability
   - Use cross-validation for optimal parameters

4. **Model Persistence**
   - Save models regularly
   - Use descriptive prefixes for different model versions
   - Keep track of training dates

## Error Handling

The module includes comprehensive error handling:

```python
try:
    labels, probs = detector.detect_regimes_hmm(returns, volumes)
except ValueError as e:
    print(f"Error: {e}")  # e.g., "Insufficient data for regime detection"
```

Common errors:
- Insufficient data
- Invalid input types
- NaN or infinite values
- Zero variance features

## Performance Considerations

1. **Computational Efficiency**
   - HMM and GMM training can be computationally intensive
   - Consider using smaller windows for real-time applications
   - Cache results for frequently accessed periods

2. **Memory Usage**
   - Large datasets may require batch processing
   - Consider downsampling for very long historical periods
   - Use appropriate data types (np.float64)

## Advanced Examples

### Example 1: Multi-Asset Regime Analysis

```python
import numpy as np
import pandas as pd
from ml.features.market_regime import MarketRegimeDetector
from ml.visualization.regime_visualizer import RegimeVisualizer

# Initialize detector and visualizer
detector = MarketRegimeDetector()
visualizer = RegimeVisualizer()

# Load multi-asset data
data = pd.read_csv('market_data.csv')
dates = data['date'].values
assets = ['SPY', 'TLT', 'GLD', 'VIX']
returns = np.array([data[asset + '_returns'].values for asset in assets]).T
volumes = np.array([data[asset + '_volume'].values for asset in assets]).T
prices = np.array([data[asset + '_close'].values for asset in assets]).T

# Detect regimes for each asset
regime_labels = {}
regime_stats = {}

for i, asset in enumerate(assets):
    # Detect different types of regimes
    momentum = detector.detect_momentum_regimes(returns[:, i])
    volatility = detector.detect_volatility_regimes(returns[:, i])
    trend = detector.detect_trend_regimes(prices[:, i])
    
    # Store labels
    regime_labels[f'{asset}_momentum'] = momentum
    regime_labels[f'{asset}_volatility'] = volatility
    regime_labels[f'{asset}_trend'] = trend
    
    # Calculate statistics
    regime_stats[asset] = detector.get_regime_statistics(
        returns[:, i],
        momentum  # Using momentum regimes for stats
    )

# Create regime matrix for heatmap
regime_matrix = np.vstack([
    labels for labels in regime_labels.values()
])

# Visualize results
transitions_fig = visualizer.plot_regime_transitions(
    dates=dates,
    labels=regime_labels['SPY_momentum'],
    prices=prices[:, 0],
    regime_names=['Negative', 'Neutral', 'Positive'],
    title='SPY Momentum Regimes'
)

heatmap_fig = visualizer.plot_regime_heatmap(
    regime_matrix=regime_matrix,
    dates=dates,
    method_names=list(regime_labels.keys()),
    title='Cross-Asset Regime Comparison'
)

# Create 3D visualization of regime transitions
features = np.column_stack([
    returns[:, 0],  # SPY returns
    volumes[:, 0],  # SPY volume
    returns[:, -1]  # VIX returns
])

transitions_3d = visualizer.plot_regime_transitions_3d(
    features=features,
    labels=regime_labels['SPY_momentum'],
    feature_names=['Returns', 'Volume', 'VIX'],
    title='SPY Regime Transitions in 3D'
)

# Create comprehensive dashboard
dashboard = visualizer.plot_regime_dashboard(
    dates=dates,
    prices=prices[:, 0],
    returns=returns[:, 0],
    volumes=volumes[:, 0],
    labels=regime_labels,
    stats=regime_stats,
    title='Market Regime Analysis Dashboard'
)
```

### Example 2: Trading Strategy with Regime Filters

```python
import numpy as np
from ml.features.market_regime import MarketRegimeDetector

class RegimeBasedStrategy:
    def __init__(self):
        self.detector = MarketRegimeDetector()
        
    def generate_signals(self, returns, volumes, spreads):
        # Detect various regimes
        momentum = self.detector.detect_momentum_regimes(returns)
        liquidity = self.detector.detect_liquidity_regimes(volumes, spreads)
        sentiment = self.detector.detect_sentiment_regimes(returns, volumes)
        volatility = self.detector.detect_volatility_regimes(returns)
        
        # Define optimal trading conditions
        good_conditions = (
            (momentum == 2) &        # Strong momentum
            (liquidity == 2) &       # High liquidity
            (sentiment == 2) &       # Bullish sentiment
            (volatility == 1)        # Normal volatility
        )
        
        bad_conditions = (
            (momentum == 0) |        # Weak momentum
            (liquidity == 0) |       # Low liquidity
            (volatility == 2)        # High volatility
        )
        
        # Generate signals
        signals = np.zeros_like(returns)
        signals[good_conditions] = 1    # Long signals
        signals[bad_conditions] = -1    # Exit signals
        
        return signals

# Usage example
strategy = RegimeBasedStrategy()
signals = strategy.generate_signals(returns, volumes, spreads)
```

### Example 3: Real-time Regime Monitoring

```python
import numpy as np
from ml.features.market_regime import MarketRegimeDetector
import time

class RegimeMonitor:
    def __init__(self, lookback_window=252):
        self.detector = MarketRegimeDetector()
        self.lookback = lookback_window
        self.current_regimes = {}
        
    def update(self, new_data):
        """Update regime detection with new data."""
        returns = new_data['returns'][-self.lookback:]
        volumes = new_data['volumes'][-self.lookback:]
        spreads = new_data['spreads'][-self.lookback:]
        
        # Detect current regimes
        self.current_regimes = {
            'momentum': self.detector.detect_momentum_regimes(returns)[-1],
            'liquidity': self.detector.detect_liquidity_regimes(
                volumes, spreads
            )[-1],
            'sentiment': self.detector.detect_sentiment_regimes(
                returns, volumes
            )[-1],
            'volatility': self.detector.detect_volatility_structure(
                returns
            )[-1]
        }
        
        # Check for regime changes
        regime_changes = self._check_regime_changes()
        
        return self.current_regimes, regime_changes
    
    def _check_regime_changes(self):
        """Identify significant regime changes."""
        # Implementation details...
        pass

# Usage example
monitor = RegimeMonitor()

while True:
    # Get new market data
    new_data = fetch_market_data()  # Implementation needed
    
    # Update regime detection
    regimes, changes = monitor.update(new_data)
    
    # Take action on regime changes
    if changes:
        print(f"Regime changes detected: {changes}")
        # Implement response to regime changes
    
    time.sleep(60)  # Update every minute
```

### Example 4: Regime-Based Risk Management

```python
import numpy as np
from ml.features.market_regime import MarketRegimeDetector

class RegimeBasedRiskManager:
    def __init__(self):
        self.detector = MarketRegimeDetector()
        
    def calculate_position_size(
        self,
        returns,
        volumes,
        capital,
        max_risk_pct=0.02
    ):
        """Calculate position size based on regime conditions."""
        # Detect regimes
        vol_regime = self.detector.detect_volatility_regimes(returns)
        liq_regime = self.detector.detect_liquidity_regimes(
            volumes,
            spreads
        )
        
        # Adjust risk based on regimes
        risk_multipliers = {
            0: 0.5,    # Reduce risk in low liquidity
            1: 1.0,    # Normal risk
            2: 0.75    # Slightly reduce risk in high volatility
        }
        
        # Use worst-case regime for risk adjustment
        risk_regime = min(vol_regime[-1], liq_regime[-1])
        risk_multiplier = risk_multipliers[risk_regime]
        
        # Calculate volatility
        volatility = np.std(returns[-21:])  # 21-day volatility
        
        # Calculate position size
        max_risk = capital * max_risk_pct * risk_multiplier
        position_size = max_risk / (volatility * np.sqrt(252))
        
        return position_size, risk_regime

# Usage example
risk_manager = RegimeBasedRiskManager()
position_size, regime = risk_manager.calculate_position_size(
    returns,
    volumes,
    capital=100000
)
```

### Example 5: Cross-Asset Regime Analysis

```python
import numpy as np
from ml.features.market_regime import MarketRegimeDetector
from ml.visualization.regime_visualizer import RegimeVisualizer

def analyze_cross_asset_regimes(returns_dict, volumes_dict):
    """Analyze regime relationships across multiple assets."""
    detector = MarketRegimeDetector()
    visualizer = RegimeVisualizer()
    
    # Store regime labels for each asset
    regime_labels = {}
    
    # Detect regimes for each asset
    for asset, returns in returns_dict.items():
        regime_labels[asset] = {
            'momentum': detector.detect_momentum_regimes(returns),
            'volatility': detector.detect_volatility_regimes(returns),
            'sentiment': detector.detect_sentiment_regimes(
                returns,
                volumes_dict[asset]
            )
        }
    
    # Calculate regime correlations
    correlations = {}
    for asset1 in regime_labels:
        for asset2 in regime_labels:
            if asset1 < asset2:
                for regime_type in ['momentum', 'volatility', 'sentiment']:
                    key = f"{asset1}_{asset2}_{regime_type}"
                    correlations[key] = np.corrcoef(
                        regime_labels[asset1][regime_type],
                        regime_labels[asset2][regime_type]
                    )[0, 1]
    
    return regime_labels, correlations

# Usage example
assets = {
    'SPY': {'returns': spy_returns, 'volumes': spy_volumes},
    'TLT': {'returns': tlt_returns, 'volumes': tlt_volumes},
    'GLD': {'returns': gld_returns, 'volumes': gld_volumes}
}

returns_dict = {k: v['returns'] for k, v in assets.items()}
volumes_dict = {k: v['volumes'] for k, v in assets.items()}

regime_labels, correlations = analyze_cross_asset_regimes(
    returns_dict,
    volumes_dict
)
```

These examples demonstrate:
1. Multi-asset regime analysis with visualization
2. Regime-based trading strategy implementation
3. Real-time regime monitoring system
4. Risk management using regime information
5. Cross-asset regime analysis and correlation

Each example includes:
- Complete implementation details
- Practical usage scenarios
- Integration with visualization tools
- Error handling considerations
- Performance optimization techniques

## Contributing

When contributing to this module:
1. Add tests for new features
2. Document your changes
3. Follow the existing code style
4. Update the documentation 