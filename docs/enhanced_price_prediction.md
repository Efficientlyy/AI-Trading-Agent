# Enhanced Price Prediction Components

## Overview

The Enhanced Price Prediction system is a modular machine learning framework designed for financial market prediction. It combines technical analysis, sentiment analysis, and market microstructure data to generate trading signals.

## Components

### 1. Technical Features (`TechnicalFeatures`)

A class that calculates various technical indicators from price data.

#### Methods:
- `calculate_rsi(prices: NDArray[np.float64], period: int = 14) -> float`
  - Calculates the Relative Strength Index
  - Returns a value between 0 and 100
  - Handles edge cases like insufficient data

- `calculate_bb_position(prices: NDArray[np.float64], current_price: float) -> float`
  - Calculates the position within Bollinger Bands
  - Returns a value between 0 (at/below lower band) and 1 (at/above upper band)
  - Uses 20-period moving average and 2 standard deviations

- `calculate_trend_strength(prices: NDArray[np.float64], period: int = 14) -> float`
  - Calculates trend strength using price momentum and volatility
  - Returns a value between 0 (no trend) and 1 (strong trend)
  - Normalizes by volatility for comparison across assets

### 2. Feature Extractor (`FeatureExtractor`)

A class that combines different types of features into a unified feature vector.

#### Feature Types:
1. Technical Features (3 dimensions):
   - RSI
   - Bollinger Band Position
   - Trend Strength

2. Sentiment Features (4 dimensions):
   - Social Media Sentiment
   - News Sentiment
   - Order Flow Sentiment
   - Fear/Greed Index

3. Market Features (3 dimensions):
   - Liquidity Score
   - Volatility
   - Correlation Score

#### Methods:
- `extract_features(prices: NDArray[np.float64], sentiment_data: Dict[str, float], market_data: Dict[str, float]) -> FeatureVector`
  - Extracts all features from input data
  - Returns a structured dictionary of feature arrays

- `combine_features(features: FeatureVector) -> NDArray[np.float64]`
  - Combines all feature arrays into a single vector
  - Returns a 10-dimensional feature vector

### 3. Model Predictor (`ModelPredictor`)

A class that handles model prediction with proper scaling and type safety.

#### Methods:
- `predict(features: NDArray[np.float64], min_price_move: float = 0.0001) -> ModelPrediction`
  - Generates predictions from feature vectors
  - Returns a dictionary containing:
    - prediction: Expected price movement
    - confidence: Model confidence score
    - direction: Trading signal (-1, 0, 1)
    - timestamp: Prediction timestamp
    - features: Input features used

### 4. Model Trainer (`ModelTrainer`)

A class for training and validating models with cross-validation.

#### Supported Models:
- Random Forest Classifier
- Gradient Boosting Classifier

#### Methods:
- `train_and_validate(features: NDArray[np.float64], labels: NDArray[np.float64], sample_weights: Optional[NDArray[np.float64]] = None) -> List[TrainingMetrics]`
  - Trains model using time series cross-validation
  - Returns metrics for each fold:
    - accuracy
    - precision
    - recall
    - f1_score
    - train_samples
    - validation_samples
    - training_time
    - timestamp

### 5. Model Validator (`ModelValidator`)

A class for calculating trading performance metrics.

#### Methods:
- `calculate_profit_factor(predictions: NDArray[np.float64], actual_returns: NDArray[np.float64]) -> float`
  - Calculates ratio of winning trades to losing trades
  - Handles edge cases like no trades or no losses

- `calculate_sharpe_ratio(predictions: NDArray[np.float64], actual_returns: NDArray[np.float64], risk_free_rate: float = 0.02) -> float`
  - Calculates risk-adjusted returns
  - Annualizes returns and volatility
  - Handles edge case of zero volatility

- `calculate_max_drawdown(predictions: NDArray[np.float64], actual_returns: NDArray[np.float64]) -> float`
  - Calculates maximum peak-to-trough decline
  - Returns a value between 0 and 1
  - Handles edge case of constant equity curve

## Usage Example

See `examples/enhanced_price_prediction_example.py` for a complete example that demonstrates:
1. Data generation for different market scenarios
2. Feature extraction and engineering
3. Model training with cross-validation
4. Real-time prediction simulation
5. Performance analysis with multiple metrics

## Market Scenarios

The system is tested against various market scenarios to ensure robust performance:

### Basic Market Conditions
1. Strong Bull Market
   - Upward trend with low volatility
   - High liquidity and bullish sentiment
   - Optimal conditions for trend following

2. Strong Bear Market
   - Downward trend with high volatility
   - Low liquidity and bearish sentiment
   - Challenging conditions for most strategies

3. Sideways Market
   - Range-bound price action
   - Neutral sentiment and normal liquidity
   - Tests mean reversion capabilities

### Mixed Conditions
1. Bull Market with High Volatility
   - Tests robustness of trend following
   - Challenges position sizing
   - Higher risk of whipsaws

2. Bear Market with Low Volatility
   - Gradual decline scenarios
   - Tests early warning capabilities
   - Important for risk management

### Sentiment-Price Divergence
1. Bearish Sentiment in Bull Market
   - Tests handling of conflicting signals
   - Important for contrarian strategies
   - Validates sentiment weighting

2. Bullish Sentiment in Bear Market
   - Tests false positive detection
   - Important for avoiding bull traps
   - Validates risk controls

### Liquidity Scenarios
1. High Volatility Low Liquidity
   - Stress test conditions
   - Tests execution assumptions
   - Important for risk limits

2. Low Volatility High Liquidity
   - Optimal trading conditions
   - Tests profit taking strategies
   - Validates transaction cost models

### Market Regime Changes
1. Transition to High Volatility
   - Tests adaptation to changing conditions
   - Important for dynamic position sizing
   - Validates stop-loss mechanisms

2. Recovery from Bear Market
   - Tests trend reversal detection
   - Important for re-entry strategies
   - Validates bottom detection

### Random Walk Variations
1. Efficient Market
   - Tests performance in efficient conditions
   - Validates transaction cost impact
   - Important for strategy viability

2. Inefficient Market
   - Tests exploitation of inefficiencies
   - Validates alpha generation
   - Important for strategy differentiation

## Test Cases

### Technical Features
1. Basic Calculations
   - RSI with standard parameters
   - Bollinger Band positioning
   - Trend strength measurement

2. Edge Cases
   - Insufficient data handling
   - Constant price sequences
   - Missing or invalid data

3. Error Conditions
   - NaN values in price data
   - Infinite values
   - Zero-length sequences

### Feature Extraction
1. Data Validation
   - Price data integrity
   - Sentiment range validation
   - Market data completeness

2. Missing Data
   - Partial sentiment data
   - Incomplete market data
   - Missing technical data

3. Feature Combination
   - Dimension verification
   - Type consistency
   - Numerical stability

### Model Prediction
1. Input Validation
   - Feature vector shape
   - Data type consistency
   - Value range checks

2. Edge Cases
   - Zero feature vectors
   - Extreme feature values
   - NaN/Infinite values

3. Error Handling
   - Invalid feature counts
   - Scaling errors
   - Model prediction failures

### Model Training
1. Data Quality
   - Feature-label alignment
   - Sample weight validation
   - Cross-validation splits

2. Edge Cases
   - Single class training
   - Imbalanced classes
   - Minimal training data

3. Error Conditions
   - NaN in features/labels
   - Mismatched dimensions
   - Invalid model parameters

### Performance Validation
1. Basic Metrics
   - Profit factor calculation
   - Sharpe ratio computation
   - Maximum drawdown

2. Edge Cases
   - No trades
   - No losing trades
   - Constant equity curve

3. Error Handling
   - Mismatched array lengths
   - Invalid return sequences
   - Negative rates

## Performance Analysis

The system provides detailed performance analysis across scenarios:

### Market Condition Analysis
- Average metrics by market type (Bull/Bear/Neutral)
- Impact of volatility on performance
- Consistency across different conditions

### Risk Metrics
- Maximum drawdown by scenario
- Sharpe ratio distribution
- Profit factor analysis

### Consistency Analysis
- Standard deviation of metrics
- Performance persistence
- Scenario sensitivity

## Best Practices

1. Data Preparation
   - Validate input data quality
   - Handle missing values appropriately
   - Normalize feature ranges

2. Model Training
   - Use appropriate cross-validation
   - Balance training data
   - Monitor for overfitting

3. Risk Management
   - Implement position sizing
   - Use appropriate stop-losses
   - Monitor drawdown limits

4. Performance Monitoring
   - Track consistency metrics
   - Monitor regime changes
   - Validate assumptions regularly

## Error Handling

All components implement robust error handling:
1. Return safe default values on error
2. Log warnings for unexpected conditions
3. Handle edge cases gracefully
4. Validate input data types and shapes

## Type Safety

The system uses strict type hints throughout:
1. NumPy array types with specific dtypes
2. Custom TypedDict definitions for structured data
3. Optional types for nullable values
4. Union types for polymorphic functions

## Performance Considerations

1. Feature Calculation:
   - Vectorized operations for speed
   - Efficient numpy operations
   - Minimal data copying

2. Model Training:
   - Time series cross-validation
   - Proper train/test splitting
   - Sample weight support

3. Prediction:
   - Fast feature extraction
   - Efficient model inference
   - Minimal data transformation

## Dependencies

- numpy: Array operations and numerical computations
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning models and utilities
- typing_extensions: Advanced type hints 