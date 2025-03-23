# Online Learning Integration

This document explains how the Online Learning Pipeline is integrated with the Adaptive ML Strategy in the AI Trading Agent.

## Overview

The Online Learning integration enables the Adaptive ML Strategy to automatically detect concept drift in market data and update its models in response. This allows the strategy to adapt to changing market conditions in real-time, maintaining model performance even as market regimes change.

## Key Components

### 1. OnlineLearningPipeline

The core component that handles:
- **Concept Drift Detection**: Monitors data distributions for significant changes
- **Incremental Model Training**: Updates models when drift is detected
- **Performance Tracking**: Records model performance before and after updates
- **Pipeline State Management**: Exports/imports pipeline state for persistence

### 2. AdaptiveMLOnlineAdapter

An adapter class that bridges the gap between the `OnlineLearningPipeline` and the `AdaptiveMLStrategy`:
- Monitors market data for concept drift
- Triggers model updates when drift is detected
- Tracks performance metrics
- Provides visualization of drift detection and model updates
- Handles regime change notifications
- Manages dynamic feature selection

### 3. Strategy Integration

The `AdaptiveMLStrategy` has been extended to:
- Initialize the online learning adapter
- Process regime change events
- Apply strategy adaptations based on detected regimes
- Maintain a history of adaptations

### 4. Dynamic Feature Selection

The system includes dynamic feature selection capabilities:
- **Feature Importance Tracking**: Monitors importance of features over time
- **Stability Analysis**: Identifies which features have stable importance
- **Drift Sensitivity**: Detects features that change importance during regime shifts
- **Regime-Specific Selection**: Creates feature sets optimized for specific market regimes
- **Visualization**: Provides visualizations of feature importance and stability

## Usage

### Initialization

```python
# Initialize strategy
strategy = AdaptiveMLStrategy(strategy_id="my_strategy")

# Initialize online learning with dynamic feature selection
strategy.initialize_online_learning(
    model_path="models/initial_model.joblib",
    models_dir="models",
    reference_data=initial_features,
    auto_update=True,
    dynamic_feature_selection=True,
    min_features=5,
    max_features=15
)
```

### Configuration Parameters

The following configuration parameters can be set in the strategy config:

```yaml
strategies:
  adaptive_ml:
    # Online Learning parameters
    ol_reference_window_size: 1000     # Size of reference window for drift detection
    ol_test_window_size: 500           # Size of test window for drift detection
    ol_drift_detection_method: ks_test  # Method for drift detection (ks_test or distribution)
    ol_drift_threshold: 0.05           # Threshold for drift detection
    ol_model_update_strategy: full_retrain  # Strategy for model updates
    ol_update_interval: 3600           # Minimum seconds between updates
    ol_auto_monitoring: true           # Auto-start monitoring
    ol_monitoring_interval: 300        # Seconds between monitoring checks
    
    # Dynamic Feature Selection parameters
    ol_dynamic_feature_selection: true  # Enable dynamic feature selection
    ol_min_features: 5                  # Minimum number of features to select
    ol_max_features: 15                 # Maximum number of features to select
    ol_importance_threshold: 0.01       # Minimum importance threshold for features
```

## Workflow

1. **Initialization**: The strategy loads initial models and sets reference data
2. **Monitoring**: The adapter continuously monitors market data for concept drift
3. **Drift Detection**: When drift is detected, the strategy is notified
4. **Model Update**: If auto-update is enabled, models are automatically retrained
5. **Feature Selection**: The system analyzes feature importance and updates feature selection
6. **Regime-Specific Features**: Feature sets are optimized for the detected market regime
7. **Strategy Adaptation**: The strategy parameters are adjusted based on the detected regime
8. **Performance Tracking**: Model performance is tracked before and after updates

## Visualization and Monitoring

The integration provides several tools for monitoring and visualization:

1. **Performance Metrics Dashboard**: Visualizes performance metrics over time
2. **Drift History**: Tracks when drift was detected and what actions were taken
3. **Model Version History**: Records all model versions and their performance
4. **Feature Importance Visualization**: Shows how feature importance evolves over time
5. **Feature Stability Analysis**: Maps features by stability and importance
6. **Regime-Specific Feature Sets**: Tracks which features are important in each regime
7. **Status Reports**: Provides comprehensive status reports of the online learning system
8. **Model Prediction Analysis**: Visualizes prediction accuracy and confidence across regimes
9. **Regime Transition Prediction**: Forecasts upcoming regime changes with confidence levels
10. **Configuration Recommendations**: Provides AI-driven suggestions for parameter optimization
11. **Regime-Specific Presets**: Maintains optimized configuration presets for different market regimes
12. **Configuration Impact Analysis**: Evaluates the performance impact of configuration changes with statistical confidence metrics

## Example

See `examples/online_learning_example.py` for a complete demonstration of the online learning integration.

## Benefits

1. **Adaptability**: Strategies automatically adapt to changing market conditions
2. **Continuous Learning**: Models continuously improve as new data arrives
3. **Regime Awareness**: Strategy parameters adjust based on detected market regimes
4. **Feature Optimization**: Feature sets are dynamically adjusted for each market regime
5. **Noise Reduction**: Unstable and unimportant features are pruned automatically
6. **Drift Sensitivity**: Features that are sensitive to regime changes are identified
7. **Performance Monitoring**: Comprehensive tracking of model performance over time
8. **Explainability**: Clear visibility into when and why models and features are updated
9. **Configuration Optimization**: AI-driven recommendations help optimize configuration parameters
10. **Regime-Specific Configurations**: Optimal configurations for different market conditions
11. **Impact Analysis**: Statistical evaluation of configuration change impacts

## Configuration Impact Analysis

The Configuration Impact Analysis system provides a comprehensive framework for evaluating how parameter changes affect strategy performance. It enables users to:

### Key Features

1. **Before/After Performance Comparison**: Analyze performance metrics before and after configuration changes
2. **Statistical Significance Testing**: Determine confidence levels for observed performance impacts
3. **Parameter Impact Correlation**: Identify which parameters most strongly influence specific metrics
4. **Visualization Tools**: Interactive charts showing performance trends and parameter relationships
5. **Historical Analysis**: Track all configuration changes and their impacts over time
6. **Market Context Awareness**: Considers market conditions when evaluating performance changes
7. **Category Filtering**: Filter configuration changes by impact type (positive, significant, etc.)

### Implementation

The system is implemented as a React component integrated into the OnlineLearningDashboard. It fetches data from the `/api/ml/configuration-impact` endpoint, which provides:

- List of configuration changes with parameter details
- Performance metrics before and after each change
- Statistical significance analysis
- Market condition context
- Historical performance data

The UI provides three main views:
1. **Overview**: Shows parameter changes and basic metrics
2. **Performance Trends**: Visualizes performance over time with change markers
3. **Parameter Impact**: Analyzes the correlation between parameter changes and performance impacts

This tool helps traders understand the effectiveness of their configuration changes and make data-driven decisions when optimizing trading strategies.