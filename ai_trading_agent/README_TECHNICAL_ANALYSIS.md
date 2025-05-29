# Technical Analysis Agent Documentation

## Overview

The Technical Analysis Agent is a comprehensive component of the AI Trading Agent system that provides advanced market analysis capabilities through a hybrid Python-Rust architecture. This agent supports both mock and real data sources, advanced pattern recognition, market regime detection, and intelligent signal generation.

## Key Features

- **Hybrid Python-Rust Architecture**: Critical calculations performed in Rust for maximum performance
- **Mock/Real Data Toggle**: Seamless switching between mock and real data sources
- **Advanced Pattern Recognition**: Detection of candlestick and chart patterns
- **Market Regime Classification**: Intelligent market state detection for adaptive strategies
- **Performance Optimization**: Caching, parallel processing, and memory management
- **Production Monitoring**: Comprehensive health checks and performance metrics
- **Orchestration**: Integration with the main Trading Orchestrator for signal routing

## Components

### 1. Core Technical Analysis Engine

The core engine processes market data and generates technical insights:

- **AdvancedTechnicalAnalysisAgent**: Main agent class providing high-level analysis functions
- **Indicator Engine**: Calculates technical indicators (trend, momentum, volatility, volume)
- **Pattern Detection**: Identifies candlestick and chart patterns
- **Strategy Framework**: Executes trading strategies based on technical conditions

### 2. Pattern Recognition

The pattern recognition system includes multiple components:

- **ThreeCandlePatternDetector**: Detects standard three-candle patterns (Morning Star, Evening Star, etc.)
- **AdvancedPatternDetector**: Identifies complex chart patterns (Head & Shoulders, Double Tops, etc.)
- **Pattern Validation**: Validates detected patterns against historical context

### 3. Orchestration & Monitoring

Production-grade features for integration and monitoring:

- **TechnicalAgentOrchestrator**: Manages the agent lifecycle and signal routing
- **TAAgentMonitor**: Provides health checks, metrics, and alerting
- **EventBus**: Enables decoupled communication between components

### 4. Mock/Real Data Integration

Seamless switching between data sources:

- **DataSourceConfig**: Manages toggle state between mock and real data
- **DataSourceFactory**: Creates appropriate data provider based on configuration
- **DataSourceToggle UI**: React component for controlling the data source

## API Endpoints

The Technical Analysis Agent exposes several API endpoints:

### Analysis Endpoints

- `GET /api/technical-analysis/indicators`: Get technical indicators for a symbol and timeframe
- `GET /api/technical-analysis/patterns`: Get detected candlestick patterns
- `GET /api/technical-analysis/advanced-patterns`: Get detected chart patterns
- `GET /api/technical-analysis/analysis`: Get comprehensive analysis (indicators + patterns)

### Orchestration Endpoints

- `POST /api/technical-analysis/orchestrator/start`: Start the technical analysis orchestration
- `POST /api/technical-analysis/orchestrator/stop`: Stop the technical analysis orchestration
- `GET /api/technical-analysis/orchestrator/status`: Get orchestration status

### Monitoring Endpoints

- `GET /api/technical-analysis/monitoring/health`: Get health status of the Technical Analysis Agent
- `GET /api/technical-analysis/monitoring/metrics`: Get detailed monitoring metrics
- `GET /api/technical-analysis/metrics`: Get API performance metrics

## Usage Examples

### Getting Technical Analysis

```python
import requests

# Get indicators
response = requests.get(
    "http://localhost:8000/api/technical-analysis/indicators",
    params={"symbol": "BTC/USD", "timeframe": "1h"}
)
indicators = response.json()

# Get patterns
response = requests.get(
    "http://localhost:8000/api/technical-analysis/patterns",
    params={"symbol": "BTC/USD", "timeframe": "1h"}
)
patterns = response.json()

# Get comprehensive analysis
response = requests.get(
    "http://localhost:8000/api/technical-analysis/analysis",
    params={"symbol": "BTC/USD", "timeframe": "1h", "include_advanced_patterns": True}
)
analysis = response.json()
```

### Starting Orchestration

```python
import requests

# Start orchestration
response = requests.post(
    "http://localhost:8000/api/technical-analysis/orchestrator/start",
    params={
        "symbols": ["BTC/USD", "ETH/USD"],
        "timeframes": ["1h", "4h", "1d"]
    }
)
result = response.json()

# Check status
response = requests.get(
    "http://localhost:8000/api/technical-analysis/orchestrator/status"
)
status = response.json()
```

### Monitoring Health

```python
import requests

# Get health status
response = requests.get(
    "http://localhost:8000/api/technical-analysis/monitoring/health"
)
health = response.json()

# Get detailed metrics
response = requests.get(
    "http://localhost:8000/api/technical-analysis/monitoring/metrics"
)
metrics = response.json()
```

## UI Components

The Technical Analysis Agent includes several React UI components:

- **TechnicalAnalysisView**: Main view component for technical analysis visualization
- **TechnicalChartViewer**: Chart viewer with indicator overlays
- **PatternRecognitionView**: Visual display of detected patterns
- **TechnicalAnalysisAdmin**: Admin panel for monitoring and controlling the agent

## Mock/Real Data Toggle

The Technical Analysis Agent fully supports the Mock/Real data toggle:

1. **Toggle Control**: Use the UI toggle switch (`Mock ○⚪️ Real`) to switch between data sources
2. **Visual Indicators**: The UI displays clear indicators when using mock data
3. **Data Source Awareness**: All components are aware of the current data source
4. **API Integration**: All API endpoints respect the current data source setting

## Performance Optimization

The Technical Analysis Agent includes several performance optimizations:

- **Caching**: Frequently accessed calculations are cached
- **Parallel Processing**: Multi-symbol and multi-timeframe calculations run in parallel
- **Memory Management**: Dataframes are optimized for reduced memory usage
- **FFI Optimization**: Efficient data transfer between Python and Rust components

## Production Monitoring

Comprehensive monitoring capabilities are included:

- **Health Checks**: Regular checks of all components (data source, pattern detection, etc.)
- **Metrics Collection**: Performance metrics for all operations
- **Alerting**: Detection of error conditions with severity levels
- **Status Reporting**: Detailed status information for all components

## Testing

The Technical Analysis Agent includes comprehensive tests:

- **Unit Tests**: Tests for individual components
- **Integration Tests**: Tests for component interactions
- **End-to-End Tests**: Tests for complete workflows
- **Mock/Real Data Tests**: Tests for data source toggle functionality

## Troubleshooting

Common issues and solutions:

### API Errors

- **404 Not Found**: Check the endpoint URL and parameters
- **500 Internal Server Error**: Check the server logs for details

### Data Source Issues

- **No Data**: Verify the data source configuration and connectivity
- **Mock/Real Toggle Not Working**: Check the event bus and configuration state

### Performance Issues

- **Slow Analysis**: Check the cache settings and parallel processing configuration
- **High Memory Usage**: Review the memory optimization settings

## Configuration

The Technical Analysis Agent can be configured through several mechanisms:

- **Environment Variables**: Set `TA_AGENT_CONFIG_PATH` to specify a configuration file
- **API Endpoints**: Use the configuration API endpoints to update settings
- **UI Controls**: Use the admin panel to configure the agent

## Dependencies

- **Python**: 3.8 or higher
- **Rust**: 1.54 or higher
- **FastAPI**: For API endpoints
- **React**: For UI components
- **Pandas**: For data manipulation
- **NumPy**: For numerical operations
- **Plotly**: For visualization

## Next Steps

Future enhancements may include:

- **Machine Learning Integration**: Enhanced pattern recognition with ML
- **Additional Indicators**: Expansion of the indicator library
- **Strategy Marketplace**: Community sharing of technical strategies
- **Enhanced Visualization**: More interactive visualization options
