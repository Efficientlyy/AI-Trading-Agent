# Mock/Real Data Toggle Feature

**Feature Status:** Complete ✅

## Overview

The Mock/Real Data Toggle feature allows users to seamlessly switch between using synthetic mock data and real market data for analysis and trading. This is particularly useful for:

- **Development & Testing**: Use mock data with predictable patterns to validate strategy logic
- **Demonstrations**: Showcase trading strategies without requiring real market connectivity
- **Training**: Train new users on the system without risking real trades
- **Offline Development**: Continue development when disconnected from data providers

## Architecture

The feature is implemented using a layered architecture:

1. **Configuration Layer** (`DataSourceConfig`): Manages the current state (mock vs real) and configuration settings
2. **Factory Layer** (`DataSourceFactory`): Creates appropriate data providers based on the current configuration
3. **Integration Layer** (Technical Analysis Agent): Connects the toggle system to the analysis workflow
4. **API Layer** (`data_source_api.py`): Exposes REST endpoints for the UI to control the toggle
5. **UI Layer** (`DataSourceToggle` component): Provides a visual toggle switch in the dashboard

### Key Components

#### Backend Components

- **`DataSourceConfig`**: Singleton class managing toggle state and configurations
- **`DataSourceFactory`**: Factory pattern implementation that provides the correct data source
- **`MockDataGenerator`**: Creates realistic synthetic market data with configurable properties
- **`AdvancedTechnicalAnalysisAgent`**: Integration with the analysis system

#### Frontend Components

- **`DataSourceToggle`**: React component providing a UI toggle switch
- **API endpoints**: REST API for toggle control and status retrieval

## Usage

### Programmatic Usage

You can toggle between mock and real data programmatically:

```python
from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.config.data_source_config import get_data_source_config

# Create an agent instance
agent = AdvancedTechnicalAnalysisAgent()

# Get current data source type
current_source = agent.get_data_source_type()  # Returns "mock" or "real"

# Toggle data source
new_source = agent.toggle_data_source()  # Returns "mock" or "real"

# Set specific data source
config = get_data_source_config()
config.use_mock_data = True  # For mock data
```

### UI Integration

The toggle is accessible via the UI with the `Mock ○⚪️ Real` switch in the dashboard header. It provides visual feedback of the current state and notifications when toggled.

### Configuration

Mock data generation can be configured with the following properties:

```python
# Update mock data settings
from ai_trading_agent.config.data_source_config import get_data_source_config

config = get_data_source_config()
config.update_config({
    "mock_data_settings": {
        "volatility": 0.02,         # Daily price volatility
        "trend_strength": 0.3,      # Strength of price trends
        "seed": 42,                 # Random seed for reproducibility
        "generate_regimes": True    # Whether to generate realistic regime transitions
    }
})
```

## API Endpoints

The following REST API endpoints are available:

- **GET `/api/data-source/status`**: Get current data source status
- **POST `/api/data-source/toggle`**: Toggle between mock and real data sources
- **POST `/api/data-source/mock-settings`**: Update mock data generation settings

## Example

See `examples/mock_real_toggle_demo.py` for a complete demonstration of the feature, including:

1. Configuration
2. Toggle operations
3. Analysis with different data sources
4. Updating mock data settings

## Technical Notes

- The toggle system uses an event listener pattern to notify components when the data source changes
- Configuration is stored persistently in JSON format
- The UI component is implemented with React and Material-UI
- Mock data generation can simulate various market conditions (trending, ranging, volatile) and specific chart patterns

## Integration Points

- **Technical Analysis Agent**: Directly integrated
- **Market Data Providers**: Used as data sources
- **Dashboard UI**: Via the DataSourceToggle component
- **Trading Orchestrator**: Via the Technical Analysis Agent

## Best Practices

1. **Testing**: Always test strategies with both mock and real data
2. **Notifications**: Keep users informed of the current data source
3. **Configuration**: Tune mock data parameters to match the characteristics of the assets being traded
4. **Visual Indicators**: Always provide clear visual indication when using mock data
