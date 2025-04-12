# AI Trading Agent Architecture

This document describes the architecture of the AI Trading Agent system, focusing on the agent components, their interactions, and how to configure and use them.

## Overview

The AI Trading Agent is built with a modular, component-based architecture that allows for easy customization and extension. The main components are:

1. **Data Manager**: Responsible for loading, preprocessing, and providing market data to other components
2. **Strategy Manager**: Manages trading strategies and generates trading signals
3. **Risk Manager**: Evaluates and manages risk for trading decisions
4. **Portfolio Manager**: Manages the portfolio, including positions and capital allocation
5. **Execution Handler**: Handles order execution, either simulated or live
6. **Orchestrator**: Coordinates all components and manages the trading loop

These components are designed to be interchangeable, allowing you to swap out implementations without affecting the rest of the system.

## Component Details

### Data Manager

The Data Manager is responsible for:
- Loading market data from various sources (CSV files, APIs, databases)
- Preprocessing data (cleaning, normalization, feature engineering)
- Providing data to other components in a standardized format

Available implementations:
- `SimpleDataManager`: Basic implementation for backtesting with CSV files
- `MinimalDataManager`: In-memory implementation for testing

### Strategy Manager

The Strategy Manager is responsible for:
- Managing one or more trading strategies
- Generating trading signals based on market data
- Providing recommendations for position sizing

Available implementations:
- `SimpleStrategyManager`: Basic implementation for managing a single strategy
- `SentimentStrategyManager`: Implementation for sentiment-based strategies

### Risk Manager

The Risk Manager is responsible for:
- Evaluating risk for trading decisions
- Implementing risk controls (position sizing, stop-loss, etc.)
- Preventing excessive risk exposure

Available implementations:
- `SimpleRiskManager`: Basic implementation with position sizing and stop-loss

### Portfolio Manager

The Portfolio Manager is responsible for:
- Managing the portfolio (positions, cash, etc.)
- Tracking performance metrics
- Implementing portfolio-level constraints

Available implementations:
- `PortfolioManager`: Comprehensive implementation with position tracking and performance metrics

### Execution Handler

The Execution Handler is responsible for:
- Executing trading orders (market, limit, etc.)
- Simulating order execution for backtesting
- Connecting to exchanges for live trading

Available implementations:
- `SimulatedExecutionHandler`: Simulated execution for backtesting

### Orchestrator

The Orchestrator is responsible for:
- Coordinating all components
- Managing the trading loop
- Collecting and reporting results

Available implementations:
- `BacktestOrchestrator`: Implementation for backtesting

## Factory System

The AI Trading Agent includes a factory system that makes it easy to create and connect components based on configuration. The factory system is defined in `ai_trading_agent/agent/factory.py` and includes factory functions for each component type.

Example usage:

```python
from ai_trading_agent.agent.factory import create_agent_from_config

# Load configuration from a file or create it programmatically
config = {
    "data_manager": {
        "type": "SimpleDataManager",
        "config": {
            "data_dir": "data/",
            "symbols": ["AAPL", "GOOG", "MSFT"],
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "timeframe": "1d",
            "data_types": ["ohlcv", "sentiment"]
        }
    },
    # ... other components ...
}

# Create the agent
agent = create_agent_from_config(config)

# Run the agent
results = agent.run()
```

## Configuration System

The AI Trading Agent includes a configuration system that allows you to define and validate component configurations. The configuration system is defined in `ai_trading_agent/common/config_validator.py` and includes functions for validating configurations.

Example configuration file (`config/agent_config_template.yaml`):

```yaml
# Data Management Configuration
data_manager:
  type: "SimpleDataManager"
  config:
    data_dir: "data/"
    timeframe: "1d"
    data_types: ["ohlcv", "sentiment"]
    symbols: ["BTC/USD", "ETH/USD", "ADA/USD"]

# Strategy Configuration
strategy:
  type: "SentimentStrategy"
  config:
    name: "SentimentBasedTrading"
    sentiment_threshold: 0.3
    position_size_pct: 0.1
    symbols: ["BTC/USD", "ETH/USD", "ADA/USD"]

# ... other components ...
```

## Using the Agent Architecture

### Creating a Custom Component

To create a custom component, you need to:

1. Create a new class that inherits from the appropriate base class
2. Implement the required methods
3. Register the component in the factory system

Example:

```python
from ai_trading_agent.agent.data_manager import DataManagerABC

class MyCustomDataManager(DataManagerABC):
    def __init__(self, config):
        self.config = config
        # Initialize your data manager
        
    def get_latest_data(self, symbol, n=1):
        # Implement this method
        pass
        
    # Implement other required methods
    
# Register the component in the factory
from ai_trading_agent.agent.factory import DATA_MANAGER_REGISTRY
DATA_MANAGER_REGISTRY["MyCustomDataManager"] = MyCustomDataManager
```

### Running a Backtest

To run a backtest, you can use the `build_agent.py` script:

```bash
python scripts/build_agent.py --config config/my_agent_config.yaml
```

Or you can use the `minimal_backtest.py` script for a simpler approach:

```bash
python minimal_backtest.py
```

### Extending the Architecture

The AI Trading Agent architecture is designed to be extensible. You can:

1. Add new component types by creating new base classes and factory functions
2. Add new implementations of existing component types
3. Customize the configuration system to support new component types
4. Add new orchestrators for different use cases (live trading, paper trading, etc.)

## Integration with Rust Components

The AI Trading Agent supports integration with Rust-accelerated components for improved performance. The Rust components are defined in the `rust_backtester` module and can be used as drop-in replacements for their Python counterparts.

To use the Rust-accelerated components:

1. Build the Rust components using Cargo
2. Use the `RustBacktester` class instead of the Python backtester

Example:

```python
from ai_trading_agent.backtesting.rust_backtester import RustBacktester

# Create the backtester
backtester = RustBacktester(
    data_manager=data_manager,
    strategy_manager=strategy_manager,
    portfolio_manager=portfolio_manager,
    risk_manager=risk_manager,
    execution_handler=execution_handler,
    config=config
)

# Run the backtest
results = backtester.run()
```

## Performance Considerations

When using the AI Trading Agent for backtesting, consider the following performance tips:

1. Use the Rust-accelerated components for large datasets
2. Use in-memory data when possible to avoid I/O bottlenecks
3. Optimize your strategies for performance
4. Use the appropriate timeframe for your backtests (daily for long-term strategies, minute for intraday strategies)
5. Consider using parallel processing for multiple backtests

## Error Handling

The AI Trading Agent includes comprehensive error handling to ensure that errors are caught and reported properly. The main error handling mechanisms are:

1. Configuration validation to catch configuration errors early
2. Exception handling in component methods to prevent crashes
3. Logging of errors and warnings to help with debugging

## Logging

The AI Trading Agent uses the Python logging module for logging. You can configure the logging level and output in the configuration file:

```yaml
logging:
  level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_to_file: true
  log_file: "logs/agent.log"
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Example Usage

Here's a complete example of using the AI Trading Agent for backtesting:

```python
import os
import yaml
from ai_trading_agent.agent.factory import create_agent_from_config
from ai_trading_agent.common.config_validator import validate_agent_config

# Load configuration from a file
with open("config/my_agent_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Validate the configuration
is_valid, error_message = validate_agent_config(config)
if not is_valid:
    raise ValueError(f"Invalid configuration: {error_message}")

# Create the agent
agent = create_agent_from_config(config)

# Run the agent
results = agent.run()

# Print the results
if results:
    print("Backtest completed successfully")
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print("Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
else:
    print("Backtest did not return results")
```

## Conclusion

The AI Trading Agent architecture provides a flexible, modular framework for building and testing trading strategies. By using the factory system and configuration validation, you can easily create and customize agents for different use cases.

For more information, see the API documentation and example scripts in the repository.
