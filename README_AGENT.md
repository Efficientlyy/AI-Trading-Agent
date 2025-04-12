# AI Trading Agent Architecture

## Overview

The AI Trading Agent is a modular, component-based trading system designed for backtesting and live trading of algorithmic strategies. The architecture focuses on flexibility, extensibility, and performance, with support for both Python and Rust-accelerated components.

## Key Features

- **Modular Component Architecture**: Easily swap components to test different strategies, risk models, and execution methods
- **Configuration-Driven**: Define your trading agent through YAML configuration files
- **Factory System**: Create and connect components programmatically
- **Rust Acceleration**: Optional Rust components for high-performance backtesting
- **Multi-Asset Support**: Trade and backtest across multiple assets simultaneously
- **Comprehensive Metrics**: Calculate detailed performance metrics for strategy evaluation
- **Extensible**: Add custom components to extend functionality

## Components

The agent architecture consists of the following core components:

### Data Manager

Responsible for loading, preprocessing, and providing market data to other components.

```python
from ai_trading_agent.agent import SimpleDataManager

data_manager = SimpleDataManager(config={
    "data_dir": "data/",
    "symbols": ["BTC/USD", "ETH/USD"],
    "timeframe": "1d"
})
```

### Strategy Manager

Manages trading strategies and generates trading signals.

```python
from ai_trading_agent.agent import SentimentStrategy, SimpleStrategyManager

strategy = SentimentStrategy(
    name="SentimentStrategy",
    config={
        "sentiment_threshold": 0.3,
        "position_size_pct": 0.1
    }
)
strategy_manager = SimpleStrategyManager(strategy)
```

### Risk Manager

Evaluates and manages risk for trading decisions.

```python
from ai_trading_agent.agent import SimpleRiskManager

risk_manager = SimpleRiskManager(config={
    "max_portfolio_risk_pct": 0.05,
    "stop_loss_pct": 0.05
})
```

### Portfolio Manager

Manages the portfolio, including positions and capital allocation.

```python
from ai_trading_agent.trading_engine import PortfolioManager

portfolio_manager = PortfolioManager(
    initial_capital=100000.0,
    risk_per_trade=0.02
)
```

### Execution Handler

Handles order execution, either simulated or live.

```python
from ai_trading_agent.agent import SimulatedExecutionHandler

execution_handler = SimulatedExecutionHandler(
    portfolio_manager=portfolio_manager,
    config={
        "commission_rate": 0.001,
        "slippage_pct": 0.001
    }
)
```

### Orchestrator

Coordinates all components and manages the trading loop.

```python
from ai_trading_agent.agent import BacktestOrchestrator

orchestrator = BacktestOrchestrator(
    data_manager=data_manager,
    strategy_manager=strategy_manager,
    portfolio_manager=portfolio_manager,
    risk_manager=risk_manager,
    execution_handler=execution_handler,
    config={
        "start_date": "2020-01-01",
        "end_date": "2020-12-31"
    }
)

# Run the backtest
results = orchestrator.run()
```

## Factory System

The AI Trading Agent includes a factory system that makes it easy to create and connect components based on configuration.

```python
from ai_trading_agent.agent.factory import create_agent_from_config

# Load configuration from a file or create it programmatically
config = {
    "data_manager": {
        "type": "SimpleDataManager",
        "config": {
            "data_dir": "data/",
            "symbols": ["AAPL", "GOOG", "MSFT"],
            "timeframe": "1d"
        }
    },
    "strategy": {
        "type": "SentimentStrategy",
        "config": {
            "sentiment_threshold": 0.3,
            "position_size_pct": 0.1
        }
    },
    # ... other components ...
}

# Create the agent (with optional Rust acceleration)
agent = create_agent_from_config(config, use_rust=True)

# Run the agent
results = agent.run()
```

## Configuration System

The AI Trading Agent uses YAML configuration files to define the agent components and their parameters.

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

## Rust Acceleration

The AI Trading Agent supports Rust-accelerated components for improved performance. To use Rust acceleration:

1. Ensure the Rust components are built and available
2. Set `use_rust: true` in your configuration or use the `--use-rust` flag with `build_agent.py`

```yaml
# Backtest Configuration
backtest:
  # ... other settings ...
  
  # Rust acceleration options
  use_rust: true
  rust_options:
    parallel: true
    optimization_level: 3
    precision: "double"
```

## Custom Components

You can extend the AI Trading Agent with custom components:

1. Create a new class that inherits from the appropriate base class
2. Register the component in the factory system

```python
from ai_trading_agent.agent.data_manager import DataManagerABC
from ai_trading_agent.agent.factory import register_custom_component

class MyCustomDataManager(DataManagerABC):
    def __init__(self, config):
        self.config = config
        # Initialize your data manager
        
    # Implement required methods
    
# Register the component
register_custom_component(
    component_type="data_manager",
    name="MyCustomDataManager",
    module_path="my_project.data_managers",
    class_name="MyCustomDataManager"
)
```

Alternatively, register custom components in your configuration:

```yaml
# Custom Components Configuration
custom_components:
  - type: "data_manager"
    name: "MyCustomDataManager"
    module_path: "my_project.data_managers"
    class_name: "MyCustomDataManager"
```

## Running Backtests

To run a backtest, you can use the `build_agent.py` script:

```bash
python scripts/build_agent.py --config config/my_agent_config.yaml --use-rust --output results/backtest_results.csv
```

Or you can use the `minimal_backtest.py` script for a simpler approach:

```bash
python minimal_backtest.py
```

## Performance Metrics

The AI Trading Agent calculates comprehensive performance metrics for strategy evaluation:

- Total Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio
- Omega Ratio
- And more...

## Documentation

For more detailed documentation, see:

- [Agent Architecture Documentation](docs/agent_architecture.md)
- [API Reference](docs/api_reference.md)
- [Configuration Guide](docs/configuration_guide.md)
- [Custom Components Guide](docs/custom_components.md)

## Examples

The repository includes several example scripts and configurations:

- `minimal_backtest.py`: A simple backtest script
- `scripts/build_agent.py`: A script for building and running agents from configuration
- `config/agent_config_template.yaml`: A template configuration file

## Testing

The AI Trading Agent includes comprehensive tests for all components:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test modules
python -m unittest tests.agent.test_factory
python -m unittest tests.agent.test_integration
```

## Contributing

Contributions to the AI Trading Agent are welcome! Please see the [Contributing Guide](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
