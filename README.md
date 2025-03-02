# AI Trading Agent

An advanced cryptocurrency trading system with automated analysis, execution, and risk management capabilities. This project provides a comprehensive framework for developing, testing, and deploying trading strategies across multiple exchanges.

## System Architecture

The AI Trading Agent consists of several modular components that work together:

### Core Components

- **Exchange Connectors**: Interfaces with multiple cryptocurrency exchanges
- **Order Management System**: Handles order submission, tracking, and execution
- **Position Management**: Tracks and manages trading positions across exchanges
- **Risk Management**: Enforces risk limits and manages exposure
- **Fee Management**: Tracks, analyzes, and optimizes trading fees
- **Strategy Engine**: Executes trading strategies and generates signals
- **Data Collection**: Gathers market data, order book data, and trading signals
- **Performance Monitoring**: Tracks strategy and system performance

### Supporting Components

- **Alert System**: Monitors and notifies of important system events
- **Dashboard**: Provides real-time monitoring of trading activities
- **Logging System**: Records detailed system activities for analysis
- **Configuration System**: Manages system settings and parameters
- **Backtesting Framework**: Tests strategies against historical data

## Implemented Components

### Alert System

The Alert System provides comprehensive monitoring and notification capabilities:

- Different alert levels (INFO, WARNING, ERROR, CRITICAL)
- Multiple alert categories (SYSTEM, EXCHANGE, ORDER, RISK, etc.)
- Alert management including resolution tracking and expiry
- Pluggable alert handlers (logging, file, etc.)

### Fee Management System

The Fee Management System tracks and optimizes trading costs:

- **Fee Tracking**: Records and stores trading fees from all transactions
- **Fee Estimation**: Estimates fees for planned transactions
- **Fee Scheduling**: Maintains up-to-date fee schedules for exchanges
- **Volume-based Tiers**: Supports exchange fee tiers based on trading volume
- **Fee Discounts**: Tracks and applies exchange-specific fee discounts
- **Fee Analytics**: Generates summaries and reports on fee expenditures
- **Fee Optimization**: Recommends optimal exchange allocation for strategies
- **Fee Visualization**: Generates visual reports of fee data and insights

### Order Routing System

The Order Routing System intelligently directs orders to optimal exchanges:

- **Multi-criteria Routing**: Routes based on fees, liquidity, latency, or balanced approach
- **Exchange Scoring**: Scores exchanges based on multiple performance metrics
- **Fee Optimization**: Minimizes transaction costs across multiple exchanges
- **Savings Estimation**: Calculates potential savings compared to alternative routing
- **Alternative Recommendations**: Provides ranked alternative exchanges
- **Decision Explanation**: Documents reasoning behind routing decisions

### Monitoring Dashboard

The Monitoring Dashboard provides real-time visibility into system operations:

- System component status monitoring
- Active order tracking and management
- Recent trade history and analysis
- Performance metrics visualization
- Integrated alert system with color-coded severity

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```
git clone https://github.com/your-username/ai-trading-agent.git
cd ai-trading-agent
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up configuration:
```
cp config/example_config.yaml config/config.yaml
# Edit config.yaml with your settings
```

### Running Components

#### Fee Management System

To run the fee management example:

```
python examples/fee_management_example.py
```

For fee visualizations (requires matplotlib and pandas):

```
python examples/fee_visualization.py
```

#### Order Routing System

To run the order routing example:

```
python examples/order_routing_standalone.py
```

This will demonstrate the Order Routing System with:
- Multiple sample orders with different characteristics
- Routing using various criteria (fees, liquidity, latency)
- Detailed output of routing decisions and fee estimates

#### Alert System

To run the alerts example:

```
python examples/alerts_standalone.py
```

#### Dashboard

To launch the monitoring dashboard:

```
python dashboard_standalone.py
```

Then access the dashboard at `http://127.0.0.1:8080` in your browser.

## Project Structure

```
ai-trading-agent/
├── config/                 # Configuration files
├── dashboard/              # Dashboard components
├── data/                   # Data storage
│   ├── market/             # Market data
│   ├── orders/             # Order data
│   └── fees/               # Fee data
├── examples/               # Example scripts
├── src/                    # Source code
│   ├── alerts/             # Alert system
│   ├── exchange/           # Exchange connectors
│   ├── fees/               # Fee management
│   ├── orders/             # Order management
│   ├── risk/               # Risk management
│   └── strategy/           # Trading strategies
├── tests/                  # Unit and integration tests
├── dashboard.py            # Dashboard application
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Thanks to all the open-source projects that made this possible
- Cryptocurrency exchange APIs that provided the trading interfaces 