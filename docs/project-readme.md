# AI Crypto Trading System

## Project Overview

This modular AI crypto trading system uses specialized agents to analyze market data and make high-confidence trading decisions. The system aims to achieve prediction accuracy and win rates exceeding 75% through systematic identification and measurement of high-impact market factors.

## System Architecture

The system consists of several interconnected components:

1. **Data Collection Framework**: Gathers real-time and historical market data from cryptocurrency exchanges
2. **Impact Factor Analysis Engine**: Identifies and measures factors with the highest predictive power
3. **Analysis Agents**:
   - Technical Analysis Agent: Processes indicators and chart patterns
   - Pattern Recognition Agent: Identifies high-probability chart patterns
   - Sentiment Analysis Agent: Analyzes market sentiment from available sources
4. **Decision Engine**: Aggregates predictions and generates trading signals
5. **Virtual Trading Environment**: Simulates trading for testing and validation
6. **Notification & Control System**: Enables remote monitoring and control via messaging platforms
7. **Monitoring Dashboard**: Provides visualization and control capabilities
8. **Authentication System**: Manages secure access to the system

## Key Features

- Factor-based analysis for identifying high-impact market drivers
- Selective trade execution with strict confidence thresholds
- Comprehensive logging infrastructure for system improvement
- Multi-agent consensus for trade decisions
- Remote monitoring and control via WhatsApp/Telegram
- Adaptive learning based on performance feedback

## Performance Targets

- Prediction Accuracy: >75%
- Win Rate: >75%
- Risk-Adjusted Return: Sharpe ratio >2.0
- Maximum Drawdown: <12%
- System Reliability: 99.95% uptime

## Budget Constraints

- Total Monthly Budget: $300
  - Premium Data Services: $150/month
  - Computational Resources: $100/month
  - Performance Analytics Tools: $50/month

## Development Approach

The system follows a phased implementation approach:
1. Foundation Phase: Core infrastructure and data collection
2. Intelligence Development: Analysis agents and impact measurement
3. Integration and Validation: Component connection and testing
4. Performance Optimization: Refinement for target metrics
5. Simulation Deployment: Risk-free performance validation
6. Production Transition: Phased live deployment

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL/TimescaleDB for data storage
- Redis for event messaging
- Binance API credentials

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-crypto-trading.git
cd ai-crypto-trading

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize the database
python scripts/init_db.py
```

### Configuration

Edit the configuration files in the `config/` directory to customize:
- Exchange API settings
- Analysis agent parameters
- Risk management rules
- Notification preferences

### Running the System

```bash
# Start the system in development mode
python run.py --mode=development

# Start the system in simulation mode
python run.py --mode=simulation
```

## Directory Structure

```
ai-crypto-trading/
├── src/                      # Source code
│   ├── data_collection/      # Data collection components
│   ├── analysis_agents/      # Analysis agent implementations
│   ├── decision_engine/      # Decision making components
│   ├── virtual_trading/      # Simulation environment
│   ├── notification/         # Notification and control system
│   ├── dashboard/            # Web dashboard
│   ├── common/               # Shared utilities
│   └── models/               # Data models
├── config/                   # Configuration files
├── logs/                     # Log files
├── tests/                    # Test cases
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
