# AI Trading Agent with Sentiment Analysis & Market Regime Detection

This project implements a comprehensive AI trading system with sentiment analysis, market regime detection, and adaptive strategy capabilities, allowing you to create sophisticated trading strategies based on social media sentiment, market data, and machine learning models.

## Features

- Multi-source sentiment analysis (Twitter, Reddit, news)
- Market regime detection with ensemble methods
- Early event detection system for market-moving events
- Real-time market data integration
- Enhanced trading strategies combining sentiment with technical indicators
- Visualization tools for sentiment-price relationships
- Robust Provider Failover System for LLM APIs
- Comprehensive monitoring and alerting system
- Continuous Improvement System with automated A/B testing
- Production-ready deployment infrastructure

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-trading-agent.git
   cd ai-trading-agent
   ```

2. Install dependencies using the Makefile:
   ```bash
   make setup
   ```

   Or manually:
   ```bash
   pip install -r requirements.txt
   ```

3. Optional: Build Rust components for performance:
   ```bash
   cd rust && cargo build --release
   ```

## API Credentials Setup

This project uses various APIs that require credentials:

1. Set up exchange credentials:
   ```bash
   # Interactive setup
   python setup_exchange_credentials.py add --exchange binance --validate
   
   # List configured exchanges
   python setup_exchange_credentials.py list
   
   # Validate credentials
   python setup_exchange_credentials.py validate
   ```

2. Set up other API credentials:
   - Copy the environment template: `cp .env.example .env`
   - Edit the `.env` file with your API credentials (Twitter, Reddit, etc.)
   - For Twitter API setup, see [TWITTER_SETUP.md](TWITTER_SETUP.md)

3. Never commit your `.env` file or API keys to version control!

## Running the System

The system can be run in different modes using the Makefile:

```bash
# Development mode (with mock exchange)
make run-dev

# Testing mode
make run-test

# Production mode
make run-prod

# Run with specific components
make run-sentiment

# Run dashboard
make dashboard
```

Or directly:

```bash
# Run with advanced options
python run_trading_system.py --env production --validate-keys

# Command-line help
python run_trading_system.py --help
```

## Usage Examples

### Trading System Components

```bash
# Full trading system demo
python examples/ai_trading_agent_demo.py

# Sentiment-based trading
python examples/enhanced_sentiment_trading_strategy.py

# Market regime detection
python examples/regime_detection_demo.py

# Multi-strategy system
python examples/multi_strategy_demo.py
```

### Analysis Tools

```bash
# Sentiment analysis demo
python examples/sentiment_real_integration_demo.py

# Early event detection
python examples/early_event_detection_demo.py

# Provider failover system
python examples/provider_failover_demo.py
```

### Dashboards

```bash
# Main system dashboard
python run_dashboard.py

# Sentiment analysis dashboard
python run_sentiment_dashboard.py

# Provider health monitoring
python run_provider_health_dashboard.py

# Performance analysis
python run_performance_dashboard.py
```

## Production Deployment

For production deployment, refer to our comprehensive guide:

[Production Deployment Guide](docs/PRODUCTION_DEPLOYMENT.md)

Key production features:
- Systemd service configuration
- Environment-specific settings
- API key security
- Monitoring and alerting
- High availability configuration
- Backup and recovery procedures

## Documentation

- [Architecture Overview](ARCHITECTURE.md)
- [Sentiment Analysis Guide](docs/SENTIMENT_ANALYSIS_GUIDE.md)
- [Market Regime Detection](docs/market_regime_detection.md)
- [Early Event Detection System](docs/EARLY_EVENT_DETECTION_SYSTEM.md)
- [Provider Failover System](docs/PROVIDER_FAILOVER_SYSTEM.md)
- [Continuous Improvement System](docs/CONTINUOUS_IMPROVEMENT_SYSTEM.md)
- [Production Deployment Guide](docs/PRODUCTION_DEPLOYMENT.md)
- [Configuration Management](CONFIG_MANAGEMENT.md)
- [Project Standards](PROJECT_STANDARDS.md)
- [Twitter Setup Guide](TWITTER_SETUP.md)
- [Implementation Summary](SENTIMENT_IMPLEMENTATION_SUMMARY.md)

## Security Best Practices

- All API credentials are managed by a secure key manager
- Environment-specific configurations with production hardening
- Encrypted API key storage and secure transport
- Network security guidelines in the deployment documentation
- Production-ready permissions and access controls

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [PROJECT_STANDARDS.md](PROJECT_STANDARDS.md) for coding standards and guidelines.

## License

Distributed under the MIT License. See `LICENSE` for more information.