# AI Trading Agent with Sentiment Analysis

This project implements an AI trading agent with sentiment analysis capabilities, allowing you to create trading strategies based on social media sentiment and market data.

## Features

- Multi-source sentiment analysis (Twitter, Reddit, news)
- Real-time market data integration
- Enhanced trading strategies combining sentiment with technical indicators
- Market regime detection
- Visualization tools for sentiment-price relationships

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-trading-agent.git
   cd ai-trading-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## API Credentials Setup

This project uses various APIs that require credentials. Follow these steps to set them up:

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API credentials:
   - Twitter API (for social media sentiment)
   - Reddit API (for social media sentiment)
   - Exchange APIs (for market data)

3. For Twitter API setup:
   - Create a [Twitter Developer](https://developer.twitter.com/) account
   - Create a project and app to get your credentials
   - See [TWITTER_SETUP.md](TWITTER_SETUP.md) for detailed instructions

4. Never commit your `.env` file to version control\!

## Usage Examples

### Sentiment Analysis Demo

```bash
python examples/sentiment_real_integration_demo.py
```

### Enhanced Sentiment Trading Strategy

```bash
python examples/enhanced_sentiment_trading_strategy.py
```

## Documentation

- [Sentiment Analysis Guide](docs/SENTIMENT_ANALYSIS_GUIDE.md)
- [Twitter Setup Guide](TWITTER_SETUP.md)
- [Implementation Summary](SENTIMENT_IMPLEMENTATION_SUMMARY.md)

## Security Best Practices

- All API credentials should be stored in the `.env` file
- The `.gitignore` file prevents committing sensitive information
- For team projects, consider using a secure secrets manager
- For deployments, use environment variables or a cloud secrets service

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
