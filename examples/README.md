# AI Trading Agent Examples

This directory contains example scripts demonstrating various features of the AI Trading Agent.

## News API Integration

The `news_api_example.py` script demonstrates how to use the NewsAPI integration to fetch and analyze news articles about cryptocurrencies.

### Features Demonstrated:

- Fetching news articles from NewsAPI.org
- Cryptocurrency-specific news searches
- Integration with the NewsAnalyzer system
- Sentiment analysis of news articles
- Extraction of market-relevant events

### Usage:

```bash
# Set your NewsAPI key in environment variables (optional)
export NEWS_API_KEY="your_api_key_here"

# Run the example
python examples/news_api_example.py
```

### Notes:

- If no API key is provided, the example will use mock data
- For full functionality, obtain a NewsAPI key from [newsapi.org](https://newsapi.org/)
- See the documentation in `/docs/news_api_integration.md` for more details

## Other Examples

- Coming soon