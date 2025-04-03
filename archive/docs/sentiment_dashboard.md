# Sentiment Analysis Dashboard

The Sentiment Analysis Dashboard provides a comprehensive visualization interface for monitoring market sentiment from multiple sources. This component is part of Phase 7 (Visualization and Monitoring) of the Sentiment Analysis Implementation Plan.

## Overview

The dashboard aggregates sentiment data from various sources including:

- Fear & Greed Index
- News sentiment
- Social media sentiment (Twitter/X, Reddit)
- On-chain metrics

It provides real-time and historical views of sentiment data, correlations with price movements, and notifications for extreme sentiment signals that may indicate contrarian trading opportunities.

## Features

### Overall Market Sentiment

- Current sentiment status (bullish, neutral, bearish)
- Confidence measurement
- Visual sentiment meter
- Historical sentiment trends

### Fear & Greed Index

- Current index value with classification
- Visual gauge for easy interpretation
- Historical index values for comparison

### News Sentiment

- Current sentiment analysis from crypto news sources
- Recent headlines with individual sentiment scores
- Article count and confidence metrics

### Social Media Sentiment

- Platform-specific sentiment analysis (Twitter/X, Reddit)
- Post volume and mention frequency metrics
- Trending keywords and hashtags

### On-Chain Sentiment

- Key blockchain metrics with sentiment interpretation
- Historical trend visualization
- Network health indicators

### Sentiment Extremes & Contrarian Signals

- Detection of extreme sentiment levels
- Contrarian signal generation with recommendations
- Historical accuracy of contrarian signals

### Sentiment-Price Correlation

- Correlation coefficients for each sentiment source
- Short-term and long-term correlation analysis
- Visual correlation charts

### Historical Sentiment Data

- Daily sentiment records across all sources
- Price change correlation
- Downloadable data for further analysis

## Implementation

The dashboard is implemented using:

- FastAPI for the server-side logic
- Jinja2 Templates for HTML rendering
- Pandas for data manipulation
- Matplotlib/Plotly for chart generation

## Usage

### Running the Dashboard

You can start the sentiment dashboard using the provided example script:

```bash
python examples/start_sentiment_dashboard.py
```

Or by running the main dashboard with sentiment components included:

```bash
python dashboard.py
```

The dashboard will be accessible at: http://localhost:8080/sentiment

### API Access

The dashboard also provides API endpoints for programmatic access to sentiment data:

```
GET /sentiment/api/data?symbol=BTC
```

Returns sentiment data in JSON format for the specified symbol.

## Architecture

The sentiment dashboard follows a modular architecture:

1. **Data Collection Layer**: Collects data from various sentiment sources
2. **Processing Layer**: Normalizes and analyzes sentiment data
3. **Presentation Layer**: Renders the dashboard interface

### Key Components

- `SentimentDashboard`: Main controller that orchestrates data collection and processing
- `SentimentCollector`: Collects historical sentiment data from various sources
- `FastAPI Router`: Handles HTTP requests and serves the dashboard

## Configuration

The dashboard uses the standard sentiment analysis configuration from `config/sentiment_analysis.yaml`. Key configuration options include:

- `update_interval`: How frequently to refresh sentiment data
- `cache_ttl`: How long to cache sentiment data before refreshing
- Sources to include in the dashboard (Fear & Greed, news, social media, on-chain)

## Future Enhancements

Planned enhancements for the sentiment dashboard include:

1. **Real-time Updates**: WebSocket implementation for live updates
2. **Custom Alerts**: Configurable alerts for specific sentiment conditions
3. **Advanced Visualizations**: More sophisticated chart types and visualizations
4. **User Customization**: Ability to customize dashboard layout and sources
5. **Machine Learning Integration**: Predictive models based on sentiment patterns
6. **Mobile Optimization**: Responsive design for mobile device access

## Conclusion

The Sentiment Analysis Dashboard provides traders and researchers with a powerful tool for monitoring and analyzing market sentiment. By visualizing data from multiple sources in a unified interface, it enables more informed trading decisions based on the overall sentiment landscape.