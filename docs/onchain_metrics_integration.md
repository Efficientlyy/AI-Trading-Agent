# On-Chain Metrics Integration

## Overview

On-chain metrics provide valuable blockchain data that can be used to analyze cryptocurrency market sentiment and trends. This document outlines the integration of on-chain metrics from Blockchain.com and Glassnode into the AI Trading Agent's sentiment analysis system.

## Implementation

The on-chain metrics integration is implemented through multiple client classes within the `src/analysis_agents/sentiment/blockchain_client.py` file:

1. **Base Client**: `BaseBlockchainClient` - Abstract base class for all blockchain data providers
2. **Provider-Specific Clients**:
   - `BlockchainComClient`: Client for Blockchain.com's API
   - `GlassnodeClient`: Client for Glassnode's API 
3. **Unified Client**: `BlockchainClient` - Aggregate client that combines data from multiple providers with automatic fallback

The `OnchainSentimentAgent` class in `src/analysis_agents/sentiment/onchain_sentiment.py` uses these clients to analyze on-chain data and generate sentiment signals.

## Key Metrics

The implementation focuses on four primary on-chain metrics:

1. **Large Transactions**:
   - Tracks high-value transactions (typically >$100,000)
   - Identifies potential institutional/whale activity
   - Provides volume and count of significant transfers

2. **Active Addresses**:
   - Counts daily active addresses on the blockchain
   - Indicates network usage and adoption
   - Growth trends suggest increasing/decreasing interest

3. **Hash Rate** (for PoW blockchains):
   - Measures the total computational power securing the network
   - Indicates miner confidence and network security
   - Available only for PoW chains like Bitcoin

4. **Exchange Reserves**:
   - Tracks the amount of cryptocurrency held on exchanges
   - Decreasing reserves often indicate accumulation (bullish)
   - Increasing reserves may indicate potential selling pressure (bearish)

## Data Sources

### Blockchain.com API

The Blockchain.com API provides basic blockchain data primarily focused on Bitcoin:

```python
# Examples of endpoints used:
# - /unconfirmed-transactions (for large transactions)
# - /charts/n-unique-addresses (for active addresses)
# - /charts/hash-rate (for hash rate)
```

Key features of the Blockchain.com integration:
- No API key required for most endpoints
- Limited to Bitcoin data
- Simple authentication
- Daily and historical data available
- Rate limiting handled with caching

### Glassnode API

The Glassnode API provides comprehensive on-chain metrics for multiple cryptocurrencies:

```python
# Examples of endpoints used:
# - /metrics/transactions/transfers_volume_more_than_100k_count
# - /metrics/addresses/active_count
# - /metrics/mining/hash_rate_mean
# - /metrics/distribution/balance_exchanges
```

Key features of the Glassnode integration:
- API key required (subscription-based)
- Multiple cryptocurrencies supported
- Advanced metrics and indicators
- Customizable time periods
- Historical data with various resolutions

## Sentiment Analysis

The `OnchainSentimentAgent` processes on-chain metrics to generate sentiment signals:

1. **Data Normalization**: Converts raw metrics to 0-1 scale sentiment values
2. **Metric Weighting**: Applies weights to different metrics based on reliability and relevance
3. **Confidence Calculation**: Determines signal confidence based on data quality and agreement
4. **Divergence Detection**: Identifies situations where on-chain data diverges from price action

```python
# Example sentiment calculation for on-chain metrics
sentiment_metrics = {
    "large_transactions": large_tx_normalized,
    "active_addresses": active_addr_normalized,
    "hash_rate": hash_rate_normalized if hash_rate_data else 0.5,
    "exchange_reserves": exchange_reserves_normalized
}

# Calculate weighted sentiment
metric_weights = {
    "large_transactions": 0.3,
    "active_addresses": 0.3,
    "hash_rate": 0.2 if hash_rate_data else 0,
    "exchange_reserves": 0.2
}

sentiment_value = sum(
    sentiment_metrics[metric] * metric_weights[metric]
    for metric in sentiment_metrics
) / sum(metric_weights.values())
```

## Configuration

The on-chain metrics integration is configured in the `sentiment_analysis.yaml` file under the `sentiment.apis` section:

```yaml
sentiment:
  apis:
    blockchain_com:
      api_key: "${BLOCKCHAIN_COM_API_KEY}"
      base_url: "https://api.blockchain.info"
      cache_expiry: 900  # 15 minutes in seconds
    
    glassnode:
      api_key: "${GLASSNODE_API_KEY}"
      base_url: "https://api.glassnode.com/v1"
      cache_expiry: 900  # 15 minutes in seconds
```

## Interpreting Metrics

### Large Transactions

| Change | Interpretation |
|--------|----------------|
| Increasing | Potential institutional activity, can be bullish or bearish depending on market context |
| Spike | Possible large player accumulation or distribution |
| Declining | Reduced institutional interest or whale activity |

### Active Addresses

| Change | Interpretation |
|--------|----------------|
| Increasing | Growing network usage and adoption (bullish) |
| Decreasing | Reducing network usage and interest (bearish) |
| Plateau | Network maturity or stagnation |

### Hash Rate

| Change | Interpretation |
|--------|----------------|
| Increasing | Growing miner confidence and security (bullish) |
| Decreasing | Reduced mining profitability or confidence (bearish) |
| Sharp drop | Potential mining disruption or regulation |

### Exchange Reserves

| Change | Interpretation |
|--------|----------------|
| Decreasing | Coins moving off exchanges for holding (bullish) |
| Increasing | Coins moving to exchanges for potential selling (bearish) |
| Sharp drop | Strong accumulation signal |

## Example Usage

See the `examples/onchain_metrics_demo.py` file for a comprehensive demonstration of the on-chain metrics integration, including:

- Fetching metrics from multiple providers
- Visualizing on-chain data
- Generating sentiment signals

```python
import asyncio
from src.analysis_agents.sentiment.blockchain_client import BlockchainClient

async def get_onchain_data():
    # Initialize the client
    client = BlockchainClient(
        blockchain_com_api_key="your_api_key",
        glassnode_api_key="your_api_key"
    )
    
    # Get active addresses
    active_addr_data = await client.get_active_addresses(
        asset="BTC",
        time_period="24h"
    )
    print(f"Active addresses: {active_addr_data.get('count', 'N/A'):,}")
    print(f"Change: {active_addr_data.get('change_percentage', 'N/A'):.2f}%")
    
    # Get exchange reserves
    exchange_data = await client.get_exchange_reserves(
        asset="BTC",
        time_period="7d"
    )
    print(f"Exchange reserves: {exchange_data.get('reserves', 'N/A'):,.2f}")
    print(f"Change: {exchange_data.get('change_percentage', 'N/A'):.2f}%")
    
    # Close the client
    await client.close()

# Run the example
asyncio.run(get_onchain_data())
```

## Integration With Sentiment Analysis System

The on-chain metrics feed into the larger sentiment analysis system:

1. `OnchainSentimentAgent` analyzes metrics and generates sentiment signals
2. Signals are published as sentiment events with detailed metadata
3. `SentimentAggregator` combines on-chain signals with other sentiment sources
4. Trading strategies consume aggregated sentiment to make informed decisions

## Advanced Features

1. **Fallback Mechanism**: Automatically falls back to alternative providers if one fails
2. **Caching System**: Minimizes API calls with configurable cache duration
3. **Mock Data Support**: Provides realistic mock data when APIs are unavailable
4. **Metric Aggregation**: Intelligently combines data from multiple sources
5. **Divergence Detection**: Identifies when on-chain metrics diverge from price action

## Next Steps

1. Add more advanced on-chain metrics like Network Value to Transactions Ratio (NVT)
2. Implement token-specific metrics for DeFi protocols
3. Add support for more blockchains (e.g., Solana, Cardano)
4. Integrate on-chain metrics into specialized trading strategies
5. Develop real-time alerts for significant on-chain events