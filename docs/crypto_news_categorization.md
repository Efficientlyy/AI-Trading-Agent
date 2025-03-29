# Cryptocurrency News Categorization System

## Overview

The Cryptocurrency News Categorization System is a specialized component designed to analyze and categorize cryptocurrency news articles into relevant market categories. This system enhances the sentiment analysis pipeline by providing more granular and targeted insights into the cryptocurrency market dynamics through news content analysis.

## Features

- **Specialized Categorization**: Categorizes news into 24+ cryptocurrency-specific categories
- **Multi-method Analysis**: Combines rule-based and ML-based approaches for robust categorization
- **Topic Graph**: Builds and analyzes relationships between news articles, topics, and assets
- **Narrative Detection**: Identifies emerging narrative clusters in the cryptocurrency news landscape
- **Trend Analysis**: Tracks trending categories and topics over time
- **Asset-specific Insights**: Provides customized analysis for different cryptocurrency assets

## Architecture

The system consists of two main components:

1. **CryptoNewsCategorizer**: Responsible for categorizing individual news articles
2. **CryptoNewsTopicGraph**: Builds and analyzes a graph representation of news articles and their relationships

These components work together to provide a comprehensive understanding of the cryptocurrency news landscape, enabling the trading system to make more informed decisions based on specialized news categories and emerging narratives.

## Category Types

The categorization system includes the following specialized categories for cryptocurrency news:

### Market and Trading
- `price_movement`: News about significant price changes
- `market_analysis`: Technical and fundamental analysis
- `trading_strategy`: Trading approaches and methods
- `market_sentiment`: Market mood and investor sentiment

### Regulatory and Legal
- `regulation`: Regulatory news and developments
- `legal`: Legal cases and proceedings
- `compliance`: Compliance and KYC/AML issues
- `tax`: Tax implications and regulations

### Technology and Development
- `protocol_update`: Network upgrades and protocol changes
- `development`: Development progress and roadmaps
- `security`: Security issues, bugs, and fixes
- `scaling`: Scaling solutions and performance

### Industry and Business
- `partnership`: Business partnerships and collaborations
- `adoption`: Adoption by users, businesses, countries
- `investment`: Venture capital and fundraising
- `exchange`: Exchange listings, delistings, issues

### Ecosystem
- `defi`: Decentralized finance news
- `nft`: Non-fungible tokens
- `governance`: DAO and governance proposals
- `staking`: Staking, yield farming, rewards

### Macro and Fundamental
- `macroeconomic`: Broader economic factors
- `institutional`: Institutional involvement
- `mining`: Mining operations and hash rate
- `supply_metrics`: Token supply, burns, emissions

### Impact-based Categories
- `high_impact`: News with significant market impact
- `trending`: Currently trending topics
- `speculative`: Rumors and speculative news
- `contrarian`: News that goes against market consensus

## Usage

### Basic Usage

```python
from src.analysis_agents.news.news_analyzer import NewsArticle
from src.analysis_agents.news.crypto_news_categorizer import CryptoNewsCategorizer

# Initialize categorizer
categorizer = CryptoNewsCategorizer()
await categorizer.initialize()

# Categorize an article
categories = await categorizer.categorize_article(article)

# Print categories with confidence scores
for category, score in categories.items():
    print(f"{category}: {score:.2f}")
```

### Building a Topic Graph

```python
from src.analysis_agents.news.crypto_news_categorizer import CryptoNewsTopicGraph

# Initialize topic graph
topic_graph = CryptoNewsTopicGraph()

# Add articles to the graph
for article in articles:
    categories = await categorizer.categorize_article(article)
    topic_graph.add_article(article, categories)

# Get trending categories
trending_categories = topic_graph.get_trending_categories(timeframe_hours=24)

# Find narrative clusters
narratives = topic_graph.find_narrative_clusters()
```

## Integration with News Analyzer

The Cryptocurrency News Categorization System integrates with the existing `NewsAnalyzer` class to provide specialized categorization for cryptocurrency news articles:

```python
from src.analysis_agents.news.news_analyzer import NewsAnalyzer
from src.analysis_agents.news.crypto_news_categorizer import CryptoNewsCategorizer

# Initialize components
news_analyzer = NewsAnalyzer()
await news_analyzer.initialize()

categorizer = CryptoNewsCategorizer()
await categorizer.initialize()

# Collect articles
articles = await news_analyzer.collect_articles(timeframe="24h", assets=["BTC", "ETH"])

# Analyze articles
await news_analyzer.analyze_articles(articles)

# Categorize articles
for article in articles:
    categories = await categorizer.categorize_article(article)
    print(f"Article: {article.title}")
    print(f"Categories: {categories}")
```

## ML-based Categorization

The system supports ML-based categorization using a zero-shot classification model from Hugging Face:

```python
# Enable ML-based categorization in configuration
config.set("news_analyzer.use_ml_categorization", True)
config.set("news_analyzer.min_category_confidence", 0.6)

# Initialize categorizer with ML enabled
categorizer = CryptoNewsCategorizer()
await categorizer.initialize()
```

The ML model will automatically fall back to rule-based categorization if the model fails to load or process an article.

## Trading Strategy Application

This categorization system can enhance trading strategies by providing more targeted signals based on specific news categories:

```python
# Example trading signal generation based on news categories
if CryptoNewsCategory.REGULATION in categories and categories[CryptoNewsCategory.REGULATION] > 0.8:
    # High confidence regulation news might have significant market impact
    # Adjust risk parameters or position sizing
    adjust_risk_parameters(asset, "increase", 0.2)

if CryptoNewsCategory.SECURITY in categories and categories[CryptoNewsCategory.SECURITY] > 0.7:
    # Security issues might lead to negative price action
    generate_signal(asset, SignalType.SHORT, confidence=categories[CryptoNewsCategory.SECURITY])
```

## Example Script

An example script demonstrating the Cryptocurrency News Categorization System is provided in `examples/crypto_news_categorization_example.py`. This script shows how to:

1. Collect news articles from different sources
2. Categorize the articles using the `CryptoNewsCategorizer`
3. Build a topic graph and analyze relationships
4. Identify trending categories and narrative clusters
5. Export the results to JSON for further analysis

## Performance Considerations

- The ML-based categorization requires loading a transformer model, which may take significant memory and time during initialization
- For production environments with limited resources, consider setting `use_ml_categorization` to `False` to use only the rule-based approach
- The topic graph analysis scales with the number of articles, so consider limiting the timeframe or number of articles for large-scale applications

## Further Customization

The system can be customized in several ways:

- Add new categories by extending the `CryptoNewsCategory` class
- Customize category keywords in the `_initialize_category_keywords` method
- Adjust confidence thresholds in the configuration
- Implement custom asset-specific rules in the `_apply_asset_specific_rules` method