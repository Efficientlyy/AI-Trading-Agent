"""
Sentiment data collectors for the AI Trading Agent.

This package contains collectors for various sentiment data sources:
- Twitter API
- Reddit API
- News APIs
- Fear & Greed Index
"""

from .twitter_collector import TwitterAPICollector
from .news_collector import NewsAPICollector
from .fear_greed_collector import FearGreedIndexCollector

__all__ = [
    'TwitterAPICollector',
    'NewsAPICollector',
    'FearGreedIndexCollector',
]
