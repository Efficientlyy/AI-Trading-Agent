"""Market sentiment analysis module.

This module provides tools for analyzing market sentiment from various sources:
1. Social media sentiment (Twitter, Reddit)
2. News sentiment
3. Market fear/greed indicators
4. Exchange order flow
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict


class MarketSentiment:
    """Market sentiment analyzer."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.social_sentiment: Dict[str, float] = {}
        self.news_sentiment: Dict[str, float] = {}
        self.order_flow_sentiment: Dict[str, float] = {}
        self.fear_greed_index: float = 50.0  # Neutral by default
    
    def update_social_sentiment(self, symbol: str, score: float) -> None:
        """Update social media sentiment score."""
        self.social_sentiment[symbol] = float(score)
    
    def update_news_sentiment(self, symbol: str, score: float) -> None:
        """Update news sentiment score."""
        self.news_sentiment[symbol] = float(score)
    
    def update_fear_greed(self, score: float) -> None:
        """Update fear/greed index."""
        self.fear_greed_index = float(score)
    
    def update_order_flow_sentiment(self, symbol: str, buy_volume: float, sell_volume: float) -> None:
        """Update order flow sentiment based on buy/sell volume ratio."""
        if sell_volume > 0:
            ratio = buy_volume / sell_volume
            self.order_flow_sentiment[symbol] = float(ratio - 1.0)  # Normalize around 0
        else:
            self.order_flow_sentiment[symbol] = 1.0 if buy_volume > 0 else 0.0
    
    def get_social_sentiment(self, symbol: str) -> float:
        """Get social sentiment score for a symbol."""
        return self.social_sentiment.get(symbol, 0.0)
    
    def get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment score for a symbol."""
        return self.news_sentiment.get(symbol, 0.0)
    
    def get_order_flow_sentiment(self, symbol: str) -> float:
        """Get order flow sentiment score for a symbol."""
        return self.order_flow_sentiment.get(symbol, 0.0)
    
    def get_aggregate_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get aggregate sentiment metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary containing sentiment scores
        """
        return {
            "sentiment_score": (
                self.get_social_sentiment(symbol) * 0.3 +
                self.get_news_sentiment(symbol) * 0.3 +
                self.get_order_flow_sentiment(symbol) * 0.4
            ),
            "social_sentiment": self.get_social_sentiment(symbol),
            "news_sentiment": self.get_news_sentiment(symbol),
            "order_flow_sentiment": self.get_order_flow_sentiment(symbol),
            "fear_greed_index": self.fear_greed_index
        } 