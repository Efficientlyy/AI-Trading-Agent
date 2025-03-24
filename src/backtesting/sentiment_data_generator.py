"""
Sentiment Data Generator for Backtesting.

This module provides utilities for generating synthetic sentiment data 
for backtesting sentiment-based trading strategies. It allows for:
- Generation of random sentiment data
- Creation of trend-based sentiment data
- Correlation with price movements (leading or lagging)
- Multiple sentiment sources with different characteristics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import random
from dataclasses import dataclass

from src.models.events import SentimentEvent
from src.models.market_data import TimeFrame


@dataclass
class SentimentSourceConfig:
    """Configuration for a sentiment data source."""
    source_name: str
    base_sentiment: float = 0.0  # Base sentiment (-1 to 1)
    volatility: float = 0.2  # Volatility of sentiment changes
    trend_strength: float = 0.1  # How strongly it follows the trend
    price_correlation: float = 0.5  # Correlation with price movements
    price_lag: int = 0  # Lag behind price movements (0 for real-time, negative for leading)
    update_frequency_hours: float = 4.0  # How often sentiment updates (in hours)
    noise_level: float = 0.2  # Random noise added to sentiment
    confidence_mean: float = 0.7  # Mean confidence level
    confidence_std: float = 0.15  # Standard deviation of confidence


class SentimentDataGenerator:
    """Generator for synthetic sentiment data used in backtesting."""
    
    def __init__(self, 
                 price_data: pd.DataFrame,
                 sources_config: Optional[List[SentimentSourceConfig]] = None,
                 random_seed: Optional[int] = None):
        """Initialize the sentiment data generator.
        
        Args:
            price_data: DataFrame with historical price data (must have 'timestamp' and 'close' columns)
            sources_config: Configuration for different sentiment sources
            random_seed: Optional seed for reproducibility
        """
        self.price_data = price_data
        self.sources_config = sources_config or self._default_sources_config()
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Validate price data
        if 'timestamp' not in price_data.columns:
            raise ValueError("Price data must contain a 'timestamp' column")
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain a 'close' column")
            
        # Convert timestamps to datetime if they're not already
        if not isinstance(price_data['timestamp'].iloc[0], datetime):
            self.price_data["timestamp"] = pd.to_datetime(price_data['timestamp'])
            
        # Sort data by timestamp
        self.price_data = self.price_data.sort_values('timestamp')
            
        # Calculate price percent changes
        self.price_data["pct_change"] = self.price_data['close'].pct_change()
        
        # Initialize sentiment data dict
        self.sentiment_data: Dict[str, List[SentimentEvent]] = {
            source.source_name: [] for source in self.sources_config
        }
    
    def _default_sources_config(self) -> List[SentimentSourceConfig]:
        """Create default configuration for sentiment sources."""
        return [
            SentimentSourceConfig(
                source_name="twitter",
                base_sentiment=0.1,
                volatility=0.3,
                trend_strength=0.15,
                price_correlation=0.4,
                price_lag=1,  # Lags behind price slightly
                update_frequency_hours=2.0,
                noise_level=0.25,
                confidence_mean=0.65,
                confidence_std=0.2
            ),
            SentimentSourceConfig(
                source_name="news",
                base_sentiment=0.0,
                volatility=0.2,
                trend_strength=0.3,
                price_correlation=0.6,
                price_lag=0,  # Real-time with price
                update_frequency_hours=4.0,
                noise_level=0.1,
                confidence_mean=0.8,
                confidence_std=0.1
            ),
            SentimentSourceConfig(
                source_name="reddit",
                base_sentiment=-0.05,
                volatility=0.4,
                trend_strength=0.1,
                price_correlation=0.3,
                price_lag=2,  # Lags behind price more
                update_frequency_hours=3.0,
                noise_level=0.3,
                confidence_mean=0.6,
                confidence_std=0.2
            )
        ]
    
    def _calculate_sentiment_value(self, 
                                  timestamp_idx: int, 
                                  source_config: SentimentSourceConfig) -> Tuple[float, float]:
        """Calculate sentiment value for a specific timestamp and source.
        
        Args:
            timestamp_idx: Index in the price data
            source_config: Configuration for the sentiment source
            
        Returns:
            Tuple of (sentiment_value, confidence)
        """
        # Get relevant price data
        price_idx = max(0, timestamp_idx - source_config.price_lag)
        price_pct_change = self.price_data['pct_change'].iloc[price_idx]
        
        # Base sentiment component
        sentiment = source_config.base_sentiment
        
        # Price correlation component
        if not np.isnan(price_pct_change):
            price_impact = price_pct_change * 10 * source_config.price_correlation
            sentiment += price_impact
        
        # Trend component (cumulative sentiment)
        if len(self.sentiment_data[source_config.source_name]) > 0:
            last_sentiment = self.sentiment_data[source_config.source_name][-1].sentiment_value
            trend_impact = last_sentiment * source_config.trend_strength
            sentiment += trend_impact
        
        # Volatility and noise component
        random_change = np.random.normal(0, source_config.volatility)
        noise = np.random.uniform(-source_config.noise_level, source_config.noise_level)
        sentiment += random_change + noise
        
        # Ensure sentiment is within [-1, 1]
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # Generate confidence value from normal distribution
        confidence = np.random.normal(source_config.confidence_mean, source_config.confidence_std)
        confidence = max(0.1, min(1.0, confidence))
        
        return sentiment, confidence
    
    def generate_sentiment_events(self, symbol: str) -> Dict[str, List[SentimentEvent]]:
        """Generate sentiment events for the entire price data period.
        
        Args:
            symbol: The trading symbol to use for the sentiment events
            
        Returns:
            Dictionary of sentiment events by source
        """
        # Reset sentiment data
        self.sentiment_data = {source.source_name: [] for source in self.sources_config}
        
        # Generate sentiment data for each source
        for source_config in self.sources_config:
            # Determine timestamps for this source based on update frequency
            update_interval = timedelta(hours=source_config.update_frequency_hours)
            current_time = self.price_data['timestamp'].iloc[0]
            end_time = self.price_data['timestamp'].iloc[-1]
            
            # Find price data points closest to our update times
            while current_time <= end_time:
                # Find closest price data point
                closest_idx = self.price_data['timestamp'].searchsorted(current_time)
                if closest_idx >= len(self.price_data):
                    break
                    
                timestamp = self.price_data['timestamp'].iloc[closest_idx]
                
                # Calculate sentiment value and confidence
                sentiment_value, confidence = self._calculate_sentiment_value(
                    closest_idx, source_config
                )
                
                # Determine sentiment direction
                sentiment_direction = "neutral"
                if sentiment_value > 0.2:
                    sentiment_direction = "bullish"
                elif sentiment_value < -0.2:
                    sentiment_direction = "bearish"
                
                # Create sentiment event
                event = SentimentEvent(
                    source=source_config.source_name,
                    symbol=symbol,
                    sentiment_value=sentiment_value,
                    sentiment_direction=sentiment_direction,
                    confidence=confidence,
                    details={"generated": True, "price": self.price_data['close'].iloc[closest_idx]},
                    timestamp=timestamp
                )
                
                # Add to sentiment data
                self.sentiment_data[source_config.source_name].append(event)
                
                # Move to next update time
                current_time += update_interval
        
        return self.sentiment_data
    
    def get_sentiment_events_for_timestamp(self, 
                                           timestamp: datetime, 
                                           lookback_hours: float = 24.0) -> Dict[str, List[SentimentEvent]]:
        """Get sentiment events for a specific timestamp, considering a lookback period.
        
        Args:
            timestamp: The timestamp to get sentiment events for
            lookback_hours: How many hours to look back for sentiment events
            
        Returns:
            Dictionary of recent sentiment events by source
        """
        lookback_start = timestamp - timedelta(hours=lookback_hours)
        
        # Filter events by timestamp
        recent_events = {}
        for source, events in self.sentiment_data.items():
            recent_events[source] = [
                event for event in events 
                if lookback_start <= event.timestamp <= timestamp
            ]
            
        return recent_events
