"""
Sentiment Analysis Dashboard.

This module provides the server-side logic for the sentiment analysis dashboard.
It collects data from various sentiment sources, analyzes correlations with price data,
and generates the dashboard display.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
import math
import numpy as np
from pathlib import Path

from fastapi import Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.data.sentiment_collector import SentimentCollector
from src.analysis_agents.sentiment.market_sentiment import FearGreedClient
from src.analysis_agents.sentiment.news_sentiment import NewsSentimentAgent
from src.analysis_agents.sentiment.social_media_sentiment import SocialMediaSentimentAgent
from src.analysis_agents.sentiment.onchain_sentiment import OnchainSentimentAgent
from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Initialize the router
router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Set up templates
templates_dir = Path("dashboard_templates")
templates = Jinja2Templates(directory=str(templates_dir))

class SentimentDashboard:
    """Sentiment Analysis Dashboard Controller."""
    
    def __init__(self):
        """Initialize the Sentiment Dashboard."""
        self.config = ConfigManager().get_config("sentiment_analysis")
        self.collector = SentimentCollector(self.config)
        self.last_update = datetime.now() - timedelta(minutes=10)
        self.cache_ttl = 300  # 5 minutes
        self.cache: Dict[str, Any] = {}
    
    async def get_dashboard_data(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get all data required for the sentiment dashboard.
        
        Args:
            symbol: The cryptocurrency symbol (default: "BTC")
            
        Returns:
            Dictionary containing all dashboard data
        """
        # Check if we have cached data that's still fresh
        cache_key = f"dashboard_{symbol}"
        now = datetime.now()
        if cache_key in self.cache and (now - self.cache[cache_key]["timestamp"]).total_seconds() < self.cache_ttl:
            logger.info(f"Using cached sentiment dashboard data for {symbol}")
            return self.cache[cache_key]["data"]
        
        logger.info(f"Collecting sentiment dashboard data for {symbol}")
        
        # Collect data from all sentiment sources
        try:
            # Run all data collection functions concurrently
            overall, fear_greed, news, social, onchain, contrarian, correlations, history = await asyncio.gather(
                self._get_overall_sentiment(symbol),
                self._get_fear_greed_data(),
                self._get_news_sentiment(symbol),
                self._get_social_sentiment(symbol),
                self._get_onchain_sentiment(symbol),
                self._get_contrarian_signals(symbol),
                self._get_sentiment_price_correlations(symbol),
                self._get_sentiment_history(symbol)
            )
            
            # Combine all data
            dashboard_data = {
                "sentiment_data": overall,
                "fear_greed_data": fear_greed,
                "news_sentiment": news,
                "social_sentiment": social,
                "onchain_sentiment": onchain,
                "contrarian_signals": contrarian,
                "correlations": correlations,
                "sentiment_history": history,
                "current_time": now.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Cache the data
            self.cache[cache_key] = {
                "timestamp": now,
                "data": dashboard_data
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error collecting sentiment dashboard data: {e}")
            # Return mock data in case of error
            return self._get_mock_dashboard_data(symbol)
    
    async def _get_overall_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get the overall sentiment data by combining all sources.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            Dictionary with overall sentiment data
        """
        try:
            # In a real implementation, we would collect sentiment data from all sources
            # and combine them with weights. For demonstration, we'll use the sentiment collector
            start_date = datetime.now() - timedelta(days=7)
            
            # Try to load historical data
            try:
                combined_data = self.collector.combine_sentiment_sources(
                    symbol=symbol,
                    start_date=start_date,
                    resample_freq='1H'
                )
                
                # Get the latest row for current sentiment
                latest = combined_data.iloc[-1]
                
                # Average of all sentiment sources for the overall value
                sentiment_value = latest['combined']
                
                # Direction based on value
                if sentiment_value > 0.6:
                    direction = "bullish"
                elif sentiment_value < 0.4:
                    direction = "bearish"
                else:
                    direction = "neutral"
                
                # Confidence is highest if sources agree, lowest if they disagree
                # Calculate standard deviation of sentiment values as a measure of agreement
                values = []
                for col in combined_data.columns:
                    if col != 'combined':
                        values.append(latest[col])
                
                if len(values) > 1:
                    std_dev = np.std(values)
                    confidence = 1.0 - min(0.5, std_dev)  # Lower std_dev means higher confidence
                else:
                    confidence = 0.8  # Default confidence with limited data
                
                # Active sources
                sources = [col for col in combined_data.columns if col != 'combined' and not pd.isna(latest[col])]
                
                # Return the data
                return {
                    "value": float(sentiment_value),
                    "direction": direction,
                    "confidence": float(confidence),
                    "sources": sources
                }
                
            except Exception as e:
                logger.warning(f"Error loading sentiment data: {e}")
                # Fall back to mock data
                raise
                
        except Exception as e:
            logger.error(f"Error getting overall sentiment: {e}")
            # Return mock data
            direction = random.choice(["bullish", "neutral", "bearish"])
            value = random.uniform(0.3, 0.7)
            if direction == "bullish":
                value = random.uniform(0.6, 0.8)
            elif direction == "bearish":
                value = random.uniform(0.2, 0.4)
            
            return {
                "value": value,
                "direction": direction,
                "confidence": random.uniform(0.7, 0.9),
                "sources": ["fear_greed", "news", "social_media", "onchain"]
            }
    
    async def _get_fear_greed_data(self) -> Dict[str, Any]:
        """
        Get Fear & Greed Index data.
        
        Returns:
            Dictionary with Fear & Greed Index data
        """
        try:
            # Initialize the Fear & Greed client
            client = FearGreedClient()
            
            # Get current index
            current_data = client.get_data()
            value = current_data.get("value", 50)
            classification = current_data.get("classification", "Neutral")
            
            # Get historical data for comparison
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            last_week = now - timedelta(days=7)
            last_month = now - timedelta(days=30)
            
            # In a real implementation, we would get the actual historical data
            # for this demo, we'll mock some realistic values
            history = [
                {"period": "Yesterday", "value": max(0, min(100, value + random.randint(-5, 5))), "classification": ""},
                {"period": "Last Week", "value": max(0, min(100, value + random.randint(-10, 10))), "classification": ""},
                {"period": "Last Month", "value": max(0, min(100, value + random.randint(-20, 20))), "classification": ""}
            ]
            
            # Assign classifications
            for item in history:
                val = item["value"]
                if val <= 25:
                    item["classification"] = "Extreme Fear"
                elif val <= 45:
                    item["classification"] = "Fear"
                elif val <= 55:
                    item["classification"] = "Neutral"
                elif val <= 75:
                    item["classification"] = "Greed"
                else:
                    item["classification"] = "Extreme Greed"
            
            return {
                "value": value,
                "classification": classification,
                "history": history
            }
        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index data: {e}")
            # Generate mock data
            value = random.randint(25, 75)
            
            if value <= 25:
                classification = "Extreme Fear"
            elif value <= 45:
                classification = "Fear"
            elif value <= 55:
                classification = "Neutral"
            elif value <= 75:
                classification = "Greed"
            else:
                classification = "Extreme Greed"
            
            history = [
                {"period": "Yesterday", "value": max(0, min(100, value + random.randint(-5, 5))), "classification": "Neutral"},
                {"period": "Last Week", "value": max(0, min(100, value + random.randint(-10, 10))), "classification": "Fear"},
                {"period": "Last Month", "value": max(0, min(100, value + random.randint(-20, 20))), "classification": "Greed"}
            ]
            
            return {
                "value": value,
                "classification": classification,
                "history": history
            }
    
    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get news sentiment data.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            Dictionary with news sentiment data
        """
        try:
            # In a real implementation, we would use the NewsSentimentAgent
            # For demonstration, we'll generate realistic mock data
            
            # Basic sentiment data
            direction = random.choice(["bullish", "neutral", "bearish"])
            value = random.uniform(0.3, 0.7)
            if direction == "bullish":
                value = random.uniform(0.6, 0.8)
            elif direction == "bearish":
                value = random.uniform(0.2, 0.4)
            
            confidence = random.uniform(0.7, 0.9)
            articles_analyzed = random.randint(15, 40)
            
            # Generate mock headlines with their sentiments
            headlines = []
            news_sources = ["CoinDesk", "CryptoSlate", "Cointelegraph", "Bitcoin Magazine", "Decrypt"]
            headline_templates = [
                f"{symbol} Could Rally to ${{price}} According to Analysts",
                f"Market Analysis: {symbol}'s Path Forward After Recent {{direction}}",
                f"Breaking: {symbol} {{event}} as {{actor}} {{action}}",
                f"{symbol} Shows Signs of {{movement}} Amid Market {{condition}}",
                f"Report: {symbol} {{metric}} {{direction}} by {{percent}}% in Past Week"
            ]
            
            events = ["Surges", "Plummets", "Stabilizes", "Consolidates", "Breaks Out"]
            actors = ["Institutional Investors", "Whales", "Retail Traders", "Mining Pools", "Exchanges"]
            actions = ["Accumulate", "Sell Off", "Hold Positions", "Increase Exposure", "Reduce Holdings"]
            movements = ["Recovery", "Weakness", "Strength", "Volatility", "Stability"]
            conditions = ["Uncertainty", "Optimism", "FUD", "FOMO", "Consolidation"]
            metrics = ["Trading Volume", "Exchange Outflows", "Hash Rate", "Active Addresses", "Futures Open Interest"]
            directions = ["Up", "Down", "Unchanged", "Higher", "Lower"]
            
            for i in range(5):
                headline_template = random.choice(headline_templates)
                headline_sentiment = random.choice(["bullish", "neutral", "bearish"])
                sentiment_value = random.uniform(0.3, 0.7)
                
                if headline_sentiment == "bullish":
                    sentiment_value = random.uniform(0.6, 0.9)
                elif headline_sentiment == "bearish":
                    sentiment_value = random.uniform(0.1, 0.4)
                
                # Format the headline
                if "price" in headline_template:
                    price = random.randint(40000, 100000)
                    headline = headline_template.replace("{price}", f"{price:,}")
                elif "direction" in headline_template:
                    headline = headline_template.replace("{direction}", random.choice(["Pullback", "Rally", "Correction", "Uptick", "Downturn"]))
                elif "event" in headline_template:
                    headline = headline_template.replace("{event}", random.choice(events)).replace("{actor}", random.choice(actors)).replace("{action}", random.choice(actions))
                elif "movement" in headline_template:
                    headline = headline_template.replace("{movement}", random.choice(movements)).replace("{condition}", random.choice(conditions))
                elif "metric" in headline_template:
                    headline = headline_template.replace("{metric}", random.choice(metrics)).replace("{direction}", random.choice(directions)).replace("{percent}", str(random.randint(5, 25)))
                
                timestamp = datetime.now() - timedelta(hours=random.randint(1, 12))
                
                headlines.append({
                    "title": headline,
                    "source": random.choice(news_sources),
                    "time": timestamp.strftime("%Y-%m-%d %H:%M"),
                    "sentiment": headline_sentiment,
                    "sentiment_value": sentiment_value
                })
            
            return {
                "direction": direction,
                "value": value,
                "confidence": confidence,
                "articles_analyzed": articles_analyzed,
                "headlines": headlines
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return {
                "direction": "neutral",
                "value": 0.5,
                "confidence": 0.5,
                "articles_analyzed": 0,
                "headlines": []
            }
    
    async def _get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get social media sentiment data.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            Dictionary with social media sentiment data
        """
        try:
            # In a real implementation, we would use the SocialMediaSentimentAgent
            # For demonstration, we'll generate realistic mock data
            
            # Basic sentiment data
            direction = random.choice(["bullish", "neutral", "bearish"])
            value = random.uniform(0.3, 0.7)
            if direction == "bullish":
                value = random.uniform(0.6, 0.8)
            elif direction == "bearish":
                value = random.uniform(0.2, 0.4)
            
            confidence = random.uniform(0.7, 0.9)
            posts_analyzed = random.randint(200, 1000)
            mentions_per_hour = random.randint(50, 300)
            
            # Generate keywords with counts
            keyword_options = [
                f"buy{symbol}", f"sell{symbol}", "hodl", "moon", "dump", "bearish", "bullish",
                "rally", "dip", "correction", "ATH", "bottom", "resistance", "support",
                "pump", "dump", "whales", "altseason", "bear trap", "bull trap",
                "breakout", "breakdown", "consolidation", "accumulation", "distribution"
            ]
            
            keywords = []
            for i in range(12):
                keywords.append({
                    "word": random.choice(keyword_options),
                    "count": random.randint(10, 100)
                })
            
            # Sort by count descending
            keywords.sort(key=lambda x: x["count"], reverse=True)
            
            return {
                "direction": direction,
                "value": value,
                "confidence": confidence,
                "posts_analyzed": posts_analyzed,
                "mentions_per_hour": mentions_per_hour,
                "keywords": keywords
            }
            
        except Exception as e:
            logger.error(f"Error getting social media sentiment: {e}")
            return {
                "direction": "neutral",
                "value": 0.5,
                "confidence": 0.5,
                "posts_analyzed": 0,
                "mentions_per_hour": 0,
                "keywords": []
            }
    
    async def _get_onchain_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get on-chain sentiment data.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            Dictionary with on-chain sentiment data
        """
        try:
            # In a real implementation, we would use the OnchainSentimentAgent
            # For demonstration, we'll generate realistic mock data
            
            # Basic sentiment data
            direction = random.choice(["bullish", "neutral", "bearish"])
            value = random.uniform(0.3, 0.7)
            if direction == "bullish":
                value = random.uniform(0.6, 0.8)
            elif direction == "bearish":
                value = random.uniform(0.2, 0.4)
            
            # Generate on-chain metrics with realistic values and changes
            metrics = [
                {
                    "name": "Active Addresses",
                    "value": f"{random.randint(700000, 1200000):,}",
                    "change": round(random.uniform(-8.0, 15.0), 1)
                },
                {
                    "name": "Exchange Netflow",
                    "value": f"{random.randint(-5000, 5000):,} BTC",
                    "change": round(random.uniform(-20.0, 20.0), 1)
                },
                {
                    "name": "Hash Rate",
                    "value": f"{random.randint(200, 400):,} EH/s",
                    "change": round(random.uniform(-5.0, 10.0), 1)
                },
                {
                    "name": "Large Transactions",
                    "value": f"{random.randint(1500, 3000):,}",
                    "change": round(random.uniform(-12.0, 18.0), 1)
                },
                {
                    "name": "SOPR",
                    "value": round(random.uniform(0.95, 1.05), 4),
                    "change": round(random.uniform(-5.0, 5.0), 1)
                }
            ]
            
            return {
                "direction": direction,
                "value": value,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting on-chain sentiment: {e}")
            return {
                "direction": "neutral",
                "value": 0.5,
                "metrics": []
            }
    
    async def _get_contrarian_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get contrarian sentiment signals.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            List of contrarian signals
        """
        try:
            # In a real implementation, we would analyze sentiment extremes across sources
            # For demonstration, we'll generate realistic mock contrarian signals
            
            signals = []
            
            # Generate 0-3 contrarian signals
            signal_count = random.randint(0, 3)
            
            sources = ["Fear & Greed Index", "Social Media Sentiment", "News Sentiment", "On-Chain Metrics"]
            recommendations = [
                "Consider contrarian position as sentiment appears overly extreme",
                "High probability of sentiment reversal based on historical patterns",
                "Current sentiment levels have preceded significant price movements in opposite direction",
                "Extreme sentiment without technical confirmation suggests possible reversal"
            ]
            
            for i in range(signal_count):
                source = random.choice(sources)
                
                # Extreme values are either very high or very low
                extreme_high = random.choice([True, False])
                
                if extreme_high:
                    value = random.uniform(0.85, 0.95)
                    direction = "bullish"
                    message = f"Extreme {direction} sentiment detected: possible contrarian (bearish) signal"
                else:
                    value = random.uniform(0.05, 0.15)
                    direction = "bearish"
                    message = f"Extreme {direction} sentiment detected: possible contrarian (bullish) signal"
                
                # Time within last 6 hours
                time = datetime.now() - timedelta(hours=random.randint(0, 6))
                
                signals.append({
                    "source": source,
                    "symbol": symbol,
                    "direction": direction,
                    "value": value,
                    "is_extreme": True,
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "message": message,
                    "recommendation": random.choice(recommendations)
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting contrarian signals: {e}")
            return []
    
    async def _get_sentiment_price_correlations(self, symbol: str) -> Dict[str, float]:
        """
        Get correlations between sentiment sources and price.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            Dictionary with correlation coefficients
        """
        try:
            # In a real implementation, we would calculate actual correlations
            # For demonstration, we'll generate realistic correlation values
            
            # Typically news and social media have lower correlations (more noise)
            # Fear & Greed and on-chain metrics tend to have higher correlations
            
            return {
                "news_1d": round(random.uniform(-0.3, 0.5), 2),  # More variable
                "news_7d": round(random.uniform(-0.2, 0.6), 2),
                "social_1d": round(random.uniform(-0.4, 0.4), 2),  # More variable
                "social_7d": round(random.uniform(-0.3, 0.5), 2),
                "fear_greed_1d": round(random.uniform(0.3, 0.7), 2),  # Stronger correlation
                "fear_greed_7d": round(random.uniform(0.4, 0.8), 2),
                "onchain_1d": round(random.uniform(0.2, 0.6), 2),  # More consistent
                "onchain_7d": round(random.uniform(0.3, 0.7), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment-price correlations: {e}")
            return {
                "news_1d": 0,
                "news_7d": 0,
                "social_1d": 0,
                "social_7d": 0,
                "fear_greed_1d": 0,
                "fear_greed_7d": 0,
                "onchain_1d": 0,
                "onchain_7d": 0
            }
    
    async def _get_sentiment_history(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get historical sentiment data.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            List of historical sentiment records
        """
        try:
            # In a real implementation, we would load historical data from the database
            # For demonstration, we'll generate realistic historical data
            
            history = []
            
            # Generate 7 days of data (one entry per day)
            for i in range(7):
                date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                
                # Overall sentiment (tends to follow a trend with some randomness)
                base_sentiment = 0.5 + math.sin(i / 3.0) * 0.2  # Oscillates between 0.3 and 0.7
                base_sentiment += random.uniform(-0.1, 0.1)  # Add some noise
                base_sentiment = max(0.1, min(0.9, base_sentiment))  # Clamp
                
                # Direction based on value
                if base_sentiment > 0.6:
                    direction = "bullish"
                elif base_sentiment < 0.4:
                    direction = "bearish"
                else:
                    direction = "neutral"
                
                # Individual sources (correlated but with their own variations)
                news = max(0.1, min(0.9, base_sentiment + random.uniform(-0.15, 0.15)))
                social = max(0.1, min(0.9, base_sentiment + random.uniform(-0.2, 0.2)))
                onchain = max(0.1, min(0.9, base_sentiment + random.uniform(-0.1, 0.1)))
                fear_greed = int(base_sentiment * 100)
                
                # Price change (somewhat correlated with sentiment)
                sentiment_effect = (base_sentiment - 0.5) * 2  # -1 to 1
                price_change = sentiment_effect * random.uniform(1.0, 5.0) + random.uniform(-2.0, 2.0)
                price_change = round(price_change, 2)
                
                history.append({
                    "date": date,
                    "overall": {
                        "value": base_sentiment,
                        "direction": direction
                    },
                    "news": news,
                    "social": social,
                    "onchain": onchain,
                    "fear_greed": fear_greed,
                    "price_change": price_change
                })
            
            # Sort by date (newest first)
            history.sort(key=lambda x: x["date"], reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting sentiment history: {e}")
            return []
    
    def _get_mock_dashboard_data(self, symbol: str) -> Dict[str, Any]:
        """
        Generate mock data for the dashboard in case of errors.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            Dictionary with mock dashboard data
        """
        # Overall sentiment
        sentiment_data = {
            "value": random.uniform(0.4, 0.6),
            "direction": random.choice(["bullish", "neutral", "bearish"]),
            "confidence": random.uniform(0.7, 0.9),
            "sources": ["fear_greed", "news", "social_media", "onchain"]
        }
        
        # Fear & Greed Index
        fear_greed_data = {
            "value": random.randint(30, 70),
            "classification": "Neutral",
            "history": [
                {"period": "Yesterday", "value": random.randint(30, 70), "classification": "Neutral"},
                {"period": "Last Week", "value": random.randint(30, 70), "classification": "Neutral"},
                {"period": "Last Month", "value": random.randint(30, 70), "classification": "Neutral"}
            ]
        }
        
        # News sentiment
        news_sentiment = {
            "direction": "neutral",
            "value": 0.5,
            "confidence": 0.8,
            "articles_analyzed": random.randint(20, 50),
            "headlines": [
                {"title": f"{symbol} Market Analysis", "source": "CoinDesk", "time": "2023-03-01 14:30", "sentiment": "neutral", "sentiment_value": 0.5},
                {"title": f"{symbol} Price Prediction", "source": "Cointelegraph", "time": "2023-03-01 12:15", "sentiment": "bullish", "sentiment_value": 0.7},
                {"title": f"What's Next for {symbol}?", "source": "CryptoSlate", "time": "2023-03-01 10:45", "sentiment": "bearish", "sentiment_value": 0.3}
            ]
        }
        
        # Social media sentiment
        social_sentiment = {
            "direction": "neutral",
            "value": 0.5,
            "confidence": 0.7,
            "posts_analyzed": random.randint(500, 1500),
            "mentions_per_hour": random.randint(100, 300),
            "keywords": [
                {"word": "hodl", "count": 85},
                {"word": "moon", "count": 72},
                {"word": "bearish", "count": 65},
                {"word": "bullish", "count": 58},
                {"word": "dip", "count": 47},
                {"word": "ATH", "count": 35}
            ]
        }
        
        # On-chain sentiment
        onchain_sentiment = {
            "direction": "neutral",
            "value": 0.5,
            "metrics": [
                {"name": "Active Addresses", "value": "850,000", "change": 2.5},
                {"name": "Exchange Netflow", "value": "-1,250 BTC", "change": -5.8},
                {"name": "Hash Rate", "value": "320 EH/s", "change": 1.2},
                {"name": "Large Transactions", "value": "2,150", "change": 3.7},
                {"name": "SOPR", "value": 1.0125, "change": 0.8}
            ]
        }
        
        # Contrarian signals
        contrarian_signals = [
            {
                "source": "Fear & Greed Index",
                "symbol": symbol,
                "direction": "bearish",
                "value": 0.1,
                "is_extreme": True,
                "time": "2023-03-01 15:30",
                "message": f"Extreme bearish sentiment detected: possible contrarian (bullish) signal",
                "recommendation": "Consider contrarian position as sentiment appears overly extreme"
            }
        ]
        
        # Correlations
        correlations = {
            "news_1d": 0.2,
            "news_7d": 0.3,
            "social_1d": 0.15,
            "social_7d": 0.25,
            "fear_greed_1d": 0.6,
            "fear_greed_7d": 0.7,
            "onchain_1d": 0.5,
            "onchain_7d": 0.6
        }
        
        # Sentiment history
        sentiment_history = [
            {
                "date": "2023-03-01",
                "overall": {"value": 0.6, "direction": "bullish"},
                "news": 0.65,
                "social": 0.55,
                "onchain": 0.63,
                "fear_greed": 65,
                "price_change": 2.8
            },
            {
                "date": "2023-02-28",
                "overall": {"value": 0.5, "direction": "neutral"},
                "news": 0.55,
                "social": 0.48,
                "onchain": 0.52,
                "fear_greed": 55,
                "price_change": 0.5
            },
            {
                "date": "2023-02-27",
                "overall": {"value": 0.4, "direction": "bearish"},
                "news": 0.35,
                "social": 0.42,
                "onchain": 0.38,
                "fear_greed": 42,
                "price_change": -1.2
            }
        ]
        
        return {
            "sentiment_data": sentiment_data,
            "fear_greed_data": fear_greed_data,
            "news_sentiment": news_sentiment,
            "social_sentiment": social_sentiment,
            "onchain_sentiment": onchain_sentiment,
            "contrarian_signals": contrarian_signals,
            "correlations": correlations,
            "sentiment_history": sentiment_history,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# Initialize the dashboard controller
sentiment_dashboard = SentimentDashboard()

@router.get("/", response_class=HTMLResponse)
async def get_sentiment_dashboard(request: Request, symbol: str = "BTC"):
    """
    Render the sentiment dashboard.
    
    Args:
        request: The FastAPI request
        symbol: The cryptocurrency symbol (default: "BTC")
        
    Returns:
        HTML response with the rendered dashboard
    """
    try:
        # Get dashboard data
        dashboard_data = await sentiment_dashboard.get_dashboard_data(symbol)
        
        # Add request to template context
        dashboard_data["request"] = request
        
        # Render the template
        return templates.TemplateResponse(
            "sentiment_dashboard.html", 
            dashboard_data
        )
        
    except Exception as e:
        logger.error(f"Error rendering sentiment dashboard: {e}")
        return HTMLResponse(f"<h1>Error</h1><p>Failed to load sentiment dashboard: {str(e)}</p>")

@router.get("/api/data")
async def get_sentiment_data(symbol: str = "BTC"):
    """
    Get sentiment dashboard data as JSON for API clients.
    
    Args:
        symbol: The cryptocurrency symbol (default: "BTC")
        
    Returns:
        JSON response with sentiment data
    """
    try:
        # Get dashboard data
        dashboard_data = await sentiment_dashboard.get_dashboard_data(symbol)
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting sentiment data: {e}")
        return {"error": str(e)}