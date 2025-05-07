"""
Alpha Vantage Client for News Sentiment Data

This module provides a client for the Alpha Vantage News Sentiment API.
It allows fetching news sentiment data for various topics and tickers.
"""

import os
import requests
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """
    Client for interacting with the Alpha Vantage News Sentiment API.
    
    This client handles authentication, rate limiting, and data retrieval
    from the Alpha Vantage News Sentiment API.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_per_minute: int = 5):
        """
        Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key. If None, will try to load from environment variable.
            rate_limit_per_minute: Maximum number of requests per minute (default: 5 for free tier)
        """
        # Load API key from environment if not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            
        if not api_key:
            raise ValueError("Alpha Vantage API key is required. Set ALPHA_VANTAGE_API_KEY environment variable or pass as parameter.")
        
        self.api_key = api_key
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_timestamps: List[float] = []
        
        logger.info("Alpha Vantage client initialized")
    
    def _enforce_rate_limit(self):
        """
        Enforce rate limiting by sleeping if necessary.
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if current_time - ts < 60]
        
        # If we've reached the rate limit, sleep until we can make another request
        if len(self.request_timestamps) >= self.rate_limit_per_minute:
            oldest_timestamp = min(self.request_timestamps)
            sleep_time = 60 - (current_time - oldest_timestamp)
            
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Add current timestamp to the list
        self.request_timestamps.append(time.time())
    
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Make a request to the Alpha Vantage API.
        
        Args:
            params: Request parameters
            
        Returns:
            API response as a dictionary
        """
        # Add API key to parameters
        params["apikey"] = self.api_key
        
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        # Make the request
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
            if "Information" in data and "please consider optimizing your API call frequency" in data["Information"]:
                logger.warning(f"Alpha Vantage rate limit warning: {data['Information']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Alpha Vantage failed: {e}")
            raise
    
    def get_news_sentiment(self, topics: Optional[List[str]] = None, 
                          tickers: Optional[List[str]] = None,
                          time_from: Optional[str] = None,
                          time_to: Optional[str] = None,
                          limit: int = 50,
                          sort: str = "LATEST") -> Dict[str, Any]:
        """
        Get news sentiment data from Alpha Vantage.
        
        Args:
            topics: List of topics to filter by (e.g., ["blockchain", "technology"])
            tickers: List of tickers to filter by (e.g., ["BTC", "ETH"])
            time_from: Start time in format YYYYMMDDTHHMM (e.g., "202304010900")
            time_to: End time in format YYYYMMDDTHHMM (e.g., "202304011600")
            limit: Maximum number of results to return
            sort: Sorting order ("LATEST", "EARLIEST", "RELEVANCE")
            
        Returns:
            News sentiment data as a dictionary
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "limit": str(limit),
            "sort": sort
        }
        
        # Add optional parameters if provided
        if topics:
            params["topics"] = ",".join(topics)
        
        if tickers:
            params["tickers"] = ",".join(tickers)
        
        if time_from:
            params["time_from"] = time_from
        
        if time_to:
            params["time_to"] = time_to
        
        return self._make_request(params)
    
    def get_sentiment_by_topic(self, topic: str, days_back: int = 7) -> pd.DataFrame:
        """
        Get sentiment data for a specific topic over the past N days.
        
        Args:
            topic: Topic to get sentiment for (e.g., "blockchain", "technology")
            days_back: Number of days to look back
            
        Returns:
            DataFrame with sentiment data
        """
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Format times for API
        time_from = start_time.strftime("%Y%m%dT%H%M")
        time_to = end_time.strftime("%Y%m%dT%H%M")
        
        # Get sentiment data
        data = self.get_news_sentiment(
            topics=[topic],
            time_from=time_from,
            time_to=time_to,
            limit=100
        )
        
        # Check if we have feed data
        if "feed" not in data or not data["feed"]:
            logger.warning(f"No news data found for topic '{topic}'")
            return pd.DataFrame()
        
        # Extract relevant information
        articles = []
        
        for article in data["feed"]:
            # Extract sentiment scores
            overall_sentiment_score = float(article.get("overall_sentiment_score", 0))
            overall_sentiment_label = article.get("overall_sentiment_label", "neutral")
            
            # Extract ticker-specific sentiment if available
            ticker_sentiments = {}
            if "ticker_sentiment" in article:
                for ticker_data in article["ticker_sentiment"]:
                    ticker = ticker_data.get("ticker")
                    if ticker:
                        ticker_sentiments[ticker] = {
                            "score": float(ticker_data.get("ticker_sentiment_score", 0)),
                            "label": ticker_data.get("ticker_sentiment_label", "neutral")
                        }
            
            # Create article entry
            article_entry = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "time_published": article.get("time_published", ""),
                "authors": article.get("authors", []),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "category_within_source": article.get("category_within_source", ""),
                "source_domain": article.get("source_domain", ""),
                "overall_sentiment_score": overall_sentiment_score,
                "overall_sentiment_label": overall_sentiment_label,
                "ticker_sentiments": ticker_sentiments
            }
            
            articles.append(article_entry)
        
        # Create DataFrame
        df = pd.DataFrame(articles)
        
        # Convert time_published to datetime
        if "time_published" in df.columns:
            df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S")
        
        return df
    
    def get_sentiment_by_crypto(self, crypto_symbol: str, days_back: int = 7) -> pd.DataFrame:
        """
        Get sentiment data for a specific cryptocurrency over the past N days.
        
        Note: This may not work on the free tier of Alpha Vantage.
        
        Args:
            crypto_symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            days_back: Number of days to look back
            
        Returns:
            DataFrame with sentiment data
        """
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Format times for API
        time_from = start_time.strftime("%Y%m%dT%H%M")
        time_to = end_time.strftime("%Y%m%dT%H%M")
        
        # Get sentiment data
        try:
            data = self.get_news_sentiment(
                tickers=[crypto_symbol],
                time_from=time_from,
                time_to=time_to,
                limit=100
            )
            
            # Check if we have feed data
            if "feed" not in data or not data["feed"]:
                logger.warning(f"No news data found for crypto '{crypto_symbol}'")
                return pd.DataFrame()
            
            # Extract relevant information (same as get_sentiment_by_topic)
            articles = []
            
            for article in data["feed"]:
                # Extract sentiment scores
                overall_sentiment_score = float(article.get("overall_sentiment_score", 0))
                overall_sentiment_label = article.get("overall_sentiment_label", "neutral")
                
                # Extract ticker-specific sentiment if available
                ticker_sentiments = {}
                if "ticker_sentiment" in article:
                    for ticker_data in article["ticker_sentiment"]:
                        ticker = ticker_data.get("ticker")
                        if ticker:
                            ticker_sentiments[ticker] = {
                                "score": float(ticker_data.get("ticker_sentiment_score", 0)),
                                "label": ticker_data.get("ticker_sentiment_label", "neutral")
                            }
                
                # Create article entry
                article_entry = {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "time_published": article.get("time_published", ""),
                    "authors": article.get("authors", []),
                    "summary": article.get("summary", ""),
                    "source": article.get("source", ""),
                    "category_within_source": article.get("category_within_source", ""),
                    "source_domain": article.get("source_domain", ""),
                    "overall_sentiment_score": overall_sentiment_score,
                    "overall_sentiment_label": overall_sentiment_label,
                    "ticker_sentiments": ticker_sentiments
                }
                
                articles.append(article_entry)
            
            # Create DataFrame
            df = pd.DataFrame(articles)
            
            # Convert time_published to datetime
            if "time_published" in df.columns:
                df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S")
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to get sentiment data for crypto '{crypto_symbol}': {e}")
            logger.info("Falling back to topic-based sentiment for 'blockchain' and 'cryptocurrency'")
            
            # Fall back to topic-based sentiment
            return self.get_sentiment_by_topic("blockchain", days_back)
    
    def get_aggregated_sentiment(self, topic_or_symbol: str, 
                                is_topic: bool = True, 
                                days_back: int = 7,
                                resample_freq: str = 'D') -> pd.DataFrame:
        """
        Get aggregated sentiment data for a topic or symbol over time.
        
        Args:
            topic_or_symbol: Topic or symbol to get sentiment for
            is_topic: Whether topic_or_symbol is a topic (True) or a ticker symbol (False)
            days_back: Number of days to look back
            resample_freq: Frequency to resample data to ('D' for daily, 'H' for hourly, etc.)
            
        Returns:
            DataFrame with aggregated sentiment data over time
        """
        # Get raw sentiment data
        if is_topic:
            df = self.get_sentiment_by_topic(topic_or_symbol, days_back)
        else:
            df = self.get_sentiment_by_crypto(topic_or_symbol, days_back)
        
        if df.empty:
            return pd.DataFrame()
        
        # Set time as index for resampling
        df = df.set_index('time_published')
        
        # Resample and aggregate sentiment scores
        agg_df = df.resample(resample_freq)['overall_sentiment_score'].agg(['mean', 'count', 'std'])
        
        # Rename columns for clarity
        agg_df.columns = ['sentiment_score_avg', 'article_count', 'sentiment_score_std']
        
        # Add sentiment label based on average score
        def get_sentiment_label(score):
            if score >= 0.35:
                return "bullish"
            elif score <= -0.35:
                return "bearish"
            else:
                return "neutral"
        
        agg_df['sentiment_label'] = agg_df['sentiment_score_avg'].apply(get_sentiment_label)
        
        # Reset index to get time as a column
        agg_df = agg_df.reset_index()
        
        return agg_df


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = AlphaVantageClient()
    
    # Get sentiment data for blockchain topic
    sentiment_df = client.get_sentiment_by_topic("blockchain", days_back=3)
    print(f"Retrieved {len(sentiment_df)} articles for blockchain topic")
    
    # Get aggregated sentiment
    agg_sentiment = client.get_aggregated_sentiment("blockchain", is_topic=True, days_back=7)
    print("\nAggregated Sentiment:")
    print(agg_sentiment)
    
    # Try getting crypto-specific sentiment (may not work on free tier)
    try:
        btc_sentiment = client.get_sentiment_by_crypto("BTC", days_back=3)
        print(f"\nRetrieved {len(btc_sentiment)} articles for BTC")
    except Exception as e:
        print(f"\nFailed to get BTC sentiment: {e}")
