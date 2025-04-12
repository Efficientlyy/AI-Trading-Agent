"""
News API Collector for sentiment analysis.

This module provides functionality to collect news articles from various
financial news APIs for sentiment analysis in the AI Trading Agent.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import requests
import pandas as pd

from ...common.logging_config import setup_logging
from ..providers.base_provider import BaseSentimentProvider

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class NewsAPICollector(BaseSentimentProvider):
    """
    Collects news articles from various financial news APIs for sentiment analysis.
    
    Supports multiple news APIs:
    - NewsAPI.org
    - Alpha Vantage News
    - Finnhub
    
    Attributes:
        api_keys (Dict[str, str]): API keys for different news services
        cache_dir (str): Directory to cache API responses
        cache_expiry (int): Cache expiry time in seconds
        default_api (str): Default API to use if not specified
    """
    
    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alphavantage_key: Optional[str] = None,
        finnhub_key: Optional[str] = None,
        cache_dir: str = "cache/news",
        cache_expiry: int = 3600,
        default_api: str = "newsapi",
        rate_limit_wait: int = 15,
    ):
        """
        Initialize the News API collector.
        
        Args:
            newsapi_key: NewsAPI.org API key (defaults to NEWS_API_KEY env var)
            alphavantage_key: Alpha Vantage API key (defaults to ALPHA_VANTAGE_KEY env var)
            finnhub_key: Finnhub API key (defaults to FINNHUB_KEY env var)
            cache_dir: Directory to cache API responses
            cache_expiry: Cache expiry time in seconds
            default_api: Default API to use if not specified
            rate_limit_wait: Time to wait when rate limited (seconds)
        """
        # Get API credentials from environment variables if not provided
        self.api_keys = {
            "newsapi": newsapi_key or os.environ.get("NEWS_API_KEY"),
            "alphavantage": alphavantage_key or os.environ.get("ALPHA_VANTAGE_KEY"),
            "finnhub": finnhub_key or os.environ.get("FINNHUB_KEY"),
        }
        
        # Check if any API keys are available
        if not any(self.api_keys.values()):
            logger.warning("No News API credentials provided. Will use mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
            # Log available APIs
            available_apis = [api for api, key in self.api_keys.items() if key]
            logger.info(f"Available news APIs: {', '.join(available_apis)}")
        
        # Set up caching
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        os.makedirs(cache_dir, exist_ok=True)
        
        # API parameters
        self.default_api = default_api
        self.rate_limit_wait = rate_limit_wait
        
        # Initialize sessions for each API
        self.sessions = {}
        for api, key in self.api_keys.items():
            if key:
                session = requests.Session()
                if api == "newsapi":
                    session.headers.update({"X-Api-Key": key})
                elif api == "finnhub":
                    session.headers.update({"X-Finnhub-Token": key})
                # Alpha Vantage uses key as a query parameter
                self.sessions[api] = session
    
    def fetch_sentiment_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        api: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch news sentiment data for the specified symbols and date range.
        
        Args:
            symbols: List of ticker symbols to fetch news for
            start_date: Start date for news data
            end_date: End date for news data
            api: Specific API to use (newsapi, alphavantage, finnhub)
            **kwargs: Additional parameters for the API query
            
        Returns:
            DataFrame with news sentiment data
        """
        if self.use_mock_data:
            return self._generate_mock_data(symbols, start_date, end_date)
        
        # Determine which API to use
        api = api or self.default_api
        if api not in self.api_keys or not self.api_keys[api]:
            available_apis = [a for a, k in self.api_keys.items() if k]
            if not available_apis:
                logger.warning(f"API {api} not available and no alternatives. Using mock data.")
                return self._generate_mock_data(symbols, start_date, end_date)
            api = available_apis[0]
            logger.info(f"API {api} not available. Using {api} instead.")
        
        # Fetch news for each symbol
        all_articles = []
        
        for symbol in symbols:
            # Fetch news for the symbol
            if api == "newsapi":
                articles = self._fetch_newsapi(symbol, start_date, end_date, **kwargs)
            elif api == "alphavantage":
                articles = self._fetch_alphavantage(symbol, start_date, end_date, **kwargs)
            elif api == "finnhub":
                articles = self._fetch_finnhub(symbol, start_date, end_date, **kwargs)
            else:
                logger.error(f"Unknown API: {api}")
                continue
            
            # Add symbol to each article
            for article in articles:
                article["symbol"] = symbol
            
            all_articles.extend(articles)
        
        # Convert to DataFrame
        if not all_articles:
            logger.warning(f"No news found for symbols {symbols} in date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_articles)
        
        # Process and clean the DataFrame
        df = self._process_news_dataframe(df)
        
        return df
    
    def stream_sentiment_data(
        self,
        symbols: List[str],
        callback: callable,
        **kwargs
    ) -> None:
        """
        Stream news sentiment data for the specified symbols.
        
        Args:
            symbols: List of ticker symbols to stream news for
            callback: Callback function to process streamed data
            **kwargs: Additional parameters for the API query
        """
        if self.use_mock_data:
            logger.warning("Streaming not supported with mock data")
            return
        
        # Only Finnhub supports streaming
        if "finnhub" not in self.api_keys or not self.api_keys["finnhub"]:
            logger.warning("Finnhub API key required for streaming. Using mock data.")
            return
        
        # Set up streaming parameters
        api_key = self.api_keys["finnhub"]
        
        # Import websocket if available
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client package required for streaming. Install with 'pip install websocket-client'")
            return
        
        # Define WebSocket callbacks
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if "data" in data:
                    for news in data["data"]:
                        # Add source and convert to DataFrame
                        news["source"] = "finnhub"
                        df = pd.DataFrame([news])
                        df = self._process_news_dataframe(df)
                        callback(df)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
            # Subscribe to news for each symbol
            for symbol in symbols:
                ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
        
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            f"wss://ws.finnhub.io?token={api_key}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket connection
        logger.info(f"Starting Finnhub news stream for symbols: {symbols}")
        try:
            ws.run_forever()
        except KeyboardInterrupt:
            logger.info("Finnhub news stream stopped by user")
        except Exception as e:
            logger.error(f"Error in Finnhub news stream: {e}")
    
    def _fetch_newsapi(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch news from NewsAPI.org.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            List of news articles
        """
        # Check cache first
        cache_key = self._get_cache_key("newsapi", symbol, start_date, end_date, kwargs)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Using cached data for NewsAPI query: {symbol}")
            return cached_data
        
        # Format dates for NewsAPI
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Set up parameters
        params = {
            "q": symbol,
            "from": start_date_str,
            "to": end_date_str,
            "language": "en",
            "sortBy": "publishedAt",
        }
        
        # Add additional parameters
        params.update(kwargs)
        
        # Make API request
        url = "https://newsapi.org/v2/everything"
        session = self.sessions.get("newsapi")
        
        if not session:
            logger.error("NewsAPI session not initialized")
            return []
        
        try:
            response = session.get(url, params=params)
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning(f"Rate limited by NewsAPI. Waiting {self.rate_limit_wait} seconds.")
                time.sleep(self.rate_limit_wait)
                response = session.get(url, params=params)
            
            # Handle other errors
            if response.status_code != 200:
                logger.error(f"Error from NewsAPI: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            
            # Extract articles
            articles = []
            if "articles" in data:
                for article in data["articles"]:
                    # Add source information
                    article["api_source"] = "newsapi"
                    articles.append(article)
            
            # Cache the results
            self._save_to_cache(cache_key, articles)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from NewsAPI: {e}")
            return []
    
    def _fetch_alphavantage(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch news from Alpha Vantage.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            List of news articles
        """
        # Check cache first
        cache_key = self._get_cache_key("alphavantage", symbol, start_date, end_date, kwargs)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Using cached data for Alpha Vantage query: {symbol}")
            return cached_data
        
        # Alpha Vantage doesn't support date filtering in the API, so we'll filter after
        
        # Set up parameters
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.api_keys.get("alphavantage", ""),
        }
        
        # Add additional parameters
        params.update(kwargs)
        
        # Make API request
        url = "https://www.alphavantage.co/query"
        session = self.sessions.get("alphavantage") or requests.Session()
        
        try:
            response = session.get(url, params=params)
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning(f"Rate limited by Alpha Vantage. Waiting {self.rate_limit_wait} seconds.")
                time.sleep(self.rate_limit_wait)
                response = session.get(url, params=params)
            
            # Handle other errors
            if response.status_code != 200:
                logger.error(f"Error from Alpha Vantage: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            
            # Extract articles
            articles = []
            if "feed" in data:
                for article in data["feed"]:
                    # Convert time_published to datetime
                    if "time_published" in article:
                        try:
                            published_date = datetime.strptime(
                                article["time_published"],
                                "%Y%m%dT%H%M%S"
                            )
                            # Filter by date
                            if published_date < start_date or published_date > end_date:
                                continue
                        except ValueError:
                            # If date parsing fails, include the article anyway
                            pass
                    
                    # Add source information
                    article["api_source"] = "alphavantage"
                    articles.append(article)
            
            # Cache the results
            self._save_to_cache(cache_key, articles)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from Alpha Vantage: {e}")
            return []
    
    def _fetch_finnhub(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch news from Finnhub.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            List of news articles
        """
        # Check cache first
        cache_key = self._get_cache_key("finnhub", symbol, start_date, end_date, kwargs)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Using cached data for Finnhub query: {symbol}")
            return cached_data
        
        # Format dates for Finnhub
        start_date_str = int(start_date.timestamp())
        end_date_str = int(end_date.timestamp())
        
        # Set up parameters
        params = {
            "symbol": symbol,
            "from": start_date_str,
            "to": end_date_str,
        }
        
        # Add additional parameters
        params.update(kwargs)
        
        # Make API request
        url = "https://finnhub.io/api/v1/company-news"
        session = self.sessions.get("finnhub")
        
        if not session:
            logger.error("Finnhub session not initialized")
            return []
        
        try:
            response = session.get(url, params=params)
            
            # Handle rate limiting
            if response.status_code == 429:
                logger.warning(f"Rate limited by Finnhub. Waiting {self.rate_limit_wait} seconds.")
                time.sleep(self.rate_limit_wait)
                response = session.get(url, params=params)
            
            # Handle other errors
            if response.status_code != 200:
                logger.error(f"Error from Finnhub: {response.status_code} - {response.text}")
                return []
            
            articles = response.json()
            
            # Add source information to each article
            for article in articles:
                article["api_source"] = "finnhub"
            
            # Cache the results
            self._save_to_cache(cache_key, articles)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news from Finnhub: {e}")
            return []
    
    def _process_news_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the news DataFrame.
        
        Args:
            df: DataFrame with raw news data
            
        Returns:
            Processed DataFrame
        """
        # Handle different API formats
        if "api_source" in df.columns:
            # Process based on the API source
            api_sources = df["api_source"].unique()
            
            if len(api_sources) == 1:
                api_source = api_sources[0]
                if api_source == "newsapi":
                    df = self._process_newsapi_data(df)
                elif api_source == "alphavantage":
                    df = self._process_alphavantage_data(df)
                elif api_source == "finnhub":
                    df = self._process_finnhub_data(df)
            else:
                # Multiple API sources, process each separately and concatenate
                processed_dfs = []
                for api_source in api_sources:
                    subset = df[df["api_source"] == api_source]
                    if api_source == "newsapi":
                        processed = self._process_newsapi_data(subset)
                    elif api_source == "alphavantage":
                        processed = self._process_alphavantage_data(subset)
                    elif api_source == "finnhub":
                        processed = self._process_finnhub_data(subset)
                    else:
                        continue
                    processed_dfs.append(processed)
                
                if processed_dfs:
                    df = pd.concat(processed_dfs, ignore_index=True)
                else:
                    # Create empty DataFrame with required columns
                    df = pd.DataFrame(columns=["timestamp", "content", "symbol", "source", "title", "url"])
        
        # Ensure required columns exist
        required_columns = ["timestamp", "content", "symbol", "source"]
        for col in required_columns:
            if col not in df.columns:
                if col == "timestamp":
                    df[col] = datetime.now()
                elif col == "source":
                    df[col] = "news"
                else:
                    df[col] = ""
        
        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        
        return df
    
    def _process_newsapi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process NewsAPI data.
        
        Args:
            df: DataFrame with NewsAPI data
            
        Returns:
            Processed DataFrame
        """
        # Create a new DataFrame with standardized columns
        processed = pd.DataFrame()
        
        # Map columns
        if "publishedAt" in df.columns:
            processed["timestamp"] = pd.to_datetime(df["publishedAt"])
        
        if "title" in df.columns:
            processed["title"] = df["title"]
        
        if "description" in df.columns and "content" in df.columns:
            # Use description if available, otherwise content
            processed["content"] = df["description"].fillna(df["content"])
        elif "description" in df.columns:
            processed["content"] = df["description"]
        elif "content" in df.columns:
            processed["content"] = df["content"]
        
        if "url" in df.columns:
            processed["url"] = df["url"]
        
        if "symbol" in df.columns:
            processed["symbol"] = df["symbol"]
        
        # Add source
        processed["source"] = "news_newsapi"
        
        return processed
    
    def _process_alphavantage_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Alpha Vantage data.
        
        Args:
            df: DataFrame with Alpha Vantage data
            
        Returns:
            Processed DataFrame
        """
        # Create a new DataFrame with standardized columns
        processed = pd.DataFrame()
        
        # Map columns
        if "time_published" in df.columns:
            processed["timestamp"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
        
        if "title" in df.columns:
            processed["title"] = df["title"]
        
        if "summary" in df.columns:
            processed["content"] = df["summary"]
        
        if "url" in df.columns:
            processed["url"] = df["url"]
        
        if "symbol" in df.columns:
            processed["symbol"] = df["symbol"]
        
        # Add source
        processed["source"] = "news_alphavantage"
        
        return processed
    
    def _process_finnhub_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Finnhub data.
        
        Args:
            df: DataFrame with Finnhub data
            
        Returns:
            Processed DataFrame
        """
        # Create a new DataFrame with standardized columns
        processed = pd.DataFrame()
        
        # Map columns
        if "datetime" in df.columns:
            processed["timestamp"] = pd.to_datetime(df["datetime"], unit="s")
        
        if "headline" in df.columns:
            processed["title"] = df["headline"]
        
        if "summary" in df.columns:
            processed["content"] = df["summary"]
        
        if "url" in df.columns:
            processed["url"] = df["url"]
        
        if "symbol" in df.columns:
            processed["symbol"] = df["symbol"]
        
        # Add source
        processed["source"] = "news_finnhub"
        
        return processed
    
    def _get_cache_key(self, api: str, symbol: str, start_date: datetime, end_date: datetime, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for the query.
        
        Args:
            api: API name
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            params: Additional parameters
            
        Returns:
            Cache key string
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        symbol_clean = symbol.replace("/", "_").replace("\\", "_")
        return f"{api}_{symbol_clean}_{start_str}_{end_str}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get data from cache if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is expired
        file_time = os.path.getmtime(cache_file)
        if time.time() - file_time > self.cache_expiry:
            logger.debug(f"Cache expired for key: {cache_key}")
            return None
        
        # Load cache
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: List[Dict[str, Any]]) -> None:
        """
        Save data to cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _generate_mock_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate mock news data for testing.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for mock data
            end_date: End date for mock data
            
        Returns:
            DataFrame with mock news data
        """
        logger.info("Generating mock news data")
        
        # Calculate date range
        date_range = (end_date - start_date).days
        if date_range <= 0:
            date_range = 1
        
        # Generate random data
        import numpy as np
        
        all_data = []
        
        for symbol in symbols:
            # Generate 2-5 news articles per day per symbol
            for day in range(date_range):
                current_date = start_date + timedelta(days=day)
                num_articles = np.random.randint(2, 6)
                
                for _ in range(num_articles):
                    # Random time during the day
                    hour = np.random.randint(0, 24)
                    minute = np.random.randint(0, 60)
                    second = np.random.randint(0, 60)
                    timestamp = current_date.replace(hour=hour, minute=minute, second=second)
                    
                    # Ensure timestamp is within the specified range
                    if timestamp < start_date:
                        timestamp = start_date
                    if timestamp > end_date:
                        timestamp = end_date
                    
                    # Generate article sentiment
                    sentiment = np.random.choice(["positive", "negative", "neutral"], p=[0.4, 0.3, 0.3])
                    
                    if sentiment == "positive":
                        title = np.random.choice([
                            f"{symbol} Beats Earnings Expectations, Shares Surge",
                            f"{symbol} Announces New Product Line, Analysts Bullish",
                            f"{symbol} Expands into New Markets, Growth Prospects Strong",
                            f"Analysts Upgrade {symbol} on Strong Fundamentals",
                            f"{symbol} Reports Record Quarterly Revenue"
                        ])
                        content = np.random.choice([
                            f"{symbol} reported quarterly earnings that exceeded Wall Street expectations, driven by strong product demand and effective cost management. The company also raised its full-year guidance.",
                            f"{symbol} unveiled a new product line that analysts believe will significantly boost revenue in the coming quarters. The announcement was well-received by investors.",
                            f"{symbol} announced plans to expand into new markets, which could potentially double its addressable market size. The company expects the expansion to contribute to revenue growth starting next quarter.",
                            f"Several analysts upgraded {symbol} citing strong fundamentals and attractive valuation. The consensus price target implies significant upside potential from current levels.",
                            f"{symbol} reported record quarterly revenue, surpassing the $1 billion mark for the first time in company history. Management attributed the strong performance to increased market share and pricing power."
                        ])
                    elif sentiment == "negative":
                        title = np.random.choice([
                            f"{symbol} Misses Earnings Estimates, Shares Plunge",
                            f"{symbol} Faces Regulatory Scrutiny, Legal Challenges Ahead",
                            f"{symbol} Cuts Guidance, Cites Market Headwinds",
                            f"Analysts Downgrade {symbol} on Weak Outlook",
                            f"{symbol} Announces Restructuring, Job Cuts"
                        ])
                        content = np.random.choice([
                            f"{symbol} reported quarterly earnings that fell short of analyst expectations, primarily due to supply chain disruptions and rising input costs. The company maintained its full-year guidance despite the miss.",
                            f"{symbol} is facing increased regulatory scrutiny that could result in significant fines and operational restrictions. Legal experts suggest the company may need to modify its business practices.",
                            f"{symbol} lowered its full-year guidance, citing macroeconomic headwinds and intensifying competition. Management expects these challenges to persist in the near term.",
                            f"Several analysts downgraded {symbol} following disappointing quarterly results and weak forward guidance. Concerns about margin pressure and slowing growth were highlighted in research notes.",
                            f"{symbol} announced a comprehensive restructuring plan that includes significant workforce reductions and facility closures. The company expects to incur substantial one-time charges related to the restructuring."
                        ])
                    else:  # neutral
                        title = np.random.choice([
                            f"{symbol} Reports In-Line Results, Maintains Guidance",
                            f"{symbol} Announces Management Changes",
                            f"{symbol} Completes Previously Announced Acquisition",
                            f"Industry Report Highlights Competitive Landscape for {symbol}",
                            f"{symbol} to Present at Upcoming Investor Conference"
                        ])
                        content = np.random.choice([
                            f"{symbol} reported quarterly results that were largely in line with consensus estimates. The company maintained its full-year guidance and highlighted both opportunities and challenges in the current market environment.",
                            f"{symbol} announced several key management changes, including the appointment of a new Chief Financial Officer. The transition is expected to be completed by the end of the quarter.",
                            f"{symbol} completed its previously announced acquisition of a smaller competitor. The integration process is underway, and management expects the transaction to be accretive to earnings within 12 months.",
                            f"A comprehensive industry report highlighted the competitive landscape for {symbol} and its peers. The report noted that while competition is intensifying, the overall market is expanding, creating opportunities for well-positioned companies.",
                            f"{symbol} will be presenting at several upcoming investor conferences. Management is expected to provide updates on the company's strategic initiatives and market outlook."
                        ])
                    
                    # Create article object
                    article = {
                        "timestamp": timestamp,
                        "title": title,
                        "content": content,
                        "symbol": symbol,
                        "source": "news_mock",
                        "url": f"https://example.com/news/{symbol.lower()}/{timestamp.strftime('%Y%m%d%H%M%S')}",
                    }
                    
                    all_data.append(article)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        return df
