"""
Fear & Greed Index Collector for sentiment analysis.

This module provides functionality to collect CNN's Fear & Greed Index data
for sentiment analysis in the AI Trading Agent.
"""

import os
import json
import time
import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import requests
import pandas as pd
from bs4 import BeautifulSoup

from ...common.logging_config import setup_logging
from ..providers.base_provider import BaseSentimentProvider

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class FearGreedIndexCollector(BaseSentimentProvider):
    """
    Collects Fear & Greed Index data for sentiment analysis.
    
    The Fear & Greed Index is a sentiment indicator created by CNN Money that
    measures two of the primary emotions that influence investors: fear and greed.
    
    This collector scrapes the CNN Money Fear & Greed Index page and also provides
    historical data through alternative sources.
    
    Attributes:
        cache_dir (str): Directory to cache scraped data
        cache_expiry (int): Cache expiry time in seconds
        historical_data_source (str): Source for historical data
    """
    
    def __init__(
        self,
        cache_dir: str = "cache/fear_greed",
        cache_expiry: int = 3600,
        historical_data_source: str = "alternative",
        rate_limit_wait: int = 15,
    ):
        """
        Initialize the Fear & Greed Index collector.
        
        Args:
            cache_dir: Directory to cache scraped data
            cache_expiry: Cache expiry time in seconds
            historical_data_source: Source for historical data ('alternative' or 'cnn')
            rate_limit_wait: Time to wait when rate limited (seconds)
        """
        # Set up caching
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        os.makedirs(cache_dir, exist_ok=True)
        
        # API parameters
        self.historical_data_source = historical_data_source
        self.rate_limit_wait = rate_limit_wait
        
        # CNN Fear & Greed Index URL
        self.cnn_url = "https://money.cnn.com/data/fear-and-greed/"
        
        # Alternative API for historical data
        self.alternative_api_url = "https://api.alternative.me/fng/"
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        
        # Mock data flag
        self.use_mock_data = False
    
    def fetch_sentiment_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch Fear & Greed Index data for the specified date range.
        
        Note: The Fear & Greed Index is a market-wide indicator and not specific to
        individual symbols. However, to maintain compatibility with the BaseSentimentProvider
        interface, this method accepts a list of symbols but returns the same index data
        for all symbols.
        
        Args:
            symbols: List of ticker symbols (not used for this collector)
            start_date: Start date for index data
            end_date: End date for index data
            **kwargs: Additional parameters for the data fetch
            
        Returns:
            DataFrame with Fear & Greed Index data
        """
        if self.use_mock_data:
            return self._generate_mock_data(symbols, start_date, end_date)
        
        # Check cache first
        cache_key = self._get_cache_key(start_date, end_date)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            logger.debug(f"Using cached Fear & Greed Index data")
            df = pd.DataFrame(cached_data)
            
            # Ensure timestamp is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            return df
        
        # Fetch historical data
        if self.historical_data_source == "alternative":
            data = self._fetch_alternative_historical_data(start_date, end_date)
        else:
            data = self._fetch_cnn_historical_data(start_date, end_date)
        
        # If no historical data, try to get current data
        if not data:
            current_data = self._fetch_current_fear_greed_index()
            if current_data:
                data = [current_data]
        
        # If still no data, use mock data
        if not data:
            logger.warning("Failed to fetch Fear & Greed Index data. Using mock data.")
            return self._generate_mock_data(symbols, start_date, end_date)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Add symbol and source columns to match BaseSentimentProvider interface
        df["source"] = "fear_greed_index"
        
        # Create duplicate rows for each symbol
        all_data = []
        for symbol in symbols:
            symbol_df = df.copy()
            symbol_df["symbol"] = symbol
            all_data.append(symbol_df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp")
        
        # Cache the data
        self._save_to_cache(cache_key, data)
        
        return combined_df
    
    def stream_sentiment_data(
        self,
        symbols: List[str],
        callback: callable,
        **kwargs
    ) -> None:
        """
        Stream Fear & Greed Index data.
        
        Since the Fear & Greed Index is updated only once per day, this method
        fetches the current index and then schedules periodic updates.
        
        Args:
            symbols: List of ticker symbols (not used for this collector)
            callback: Callback function to process streamed data
            **kwargs: Additional parameters for the data stream
        """
        import threading
        
        def fetch_and_callback():
            try:
                # Fetch current Fear & Greed Index
                data = self._fetch_current_fear_greed_index()
                
                if data:
                    # Convert to DataFrame
                    df = pd.DataFrame([data])
                    
                    # Add symbol and source columns
                    df["source"] = "fear_greed_index"
                    
                    # Create duplicate rows for each symbol
                    all_data = []
                    for symbol in symbols:
                        symbol_df = df.copy()
                        symbol_df["symbol"] = symbol
                        all_data.append(symbol_df)
                    
                    # Combine all data
                    combined_df = pd.concat(all_data, ignore_index=True)
                    
                    # Call the callback
                    callback(combined_df)
                
                # Schedule next update (every 4 hours)
                threading.Timer(4 * 60 * 60, fetch_and_callback).start()
            except Exception as e:
                logger.error(f"Error in Fear & Greed Index stream: {e}")
        
        # Start the first fetch
        fetch_and_callback()
    
    def _fetch_current_fear_greed_index(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the current Fear & Greed Index from CNN Money.
        
        Returns:
            Dictionary with current Fear & Greed Index data or None if failed
        """
        try:
            # Fetch CNN Fear & Greed Index page
            response = self.session.get(self.cnn_url)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch CNN Fear & Greed Index page: {response.status_code}")
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract current index value
            index_value_elem = soup.select_one(".feargreed-dial")
            if not index_value_elem:
                logger.error("Could not find Fear & Greed Index value element")
                return None
            
            # Extract index value using regex
            index_match = re.search(r'data-current-value="(\d+)"', str(index_value_elem))
            if not index_match:
                logger.error("Could not extract Fear & Greed Index value")
                return None
            
            index_value = int(index_match.group(1))
            
            # Extract index classification
            classification = self._get_classification_from_value(index_value)
            
            # Extract index components if available
            components = self._extract_index_components(soup)
            
            # Create result
            result = {
                "timestamp": datetime.now(),
                "value": index_value,
                "classification": classification,
                "content": f"Fear & Greed Index: {index_value} ({classification})",
                "components": components
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching current Fear & Greed Index: {e}")
            return None
    
    def _extract_index_components(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract Fear & Greed Index components from CNN page.
        
        Args:
            soup: BeautifulSoup object of the CNN Fear & Greed Index page
            
        Returns:
            Dictionary with component data
        """
        components = {}
        
        try:
            # Extract component data
            component_elems = soup.select(".feargreed-components .feargreed-component")
            
            for elem in component_elems:
                name_elem = elem.select_one(".feargreed-component-name")
                value_elem = elem.select_one(".feargreed-component-value")
                
                if name_elem and value_elem:
                    name = name_elem.text.strip()
                    value_text = value_elem.text.strip()
                    
                    # Extract classification
                    classification_match = re.search(r'([A-Za-z\s]+)$', value_text)
                    if classification_match:
                        classification = classification_match.group(1).strip()
                    else:
                        classification = "Unknown"
                    
                    components[name] = {
                        "classification": classification
                    }
        except Exception as e:
            logger.error(f"Error extracting Fear & Greed Index components: {e}")
        
        return components
    
    def _fetch_alternative_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical Fear & Greed Index data from alternative.me API.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dictionaries with historical Fear & Greed Index data
        """
        try:
            # Calculate days difference
            days_diff = (end_date - start_date).days
            
            # API only allows up to 365 days of historical data
            if days_diff > 365:
                logger.warning("Alternative.me API only provides up to 365 days of historical data. Limiting to 365 days.")
                days_diff = 365
            
            # Set up parameters
            params = {
                "limit": days_diff + 1,  # +1 to include end date
                "format": "json",
            }
            
            # Make API request
            response = self.session.get(self.alternative_api_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch historical Fear & Greed Index data: {response.status_code}")
                return []
            
            data = response.json()
            
            # Extract data
            results = []
            if "data" in data:
                for item in data["data"]:
                    # Convert timestamp to datetime
                    timestamp = datetime.fromtimestamp(int(item["timestamp"]))
                    
                    # Skip if outside date range
                    if timestamp < start_date or timestamp > end_date:
                        continue
                    
                    # Extract values
                    value = int(item["value"])
                    classification = item["value_classification"]
                    
                    # Create result
                    result = {
                        "timestamp": timestamp,
                        "value": value,
                        "classification": classification,
                        "content": f"Fear & Greed Index: {value} ({classification})"
                    }
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching historical Fear & Greed Index data: {e}")
            return []
    
    def _fetch_cnn_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical Fear & Greed Index data from CNN.
        
        Note: This method attempts to extract historical data from CNN's JavaScript,
        but it may not be reliable as CNN doesn't provide an official API for this data.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of dictionaries with historical Fear & Greed Index data
        """
        try:
            # Fetch CNN Fear & Greed Index page
            response = self.session.get(self.cnn_url)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch CNN Fear & Greed Index page: {response.status_code}")
                return []
            
            # Extract historical data from JavaScript
            # This is a bit hacky and may break if CNN changes their page structure
            history_match = re.search(r'var fearAndGreedHistoricalData = (\[.*?\]);', response.text, re.DOTALL)
            
            if not history_match:
                logger.error("Could not extract historical Fear & Greed Index data")
                return []
            
            # Parse JSON
            try:
                history_json = history_match.group(1)
                history_data = json.loads(history_json)
            except json.JSONDecodeError:
                logger.error("Failed to parse historical Fear & Greed Index data")
                return []
            
            # Extract data
            results = []
            for item in history_data:
                # Convert date string to datetime
                try:
                    timestamp = datetime.strptime(item["x"], "%Y-%m-%d")
                except ValueError:
                    continue
                
                # Skip if outside date range
                if timestamp < start_date or timestamp > end_date:
                    continue
                
                # Extract values
                value = int(item["y"])
                classification = self._get_classification_from_value(value)
                
                # Create result
                result = {
                    "timestamp": timestamp,
                    "value": value,
                    "classification": classification,
                    "content": f"Fear & Greed Index: {value} ({classification})"
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching CNN historical Fear & Greed Index data: {e}")
            return []
    
    def _get_classification_from_value(self, value: int) -> str:
        """
        Get the classification for a Fear & Greed Index value.
        
        Args:
            value: Fear & Greed Index value (0-100)
            
        Returns:
            Classification string
        """
        if value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"
    
    def _get_cache_key(self, start_date: datetime, end_date: datetime) -> str:
        """
        Generate a cache key for the query.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache key string
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        source = self.historical_data_source
        return f"fear_greed_{source}_{start_str}_{end_str}"
    
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
            # Convert datetime objects to strings
            serializable_data = []
            for item in data:
                serializable_item = item.copy()
                if "timestamp" in serializable_item and isinstance(serializable_item["timestamp"], datetime):
                    serializable_item["timestamp"] = serializable_item["timestamp"].isoformat()
                serializable_data.append(serializable_item)
            
            with open(cache_file, "w") as f:
                json.dump(serializable_data, f)
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def _generate_mock_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate mock Fear & Greed Index data for testing.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for mock data
            end_date: End date for mock data
            
        Returns:
            DataFrame with mock Fear & Greed Index data
        """
        logger.info("Generating mock Fear & Greed Index data")
        
        # Calculate date range
        date_range = (end_date - start_date).days
        if date_range <= 0:
            date_range = 1
        
        # Generate random data
        import numpy as np
        
        all_data = []
        
        # Generate one data point per day
        for day in range(date_range + 1):  # +1 to include end date
            current_date = start_date + timedelta(days=day)
            
            # Generate random value (0-100)
            value = np.random.randint(0, 101)
            
            # Get classification
            classification = self._get_classification_from_value(value)
            
            # Create data point
            data_point = {
                "timestamp": current_date,
                "value": value,
                "classification": classification,
                "content": f"Fear & Greed Index: {value} ({classification})",
                "source": "fear_greed_index"
            }
            
            # Add data point for each symbol
            for symbol in symbols:
                symbol_data = data_point.copy()
                symbol_data["symbol"] = symbol
                all_data.append(symbol_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        return df
    
    def normalize_index_value(self, value: int) -> float:
        """
        Normalize Fear & Greed Index value to a range of -1 to 1.
        
        This is useful for integrating with other sentiment data sources.
        
        Args:
            value: Fear & Greed Index value (0-100)
            
        Returns:
            Normalized value (-1 to 1)
        """
        # Convert 0-100 scale to -1 to 1 scale
        # 0 = Extreme Fear = -1
        # 50 = Neutral = 0
        # 100 = Extreme Greed = 1
        return (value / 50) - 1
