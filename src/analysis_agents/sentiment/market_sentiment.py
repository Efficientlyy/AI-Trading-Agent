"""Market sentiment analysis.

This module provides functionality for analyzing sentiment from market indicators
such as the Fear & Greed Index, Long/Short ratio, and other market sentiment metrics.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.analysis_agents.sentiment.nlp_service import NLPService
from src.common.config import config
from src.common.logging import get_logger
from src.models.market_data import CandleData, TimeFrame


class FearGreedClient:
    """Client for fetching the Crypto Fear & Greed Index from Alternative.me API.
    
    The Fear & Greed Index represents the market sentiment on a scale from
    0 (Extreme Fear) to 100 (Extreme Greed) for the cryptocurrency market.
    It integrates multiple data points including price volatility, market momentum,
    social media sentiment, and trading volume.
    """
    
    def __init__(self):
        """Initialize the Fear & Greed client.
        
        Loads configuration, sets up cache for responses to minimize API calls,
        and prepares HTTP session for API requests.
        """
        from src.common.config import config
        import aiohttp
        from datetime import timedelta
        
        self.logger = get_logger("clients", "fear_greed")
        
        # Load configuration
        self.base_url = config.get("sentiment.apis.fear_greed.base_url", 
                                   "https://api.alternative.me/fng/")
        self.cache_expiry = config.get("sentiment.apis.fear_greed.cache_expiry", 3600)
        
        # Cache setup
        self.cache = {}
        self.last_cache_update = datetime.min
        self.cache_validity = timedelta(seconds=self.cache_expiry)
        
        # Session for API requests
        self._session = None
        
        self.logger.info("Fear & Greed client initialized", base_url=self.base_url)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session.
        
        Returns:
            An aiohttp client session for making API requests
        """
        import aiohttp
        
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
        
    async def _fetch_fear_greed_data(self, limit: int = 1) -> Dict[str, Any]:
        """Fetch Fear & Greed Index data from API.
        
        Args:
            limit: Number of days of data to fetch (default: 1 for current only)
            
        Returns:
            Dictionary containing the API response with fear & greed data
            
        Raises:
            Exception: If API request fails or returns invalid data
        """
        import aiohttp
        import json
        
        # Check cache for historical data if requested limit is larger
        current_time = datetime.utcnow()
        cache_key = f"fear_greed_{limit}"
        
        if (cache_key in self.cache and 
            (current_time - self.last_cache_update) < self.cache_validity):
            self.logger.debug("Using cached Fear & Greed data", limit=limit)
            return self.cache[cache_key]
        
        # Construct URL with limit parameter
        url = f"{self.base_url}?limit={limit}"
        
        try:
            session = await self._get_session()
            
            # Make request to API
            self.logger.debug("Fetching Fear & Greed data from API", url=url)
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.error("Failed to fetch Fear & Greed data", 
                                    status=response.status)
                    raise Exception(f"API error: {response.status}")
                
                # Parse JSON response
                data = await response.json()
                
                # Validate response structure
                if not data or 'data' not in data:
                    self.logger.error("Invalid Fear & Greed API response", response=str(data))
                    raise Exception("Invalid API response")
                
                # Store in cache
                self.cache[cache_key] = data
                self.last_cache_update = current_time
                
                return data
                
        except aiohttp.ClientError as e:
            self.logger.error("HTTP error fetching Fear & Greed data", error=str(e))
            raise Exception(f"HTTP error: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON response from Fear & Greed API", error=str(e))
            raise Exception(f"Invalid JSON: {str(e)}")
        except Exception as e:
            self.logger.error("Error fetching Fear & Greed data", error=str(e))
            raise
    
    def _classify_index_value(self, value: int) -> str:
        """Classify a Fear & Greed Index value.
        
        Args:
            value: The numerical index value (0-100)
            
        Returns:
            Classification as a string: "extreme fear", "fear", 
            "neutral", "greed", or "extreme greed"
        """
        if value <= 25:
            return "extreme fear"
        elif value <= 40:
            return "fear"
        elif value <= 60:
            return "neutral"
        elif value <= 80:
            return "greed"
        else:
            return "extreme greed"
    
    async def get_current_index(self) -> Dict[str, Any]:
        """Get the current Fear & Greed Index.
        
        Returns:
            Dictionary with fear & greed data including:
            - value: The current index value (0-100)
            - classification: The classification 
            - timestamp: When the index was generated
            
        Raises:
            Exception: If API request fails or returns invalid data
        """
        try:
            self.logger.debug("Fetching current Fear & Greed Index")
            
            # Fetch current data (limit=1)
            response = await self._fetch_fear_greed_data(limit=1)
            
            if not response['data'] or len(response['data']) == 0:
                raise Exception("No Fear & Greed data returned from API")
            
            # Get the most recent entry
            latest = response['data'][0]
            
            # Extract value
            try:
                value = int(latest['value'])
            except (KeyError, ValueError):
                self.logger.error("Invalid value in Fear & Greed response", data=latest)
                raise Exception("Invalid 'value' in Fear & Greed response")
            
            # Get timestamp (convert from UNIX time if present, otherwise use current time)
            try:
                if 'timestamp' in latest:
                    timestamp = datetime.fromtimestamp(int(latest['timestamp']))
                else:
                    timestamp = datetime.utcnow()
            except (KeyError, ValueError):
                timestamp = datetime.utcnow()
                
            # Determine classification (either from API or calculate it)
            if 'value_classification' in latest:
                classification = latest['value_classification'].lower()
            else:
                classification = self._classify_index_value(value)
            
            return {
                "value": value,
                "classification": classification,
                "timestamp": timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error getting current Fear & Greed Index", error=str(e))
            
            # Return a fallback or re-raise depending on severity
            # For now, we'll return a neutral value with low confidence
            return {
                "value": 50,
                "classification": "neutral",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "is_fallback": True
            }
    
    async def get_historical_index(
        self, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical Fear & Greed Index data.
        
        Args:
            days: Number of days of historical data to retrieve
            
        Returns:
            List of daily index values sorted by date (newest first)
            
        Raises:
            Exception: If API request fails or returns invalid data
        """
        try:
            self.logger.debug("Fetching historical Fear & Greed data", days=days)
            
            # Fetch historical data
            response = await self._fetch_fear_greed_data(limit=days)
            
            if not response['data']:
                raise Exception("No historical Fear & Greed data returned from API")
            
            # Process each data point
            historical_data = []
            for entry in response['data']:
                try:
                    # Extract value
                    value = int(entry['value'])
                    
                    # Get timestamp (convert from UNIX time if present)
                    if 'timestamp' in entry:
                        timestamp = datetime.fromtimestamp(int(entry['timestamp']))
                    else:
                        # If no timestamp, we have to approximate based on the index
                        # Assuming entries are in reverse chronological order
                        index = response['data'].index(entry)
                        timestamp = datetime.utcnow() - timedelta(days=index)
                        
                    # Determine classification
                    if 'value_classification' in entry:
                        classification = entry['value_classification'].lower()
                    else:
                        classification = self._classify_index_value(value)
                    
                    historical_data.append({
                        "value": value,
                        "classification": classification,
                        "timestamp": timestamp.isoformat()
                    })
                    
                except (KeyError, ValueError) as e:
                    self.logger.warning("Skipping invalid historical Fear & Greed entry", 
                                      entry=str(entry), error=str(e))
            
            return historical_data
            
        except Exception as e:
            self.logger.error("Error getting historical Fear & Greed data", 
                            days=days, error=str(e))
            
            # Return a fallback with a warning
            now = datetime.utcnow()
            fallback_data = []
            
            # Generate neutral historical data as a fallback
            for i in range(days):
                date = now - timedelta(days=days-i-1)
                fallback_data.append({
                    "value": 50,
                    "classification": "neutral",
                    "timestamp": date.isoformat(),
                    "is_fallback": True
                })
            
            return fallback_data
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


class ExchangeDataClient:
    """Client for fetching exchange data like Long/Short ratio."""
    
    def __init__(self, api_key: str):
        """Initialize the Exchange Data client.
        
        Args:
            api_key: API key for the exchange
        """
        self.api_key = api_key
        self.logger = get_logger("clients", "exchange_data")
        
        # In production, this would be a real API client
        # For now, we'll use a mock implementation
    
    async def get_long_short_ratio(self, symbol: str) -> Dict[str, Any]:
        """Get the long/short ratio for a trading pair.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dictionary with long/short data
        """
        # In production, this would call the Exchange API
        # For now, return mock data
        self.logger.debug("Fetching long/short ratio", symbol=symbol)
        
        # Simulate a delay
        await asyncio.sleep(0.1)
        
        # Generate random long/short ratio
        # Values > 1 indicate more longs than shorts
        # Values < 1 indicate more shorts than longs
        base_currency = symbol.split('/')[0]
        
        # Bias slightly based on currency (just for demo variation)
        ratio_bias = {
            "BTC": 0.1,    # Slightly more longs for Bitcoin
            "ETH": 0.05,   # Slightly more longs for Ethereum
            "XRP": -0.1,   # Slightly more shorts for XRP
            "SOL": 0.15,   # More longs for Solana
        }.get(base_currency, 0)
        
        # Generate long/short ratio (centered around 1.0 with the bias)
        long_short_ratio = max(0.3, min(3.0, random.normalvariate(1.0 + ratio_bias, 0.3)))
        
        return {
            "symbol": symbol,
            "longShortRatio": long_short_ratio,
            "longPosition": long_short_ratio / (1 + long_short_ratio),  # As percentage of total
            "shortPosition": 1 / (1 + long_short_ratio),  # As percentage of total
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get the funding rate for a perpetual futures contract.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dictionary with funding rate data
        """
        # In production, this would call the Exchange API
        # For now, return mock data
        self.logger.debug("Fetching funding rate", symbol=symbol)
        
        # Simulate a delay
        await asyncio.sleep(0.1)
        
        # Generate random funding rate
        # Positive values indicate longs pay shorts
        # Negative values indicate shorts pay longs
        funding_rate = random.normalvariate(0.0001, 0.0005)  # Typically small values
        
        return {
            "symbol": symbol,
            "fundingRate": funding_rate,
            "annualizedRate": funding_rate * 3 * 365,  # 3 funding periods per day
            "nextFundingTime": (datetime.utcnow() + datetime.timedelta(hours=8)).isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }


class MarketSentimentAgent(BaseSentimentAgent):
    """Analysis agent for market sentiment indicators.
    
    This agent processes sentiment data from market indicators
    and publishes sentiment events with confidence scores.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the market sentiment agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "market_sentiment")
        
        # Market indicators to monitor
        self.indicators = config.get(
            f"analysis_agents.{agent_id}.indicators", 
            ["FearGreedIndex", "LongShortRatio"]
        )
        
        # Update interval in seconds
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            3600  # Default: 1 hour
        )
        
        # API clients (will be initialized during _initialize)
        self.fear_greed_client = None
        self.exchange_data_client = None
        
        # NLP service (will be set by manager)
        self.nlp_service = None
    
    async def _initialize(self) -> None:
        """Initialize the market sentiment agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing market sentiment agent",
                       indicators=self.indicators)
                       
        # Initialize API clients
        try:
            # Fear & Greed client
            self.fear_greed_client = FearGreedClient()
            
            # Exchange data client
            self.exchange_data_client = ExchangeDataClient(
                api_key=config.get("apis.exchange_data.api_key", "")
            )
            
            self.logger.info("Initialized market sentiment API clients")
            
        except Exception as e:
            self.logger.error("Failed to initialize market sentiment API clients", error=str(e))
    
    def set_nlp_service(self, nlp_service: NLPService) -> None:
        """Set the NLP service for sentiment analysis.
        
        Args:
            nlp_service: The NLP service to use
        """
        self.nlp_service = nlp_service
        self.logger.info("NLP service set for market sentiment agent")
    
    async def _start(self) -> None:
        """Start the market sentiment agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for market sentiment
        self.update_task = self.create_task(
            self._update_sentiment_periodically()
        )
    
    async def _stop(self) -> None:
        """Stop the market sentiment agent."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Close Fear & Greed client if it exists
        if hasattr(self, "fear_greed_client") and self.fear_greed_client:
            try:
                await self.fear_greed_client.close()
                self.logger.debug("Fear & Greed client closed")
            except Exception as e:
                self.logger.warning("Error closing Fear & Greed client", error=str(e))
        
        await super()._stop()
    
    async def _update_sentiment_periodically(self) -> None:
        """Update market sentiment periodically."""
        try:
            while True:
                for symbol in self.symbols if self.symbols else ["BTC/USDT", "ETH/USDT"]:
                    await self._analyze_market_sentiment_indicators(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Market sentiment update task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in market sentiment update task", error=str(e))
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _analyze_market_sentiment_indicators(self, symbol: str) -> None:
        """Analyze market sentiment indicators for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
        """
        # Check if we need to update (respect the interval)
        if not self._should_update_sentiment(symbol, "market_sentiment", self.update_interval):
            return
        
        try:
            base_currency = symbol.split('/')[0]
            
            # Fetch Fear & Greed Index
            fear_greed_data = await self.fear_greed_client.get_current_index()
            fear_greed = fear_greed_data.get("value", 50)
            
            # Fetch long/short ratio from exchanges
            long_short_data = await self.exchange_data_client.get_long_short_ratio(
                symbol=symbol
            )
            long_short_ratio = long_short_data.get("longShortRatio", 1.0)
            
            # Calculate overall market sentiment (0-1)
            # Fear & Greed: 0=extreme fear, 100=extreme greed
            # Convert to 0-1 scale
            fg_sentiment = fear_greed / 100.0
            
            # Long/Short ratio: <1 means more shorts, >1 means more longs
            # Convert to 0-1 scale with 0.5 at ratio=1
            if long_short_ratio < 1:
                ls_sentiment = 0.5 * long_short_ratio
            else:
                ls_sentiment = 0.5 + 0.5 * min(1.0, (long_short_ratio - 1) / 2)
            
            # Combine both indicators (equal weight)
            sentiment_value = (fg_sentiment + ls_sentiment) / 2
            
            # Determine confidence based on the agreement between indicators
            indicator_agreement = 1.0 - abs(fg_sentiment - ls_sentiment)
            confidence = 0.7 + (indicator_agreement * 0.25)  # 0.7 to 0.95 range
            
            # Determine direction
            if sentiment_value > 0.6:  # Higher threshold for market indicators
                direction = "bullish"
            elif sentiment_value < 0.4:  # Lower threshold for market indicators
                direction = "bearish"
            else:
                direction = "neutral"
            
            # Store additional metadata
            additional_data = {
                "fear_greed_index": fear_greed,
                "fear_greed_classification": fear_greed_data.get("classification", ""),
                "long_short_ratio": long_short_ratio,
                "indicators": self.indicators
            }
            
            # Update the sentiment cache
            sentiment_shift = self._update_sentiment_cache(
                symbol=symbol,
                source_type="market_sentiment",
                sentiment_value=sentiment_value,
                direction=direction,
                confidence=confidence,
                additional_data=additional_data
            )
            
            # Check for extreme values
            is_extreme = fear_greed <= 20 or fear_greed >= 80
            
            # Publish event if significant shift or extreme values
            if sentiment_shift > self.sentiment_shift_threshold or is_extreme:
                # Determine if extreme sentiment should be treated as contrarian
                signal_type = "sentiment"
                if is_extreme:
                    # Extreme fear/greed can be contrarian
                    signal_type = "contrarian"
                
                # Publish event if confidence is high enough
                if confidence >= self.min_confidence:
                    await self.publish_sentiment_event(
                        symbol=symbol,
                        direction=direction,
                        value=sentiment_value,
                        confidence=confidence,
                        is_extreme=is_extreme,
                        signal_type=signal_type,
                        sources=self.indicators,
                        details={
                            "fear_greed_index": fear_greed,
                            "fear_greed_classification": fear_greed_data.get("classification", ""),
                            "long_short_ratio": long_short_ratio,
                            "event_type": "market_sentiment_shift" if sentiment_shift > self.sentiment_shift_threshold else "extreme_market_sentiment"
                        }
                    )
        
        except Exception as e:
            self.logger.error("Error analyzing market sentiment indicators", 
                           symbol=symbol,
                           error=str(e))
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        candles: List[CandleData]
    ) -> None:
        """Analyze market data in relation to market sentiment indicators.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            candles: The candles to analyze
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        if not candles or len(candles) < 10:
            return
            
        # Check if we have market sentiment data for this symbol
        if symbol not in self.sentiment_cache or "market_sentiment" not in self.sentiment_cache[symbol]:
            return
            
        # Get the latest market sentiment
        sentiment_data = self.sentiment_cache[symbol]["market_sentiment"]
        fear_greed = sentiment_data.get("fear_greed_index", 50)
        sentiment_value = sentiment_data.get("value", 0.5)
        
        # Calculate recent volatility
        if len(candles) >= 20:
            # Calculate volatility as standard deviation of returns
            closes = [candle.close for candle in candles[-20:]]
            returns = [(closes[i] / closes[i-1]) - 1 for i in range(1, len(closes))]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
            
            # Check for extreme fear and high volatility
            if fear_greed < 25 and volatility > 0.03:  # 3% daily volatility is high
                # This is often a contrarian buy signal
                await self.publish_sentiment_event(
                    symbol=symbol,
                    direction="bullish",  # Contrarian to extreme fear
                    value=0.7,  # Moderately bullish
                    confidence=0.8,
                    timeframe=timeframe,
                    is_extreme=True,
                    signal_type="contrarian",
                    sources=self.indicators,
                    details={
                        "fear_greed_index": fear_greed,
                        "volatility": volatility,
                        "event_type": "volatility_fear_contrarian"
                    }
                )