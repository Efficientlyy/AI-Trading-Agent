"""
Data Services Layer

This module provides data services for the dashboard, including mock and real data sources.
"""

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_services")

class DataService(ABC):
    """
    Abstract base class for data services.
    """
    
    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: The symbol to get data for
            timeframe: The timeframe to get data for
            limit: The maximum number of data points to return
            
        Returns:
            Dict containing the market data
        """
        pass
    
    @abstractmethod
    async def get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment data for a symbol.
        
        Args:
            symbol: The symbol to get data for
            
        Returns:
            Dict containing the sentiment data
        """
        pass
    
    @abstractmethod
    async def get_performance_data(self) -> Dict[str, Any]:
        """
        Get performance data.
        
        Returns:
            Dict containing the performance data
        """
        pass
    
    @abstractmethod
    async def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get alerts.
        
        Returns:
            List of alerts
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get positions.
        
        Returns:
            List of positions
        """
        pass
    
    @abstractmethod
    async def get_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get trades.
        
        Args:
            limit: The maximum number of trades to return
            
        Returns:
            List of trades
        """
        pass

class MockDataService(DataService):
    """
    Mock data service for development and testing.
    """
    
    def __init__(self):
        """
        Initialize the mock data service.
        """
        # Supported symbols
        self.symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'XRP/USD']
        
        # Supported timeframes
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        # Mock data cache
        self.market_data_cache = {}
        self.sentiment_data_cache = {}
        self.performance_data = self._generate_performance_data()
        self.alerts = self._generate_alerts()
        self.positions = self._generate_positions()
        self.trades = self._generate_trades(50)
        
        logger.info("Mock data service initialized")
    
    async def get_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get mock market data for a symbol.
        
        Args:
            symbol: The symbol to get data for
            timeframe: The timeframe to get data for
            limit: The maximum number of data points to return
            
        Returns:
            Dict containing the market data
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Unsupported symbol: {symbol}")
            return {'error': f"Unsupported symbol: {symbol}"}
        
        # Check if timeframe is supported
        if timeframe not in self.timeframes:
            logger.warning(f"Unsupported timeframe: {timeframe}")
            return {'error': f"Unsupported timeframe: {timeframe}"}
        
        # Check if data is in cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.market_data_cache:
            # Return cached data
            return self.market_data_cache[cache_key]
        
        # Generate mock data
        data = self._generate_market_data(symbol, timeframe, limit)
        
        # Cache data
        self.market_data_cache[cache_key] = data
        
        return data
    
    async def get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get mock sentiment data for a symbol.
        
        Args:
            symbol: The symbol to get data for
            
        Returns:
            Dict containing the sentiment data
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Unsupported symbol: {symbol}")
            return {'error': f"Unsupported symbol: {symbol}"}
        
        # Check if data is in cache
        if symbol in self.sentiment_data_cache:
            # Return cached data
            return self.sentiment_data_cache[symbol]
        
        # Generate mock data
        data = self._generate_sentiment_data(symbol)
        
        # Cache data
        self.sentiment_data_cache[symbol] = data
        
        return data
    
    async def get_performance_data(self) -> Dict[str, Any]:
        """
        Get mock performance data.
        
        Returns:
            Dict containing the performance data
        """
        # Update performance data occasionally
        if random.random() < 0.2:
            self.performance_data = self._generate_performance_data()
        
        return self.performance_data
    
    async def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get mock alerts.
        
        Returns:
            List of alerts
        """
        # Update alerts occasionally
        if random.random() < 0.1:
            # Add a new alert
            self.alerts.insert(0, self._generate_alert())
            
            # Limit number of alerts
            if len(self.alerts) > 20:
                self.alerts = self.alerts[:20]
        
        return self.alerts
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get mock positions.
        
        Returns:
            List of positions
        """
        # Update positions occasionally
        if random.random() < 0.05:
            self.positions = self._generate_positions()
        
        return self.positions
    
    async def get_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get mock trades.
        
        Args:
            limit: The maximum number of trades to return
            
        Returns:
            List of trades
        """
        # Update trades occasionally
        if random.random() < 0.1:
            # Add a new trade
            self.trades.insert(0, self._generate_trade())
            
            # Limit number of trades
            if len(self.trades) > 50:
                self.trades = self.trades[:50]
        
        # Return limited number of trades
        return self.trades[:limit]
    
    def _generate_market_data(self, symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
        """
        Generate mock market data.
        
        Args:
            symbol: The symbol to generate data for
            timeframe: The timeframe to generate data for
            limit: The number of data points to generate
            
        Returns:
            Dict containing the market data
        """
        # Set base price based on symbol
        base_price = {
            'BTC/USD': 50000,
            'ETH/USD': 3000,
            'SOL/USD': 150,
            'ADA/USD': 1.2,
            'XRP/USD': 0.8
        }.get(symbol, 100)
        
        # Set volatility based on timeframe
        volatility = {
            '1m': 0.001,
            '5m': 0.002,
            '15m': 0.003,
            '1h': 0.005,
            '4h': 0.01,
            '1d': 0.02
        }.get(timeframe, 0.005)
        
        # Set time increment based on timeframe
        time_increment = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }.get(timeframe, timedelta(hours=1))
        
        # Generate data
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        # Start time
        end_time = datetime.now()
        current_time = end_time - time_increment * limit
        
        # Current price
        current_price = base_price
        
        for _ in range(limit):
            # Generate price movement
            price_change = current_price * volatility * (random.random() * 2 - 1)
            open_price = current_price
            close_price = current_price + price_change
            high_price = max(open_price, close_price) * (1 + random.random() * volatility)
            low_price = min(open_price, close_price) * (1 - random.random() * volatility)
            volume = base_price * 10 * (0.5 + random.random())
            
            # Add data point
            timestamps.append(int(current_time.timestamp() * 1000))
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            # Update current price and time
            current_price = close_price
            current_time += time_increment
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamps': timestamps,
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'volumes': volumes
        }
    
    def _generate_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Generate mock sentiment data.
        
        Args:
            symbol: The symbol to generate data for
            
        Returns:
            Dict containing the sentiment data
        """
        # Generate sentiment scores
        social_media_sentiment = random.uniform(-1, 1)
        news_sentiment = random.uniform(-1, 1)
        market_sentiment = random.uniform(-1, 1)
        
        # Generate sentiment trends
        timestamps = []
        social_media_trends = []
        news_trends = []
        market_trends = []
        
        # Start time
        end_time = datetime.now()
        current_time = end_time - timedelta(days=7)
        
        # Current sentiment values
        current_social = social_media_sentiment * 0.5
        current_news = news_sentiment * 0.5
        current_market = market_sentiment * 0.5
        
        # Generate hourly data for a week
        for _ in range(24 * 7):
            # Add data point
            timestamps.append(int(current_time.timestamp() * 1000))
            
            # Update sentiment with random walk
            current_social += random.uniform(-0.05, 0.05)
            current_social = max(-1, min(1, current_social))
            
            current_news += random.uniform(-0.03, 0.03)
            current_news = max(-1, min(1, current_news))
            
            current_market += random.uniform(-0.02, 0.02)
            current_market = max(-1, min(1, current_market))
            
            # Add to trends
            social_media_trends.append(current_social)
            news_trends.append(current_news)
            market_trends.append(current_market)
            
            # Update time
            current_time += timedelta(hours=1)
        
        # Calculate overall sentiment
        overall_sentiment = (social_media_sentiment + news_sentiment + market_sentiment) / 3
        
        # Generate sentiment sources
        sources = []
        for _ in range(5):
            source_type = random.choice(['twitter', 'reddit', 'news', 'blog'])
            sentiment = random.uniform(-1, 1)
            impact = random.uniform(0, 1)
            
            sources.append({
                'type': source_type,
                'source': f"Example {source_type.capitalize()} Source",
                'sentiment': sentiment,
                'impact': impact,
                'timestamp': int((datetime.now() - timedelta(hours=random.randint(0, 24))).timestamp() * 1000)
            })
        
        return {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'social_media_sentiment': social_media_sentiment,
            'news_sentiment': news_sentiment,
            'market_sentiment': market_sentiment,
            'trends': {
                'timestamps': timestamps,
                'social_media': social_media_trends,
                'news': news_trends,
                'market': market_trends
            },
            'sources': sources
        }
    
    def _generate_performance_data(self) -> Dict[str, Any]:
        """
        Generate mock performance data.
        
        Returns:
            Dict containing the performance data
        """
        # Generate daily returns
        timestamps = []
        daily_returns = []
        cumulative_returns = []
        
        # Start time
        end_time = datetime.now()
        current_time = end_time - timedelta(days=30)
        
        # Cumulative return
        cumulative = 1.0
        
        # Generate daily data for a month
        for _ in range(30):
            # Add timestamp
            timestamps.append(int(current_time.timestamp() * 1000))
            
            # Generate daily return
            daily_return = random.uniform(-0.03, 0.04)
            daily_returns.append(daily_return)
            
            # Update cumulative return
            cumulative *= (1 + daily_return)
            cumulative_returns.append(cumulative - 1)
            
            # Update time
            current_time += timedelta(days=1)
        
        # Calculate performance metrics
        total_return = cumulative_returns[-1]
        max_drawdown = min(0, min(cumulative_returns))
        sharpe_ratio = random.uniform(0.5, 2.5)
        win_rate = random.uniform(0.4, 0.7)
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'daily_returns': {
                'timestamps': timestamps,
                'returns': daily_returns,
                'cumulative': cumulative_returns
            }
        }
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate mock alerts.
        
        Returns:
            List of alerts
        """
        alerts = []
        
        # Alert types
        alert_types = ['price', 'sentiment', 'pattern', 'system', 'trading']
        
        # Alert levels
        alert_levels = ['info', 'warning', 'danger']
        
        # Generate random alerts
        for _ in range(10):
            alert_type = random.choice(alert_types)
            alert_level = random.choice(alert_levels)
            symbol = random.choice(self.symbols)
            
            # Generate alert message based on type
            if alert_type == 'price':
                message = f"{symbol} price {random.choice(['above', 'below'])} threshold"
            elif alert_type == 'sentiment':
                message = f"{symbol} sentiment {random.choice(['positive', 'negative', 'neutral'])}"
            elif alert_type == 'pattern':
                message = f"{symbol} {random.choice(['bullish', 'bearish'])} pattern detected"
            elif alert_type == 'system':
                message = f"System {random.choice(['warning', 'error', 'notification'])}"
            else:  # trading
                message = f"{symbol} {random.choice(['buy', 'sell'])} signal"
            
            # Generate timestamp
            timestamp = int((datetime.now() - timedelta(minutes=random.randint(0, 60))).timestamp() * 1000)
            
            alerts.append({
                'id': f"alert_{len(alerts) + 1}",
                'type': alert_type,
                'level': alert_level,
                'message': message,
                'symbol': symbol,
                'timestamp': timestamp
            })
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return alerts
    
    def _generate_alert(self) -> Dict[str, Any]:
        """
        Generate a single mock alert.
        
        Returns:
            Dict containing the alert
        """
        # Alert types
        alert_types = ['price', 'sentiment', 'pattern', 'system', 'trading']
        
        # Alert levels
        alert_levels = ['info', 'warning', 'danger']
        
        alert_type = random.choice(alert_types)
        alert_level = random.choice(alert_levels)
        symbol = random.choice(self.symbols)
        
        # Generate alert message based on type
        if alert_type == 'price':
            message = f"{symbol} price {random.choice(['above', 'below'])} threshold"
        elif alert_type == 'sentiment':
            message = f"{symbol} sentiment {random.choice(['positive', 'negative', 'neutral'])}"
        elif alert_type == 'pattern':
            message = f"{symbol} {random.choice(['bullish', 'bearish'])} pattern detected"
        elif alert_type == 'system':
            message = f"System {random.choice(['warning', 'error', 'notification'])}"
        else:  # trading
            message = f"{symbol} {random.choice(['buy', 'sell'])} signal"
        
        # Generate timestamp
        timestamp = int(datetime.now().timestamp() * 1000)
        
        return {
            'id': f"alert_{int(time.time() * 1000)}",
            'type': alert_type,
            'level': alert_level,
            'message': message,
            'symbol': symbol,
            'timestamp': timestamp
        }
    
    def _generate_positions(self) -> List[Dict[str, Any]]:
        """
        Generate mock positions.
        
        Returns:
            List of positions
        """
        positions = []
        
        # Position types
        position_types = ['long', 'short']
        
        # Generate random positions
        for symbol in random.sample(self.symbols, random.randint(1, len(self.symbols))):
            position_type = random.choice(position_types)
            
            # Generate position details
            entry_price = {
                'BTC/USD': random.uniform(45000, 55000),
                'ETH/USD': random.uniform(2800, 3200),
                'SOL/USD': random.uniform(140, 160),
                'ADA/USD': random.uniform(1.1, 1.3),
                'XRP/USD': random.uniform(0.7, 0.9)
            }.get(symbol, 100)
            
            current_price = entry_price * (1 + random.uniform(-0.05, 0.05))
            
            size = random.uniform(0.1, 2.0)
            if symbol == 'BTC/USD':
                size = random.uniform(0.01, 0.2)
            
            # Calculate profit/loss
            if position_type == 'long':
                pnl = (current_price - entry_price) * size
                pnl_percentage = (current_price / entry_price - 1) * 100
            else:  # short
                pnl = (entry_price - current_price) * size
                pnl_percentage = (entry_price / current_price - 1) * 100
            
            # Generate timestamp
            timestamp = int((datetime.now() - timedelta(hours=random.randint(1, 48))).timestamp() * 1000)
            
            positions.append({
                'id': f"position_{len(positions) + 1}",
                'symbol': symbol,
                'type': position_type,
                'size': size,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'timestamp': timestamp
            })
        
        return positions
    
    def _generate_trades(self, count: int) -> List[Dict[str, Any]]:
        """
        Generate mock trades.
        
        Args:
            count: The number of trades to generate
            
        Returns:
            List of trades
        """
        trades = []
        
        # Trade types
        trade_types = ['buy', 'sell']
        
        # Generate random trades
        for i in range(count):
            symbol = random.choice(self.symbols)
            trade_type = random.choice(trade_types)
            
            # Generate trade details
            price = {
                'BTC/USD': random.uniform(45000, 55000),
                'ETH/USD': random.uniform(2800, 3200),
                'SOL/USD': random.uniform(140, 160),
                'ADA/USD': random.uniform(1.1, 1.3),
                'XRP/USD': random.uniform(0.7, 0.9)
            }.get(symbol, 100)
            
            size = random.uniform(0.1, 2.0)
            if symbol == 'BTC/USD':
                size = random.uniform(0.01, 0.2)
            
            # Calculate value
            value = price * size
            
            # Generate timestamp
            timestamp = int((datetime.now() - timedelta(minutes=i * 30)).timestamp() * 1000)
            
            trades.append({
                'id': f"trade_{count - i}",
                'symbol': symbol,
                'type': trade_type,
                'size': size,
                'price': price,
                'value': value,
                'timestamp': timestamp
            })
        
        # Sort by timestamp (newest first)
        trades.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return trades
    
    def _generate_trade(self) -> Dict[str, Any]:
        """
        Generate a single mock trade.
        
        Returns:
            Dict containing the trade
        """
        # Trade types
        trade_types = ['buy', 'sell']
        
        symbol = random.choice(self.symbols)
        trade_type = random.choice(trade_types)
        
        # Generate trade details
        price = {
            'BTC/USD': random.uniform(45000, 55000),
            'ETH/USD': random.uniform(2800, 3200),
            'SOL/USD': random.uniform(140, 160),
            'ADA/USD': random.uniform(1.1, 1.3),
            'XRP/USD': random.uniform(0.7, 0.9)
        }.get(symbol, 100)
        
        size = random.uniform(0.1, 2.0)
        if symbol == 'BTC/USD':
            size = random.uniform(0.01, 0.2)
        
        # Calculate value
        value = price * size
        
        # Generate timestamp
        timestamp = int(datetime.now().timestamp() * 1000)
        
        return {
            'id': f"trade_{int(time.time() * 1000)}",
            'symbol': symbol,
            'type': trade_type,
            'size': size,
            'price': price,
            'value': value,
            'timestamp': timestamp
        }

class RealDataService(DataService):
    """
    Real data service for production use.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the real data service.
        
        Args:
            config: Configuration for the data service
        """
        # Store configuration
        self.config = config
        
        # Initialize connectors
        self.exchange_connector = None
        self.sentiment_connector = None
        self.news_connector = None
        
        # Cache
        self.cache = {}
        self.cache_expiry = {}
        
        logger.info("Real data service initialized")
    
    async def get_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get real market data for a symbol.
        
        Args:
            symbol: The symbol to get data for
            timeframe: The timeframe to get data for
            limit: The maximum number of data points to return
            
        Returns:
            Dict containing the market data
        """
        # Check if exchange connector is available
        if not self.exchange_connector:
            logger.warning("Exchange connector not available")
            return {'error': "Exchange connector not available"}
        
        try:
            # Check cache
            cache_key = f"market_data_{symbol}_{timeframe}_{limit}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data from exchange
            data = await self.exchange_connector.get_candles(symbol, timeframe, limit)
            
            # Cache data
            self._cache_data(cache_key, data, 60)  # Cache for 60 seconds
            
            return data
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {'error': str(e)}
    
    async def get_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get real sentiment data for a symbol.
        
        Args:
            symbol: The symbol to get data for
            
        Returns:
            Dict containing the sentiment data
        """
        # Check if sentiment connector is available
        if not self.sentiment_connector:
            logger.warning("Sentiment connector not available")
            return {'error': "Sentiment connector not available"}
        
        try:
            # Check cache
            cache_key = f"sentiment_data_{symbol}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data from sentiment API
            data = await self.sentiment_connector.get_sentiment(symbol)
            
            # Cache data
            self._cache_data(cache_key, data, 300)  # Cache for 5 minutes
            
            return data
        except Exception as e:
            logger.error(f"Error getting sentiment data: {e}")
            return {'error': str(e)}
    
    async def get_performance_data(self) -> Dict[str, Any]:
        """
        Get real performance data.
        
        Returns:
            Dict containing the performance data
        """
        try:
            # Check cache
            cache_key = "performance_data"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data from database or API
            # This would be implemented based on the specific data source
            
            # For now, return mock data
            data = MockDataService()._generate_performance_data()
            
            # Cache data
            self._cache_data(cache_key, data, 600)  # Cache for 10 minutes
            
            return data
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {'error': str(e)}
    
    async def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get real alerts.
        
        Returns:
            List of alerts
        """
        try:
            # Check cache
            cache_key = "alerts"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data from database or API
            # This would be implemented based on the specific data source
            
            # For now, return mock data
            data = MockDataService()._generate_alerts()
            
            # Cache data
            self._cache_data(cache_key, data, 60)  # Cache for 1 minute
            
            return data
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return [{'error': str(e)}]
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get real positions.
        
        Returns:
            List of positions
        """
        # Check if exchange connector is available
        if not self.exchange_connector:
            logger.warning("Exchange connector not available")
            return [{'error': "Exchange connector not available"}]
        
        try:
            # Check cache
            cache_key = "positions"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data from exchange
            data = await self.exchange_connector.get_positions()
            
            # Cache data
            self._cache_data(cache_key, data, 30)  # Cache for 30 seconds
            
            return data
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return [{'error': str(e)}]
    
    async def get_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get real trades.
        
        Args:
            limit: The maximum number of trades to return
            
        Returns:
            List of trades
        """
        # Check if exchange connector is available
        if not self.exchange_connector:
            logger.warning("Exchange connector not available")
            return [{'error': "Exchange connector not available"}]
        
        try:
            # Check cache
            cache_key = f"trades_{limit}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data from exchange
            data = await self.exchange_connector.get_trades(limit)
            
            # Cache data
            self._cache_data(cache_key, data, 30)  # Cache for 30 seconds
            
            return data
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return [{'error': str(e)}]
    
    def _is_cache_valid(self, key: str) -> bool:
        """
        Check if cached data is valid.
        
        Args:
            key: The cache key
            
        Returns:
            True if cache is valid, False otherwise
        """
        if key not in self.cache or key not in self.cache_expiry:
            return False
        
        return time.time() < self.cache_expiry[key]
    
    def _cache_data(self, key: str, data: Any, expiry_seconds: int):
        """
        Cache data with expiry.
        
        Args:
            key: The cache key
            data: The data to cache
            expiry_seconds: The number of seconds until the cache expires
        """
        self.cache[key] = data
        self.cache_expiry[key] = time.time() + expiry_seconds

class DataServiceFactory:
    """
    Factory for creating data services.
    """
    
    @staticmethod
    def create_data_service(use_real_data: bool, config: Optional[Dict[str, Any]] = None) -> DataService:
        """
        Create a data service.
        
        Args:
            use_real_data: Whether to use real data
            config: Configuration for the data service (required for real data)
            
        Returns:
            A data service instance
        """
        if use_real_data:
            if not config:
                logger.warning("No configuration provided for real data service, using mock data instead")
                return MockDataService()
            
            return RealDataService(config)
        else:
            return MockDataService()
