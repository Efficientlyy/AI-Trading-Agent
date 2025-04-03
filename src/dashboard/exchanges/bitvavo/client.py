"""
Bitvavo Exchange Integration Module

This module provides integration with the Bitvavo cryptocurrency exchange.
It includes API client, authentication, and trading functionality.
"""

import logging
import time
import hmac
import hashlib
import json
import base64
import asyncio
from typing import Dict, List, Any, Optional, Union
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bitvavo_integration")

class BitvavoClient:
    """
    Client for interacting with the Bitvavo API.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", base_url: str = "https://api.bitvavo.com/v2"):
        """
        Initialize the Bitvavo client.
        
        Args:
            api_key: Bitvavo API key
            api_secret: Bitvavo API secret
            base_url: Base URL for the Bitvavo API
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        
        # Rate limiting
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = 0
        
        # Session
        self.session = None
        
        logger.info("Bitvavo client initialized")
    
    async def create_session(self):
        """
        Create an aiohttp session for making API requests.
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            logger.info("Created new aiohttp session")
    
    async def close_session(self):
        """
        Close the aiohttp session.
        """
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Closed aiohttp session")
    
    def _create_signature(self, timestamp: int, method: str, url_path: str, body: Dict[str, Any] = None) -> str:
        """
        Create a signature for API authentication.
        
        Args:
            timestamp: Current timestamp in milliseconds
            method: HTTP method (GET, POST, etc.)
            url_path: URL path without base URL
            body: Request body (for POST requests)
            
        Returns:
            Signature string
        """
        if body is None:
            body = {}
        
        # Create message string
        message = str(timestamp) + method + "/v2" + url_path
        if method != "GET" and body:
            message += json.dumps(body)
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """
        Handle API response and rate limiting.
        
        Args:
            response: API response
            
        Returns:
            Response data
        """
        # Update rate limiting information
        if 'Bitvavo-Ratelimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['Bitvavo-Ratelimit-Remaining'])
        
        if 'Bitvavo-Ratelimit-ResetAt' in response.headers:
            self.rate_limit_reset = int(response.headers['Bitvavo-Ratelimit-ResetAt'])
        
        # Check if rate limited
        if response.status == 429:
            reset_time = self.rate_limit_reset / 1000  # Convert to seconds
            current_time = time.time()
            sleep_time = max(0, reset_time - current_time) + 1  # Add 1 second buffer
            
            logger.warning(f"Rate limited, sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
            
            # Return error
            return {'error': 'Rate limited'}
        
        # Parse response
        try:
            data = await response.json()
            return data
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {'error': 'Error parsing response'}
    
    async def _request(self, method: str, endpoint: str, params: Dict[str, Any] = None, data: Dict[str, Any] = None, auth: bool = False) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            auth: Whether authentication is required
            
        Returns:
            Response data
        """
        # Create session if needed
        await self.create_session()
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Build headers
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Add authentication if required
        if auth:
            if not self.api_key or not self.api_secret:
                logger.error("API key and secret required for authenticated requests")
                return {'error': 'API key and secret required'}
            
            timestamp = int(time.time() * 1000)
            signature = self._create_signature(timestamp, method, endpoint, data)
            
            headers['Bitvavo-Access-Key'] = self.api_key
            headers['Bitvavo-Access-Signature'] = signature
            headers['Bitvavo-Access-Timestamp'] = str(timestamp)
            headers['Bitvavo-Access-Window'] = '10000'
        
        # Check rate limiting
        if self.rate_limit_remaining <= 1:
            reset_time = self.rate_limit_reset / 1000  # Convert to seconds
            current_time = time.time()
            sleep_time = max(0, reset_time - current_time) + 1  # Add 1 second buffer
            
            logger.warning(f"Rate limit almost reached, sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
        
        try:
            # Make request
            async with self.session.request(method, url, params=params, json=data, headers=headers) as response:
                return await self._handle_response(response)
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            return {'error': f'Request error: {e}'}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {'error': f'Unexpected error: {e}'}
    
    async def get_time(self) -> Dict[str, Any]:
        """
        Get server time.
        
        Returns:
            Server time
        """
        return await self._request('GET', '/time')
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """
        Get available markets.
        
        Returns:
            List of markets
        """
        return await self._request('GET', '/markets')
    
    async def get_assets(self) -> List[Dict[str, Any]]:
        """
        Get available assets.
        
        Returns:
            List of assets
        """
        return await self._request('GET', '/assets')
    
    async def get_ticker_price(self, market: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get ticker price for a market or all markets.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            Ticker price data
        """
        if market:
            return await self._request('GET', f'/ticker/price?market={market}')
        else:
            return await self._request('GET', '/ticker/price')
    
    async def get_ticker_book(self, market: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get order book for a market or all markets.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            Order book data
        """
        if market:
            return await self._request('GET', f'/ticker/book?market={market}')
        else:
            return await self._request('GET', '/ticker/book')
    
    async def get_ticker_24h(self, market: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get 24-hour ticker data for a market or all markets.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            24-hour ticker data
        """
        if market:
            return await self._request('GET', f'/ticker/24h?market={market}')
        else:
            return await self._request('GET', '/ticker/24h')
    
    async def get_candles(self, market: str, interval: str, limit: int = 1000, start: Optional[int] = None, end: Optional[int] = None) -> List[List[Any]]:
        """
        Get candle data for a market.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            interval: Candle interval (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d)
            limit: Maximum number of candles to return
            start: Start time in milliseconds
            end: End time in milliseconds
            
        Returns:
            Candle data
        """
        params = {
            'market': market,
            'interval': interval,
            'limit': limit
        }
        
        if start:
            params['start'] = start
        
        if end:
            params['end'] = end
        
        return await self._request('GET', '/candles', params=params)
    
    async def get_trades(self, market: str, limit: int = 1000, start: Optional[int] = None, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get trades for a market.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            limit: Maximum number of trades to return
            start: Start time in milliseconds
            end: End time in milliseconds
            
        Returns:
            Trade data
        """
        params = {
            'market': market,
            'limit': limit
        }
        
        if start:
            params['start'] = start
        
        if end:
            params['end'] = end
        
        return await self._request('GET', '/trades', params=params)
    
    async def get_book(self, market: str, depth: int = 100) -> Dict[str, Any]:
        """
        Get order book for a market.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            depth: Order book depth
            
        Returns:
            Order book data
        """
        params = {
            'market': market,
            'depth': depth
        }
        
        return await self._request('GET', '/book', params=params)
    
    # Authenticated endpoints
    
    async def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information
        """
        return await self._request('GET', '/account', auth=True)
    
    async def get_balance(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get account balance.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC')
            
        Returns:
            Balance information
        """
        if symbol:
            return await self._request('GET', f'/balance?symbol={symbol}', auth=True)
        else:
            return await self._request('GET', '/balance', auth=True)
    
    async def get_orders(self, market: Optional[str] = None, limit: int = 500, start: Optional[int] = None, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            limit: Maximum number of orders to return
            start: Start time in milliseconds
            end: End time in milliseconds
            
        Returns:
            Order data
        """
        params = {
            'limit': limit
        }
        
        if market:
            params['market'] = market
        
        if start:
            params['start'] = start
        
        if end:
            params['end'] = end
        
        return await self._request('GET', '/orders', params=params, auth=True)
    
    async def get_order(self, order_id: str, market: str) -> Dict[str, Any]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            Order data
        """
        params = {
            'orderId': order_id,
            'market': market
        }
        
        return await self._request('GET', '/order', params=params, auth=True)
    
    async def create_order(self, market: str, side: str, order_type: str, amount: Optional[str] = None, 
                          price: Optional[str] = None, amount_quote: Optional[str] = None, 
                          time_in_force: Optional[str] = None, self_trade_prevention: Optional[str] = None, 
                          disable_market_protection: Optional[bool] = None) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            side: Order side ('buy' or 'sell')
            order_type: Order type ('limit', 'market', 'stopLoss', 'stopLossLimit', 'takeProfit', 'takeProfitLimit')
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            amount_quote: Order amount in quote currency (alternative to amount)
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            self_trade_prevention: Self-trade prevention ('decrementAndCancel', 'cancelOldest', 'cancelNewest', 'cancelBoth')
            disable_market_protection: Whether to disable market protection
            
        Returns:
            Order data
        """
        data = {
            'market': market,
            'side': side,
            'orderType': order_type
        }
        
        if amount:
            data['amount'] = amount
        
        if price:
            data['price'] = price
        
        if amount_quote:
            data['amountQuote'] = amount_quote
        
        if time_in_force:
            data['timeInForce'] = time_in_force
        
        if self_trade_prevention:
            data['selfTradePrevention'] = self_trade_prevention
        
        if disable_market_protection is not None:
            data['disableMarketProtection'] = disable_market_protection
        
        return await self._request('POST', '/order', data=data, auth=True)
    
    async def cancel_order(self, order_id: str, market: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            Cancellation result
        """
        data = {
            'orderId': order_id,
            'market': market
        }
        
        return await self._request('DELETE', '/order', data=data, auth=True)
    
    async def cancel_all_orders(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Cancel all orders.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            Cancellation results
        """
        data = {}
        
        if market:
            data['market'] = market
        
        return await self._request('DELETE', '/orders', data=data, auth=True)
    
    async def get_trades_history(self, market: Optional[str] = None, limit: int = 500, start: Optional[int] = None, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            limit: Maximum number of trades to return
            start: Start time in milliseconds
            end: End time in milliseconds
            
        Returns:
            Trade history
        """
        params = {
            'limit': limit
        }
        
        if market:
            params['market'] = market
        
        if start:
            params['start'] = start
        
        if end:
            params['end'] = end
        
        return await self._request('GET', '/trades', params=params, auth=True)
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection.
        
        Returns:
            Connection test result
        """
        try:
            # Test public endpoint
            time_result = await self.get_time()
            
            if 'error' in time_result:
                return {'success': False, 'message': f"Public API error: {time_result['error']}"}
            
            # Test authenticated endpoint if credentials are provided
            if self.api_key and self.api_secret:
                account_result = await self.get_account()
                
                if 'error' in account_result:
                    return {'success': False, 'message': f"Authentication error: {account_result['error']}"}
                
                return {'success': True, 'message': "Connection successful with authentication"}
            
            return {'success': True, 'message': "Connection successful (public API only)"}
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            return {'success': False, 'message': f"Connection test error: {e}"}

class BitvavoExchangeConnector:
    """
    Connector for the Bitvavo exchange.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", paper_trading: bool = True):
        """
        Initialize the Bitvavo exchange connector.
        
        Args:
            api_key: Bitvavo API key
            api_secret: Bitvavo API secret
            paper_trading: Whether to use paper trading
        """
        self.client = BitvavoClient(api_key, api_secret)
        self.paper_trading = paper_trading
        
        # Paper trading state
        self.paper_balance = {}
        self.paper_orders = []
        self.paper_trades = []
        
        logger.info(f"Bitvavo exchange connector initialized (paper_trading={paper_trading})")
    
    async def initialize(self):
        """
        Initialize the connector.
        """
        # Initialize paper trading if enabled
        if self.paper_trading:
            await self._initialize_paper_trading()
    
    async def _initialize_paper_trading(self):
        """
        Initialize paper trading.
        """
        # Set initial paper balance
        self.paper_balance = {
            'EUR': {'available': '10000', 'inOrder': '0'},
            'BTC': {'available': '0.1', 'inOrder': '0'},
            'ETH': {'available': '1', 'inOrder': '0'},
            'SOL': {'available': '10', 'inOrder': '0'},
            'ADA': {'available': '1000', 'inOrder': '0'},
            'XRP': {'available': '1000', 'inOrder': '0'}
        }
        
        logger.info("Paper trading initialized with default balances")
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """
        Get available markets.
        
        Returns:
            List of markets
        """
        return await self.client.get_markets()
    
    async def get_assets(self) -> List[Dict[str, Any]]:
        """
        Get available assets.
        
        Returns:
            List of assets
        """
        return await self.client.get_assets()
    
    async def get_ticker(self, market: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get ticker data for a market or all markets.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            Ticker data
        """
        return await self.client.get_ticker_24h(market)
    
    async def get_candles(self, market: str, interval: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get candle data for a market.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            interval: Candle interval (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d)
            limit: Maximum number of candles to return
            
        Returns:
            Candle data in a structured format
        """
        # Get raw candle data
        candles = await self.client.get_candles(market, interval, limit)
        
        if isinstance(candles, dict) and 'error' in candles:
            return candles
        
        # Transform to structured format
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        for candle in candles:
            timestamps.append(candle[0])  # Timestamp
            opens.append(float(candle[1]))  # Open
            highs.append(float(candle[2]))  # High
            lows.append(float(candle[3]))  # Low
            closes.append(float(candle[4]))  # Close
            volumes.append(float(candle[5]))  # Volume
        
        return {
            'symbol': market,
            'timeframe': interval,
            'timestamps': timestamps,
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'volumes': volumes
        }
    
    async def get_balance(self) -> List[Dict[str, Any]]:
        """
        Get account balance.
        
        Returns:
            Balance information
        """
        if self.paper_trading:
            # Return paper balance
            result = []
            for symbol, balance in self.paper_balance.items():
                result.append({
                    'symbol': symbol,
                    'available': balance['available'],
                    'inOrder': balance['inOrder']
                })
            return result
        else:
            # Get real balance
            return await self.client.get_balance()
    
    async def get_orders(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            List of open orders
        """
        if self.paper_trading:
            # Return paper orders
            if market:
                return [order for order in self.paper_orders if order['market'] == market]
            else:
                return self.paper_orders
        else:
            # Get real orders
            return await self.client.get_orders(market)
    
    async def create_order(self, market: str, side: str, order_type: str, amount: str, price: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            market: Market symbol (e.g., 'BTC-EUR')
            side: Order side ('buy' or 'sell')
            order_type: Order type ('limit', 'market')
            amount: Order amount
            price: Order price (required for limit orders)
            
        Returns:
            Order data
        """
        if self.paper_trading:
            # Create paper order
            order_id = f"paper_{int(time.time() * 1000)}_{len(self.paper_orders)}"
            
            order = {
                'orderId': order_id,
                'market': market,
                'side': side,
                'orderType': order_type,
                'amount': amount,
                'status': 'new',
                'created': int(time.time() * 1000)
            }
            
            if price:
                order['price'] = price
            
            # Add to paper orders
            self.paper_orders.append(order)
            
            # Update paper balance
            await self._update_paper_balance_for_order(order)
            
            logger.info(f"Created paper order: {order}")
            
            return order
        else:
            # Create real order
            return await self.client.create_order(market, side, order_type, amount, price)
    
    async def cancel_order(self, order_id: str, market: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            market: Market symbol (e.g., 'BTC-EUR')
            
        Returns:
            Cancellation result
        """
        if self.paper_trading:
            # Cancel paper order
            for i, order in enumerate(self.paper_orders):
                if order['orderId'] == order_id and order['market'] == market:
                    # Update order status
                    self.paper_orders[i]['status'] = 'canceled'
                    
                    # Update paper balance
                    await self._update_paper_balance_for_cancel(self.paper_orders[i])
                    
                    logger.info(f"Canceled paper order: {order_id}")
                    
                    return {'orderId': order_id, 'market': market}
            
            return {'error': 'Order not found'}
        else:
            # Cancel real order
            return await self.client.cancel_order(order_id, market)
    
    async def get_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trades
        """
        if self.paper_trading:
            # Return paper trades
            return self.paper_trades[:limit]
        else:
            # Get real trades from multiple markets
            markets = ['BTC-EUR', 'ETH-EUR', 'SOL-EUR']
            all_trades = []
            
            for market in markets:
                trades = await self.client.get_trades_history(market, limit=limit // len(markets))
                if not isinstance(trades, dict) or 'error' not in trades:
                    all_trades.extend(trades)
            
            # Sort by timestamp (newest first)
            all_trades.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            return all_trades[:limit]
    
    async def _update_paper_balance_for_order(self, order: Dict[str, Any]):
        """
        Update paper balance for a new order.
        
        Args:
            order: Order data
        """
        # Parse market symbol
        base_asset, quote_asset = order['market'].split('-')
        
        # Update balance based on order side
        if order['side'] == 'buy':
            # Calculate order value
            if order['orderType'] == 'limit':
                order_value = float(order['amount']) * float(order['price'])
            else:
                # For market orders, use current price (simplified)
                ticker = await self.client.get_ticker_price(order['market'])
                if 'error' in ticker:
                    logger.error(f"Error getting ticker price: {ticker['error']}")
                    return
                
                order_value = float(order['amount']) * float(ticker['price'])
            
            # Update quote asset (e.g., EUR)
            if quote_asset in self.paper_balance:
                available = float(self.paper_balance[quote_asset]['available'])
                in_order = float(self.paper_balance[quote_asset]['inOrder'])
                
                if available >= order_value:
                    self.paper_balance[quote_asset]['available'] = str(available - order_value)
                    self.paper_balance[quote_asset]['inOrder'] = str(in_order + order_value)
                else:
                    logger.warning(f"Insufficient {quote_asset} balance for order")
        else:  # sell
            # Update base asset (e.g., BTC)
            if base_asset in self.paper_balance:
                available = float(self.paper_balance[base_asset]['available'])
                in_order = float(self.paper_balance[base_asset]['inOrder'])
                
                order_amount = float(order['amount'])
                
                if available >= order_amount:
                    self.paper_balance[base_asset]['available'] = str(available - order_amount)
                    self.paper_balance[base_asset]['inOrder'] = str(in_order + order_amount)
                else:
                    logger.warning(f"Insufficient {base_asset} balance for order")
    
    async def _update_paper_balance_for_cancel(self, order: Dict[str, Any]):
        """
        Update paper balance for a canceled order.
        
        Args:
            order: Order data
        """
        # Parse market symbol
        base_asset, quote_asset = order['market'].split('-')
        
        # Update balance based on order side
        if order['side'] == 'buy':
            # Calculate order value
            if order['orderType'] == 'limit':
                order_value = float(order['amount']) * float(order['price'])
            else:
                # For market orders, use current price (simplified)
                ticker = await self.client.get_ticker_price(order['market'])
                if 'error' in ticker:
                    logger.error(f"Error getting ticker price: {ticker['error']}")
                    return
                
                order_value = float(order['amount']) * float(ticker['price'])
            
            # Update quote asset (e.g., EUR)
            if quote_asset in self.paper_balance:
                available = float(self.paper_balance[quote_asset]['available'])
                in_order = float(self.paper_balance[quote_asset]['inOrder'])
                
                self.paper_balance[quote_asset]['available'] = str(available + order_value)
                self.paper_balance[quote_asset]['inOrder'] = str(max(0, in_order - order_value))
        else:  # sell
            # Update base asset (e.g., BTC)
            if base_asset in self.paper_balance:
                available = float(self.paper_balance[base_asset]['available'])
                in_order = float(self.paper_balance[base_asset]['inOrder'])
                
                order_amount = float(order['amount'])
                
                self.paper_balance[base_asset]['available'] = str(available + order_amount)
                self.paper_balance[base_asset]['inOrder'] = str(max(0, in_order - order_amount))
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection.
        
        Returns:
            Connection test result
        """
        return await self.client.test_connection()
