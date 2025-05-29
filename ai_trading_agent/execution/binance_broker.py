"""
Binance Broker Integration Module

Implements the broker interface for the Binance exchange, providing real
trading capabilities via the Binance API.
"""

import logging
import time
import hmac
import hashlib
import json
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import asyncio
import aiohttp

from ..common.enums import OrderType, OrderSide, OrderStatus, TimeInForce
from ..common.models import Order, Trade, Position, Balance, Portfolio
from ..common.enhanced_circuit_breaker import EnhancedCircuitBreaker, register_circuit_breaker
from .broker_interface import BrokerInterface

logger = logging.getLogger(__name__)


class BinanceBroker(BrokerInterface):
    """
    Binance broker implementation.
    
    Provides trading capabilities through the Binance API with proper
    error handling, rate limiting, and circuit breaking for robustness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Binance broker.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.name = "BinanceBroker"
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.testnet = config.get('testnet', True)  # Default to testnet for safety
        
        # Circuit breaker for API calls
        self.circuit_breaker = EnhancedCircuitBreaker(
            name="binance_api",
            warning_threshold=3,
            failure_threshold=5,
            recovery_time_base=10.0,
            max_recovery_time=300.0,
            reset_timeout=60.0
        )
        register_circuit_breaker(self.circuit_breaker)
        
        # Configure base URLs based on testnet setting
        if self.testnet:
            self.base_url = "https://testnet.binance.vision/api"
            logger.info("Using Binance TESTNET")
        else:
            self.base_url = "https://api.binance.com/api"
            logger.info("Using Binance PRODUCTION - REAL TRADING ENABLED")
            
        # API rate limiting configuration
        self.rate_limits = {
            'order': {'limit': 10, 'window': 60},  # 10 orders per minute
            'query': {'limit': 50, 'window': 60}   # 50 queries per minute
        }
        self.request_timestamps = {'order': [], 'query': []}
        
        # HTTP session for API calls
        self.session = None
        
        # Cache for market data
        self.price_cache = {}
        self.price_cache_time = {}
        self.price_cache_ttl = 5.0  # 5 seconds TTL
        
        # Initialize portfolio tracking
        self.portfolio = Portfolio(
            balances={},
            positions={},
            order_history=[],
            trade_history=[],
            realized_pnl=0.0,
            unrealized_pnl=0.0
        )
        
        # Symbol info cache
        self.symbol_info = {}
        
        logger.info(f"Initialized {self.name}")
    
    async def initialize(self) -> bool:
        """
        Initialize the broker connection and authenticate.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test authentication with account info endpoint
            account_info = await self._api_request('GET', '/v3/account', {}, True)
            if not account_info:
                logger.error("Failed to authenticate with Binance API")
                return False
                
            # Load exchange information for symbols
            exchange_info = await self._api_request('GET', '/v3/exchangeInfo', {}, False)
            if not exchange_info:
                logger.error("Failed to retrieve exchange information")
                return False
                
            # Cache symbol information
            for symbol in exchange_info.get('symbols', []):
                self.symbol_info[symbol['symbol']] = symbol
                
            # Load initial account balances
            await self.refresh_balances()
            
            logger.info(f"Successfully initialized {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {self.name}: {str(e)}")
            return False
            
    async def refresh_balances(self) -> None:
        """
        Refresh account balances from the exchange.
        """
        try:
            account_info = await self._api_request('GET', '/v3/account', {}, True)
            if not account_info:
                logger.error("Failed to refresh balances")
                return
                
            # Update balances
            balances = {}
            for asset in account_info.get('balances', []):
                currency = asset['asset']
                free = float(asset['free'])
                locked = float(asset['locked'])
                
                if free > 0 or locked > 0:
                    balances[currency] = Balance(
                        currency=currency,
                        free=free,
                        locked=locked
                    )
            
            self.portfolio.balances = balances
            logger.debug(f"Refreshed balances: {len(balances)} assets with balance")
            
        except Exception as e:
            logger.error(f"Error refreshing balances: {str(e)}")
    
    async def place_order(self, order: Order) -> Tuple[bool, str, Optional[str]]:
        """
        Place an order with the broker.
        
        Args:
            order: Order details
            
        Returns:
            Tuple of (success, message, order_id)
        """
        # Check circuit breaker
        if not self.circuit_breaker.is_allowed():
            return False, "Circuit breaker is open, API calls blocked", None
            
        # Check rate limits
        if not self._check_rate_limit('order'):
            return False, "Rate limit exceeded for order placement", None
            
        try:
            # Prepare parameters
            params = {
                'symbol': order.symbol.replace('/', ''),  # Remove / separator
                'side': order.side.value.upper(),
                'type': order.type.value.upper(),
                'quantity': str(order.quantity)
            }
            
            # Add price for limit orders
            if order.type == OrderType.LIMIT and order.price is not None:
                params['price'] = str(order.price)
                params['timeInForce'] = order.time_in_force.value if order.time_in_force else TimeInForce.GTC.value
                
            # Send order request
            result = await self._api_request('POST', '/v3/order', params, True)
            if not result or 'orderId' not in result:
                self.circuit_breaker.record_failure()
                error_msg = result.get('msg', 'Unknown error') if result else 'Failed to place order'
                logger.error(f"Order placement failed: {error_msg}")
                return False, error_msg, None
                
            # Update order with response data
            order_id = str(result['orderId'])
            
            logger.info(f"Order placed successfully: {order_id}")
            return True, "Order placed successfully", order_id
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Error placing order: {str(e)}")
            return False, f"Error placing order: {str(e)}", None
            
    async def cancel_order(self, order_id: str, symbol: str) -> Tuple[bool, str]:
        """
        Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Symbol for the order
            
        Returns:
            Tuple of (success, message)
        """
        # Check circuit breaker
        if not self.circuit_breaker.is_allowed():
            return False, "Circuit breaker is open, API calls blocked"
            
        # Check rate limits
        if not self._check_rate_limit('order'):
            return False, "Rate limit exceeded for order cancellation"
            
        try:
            # Prepare parameters
            params = {
                'symbol': symbol.replace('/', ''),  # Remove / separator
                'orderId': order_id
            }
            
            # Send cancel request
            result = await self._api_request('DELETE', '/v3/order', params, True)
            if not result or 'orderId' not in result:
                self.circuit_breaker.record_failure()
                error_msg = result.get('msg', 'Unknown error') if result else 'Failed to cancel order'
                logger.error(f"Order cancellation failed: {error_msg}")
                return False, error_msg
                
            logger.info(f"Order cancelled successfully: {order_id}")
            return True, "Order cancelled successfully"
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Error cancelling order: {str(e)}")
            return False, f"Error cancelling order: {str(e)}"
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Get order details by ID.
        
        Args:
            order_id: Order ID to retrieve
            symbol: Symbol for the order
            
        Returns:
            Order object if found, None otherwise
        """
        # Check circuit breaker
        if not self.circuit_breaker.is_allowed():
            logger.warning("Circuit breaker is open, API calls blocked")
            return None
            
        # Check rate limits
        if not self._check_rate_limit('query'):
            logger.warning("Rate limit exceeded for order query")
            return None
            
        try:
            # Prepare parameters
            params = {
                'symbol': symbol.replace('/', ''),  # Remove / separator
                'orderId': order_id
            }
            
            # Send query request
            result = await self._api_request('GET', '/v3/order', params, True)
            if not result or 'orderId' not in result:
                return None
                
            # Convert to Order object
            return self._binance_order_to_order(result)
            
        except Exception as e:
            logger.error(f"Error retrieving order: {str(e)}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of open orders
        """
        # Check circuit breaker
        if not self.circuit_breaker.is_allowed():
            logger.warning("Circuit breaker is open, API calls blocked")
            return []
            
        # Check rate limits
        if not self._check_rate_limit('query'):
            logger.warning("Rate limit exceeded for open orders query")
            return []
            
        try:
            # Prepare parameters
            params = {}
            if symbol:
                params['symbol'] = symbol.replace('/', '')  # Remove / separator
            
            # Send query request
            result = await self._api_request('GET', '/v3/openOrders', params, True)
            if not result:
                return []
                
            # Convert to Order objects
            return [self._binance_order_to_order(order_data) for order_data in result]
            
        except Exception as e:
            logger.error(f"Error retrieving open orders: {str(e)}")
            return []
    
    async def get_order_history(self, 
                          symbol: Optional[str] = None,
                          limit: int = 100) -> List[Order]:
        """
        Get order history.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of orders to return
            
        Returns:
            List of historical orders
        """
        # Check circuit breaker
        if not self.circuit_breaker.is_allowed():
            logger.warning("Circuit breaker is open, API calls blocked")
            return []
            
        # Check rate limits
        if not self._check_rate_limit('query'):
            logger.warning("Rate limit exceeded for order history query")
            return []
            
        try:
            # Prepare parameters
            params = {'limit': limit}
            if symbol:
                params['symbol'] = symbol.replace('/', '')  # Remove / separator
            
            # Send query request
            result = await self._api_request('GET', '/v3/allOrders', params, True)
            if not result:
                return []
                
            # Convert to Order objects
            return [self._binance_order_to_order(order_data) for order_data in result]
            
        except Exception as e:
            logger.error(f"Error retrieving order history: {str(e)}")
            return []
    
    async def get_trade_history(self,
                          symbol: Optional[str] = None,
                          limit: int = 100) -> List[Trade]:
        """
        Get trade execution history.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of trades to return
            
        Returns:
            List of trade objects
        """
        # Check circuit breaker
        if not self.circuit_breaker.is_allowed():
            logger.warning("Circuit breaker is open, API calls blocked")
            return []
            
        # Check rate limits
        if not self._check_rate_limit('query'):
            logger.warning("Rate limit exceeded for trade history query")
            return []
            
        try:
            # Prepare parameters
            params = {'limit': limit}
            if symbol:
                params['symbol'] = symbol.replace('/', '')  # Remove / separator
            
            # Send query request
            result = await self._api_request('GET', '/v3/myTrades', params, True)
            if not result:
                return []
                
            # Convert to Trade objects
            return [self._binance_trade_to_trade(trade_data) for trade_data in result]
            
        except Exception as e:
            logger.error(f"Error retrieving trade history: {str(e)}")
            return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Position]:
        """
        Get current positions.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dictionary mapping symbols to position objects
        """
        # For spot trading, positions are derived from balances
        positions = {}
        
        try:
            # Refresh account info
            await self.refresh_balances()
            
            # Get current prices for position valuation
            for currency, balance in self.portfolio.balances.items():
                if currency == 'USDT' or balance.free + balance.locked <= 0:
                    continue
                    
                if symbol and not symbol.startswith(currency):
                    continue
                    
                # Try to get price for currency/USDT
                price_symbol = f"{currency}USDT"
                price = await self.get_market_price(price_symbol)
                
                if price:
                    positions[currency] = Position(
                        symbol=price_symbol,
                        quantity=balance.free + balance.locked,
                        entry_price=0.0,  # Historical cost not available from API
                        current_price=price,
                        unrealized_pnl=0.0  # Can't calculate without entry price
                    )
            
            return positions
            
        except Exception as e:
            logger.error(f"Error retrieving positions: {str(e)}")
            return {}
    
    async def get_balances(self) -> Dict[str, Balance]:
        """
        Get account balances.
        
        Returns:
            Dictionary mapping currencies to balance objects
        """
        await self.refresh_balances()
        return self.portfolio.balances
    
    async def get_portfolio(self) -> Portfolio:
        """
        Get full portfolio information.
        
        Returns:
            Portfolio object with balances, positions, and trade history
        """
        try:
            # Refresh balances and positions
            await self.refresh_balances()
            positions = await self.get_positions()
            
            # Create updated portfolio
            portfolio = Portfolio(
                balances=self.portfolio.balances,
                positions=positions,
                order_history=self.portfolio.order_history,
                trade_history=self.portfolio.trade_history,
                realized_pnl=self.portfolio.realized_pnl,
                unrealized_pnl=0.0  # Calculate unrealized PnL
            )
            
            # Update unrealized PnL
            for symbol, position in positions.items():
                portfolio.unrealized_pnl += position.unrealized_pnl
                
            return portfolio
            
        except Exception as e:
            logger.error(f"Error retrieving portfolio: {str(e)}")
            return self.portfolio
    
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if unavailable
        """
        # Format symbol
        formatted_symbol = symbol.replace('/', '')
        
        # Check cache
        now = time.time()
        if formatted_symbol in self.price_cache:
            if now - self.price_cache_time[formatted_symbol] < self.price_cache_ttl:
                return self.price_cache[formatted_symbol]
        
        # Check circuit breaker
        if not self.circuit_breaker.is_allowed():
            logger.warning("Circuit breaker is open, API calls blocked")
            return None
            
        # Check rate limits
        if not self._check_rate_limit('query'):
            logger.warning("Rate limit exceeded for price query")
            return None
            
        try:
            # Prepare parameters
            params = {'symbol': formatted_symbol}
            
            # Send query request
            result = await self._api_request('GET', '/v3/ticker/price', params, False)
            if not result or 'price' not in result:
                return None
                
            # Cache and return price
            price = float(result['price'])
            self.price_cache[formatted_symbol] = price
            self.price_cache_time[formatted_symbol] = now
            
            return price
            
        except Exception as e:
            logger.error(f"Error retrieving market price: {str(e)}")
            return None
    
    async def close(self) -> None:
        """
        Close the broker connection and clean up resources.
        """
        try:
            if self.session:
                await self.session.close()
                self.session = None
                
            logger.info(f"Closed {self.name} connection")
            
        except Exception as e:
            logger.error(f"Error closing {self.name} connection: {str(e)}")
    
    async def _api_request(self, method: str, endpoint: str, params: Dict[str, Any], signed: bool) -> Optional[Any]:
        """
        Make an API request to Binance.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether the request needs authentication
            
        Returns:
            Response data or None on error
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.base_url}{endpoint}"
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        if signed:
            # Add timestamp for signed requests
            params['timestamp'] = str(int(time.time() * 1000))
            
            # Create signature
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
        
        try:
            # Different HTTP methods handle parameters differently
            if method == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"API error: {response.status} - {text}")
                        return None
                    return await response.json()
                    
            elif method == 'POST':
                async with self.session.post(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"API error: {response.status} - {text}")
                        return None
                    return await response.json()
                    
            elif method == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"API error: {response.status} - {text}")
                        return None
                    return await response.json()
                    
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"API request error: {str(e)}")
            return None
    
    def _check_rate_limit(self, limit_type: str) -> bool:
        """
        Check if a request exceeds the rate limit.
        
        Args:
            limit_type: Type of rate limit to check ('order' or 'query')
            
        Returns:
            True if request is allowed, False if rate limit would be exceeded
        """
        now = time.time()
        window = self.rate_limits[limit_type]['window']
        limit = self.rate_limits[limit_type]['limit']
        
        # Remove timestamps outside the window
        self.request_timestamps[limit_type] = [
            ts for ts in self.request_timestamps[limit_type] if now - ts < window
        ]
        
        # Check if adding this request would exceed the limit
        if len(self.request_timestamps[limit_type]) >= limit:
            logger.warning(f"Rate limit exceeded for {limit_type}: {limit} requests per {window}s")
            return False
            
        # Add timestamp for this request
        self.request_timestamps[limit_type].append(now)
        return True
    
    def _binance_order_to_order(self, data: Dict[str, Any]) -> Order:
        """
        Convert Binance order data to internal Order model.
        
        Args:
            data: Binance order data
            
        Returns:
            Order object
        """
        # Parse symbol to add / if needed
        symbol = data['symbol']
        for i in range(len(symbol) - 1, 0, -1):
            if symbol[i] in "USDT":
                formatted_symbol = f"{symbol[:i]}/{symbol[i:]}"
                break
        else:
            formatted_symbol = symbol
        
        # Map status
        status_map = {
            'NEW': OrderStatus.OPEN,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        
        # Map side
        side_map = {
            'BUY': OrderSide.BUY,
            'SELL': OrderSide.SELL
        }
        
        # Map order type
        type_map = {
            'LIMIT': OrderType.LIMIT,
            'MARKET': OrderType.MARKET,
            'STOP_LOSS': OrderType.STOP,
            'STOP_LOSS_LIMIT': OrderType.STOP_LIMIT,
            'TAKE_PROFIT': OrderType.TAKE_PROFIT,
            'TAKE_PROFIT_LIMIT': OrderType.TAKE_PROFIT_LIMIT
        }
        
        # Convert time in force
        tif_map = {
            'GTC': TimeInForce.GTC,
            'IOC': TimeInForce.IOC,
            'FOK': TimeInForce.FOK
        }
        
        # Create order object
        return Order(
            order_id=str(data['orderId']),
            symbol=formatted_symbol,
            type=type_map.get(data['type'], OrderType.LIMIT),
            side=side_map.get(data['side'], OrderSide.BUY),
            status=status_map.get(data['status'], OrderStatus.UNKNOWN),
            price=float(data['price']) if data.get('price') else None,
            quantity=float(data['origQty']),
            filled_quantity=float(data['executedQty']),
            remaining_quantity=float(data['origQty']) - float(data['executedQty']),
            time_in_force=tif_map.get(data.get('timeInForce'), TimeInForce.GTC),
            created_at=datetime.fromtimestamp(data['time'] / 1000.0).isoformat() if 'time' in data else None,
            updated_at=datetime.fromtimestamp(data['updateTime'] / 1000.0).isoformat() if 'updateTime' in data else None
        )
    
    def _binance_trade_to_trade(self, data: Dict[str, Any]) -> Trade:
        """
        Convert Binance trade data to internal Trade model.
        
        Args:
            data: Binance trade data
            
        Returns:
            Trade object
        """
        # Parse symbol to add / if needed
        symbol = data['symbol']
        for i in range(len(symbol) - 1, 0, -1):
            if symbol[i] in "USDT":
                formatted_symbol = f"{symbol[:i]}/{symbol[i:]}"
                break
        else:
            formatted_symbol = symbol
        
        # Map side
        side_map = {
            'BUY': OrderSide.BUY,
            'SELL': OrderSide.SELL
        }
        
        # Create trade object
        return Trade(
            trade_id=str(data['id']),
            order_id=str(data['orderId']),
            symbol=formatted_symbol,
            side=side_map.get(data['isBuyer'] and 'BUY' or 'SELL', OrderSide.BUY),
            price=float(data['price']),
            quantity=float(data['qty']),
            commission=float(data['commission']),
            commission_asset=data['commissionAsset'],
            time=datetime.fromtimestamp(data['time'] / 1000.0).isoformat() if 'time' in data else None
        )
