"""Mock exchange connector for testing.

This module provides a simulated exchange connector that can be used for testing
without connecting to a real exchange.
"""

import asyncio
import copy
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Set

from src.common.logging import get_logger
from src.models.order import Order, OrderType, OrderStatus, OrderSide, TimeInForce
from src.execution.exchange.base import BaseExchangeConnector


class MockExchangeConnector(BaseExchangeConnector):
    """Mock exchange connector for testing.
    
    This connector simulates the behavior of a real exchange for testing
    without making actual API calls.
    """
    
    def __init__(
        self,
        exchange_id: str = "mock",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        initial_balances: Optional[Dict[str, float]] = None,
        initial_prices: Optional[Dict[str, float]] = None,
        latency_ms: int = 100,
        fill_probability: float = 0.9,
        price_volatility: float = 0.002,
        is_paper_trading: bool = True
    ):
        """Initialize the mock exchange connector.
        
        Args:
            exchange_id: Unique identifier for this exchange
            api_key: API key (not used in mock)
            api_secret: API secret (not used in mock)
            initial_balances: Initial account balances
            initial_prices: Initial prices for symbols
            latency_ms: Simulated API latency in milliseconds
            fill_probability: Probability of order fills (0.0-1.0)
            price_volatility: Volatility of price movements
            is_paper_trading: Whether this is a paper trading connector
        """
        super().__init__(exchange_id, api_key, api_secret)
        
        # Logger
        self.logger = get_logger("exchange", f"mock_{exchange_id}")
        
        # Simulation parameters
        self.latency_ms = latency_ms
        self.fill_probability = fill_probability
        self.price_volatility = price_volatility
        self.is_paper_trading = is_paper_trading
        
        # Exchange state
        self.balances: Dict[str, Decimal] = {}
        self.prices: Dict[str, Decimal] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}  # Exchange order ID -> order info
        self.trade_history: Dict[str, List[Dict[str, Any]]] = {}  # Symbol -> list of trades
        self.active = False
        
        # Initial data
        if initial_balances:
            for asset, amount in initial_balances.items():
                self.balances[asset] = Decimal(str(amount))
        else:
            # Default balances
            self.balances = {
                "BTC": Decimal("1.0"),
                "ETH": Decimal("10.0"),
                "USDT": Decimal("50000"),
                "BNB": Decimal("100")
            }
        
        if initial_prices:
            for symbol, price in initial_prices.items():
                self.prices[symbol] = Decimal(str(price))
        else:
            # Default prices
            self.prices = {
                "BTC/USDT": Decimal("50000"),
                "ETH/USDT": Decimal("3000"),
                "BNB/USDT": Decimal("400"),
                "SOL/USDT": Decimal("100"),
                "XRP/USDT": Decimal("0.5")
            }
        
        # Exchange info
        self.exchange_info = {
            "name": exchange_id,
            "symbols": list(self.prices.keys()),
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "fees": {
                "maker": Decimal("0.001"),  # 0.1%
                "taker": Decimal("0.001")   # 0.1%
            }
        }
        
        # Price update task
        self.price_update_task = None
    
    async def simulate_latency(self) -> None:
        """Simulate network latency for API calls."""
        if self.latency_ms > 0:
            latency = random.uniform(0.5 * self.latency_ms, 1.5 * self.latency_ms)
            await asyncio.sleep(latency / 1000.0)
    
    async def initialize(self) -> bool:
        """Initialize the mock exchange connector.
        
        Returns:
            bool: True if initialization was successful
        """
        self.logger.info("Initializing mock exchange connector")
        self.simulate_latency()
        
        self.active = True
        
        # Start price update task
        self.price_update_task = asyncio.create_task(self._update_prices())
        
        self.logger.info("Mock exchange connector initialized")
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the mock exchange connector."""
        self.logger.info("Shutting down mock exchange connector")
        
        # Stop price update task
        if self.price_update_task:
            self.price_update_task.cancel()
            try:
                await self.price_update_task
            except asyncio.CancelledError:
                pass
        
        self.active = False
        self.logger.info("Mock exchange connector shut down")
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information and trading rules.
        
        Returns:
            Dict containing exchange information
        """
        self.simulate_latency()
        return copy.deepcopy(self.exchange_info)
    
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """Get account balances for all assets.
        
        Returns:
            Dict mapping asset symbol to balance amount
        """
        self.simulate_latency()
        return copy.deepcopy(self.balances)
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dict containing ticker information
        """
        self.simulate_latency()
        
        if symbol not in self.prices:
            return {}
        
        # Generate some random bid/ask spread
        price = self.prices[symbol]
        spread = price * Decimal("0.0004")  # 0.04% spread
        bid = price - spread / Decimal("2")
        ask = price + spread / Decimal("2")
        
        return {
            "symbol": symbol,
            "price": price,
            "bid": bid,
            "ask": ask,
            "volume": Decimal(str(random.uniform(100, 1000))),
            "timestamp": datetime.now().timestamp()
        }
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            limit: Maximum number of bids/asks to return
            
        Returns:
            Dict containing order book data
        """
        self.simulate_latency()
        
        if symbol not in self.prices:
            return {"bids": [], "asks": []}
        
        price = self.prices[symbol]
        orderbook = {
            "symbol": symbol,
            "timestamp": datetime.now().timestamp(),
            "bids": [],
            "asks": []
        }
        
        # Generate realistic looking order book
        spread = price * Decimal("0.0004")  # 0.04% spread
        mid_price = price
        bid_start = mid_price - spread / Decimal("2")
        ask_start = mid_price + spread / Decimal("2")
        
        # Generate bids (sorted by price descending)
        for i in range(min(limit, 20)):
            price_level = bid_start - (bid_start * Decimal(str(0.0001 * i)))
            size = Decimal(str(random.uniform(0.1, 10.0)))
            orderbook["bids"].append([float(price_level), float(size)])
        
        # Generate asks (sorted by price ascending)
        for i in range(min(limit, 20)):
            price_level = ask_start + (ask_start * Decimal(str(0.0001 * i)))
            size = Decimal(str(random.uniform(0.1, 10.0)))
            orderbook["asks"].append([float(price_level), float(size)])
        
        return orderbook
    
    async def create_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submit an order to the exchange.
        
        Args:
            order: Order object containing order details
            
        Returns:
            Tuple of (success, exchange_order_id, error_message)
        """
        self.simulate_latency()
        
        symbol = order.symbol
        
        # Validation checks
        if symbol not in self.prices:
            return False, None, f"Symbol {symbol} not found"
        
        # Generate an exchange order ID
        exchange_order_id = f"mock-{random.randint(100000, 999999)}"
        
        # Create an exchange order
        exchange_order = {
            "id": exchange_order_id,
            "client_order_id": order.id,
            "symbol": symbol,
            "type": order.order_type.value,
            "side": order.side.value,
            "quantity": float(order.quantity),
            "price": float(order.price) if order.price is not None else None,
            "stop_price": float(order.stop_price) if order.stop_price is not None else None,
            "time_in_force": order.time_in_force.value,
            "status": "open",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "filled_quantity": 0.0,
            "average_fill_price": None,
            "fees": {},
            "is_post_only": order.is_post_only,
            "is_reduce_only": order.is_reduce_only
        }
        
        # Store the order
        self.orders[exchange_order_id] = exchange_order
        
        # For market orders, simulate immediate fill
        if order.order_type == OrderType.MARKET:
            await self._process_market_order_fill(exchange_order_id)
        else:
            # For limit orders, start a task to check for fills
            asyncio.create_task(self._monitor_limit_order(exchange_order_id))
        
        return True, exchange_order_id, None
    
    async def cancel_order(self, order_id: str, symbol: str) -> Tuple[bool, Optional[str]]:
        """Cancel an existing order.
        
        Args:
            order_id: Exchange order ID to cancel
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (success, error_message)
        """
        self.simulate_latency()
        
        if order_id not in self.orders:
            return False, "Order not found"
        
        order = self.orders[order_id]
        
        if order["status"] not in ["open", "partially_filled"]:
            return False, f"Order status is {order['status']}, cannot cancel"
        
        # Cancel the order
        order["status"] = "cancelled"
        order["updated_at"] = datetime.now()
        
        return True, None
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific order.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            
        Returns:
            Dict containing order information or None if not found
        """
        self.simulate_latency()
        
        if order_id not in self.orders:
            return None
        
        return copy.deepcopy(self.orders[order_id])
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders.
        
        Args:
            symbol: Optional trading pair symbol to filter by
            
        Returns:
            List of dictionaries containing order information
        """
        self.simulate_latency()
        
        result = []
        for order_id, order in self.orders.items():
            if order["status"] in ["open", "partially_filled"]:
                if symbol is None or order["symbol"] = = symbol:
                    result.append(copy.deepcopy(order))
        
        return result
    
    async def get_trade_history(
        self, 
        symbol: str, 
        limit: int = 100,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get trade history for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades to return
            from_time: Optional start time for trades
            to_time: Optional end time for trades
            
        Returns:
            List of dictionaries containing trade information
        """
        self.simulate_latency()
        
        if symbol not in self.trade_history:
            return []
        
        # Filter trades by time if provided
        trades = self.trade_history[symbol]
        if from_time:
            trades = [t for t in trades if t["timestamp"] >= from_time]
        if to_time:
            trades = [t for t in trades if t["timestamp"] <= to_time]
        
        # Return most recent trades first, up to the limit
        return sorted(trades, key=lambda t: t["timestamp"], reverse=True)[:limit]
    
    async def _update_prices(self) -> None:
        """Periodically update prices to simulate market movement."""
        try:
            while self.active:
                # Update each price
                for symbol in list(self.prices.keys()):
                    current_price = self.prices[symbol]
                    
                    # Calculate a random price change
                    change_pct = Decimal(str(random.uniform(-self.price_volatility, self.price_volatility)))
                    new_price = current_price * (Decimal("1") + change_pct)
                    
                    # Update the price
                    self.prices[symbol] = new_price
                
                # Wait before next update
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
    
    async def _process_market_order_fill(self, order_id: str) -> None:
        """Process an immediate fill for a market order."""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        symbol = order["symbol"]
        
        # Get current price
        if symbol not in self.prices:
            return
        
        current_price = self.prices[symbol]
        
        # Apply a small slippage for market orders
        if order["side"] = = "buy":
            fill_price = current_price * Decimal("1.001")  # 0.1% slippage
        else:
            fill_price = current_price * Decimal("0.999")  # 0.1% slippage
        
        # Update the order
        order["status"] = "filled"
        order["filled_quantity"] = order["quantity"]
        order["average_fill_price"] = float(fill_price)
        order["updated_at"] = datetime.now()
        
        # Calculate fees
        fee_percentage = self.exchange_info["fees"]["taker"]
        fee_amount = Decimal(str(order["quantity"])) * fill_price * fee_percentage
        
        # Add fee to the order
        quote_currency = symbol.split("/")[1]
        order["fees"] = {quote_currency: float(fee_amount)}
        
        # Record the trade
        self._record_trade(symbol, order_id, order["side"], order["quantity"], float(fill_price), datetime.now())
        
        # Update balances
        self._update_balances_for_order(order)
    
    async def _monitor_limit_order(self, order_id: str) -> None:
        """Monitor a limit order for potential fills."""
        try:
            # Check the order every second
            while self.active and order_id in self.orders:
                order = self.orders[order_id]
                
                # Skip if the order is no longer active
                if order["status"] not in ["open", "partially_filled"]:
                    break
                
                symbol = order["symbol"]
                current_price = self.prices.get(symbol)
                
                if current_price is None:
                    await asyncio.sleep(1)
                    continue
                
                # Check if the limit price is reached
                limit_price = Decimal(str(order["price"])) if order["price"] is not None else None
                
                if limit_price is not None:
                    # For buy orders, fill when price <= limit price
                    # For sell orders, fill when price >= limit price
                    if (order["side"] = = "buy" and current_price <= limit_price) or \
                       (order["side"] = = "sell" and current_price >= limit_price):
                        
                        # Determine if the order should fill based on probability
                        if random.random() <= self.fill_probability:
                            # Fill the order
                            await self._fill_limit_order(order_id, current_price)
                            break
                
                # Wait before checking again
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
    
    async def _fill_limit_order(self, order_id: str, price: Decimal) -> None:
        """Fill a limit order at the specified price."""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        symbol = order["symbol"]
        
        # Determine fill price (use limit price for limit orders)
        if order["price"] is not None:
            fill_price = Decimal(str(order["price"]))
        else:
            fill_price = price
        
        # Update the order
        order["status"] = "filled"
        order["filled_quantity"] = order["quantity"]
        order["average_fill_price"] = float(fill_price)
        order["updated_at"] = datetime.now()
        
        # Calculate fees
        fee_percentage = self.exchange_info["fees"]["maker"]
        fee_amount = Decimal(str(order["quantity"])) * fill_price * fee_percentage
        
        # Add fee to the order
        quote_currency = symbol.split("/")[1]
        order["fees"] = {quote_currency: float(fee_amount)}
        
        # Record the trade
        self._record_trade(symbol, order_id, order["side"], order["quantity"], float(fill_price), datetime.now())
        
        # Update balances
        self._update_balances_for_order(order)
    
    def _record_trade(self, symbol: str, order_id: str, side: str, quantity: float, price: float, timestamp: datetime) -> None:
        """Record a trade in the trade history."""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        
        trade = {
            "id": f"trade-{random.randint(1000000, 9999999)}",
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "fee": 0.0,
            "fee_currency": symbol.split("/")[1]
        }
        
        self.trade_history[symbol].append(trade)
    
    def _update_balances_for_order(self, order: Dict[str, Any]) -> None:
        """Update account balances based on an order fill."""
        if not self.is_paper_trading:
            return  # Only update balances in paper trading mode
        
        symbol = order["symbol"]
        base_currency, quote_currency = symbol.split("/")
        
        quantity = Decimal(str(order["filled_quantity"]))
        price = Decimal(str(order["average_fill_price"]))
        side = order["side"]
        
        # Calculate trade value
        trade_value = quantity * price
        
        # Update balances based on the trade
        if side == "buy":
            # Add base currency, subtract quote currency
            self.balances[base_currency] = self.balances.get(base_currency, Decimal("0")) + quantity
            self.balances[quote_currency] = self.balances.get(quote_currency, Decimal("0")) - trade_value
        else:  # sell
            # Subtract base currency, add quote currency
            self.balances[base_currency] = self.balances.get(base_currency, Decimal("0")) - quantity
            self.balances[quote_currency] = self.balances.get(quote_currency, Decimal("0")) + trade_value
        
        # Subtract fees
        for fee_currency, fee_amount in order["fees"].items():
            fee_decimal = Decimal(str(fee_amount))
            self.balances[fee_currency] = self.balances.get(fee_currency, Decimal("0")) - fee_decimal 