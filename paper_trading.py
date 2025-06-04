#!/usr/bin/env python
"""
Paper Trading System for Flash Trading

This module provides a paper trading implementation that simulates trading
with real market data but without using real funds. It's designed for
testing flash trading strategies with zero-fee pairs (BTCUSDC, ETHUSDC).
"""

import time
import json
import logging
import os
import uuid
from datetime import datetime
from threading import RLock
from optimized_mexc_client import OptimizedMEXCClient
from flash_trading_config import FlashTradingConfig

# Import error handling utilities
from error_handling_utils import (
    safe_get, safe_get_nested, safe_list_access,
    validate_api_response, log_exception,
    parse_float_safely, parse_int_safely,
    handle_api_error
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("paper_trading")

class PaperTradingSystem:
    """Paper trading system for flash trading strategies"""
    
    def __init__(self, client=None, config=None):
        """Initialize paper trading system with API client and configuration"""
        self.client = client or OptimizedMEXCClient()
        self.config = config or FlashTradingConfig()
        
        # Load paper trading configuration
        self.paper_config = self.config.config["paper_trading"]
        
        # Initialize balances
        self.balances = self.paper_config["initial_balance"].copy()
        
        # Initialize orders and trades
        self.open_orders = {}
        self.order_history = []
        self.trade_history = []
        
        # Market data cache
        self.market_data = {}
        
        # Lock for thread safety
        self.lock = RLock()
        
        # Load state if enabled
        if self.paper_config["persist_state"]:
            self._load_state()
    
    def _load_state(self):
        """Load paper trading state from file"""
        state_file = self.paper_config["state_file"]
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore state
                if "balances" in state:
                    self.balances = state["balances"]
                if "open_orders" in state:
                    self.open_orders = state["open_orders"]
                if "order_history" in state:
                    self.order_history = state["order_history"]
                if "trade_history" in state:
                    self.trade_history = state["trade_history"]
                
                logger.info(f"Paper trading state loaded from {state_file}")
            except Exception as e:
                logger.error(f"Error loading paper trading state: {str(e)}")
    
    def _save_state(self):
        """Save paper trading state to file"""
        if not self.paper_config["persist_state"]:
            return
        
        state_file = self.paper_config["state_file"]
        try:
            state = {
                "timestamp": int(time.time() * 1000),
                "balances": self.balances,
                "open_orders": self.open_orders,
                "order_history": self.order_history,
                "trade_history": self.trade_history
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Paper trading state saved to {state_file}")
        except Exception as e:
            logger.error(f"Error saving paper trading state: {str(e)}")
    
    def get_balance(self, asset):
        """Get paper trading balance for an asset"""
        with self.lock:
            return self.balances.get(asset, 0.0)
    
    def get_all_balances(self):
        """Get all paper trading balances"""
        with self.lock:
            return self.balances.copy()
            
    def get_account(self):
        """Get account information (compatibility method for extended_testing.py)"""
        return self.get_account_info()
    
    def _update_market_data(self, symbol):
        """Update market data for a symbol"""
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                logger.error(f"Invalid symbol for market data update: {symbol}")
                return False
                
            # Get order book with robust error handling
            order_book = self.client.get_order_book(symbol, limit=10)
            
            # Validate order book data
            if not order_book:
                logger.warning(f"Empty order book response for {symbol}")
                return False
                
            if not isinstance(order_book, dict):
                logger.error(f"Invalid order book response type for {symbol}: {type(order_book)}")
                return False
                
            if 'bids' not in order_book or 'asks' not in order_book:
                logger.error(f"Missing bids or asks in order book for {symbol}")
                return False
                
            if not order_book["bids"] or not order_book["asks"]:
                logger.warning(f"Empty bids or asks in order book for {symbol}")
                return False
                
            # Store validated market data
            self.market_data[symbol] = {
                "timestamp": int(time.time() * 1000),
                "bids": order_book["bids"],
                "asks": order_book["asks"]
            }
            return True
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {str(e)}")
            return False
    
    def _get_current_price(self, symbol, side):
        """Get current price from market data"""
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                logger.error(f"Invalid symbol for price lookup: {symbol}")
                return None
                
            if not side or side not in ["BUY", "SELL"]:
                logger.error(f"Invalid side for price lookup: {side}")
                return None
            
            # Update market data if needed with validation
            if symbol not in self.market_data or time.time() * 1000 - self.market_data[symbol].get("timestamp", 0) > 5000:
                if not self._update_market_data(symbol):
                    logger.warning(f"Failed to update market data for {symbol}")
                    return None
            
            # Validate market data structure
            if not isinstance(self.market_data.get(symbol), dict):
                logger.error(f"Invalid market data structure for {symbol}: {type(self.market_data.get(symbol))}")
                return None
                
            # Validate bids and asks exist and are properly structured
            if side == "BUY":
                asks = self.market_data[symbol].get("asks")
                if not asks or not isinstance(asks, list) or len(asks) == 0:
                    logger.error(f"Invalid or empty asks data for {symbol}")
                    return None
                    
                ask_price = asks[0]
                if not isinstance(ask_price, list) or len(ask_price) < 1:
                    logger.error(f"Invalid ask price format for {symbol}: {ask_price}")
                    return None
                    
                try:
                    # Buy at ask price with validation
                    return float(asks[0][0])
                except (ValueError, TypeError, IndexError) as e:
                    logger.error(f"Error parsing ask price for {symbol}: {str(e)}")
                    return None
            else:
                bids = self.market_data[symbol].get("bids")
                if not bids or not isinstance(bids, list) or len(bids) == 0:
                    logger.error(f"Invalid or empty bids data for {symbol}")
                    return None
                    
                bid_price = bids[0]
                if not isinstance(bid_price, list) or len(bid_price) < 1:
                    logger.error(f"Invalid bid price format for {symbol}: {bid_price}")
                    return None
                    
                try:
                    # Sell at bid price with validation
                    return float(bids[0][0])
                except (ValueError, TypeError, IndexError) as e:
                    logger.error(f"Error parsing bid price for {symbol}: {str(e)}")
                    return None
        except Exception as e:
            logger.error(f"Unexpected error getting current price for {symbol}: {str(e)}")
            return None
    
    def _apply_slippage(self, price, side):
        """Apply simulated slippage to price"""
        try:
            # Validate inputs
            if price is None or not isinstance(price, (int, float)) or price <= 0:
                logger.error(f"Invalid price for slippage calculation: {price}")
                return price
                
            if not side or side not in ["BUY", "SELL"]:
                logger.error(f"Invalid side for slippage calculation: {side}")
                return price
                
            # Validate paper config with safe access
            if not isinstance(self.paper_config, dict):
                logger.error(f"Invalid paper config type: {type(self.paper_config)}")
                return price
                
            simulate_slippage = self.paper_config.get("simulate_slippage", False)
            if not simulate_slippage:
                return price
            
            # Get slippage factor with validation
            slippage_bps = self.paper_config.get("slippage_bps", 0)
            if not isinstance(slippage_bps, (int, float)):
                logger.warning(f"Invalid slippage_bps type: {type(slippage_bps)}, using 0")
                slippage_bps = 0
                
            slippage_factor = float(slippage_bps) / 10000.0
            
            # Apply slippage based on side
            if side == "BUY":
                # Higher price for buys
                return price * (1 + slippage_factor)
            else:
                # Lower price for sells
                return price * (1 - slippage_factor)
        except Exception as e:
            logger.error(f"Error applying slippage: {str(e)}")
            return price  # Return original price on error
    
    def place_order(self, symbol, side, order_type, quantity, price=None, time_in_force="GTC"):
        """Place a paper trading order"""
        with self.lock:
            try:
                # Validate basic inputs
                if not symbol or not isinstance(symbol, str):
                    logger.error(f"Invalid symbol: {symbol}")
                    return None
                    
                if not side or side not in ["BUY", "SELL"]:
                    logger.error(f"Invalid side: {side}")
                    return None
                    
                if not order_type or order_type not in ["LIMIT", "MARKET"]:
                    logger.error(f"Invalid order type: {order_type}")
                    return None
                    
                if not quantity or not isinstance(quantity, (int, float, str)):
                    logger.error(f"Invalid quantity: {quantity}")
                    return None
                    
                if order_type == "LIMIT" and (price is None or not isinstance(price, (int, float, str))):
                    logger.error(f"Price required for LIMIT orders: {price}")
                    return None
                    
                if not time_in_force or time_in_force not in ["GTC", "IOC", "FOK"]:
                    logger.warning(f"Invalid time_in_force: {time_in_force}, using GTC")
                    time_in_force = "GTC"
                
                # Validate symbol with robust error handling
                try:
                    pair_config = self.config.get_trading_pair_config(symbol)
                    if not pair_config:
                        logger.warning(f"Invalid symbol: {symbol}")
                        return None
                        
                    if not isinstance(pair_config, dict):
                        logger.error(f"Invalid pair configuration format: {type(pair_config)}")
                        return None
                except Exception as e:
                    logger.error(f"Error getting pair configuration for {symbol}: {str(e)}")
                    return None
                
                # Extract assets with validation
                try:
                    base_asset = pair_config.get("base_asset")
                    quote_asset = pair_config.get("quote_asset")
                    
                    if not base_asset or not quote_asset:
                        logger.error(f"Missing base or quote asset in pair config: {pair_config}")
                        return None
                except Exception as e:
                    logger.error(f"Error extracting assets from pair config: {str(e)}")
                    return None
                
                # Validate quantity with robust error handling
                try:
                    quantity = float(quantity)
                    min_size = pair_config.get("min_order_size", 0.0001)
                    
                    if quantity <= 0:
                        logger.error(f"Order quantity must be positive: {quantity}")
                        return None
                        
                    if quantity < min_size:
                        logger.warning(f"Order quantity {quantity} below minimum {min_size}")
                        return None
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing quantity: {str(e)}")
                    return None
                
                # Get current price if not provided with validation
                if price is None:
                    if order_type == "MARKET":
                        price = self._get_current_price(symbol, side)
                        if price is None:
                            logger.error(f"Failed to get current price for {symbol}")
                            return None
                    else:
                        logger.error("Price is required for LIMIT orders")
                        return None
                else:
                    try:
                        price = float(price)
                        if price <= 0:
                            logger.error(f"Price must be positive: {price}")
                            return None
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error parsing price: {str(e)}")
                        return None

                # Apply slippage for market orders with validation
                if order_type == "MARKET" and price is not None:
                    price = self._apply_slippage(price, side)
                
                # Check balance with robust validation
                try:
                    if side == "BUY":
                        # Check quote asset balance with validation
                        # Validate price and quantity
                        if price is None or not isinstance(price, (int, float)) or price <= 0:
                            logger.error(f"Invalid price for balance check: {price}")
                            return None
                            
                        # Calculate required balance with validation
                        try:
                            required_balance = price * quantity
                        except (TypeError, ValueError) as e:
                            logger.error(f"Error calculating required balance: {str(e)}")
                            return None
                        
                        # Check quote asset balance with safe access
                        quote_balance = self.balances.get(quote_asset, 0)
                        if not isinstance(quote_balance, (int, float)):
                            logger.error(f"Invalid balance type for {quote_asset}: {type(quote_balance)}")
                            return None
                            
                        if quote_balance < required_balance:
                            logger.warning(f"Insufficient {quote_asset} balance for order: {quote_balance} < {required_balance}")
                            return None
                    else:
                        # Check base asset balance with validation
                        base_balance = self.balances.get(base_asset, 0)
                        if not isinstance(base_balance, (int, float)):
                            logger.error(f"Invalid balance type for {base_asset}: {type(base_balance)}")
                            return None
                            
                        if base_balance < quantity:
                            logger.warning(f"Insufficient {base_asset} balance for order: {base_balance} < {quantity}")
                            return None
                except Exception as e:
                    logger.error(f"Error checking balance for order: {str(e)}")
                    return None
                
                # Create order with validation
                try:
                    order_id = str(uuid.uuid4())
                    timestamp = int(time.time() * 1000)
                    
                    # Create order object with all required fields
                    order = {
                        "orderId": order_id,
                        "symbol": symbol,
                        "side": side,
                        "type": order_type,
                        "timeInForce": time_in_force,
                        "quantity": quantity,
                        "price": price,
                        "status": "NEW",
                        "timestamp": timestamp,
                        "executedQty": 0.0,
                        "cummulativeQuoteQty": 0.0,
                        "fills": []
                    }
                    
                    # Update balances based on order
                    if side == "BUY":
                        # Decrease quote asset balance
                        self.balances[quote_asset] = self.balances.get(quote_asset, 0) - (price * quantity)
                    else:
                        # Decrease base asset balance
                        self.balances[base_asset] = self.balances.get(base_asset, 0) - quantity
                    
                    # Add to open orders
                    self.open_orders[order_id] = order
                    
                    # Add to order history
                    self.order_history.append(order.copy())
                    
                    # Process order immediately for market orders or IOC limit orders with validation
                    if order_type == "MARKET" or time_in_force == "IOC":
                        try:
                            self._process_order(order_id)
                        except Exception as e:
                            logger.error(f"Error processing immediate order: {str(e)}")
                            # Order is still valid even if immediate processing fails
                    
                    # Save state
                    try:
                        self._save_state()
                    except Exception as e:
                        logger.error(f"Error saving state: {str(e)}")
                    
                    # Log order
                    if self.paper_config.get("log_trades", False):
                        logger.info(f"Paper order placed: {side} {quantity} {symbol} @ {price}")
                    
                    return order
                except Exception as e:
                    logger.error(f"Error creating order: {str(e)}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error placing order: {str(e)}")
                return None
    
    def _process_order(self, order_id):
        """Process an open order"""
        try:
            if order_id not in self.open_orders:
                return False
            
            order = self.open_orders[order_id]
            symbol = order["symbol"]
            side = order["side"]
            order_type = order["type"]
            quantity = order["quantity"]
            price = order["price"]
            
            # Get current market price
            market_price = self._get_current_price(symbol, side)
            if market_price is None:
                return False
            
            # Check if order can be filled
            can_fill = False
            fill_price = price
            
            if order_type == "MARKET":
                can_fill = True
                fill_price = market_price
            elif side == "BUY" and market_price <= price:
                can_fill = True
                fill_price = price
            elif side == "SELL" and market_price >= price:
                can_fill = True
                fill_price = price
            
            if not can_fill:
                # Cancel order if it's IOC
                if order["timeInForce"] == "IOC":
                    self._cancel_order(order_id)
                return False
            
            # Determine fill quantity
            fill_quantity = quantity
            if self.paper_config["simulate_partial_fills"]:
                import random
                if random.random() < self.paper_config["partial_fill_probability"]:
                    fill_quantity = quantity * random.uniform(0.1, 0.9)
                    fill_quantity = round(fill_quantity, 8)
            
            # Extract assets
            pair_config = self.config.get_trading_pair_config(symbol)
            base_asset = pair_config["base_asset"]
            quote_asset = pair_config["quote_asset"]
            
            # Update order
            order["status"] = "FILLED"
            order["executedQty"] = fill_quantity
            order["cummulativeQuoteQty"] = fill_quantity * fill_price
            
            # Create fill
            fill = {
                "price": fill_price,
                "qty": fill_quantity,
                "commission": 0.0,  # Zero fee
                "commissionAsset": quote_asset,
                "tradeId": str(uuid.uuid4())
            }
            
            order["fills"].append(fill)
            
            # Update balances
            if side == "BUY":
                # Increase base asset
                self.balances[base_asset] = self.balances.get(base_asset, 0) + fill_quantity
                
                # Refund unused quote asset if partial fill
                if fill_quantity < quantity:
                    refund = (quantity - fill_quantity) * price
                    self.balances[quote_asset] = self.balances.get(quote_asset, 0) + refund
            else:
                # Increase quote asset
                self.balances[quote_asset] = self.balances.get(quote_asset, 0) + (fill_quantity * fill_price)
                
                # Refund unused base asset if partial fill
                if fill_quantity < quantity:
                    refund = quantity - fill_quantity
                    self.balances[base_asset] = self.balances.get(base_asset, 0) + refund
            
            # Create trade record
            trade = {
                "symbol": symbol,
                "id": fill["tradeId"],
                "orderId": order_id,
                "side": side,
                "price": fill_price,
                "qty": fill_quantity,
                "quoteQty": fill_quantity * fill_price,
                "commission": 0.0,
                "commissionAsset": quote_asset,
                "time": int(time.time() * 1000)
            }
            
            # Add to trade history
            self.trade_history.append(trade)
            
            # Remove from open orders
            del self.open_orders[order_id]
            
            # Update order history
            for i, hist_order in enumerate(self.order_history):
                if hist_order["orderId"] == order_id:
                    self.order_history[i] = order.copy()
                    break
            
            # Save state
            self._save_state()
            
            # Log trade
            if self.paper_config["log_trades"]:
                logger.info(f"Paper order filled: {side} {fill_quantity} {symbol} @ {fill_price}")
            
            return True
        except Exception as e:
            logger.error(f"Error processing order {order_id}: {str(e)}")
            return False
    
    def _cancel_order(self, order_id):
        """Cancel an open order"""
        try:
            if order_id not in self.open_orders:
                return False
            
            order = self.open_orders[order_id]
            symbol = order["symbol"]
            side = order["side"]
            quantity = order["quantity"]
            price = order["price"]
            
            # Extract assets
            pair_config = self.config.get_trading_pair_config(symbol)
            base_asset = pair_config["base_asset"]
            quote_asset = pair_config["quote_asset"]
            
            # Update order
            order["status"] = "CANCELED"
            
            # Refund balances
            if side == "BUY":
                # Refund quote asset
                refund = quantity * price
                self.balances[quote_asset] = self.balances.get(quote_asset, 0) + refund
            else:
                # Refund base asset
                self.balances[base_asset] = self.balances.get(base_asset, 0) + quantity
            
            # Remove from open orders
            del self.open_orders[order_id]
            
            # Update order history
            for i, hist_order in enumerate(self.order_history):
                if hist_order["orderId"] == order_id:
                    self.order_history[i] = order.copy()
                    break
            
            # Save state
            self._save_state()
            
            # Log cancellation
            if self.paper_config["log_trades"]:
                logger.info(f"Paper order canceled: {order_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    def get_order(self, order_id):
        """Get order by ID"""
        with self.lock:
            # Check open orders
            if order_id in self.open_orders:
                return self.open_orders[order_id].copy()
            
            # Check order history
            for order in self.order_history:
                if order["orderId"] == order_id:
                    return order.copy()
            
            return None
    
    def get_open_orders(self, symbol=None):
        """Get all open orders, optionally filtered by symbol"""
        with self.lock:
            if symbol:
                return [order.copy() for order in self.open_orders.values() if order["symbol"] == symbol]
            else:
                return [order.copy() for order in self.open_orders.values()]
    
    def cancel_all_orders(self, symbol=None):
        """Cancel all open orders, optionally filtered by symbol"""
        with self.lock:
            canceled = 0
            order_ids = list(self.open_orders.keys())
            
            for order_id in order_ids:
                order = self.open_orders[order_id]
                if symbol is None or order["symbol"] == symbol:
                    if self._cancel_order(order_id):
                        canceled += 1
            
            return canceled
    
    def process_open_orders(self):
        """Process all open orders"""
        with self.lock:
            processed = 0
            order_ids = list(self.open_orders.keys())
            
            for order_id in order_ids:
                if order_id in self.open_orders and self._process_order(order_id):
                    processed += 1
            
            return processed
    
    def get_account_info(self):
        """Get account information"""
        with self.lock:
            return {
                "balances": self.balances.copy(),
                "open_orders_count": len(self.open_orders),
                "total_orders": len(self.order_history),
                "total_trades": len(self.trade_history)
            }
            
    def get_account(self):
        """Get account information (alias for get_account_info for API compatibility)"""
        return self.get_account_info()
    
    def reset(self):
        """Reset paper trading to initial state"""
        with self.lock:
            # Reset balances
            self.balances = self.paper_config["initial_balance"].copy()
            
            # Reset orders and trades
            self.open_orders = {}
            self.order_history = []
            self.trade_history = []
            
            # Reset market data
            self.market_data = {}
            
            # Save state
            self._save_state()
            
            logger.info("Paper trading reset to initial state")
            return True
