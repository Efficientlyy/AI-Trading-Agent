#!/usr/bin/env python
"""
Enhanced Paper Trading System with Order Creation

This module extends the paper trading system with proper order creation,
management, and notification integration.
"""

import os
import sys
import json
import time
import logging
import threading
import uuid
from queue import Queue
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from enhanced_logging_fixed import EnhancedLogger
from optimized_mexc_client import OptimizedMEXCClient

# Initialize enhanced logger
logger = EnhancedLogger("paper_trading_fixed")

class PaperTradingSystem:
    """Enhanced paper trading system with proper order creation and management"""
    
    def __init__(self, client=None, config=None):
        """Initialize paper trading system
        
        Args:
            client: Exchange client (optional)
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logger
        self.client = client or OptimizedMEXCClient()
        
        # Initialize state
        self.balance = self.config.get('initial_balance', {
            'USDC': 10000.0,
            'BTC': 0.1,
            'ETH': 1.0,
            'SOL': 10.0
        })
        
        self.orders = {}
        self.trades = []
        self.positions = {}
        
        # Order book cache
        self.order_books = {}
        
        # Last prices
        self.last_prices = {}
        
        # Running flag
        self.running = False
        
        # Order processing queue
        self.order_queue = Queue()
        
        # Notification callback
        self.notification_callback = None
        
        # Initialize
        self.initialize()
        
        self.logger.system.info("Fixed paper trading system initialized")
    
    def initialize(self):
        """Initialize paper trading system"""
        # Initialize positions
        for symbol in ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']:
            base_asset = symbol.replace('USDC', '')
            self.positions[symbol] = {
                'symbol': symbol,
                'base_asset': base_asset,
                'quote_asset': 'USDC',
                'base_quantity': self.balance.get(base_asset, 0.0),
                'quote_quantity': 0.0,
                'entry_price': 0.0,
                'current_price': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'timestamp': int(time.time() * 1000)
            }
        
        # Initialize last prices
        self.update_market_data()
    
    def set_notification_callback(self, callback):
        """Set notification callback
        
        Args:
            callback: Notification callback function
        """
        self.notification_callback = callback
        self.logger.system.info("Notification callback set")
    
    def notify(self, notification_type, data):
        """Send notification
        
        Args:
            notification_type: Type of notification
            data: Notification data
        """
        if self.notification_callback:
            self.notification_callback(notification_type, data)
    
    def start(self):
        """Start paper trading system"""
        self.logger.system.info("Starting paper trading system")
        
        try:
            # Start order processing thread
            self.running = True
            self.order_thread = threading.Thread(target=self.process_orders)
            self.order_thread.daemon = True
            self.order_thread.start()
            
            self.logger.system.info("Paper trading system started")
        except Exception as e:
            self.logger.log_error("Error starting paper trading system", component="paper_trading")
            raise
    
    def stop(self):
        """Stop paper trading system"""
        self.logger.system.info("Stopping paper trading system")
        
        try:
            # Stop order processing
            self.running = False
            
            # Wait for thread to terminate
            if hasattr(self, 'order_thread') and self.order_thread.is_alive():
                self.order_thread.join(timeout=5.0)
            
            self.logger.system.info("Paper trading system stopped")
        except Exception as e:
            self.logger.log_error("Error stopping paper trading system", component="paper_trading")
            raise
    
    def process_orders(self):
        """Process orders from the queue"""
        self.logger.system.info("Order processing thread started")
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Send heartbeat log periodically
                current_time = time.time()
                if current_time - last_heartbeat > 60:
                    self.logger.system.debug("Order processing thread heartbeat")
                    last_heartbeat = current_time
                
                # Process orders from queue
                if not self.order_queue.empty():
                    order_action = self.order_queue.get(timeout=0.1)
                    self.execute_order_action(order_action)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.logger.log_error("Error in order processing thread", component="paper_trading")
                time.sleep(1)  # Prevent tight loop on persistent errors
        
        self.logger.system.info("Order processing thread stopped")
    
    def execute_order_action(self, order_action):
        """Execute an order action
        
        Args:
            order_action: Order action dictionary
        """
        try:
            # Extract action data
            action_type = order_action.get('type', 'unknown')
            order_id = order_action.get('order_id')
            
            if action_type == 'create':
                # Create order
                order = order_action.get('order', {})
                self.create_order_internal(order)
            elif action_type == 'cancel':
                # Cancel order
                reason = order_action.get('reason', 'User requested')
                self.cancel_order_internal(order_id, reason)
            elif action_type == 'fill':
                # Fill order
                price = order_action.get('price')
                self.fill_order_internal(order_id, price)
            else:
                self.logger.system.warning(f"Unknown order action type: {action_type}")
        except Exception as e:
            self.logger.log_error(f"Error executing order action: {str(e)}", component="paper_trading")
    
    def create_order(self, symbol, side, order_type, quantity, price=None):
        """Create a new order
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT or MARKET)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            
        Returns:
            str: Order ID
        """
        try:
            # Generate order ID
            order_id = f"ORD-{uuid.uuid4()}"
            
            # Create order
            order = {
                'orderId': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': float(quantity),
                'price': float(price) if price is not None else None,
                'status': 'NEW',
                'timestamp': int(time.time() * 1000)
            }
            
            # Add to order queue
            self.order_queue.put({
                'type': 'create',
                'order': order
            })
            
            self.logger.system.info(f"Order {order_id} queued for creation")
            
            return order_id
        except Exception as e:
            self.logger.log_error(f"Error creating order: {str(e)}", component="paper_trading")
            return None
    
    def create_order_internal(self, order):
        """Create order internally
        
        Args:
            order: Order dictionary
        """
        try:
            # Extract order data
            order_id = order.get('orderId')
            symbol = order.get('symbol')
            side = order.get('side')
            order_type = order.get('type')
            quantity = order.get('quantity')
            price = order.get('price')
            
            # Validate order
            if not self.validate_order(symbol, side, order_type, quantity, price):
                self.logger.system.warning(f"Order {order_id} validation failed")
                return
            
            # Add order to orders dictionary
            self.orders[order_id] = order
            
            # Log order creation
            self.logger.system.info(f"Order {order_id} created: {side} {quantity} {symbol} at {price}")
            
            # Send notification
            self.notify('order_created', order)
            
            # Check if order can be filled immediately
            if order_type == 'MARKET':
                # Fill market order immediately
                current_price = self.get_current_price(symbol)
                self.fill_order_internal(order_id, current_price)
        except Exception as e:
            self.logger.log_error(f"Error creating order internally: {str(e)}", component="paper_trading")
    
    def validate_order(self, symbol, side, order_type, quantity, price):
        """Validate order parameters
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT or MARKET)
            quantity: Order quantity
            price: Order price
            
        Returns:
            bool: Whether the order is valid
        """
        try:
            # Check symbol
            if symbol not in ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']:
                self.logger.system.warning(f"Invalid symbol: {symbol}")
                return False
            
            # Check side
            if side not in ['BUY', 'SELL']:
                self.logger.system.warning(f"Invalid side: {side}")
                return False
            
            # Check order type
            if order_type not in ['LIMIT', 'MARKET']:
                self.logger.system.warning(f"Invalid order type: {order_type}")
                return False
            
            # Check quantity
            if quantity <= 0:
                self.logger.system.warning(f"Invalid quantity: {quantity}")
                return False
            
            # Check price for LIMIT orders
            if order_type == 'LIMIT' and (price is None or price <= 0):
                self.logger.system.warning(f"Invalid price for LIMIT order: {price}")
                return False
            
            # Check balance
            base_asset = symbol.replace('USDC', '')
            if side == 'SELL':
                # Check if enough base asset
                if self.balance.get(base_asset, 0.0) < quantity:
                    self.logger.system.warning(f"Insufficient {base_asset} balance: {self.balance.get(base_asset, 0.0)} < {quantity}")
                    return False
            else:  # BUY
                # Check if enough quote asset
                cost = quantity * (price or self.get_current_price(symbol))
                if self.balance.get('USDC', 0.0) < cost:
                    self.logger.system.warning(f"Insufficient USDC balance: {self.balance.get('USDC', 0.0)} < {cost}")
                    return False
            
            return True
        except Exception as e:
            self.logger.log_error(f"Error validating order: {str(e)}", component="paper_trading")
            return False
    
    def cancel_order(self, order_id, reason="User requested"):
        """Cancel an order
        
        Args:
            order_id: Order ID
            reason: Cancellation reason
            
        Returns:
            bool: Whether the cancellation was successful
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                self.logger.system.warning(f"Order {order_id} not found")
                return False
            
            # Add to order queue
            self.order_queue.put({
                'type': 'cancel',
                'order_id': order_id,
                'reason': reason
            })
            
            self.logger.system.info(f"Order {order_id} queued for cancellation")
            
            return True
        except Exception as e:
            self.logger.log_error(f"Error cancelling order: {str(e)}", component="paper_trading")
            return False
    
    def cancel_order_internal(self, order_id, reason="User requested"):
        """Cancel order internally
        
        Args:
            order_id: Order ID
            reason: Cancellation reason
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                self.logger.system.warning(f"Order {order_id} not found")
                return
            
            # Get order
            order = self.orders[order_id]
            
            # Update order status
            order['status'] = 'CANCELED'
            order['cancelTime'] = int(time.time() * 1000)
            order['cancelReason'] = reason
            
            # Log order cancellation
            self.logger.system.info(f"Order {order_id} cancelled: {reason}")
            
            # Send notification
            self.notify('order_cancelled', order)
        except Exception as e:
            self.logger.log_error(f"Error cancelling order internally: {str(e)}", component="paper_trading")
    
    def fill_order(self, order_id, price=None):
        """Fill an order
        
        Args:
            order_id: Order ID
            price: Fill price (optional)
            
        Returns:
            bool: Whether the fill was successful
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                self.logger.system.warning(f"Order {order_id} not found")
                return False
            
            # Add to order queue
            self.order_queue.put({
                'type': 'fill',
                'order_id': order_id,
                'price': price
            })
            
            self.logger.system.info(f"Order {order_id} queued for filling")
            
            return True
        except Exception as e:
            self.logger.log_error(f"Error filling order: {str(e)}", component="paper_trading")
            return False
    
    def fill_order_internal(self, order_id, price=None):
        """Fill order internally
        
        Args:
            order_id: Order ID
            price: Fill price (optional)
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                self.logger.system.warning(f"Order {order_id} not found")
                return
            
            # Get order
            order = self.orders[order_id]
            
            # Check if order is already filled or cancelled
            if order['status'] in ['FILLED', 'CANCELED']:
                self.logger.system.warning(f"Order {order_id} already {order['status']}")
                return
            
            # Get fill price
            fill_price = price or order.get('price') or self.get_current_price(order['symbol'])
            
            # Update order
            order['status'] = 'FILLED'
            order['price'] = fill_price
            order['fillTime'] = int(time.time() * 1000)
            
            # Create trade
            trade = {
                'tradeId': f"TRD-{uuid.uuid4()}",
                'orderId': order_id,
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': order['quantity'],
                'price': fill_price,
                'timestamp': int(time.time() * 1000)
            }
            
            # Add trade to trades list
            self.trades.append(trade)
            
            # Update balance and position
            self.update_balance_and_position(trade)
            
            # Log order fill
            self.logger.system.info(f"Order {order_id} filled at {fill_price}")
            
            # Send notification
            self.notify('order_filled', {
                'order': order,
                'trade': trade
            })
        except Exception as e:
            self.logger.log_error(f"Error filling order internally: {str(e)}", component="paper_trading")
    
    def update_balance_and_position(self, trade):
        """Update balance and position after a trade
        
        Args:
            trade: Trade dictionary
        """
        try:
            # Extract trade data
            symbol = trade['symbol']
            side = trade['side']
            quantity = trade['quantity']
            price = trade['price']
            
            # Get base and quote assets
            base_asset = symbol.replace('USDC', '')
            quote_asset = 'USDC'
            
            # Update balance
            if side == 'BUY':
                # Increase base asset, decrease quote asset
                self.balance[base_asset] = self.balance.get(base_asset, 0.0) + quantity
                self.balance[quote_asset] = self.balance.get(quote_asset, 0.0) - (quantity * price)
            else:  # SELL
                # Decrease base asset, increase quote asset
                self.balance[base_asset] = self.balance.get(base_asset, 0.0) - quantity
                self.balance[quote_asset] = self.balance.get(quote_asset, 0.0) + (quantity * price)
            
            # Update position
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Update position
                if side == 'BUY':
                    # Calculate new entry price
                    old_quantity = position['base_quantity']
                    new_quantity = old_quantity + quantity
                    old_value = old_quantity * position['entry_price'] if position['entry_price'] > 0 else 0
                    new_value = quantity * price
                    
                    # Update position
                    position['base_quantity'] = new_quantity
                    position['entry_price'] = (old_value + new_value) / new_quantity if new_quantity > 0 else 0
                else:  # SELL
                    # Calculate realized PnL
                    realized_pnl = quantity * (price - position['entry_price'])
                    
                    # Update position
                    position['base_quantity'] = position['base_quantity'] - quantity
                    position['realized_pnl'] = position['realized_pnl'] + realized_pnl
                
                # Update current price
                position['current_price'] = price
                
                # Update unrealized PnL
                position['unrealized_pnl'] = position['base_quantity'] * (price - position['entry_price'])
                
                # Update timestamp
                position['timestamp'] = int(time.time() * 1000)
            
            self.logger.system.info(f"Updated balance and position for {symbol}")
        except Exception as e:
            self.logger.log_error(f"Error updating balance and position: {str(e)}", component="paper_trading")
    
    def update_market_data(self):
        """Update market data"""
        try:
            # Update last prices
            for symbol in ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']:
                try:
                    # Get ticker
                    ticker = self.client.get_ticker(symbol)
                    
                    # Update last price
                    if ticker and 'lastPrice' in ticker:
                        self.last_prices[symbol] = float(ticker['lastPrice'])
                        
                        # Update position current price
                        if symbol in self.positions:
                            self.positions[symbol]['current_price'] = float(ticker['lastPrice'])
                            
                            # Update unrealized PnL
                            entry_price = self.positions[symbol]['entry_price']
                            base_quantity = self.positions[symbol]['base_quantity']
                            current_price = float(ticker['lastPrice'])
                            
                            self.positions[symbol]['unrealized_pnl'] = base_quantity * (current_price - entry_price)
                except Exception as e:
                    self.logger.log_error(f"Error updating market data for {symbol}: {str(e)}", component="paper_trading")
            
            self.logger.system.debug("Market data updated")
        except Exception as e:
            self.logger.log_error(f"Error updating market data: {str(e)}", component="paper_trading")
    
    def get_current_price(self, symbol):
        """Get current price for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Current price
        """
        try:
            # Check if we have a cached price
            if symbol in self.last_prices:
                return self.last_prices[symbol]
            
            # Try to get ticker
            try:
                ticker = self.client.get_ticker(symbol)
                
                # Update last price
                if ticker and 'lastPrice' in ticker:
                    self.last_prices[symbol] = float(ticker['lastPrice'])
                    return float(ticker['lastPrice'])
            except Exception as e:
                self.logger.log_error(f"Error getting ticker for {symbol}: {str(e)}", component="paper_trading")
            
            # Fallback to default prices
            default_prices = {
                'BTCUSDC': 50000.0,
                'ETHUSDC': 3000.0,
                'SOLUSDC': 100.0
            }
            
            return default_prices.get(symbol, 0.0)
        except Exception as e:
            self.logger.log_error(f"Error getting current price for {symbol}: {str(e)}", component="paper_trading")
            return 0.0
    
    def get_balance(self):
        """Get current balance
        
        Returns:
            dict: Current balance
        """
        return self.balance
    
    def get_positions(self):
        """Get current positions
        
        Returns:
            dict: Current positions
        """
        return self.positions
    
    def get_orders(self):
        """Get current orders
        
        Returns:
            dict: Current orders
        """
        return self.orders
    
    def get_trades(self):
        """Get trades
        
        Returns:
            list: Trades
        """
        return self.trades
    
    def get_status(self):
        """Get paper trading system status
        
        Returns:
            dict: System status
        """
        return {
            'running': self.running,
            'balance': self.balance,
            'positions': self.positions,
            'orders': len(self.orders),
            'trades': len(self.trades)
        }
    
    def execute_trade(self, signal):
        """Execute a trade based on a signal
        
        Args:
            signal: Trading signal
            
        Returns:
            dict: Trade details
        """
        try:
            # Extract signal data
            symbol = signal['symbol']
            direction = signal['direction']
            strength = signal['strength']
            price = signal.get('price')
            
            # Convert symbol format if needed (BTC/USDC -> BTCUSDC)
            api_symbol = symbol.replace('/', '')
            
            # Determine side
            side = 'BUY' if direction == 'BUY' else 'SELL'
            
            # Determine quantity based on position size and balance
            position_size = self.config.get('position_size', 0.1)
            
            if side == 'BUY':
                # Calculate quantity based on USDC balance
                usdc_balance = self.balance.get('USDC', 0.0)
                trade_amount = usdc_balance * position_size
                
                # Get current price if not provided
                current_price = price or self.get_current_price(api_symbol)
                
                # Calculate quantity
                quantity = trade_amount / current_price if current_price > 0 else 0
            else:  # SELL
                # Calculate quantity based on base asset balance
                base_asset = api_symbol.replace('USDC', '')
                base_balance = self.balance.get(base_asset, 0.0)
                
                # Calculate quantity
                quantity = base_balance * position_size
            
            # Create order
            order_id = self.create_order(
                symbol=api_symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity
            )
            
            if not order_id:
                self.logger.system.warning(f"Failed to create order for signal: {signal}")
                return None
            
            # Wait for order to be filled
            max_wait = 5.0  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if order_id in self.orders and self.orders[order_id]['status'] == 'FILLED':
                    # Get order
                    order = self.orders[order_id]
                    
                    # Create trade result
                    trade_result = {
                        'id': f"SIG-{uuid.uuid4()}",
                        'signal_id': signal.get('id', 'unknown'),
                        'order_id': order_id,
                        'symbol': symbol,
                        'action': side,
                        'quantity': order['quantity'],
                        'price': order['price'],
                        'timestamp': int(time.time() * 1000),
                        'status': 'EXECUTED'
                    }
                    
                    self.logger.system.info(f"Trade executed: {side} {order['quantity']} {symbol} at {order['price']}")
                    return trade_result
                
                time.sleep(0.1)
            
            self.logger.system.warning(f"Order {order_id} not filled within timeout")
            
            # Create timeout trade result
            trade_result = {
                'id': f"SIG-{uuid.uuid4()}",
                'signal_id': signal.get('id', 'unknown'),
                'order_id': order_id,
                'symbol': symbol,
                'action': side,
                'quantity': quantity,
                'price': price or self.get_current_price(api_symbol),
                'timestamp': int(time.time() * 1000),
                'status': 'TIMEOUT'
            }
            
            return trade_result
        except Exception as e:
            self.logger.log_error(f"Error executing trade: {str(e)}", component="paper_trading")
            return None

# Alias for backward compatibility
FixedPaperTradingSystem = PaperTradingSystem
