#!/usr/bin/env python
"""
Paper Trading System Fix for Trading-Agent System

This module provides a wrapper and compatibility layer for the FixedPaperTradingSystem class,
exposing it as PaperTradingSystem for compatibility with the rest of the system.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("paper_trading_fix.log")
    ]
)

logger = logging.getLogger("paper_trading_fix")

# Import the original paper trading module
try:
    from fixed_paper_trading import FixedPaperTradingSystem
    logger.info("Successfully imported FixedPaperTradingSystem")
except ImportError as e:
    logger.error(f"Failed to import FixedPaperTradingSystem: {str(e)}")
    raise

class PaperTradingSystem:
    """Wrapper class for FixedPaperTradingSystem to provide compatibility with expected interface"""
    
    def __init__(self, client=None, client_instance=None, config=None, use_mock_data=False):
        """Initialize paper trading system
        
        Args:
            client: Exchange client (optional)
            client_instance: Alternative name for exchange client (optional)
            config: Configuration dictionary (optional)
            use_mock_data: Whether to use mock data (ignored in production mode)
        """
        # Store use_mock_data flag but ignore it in production
        self.use_mock_data = False  # Always set to False for production mode
        
        if use_mock_data:
            logger.warning("Mock data requested but disabled in production mode")
        
        # Use client_instance if provided, otherwise use client
        client = client_instance if client_instance is not None else client
        
        # Initialize underlying paper trading system
        self.paper_trading = FixedPaperTradingSystem(
            client=client,
            config=config
        )
        
        logger.info("PaperTradingSystem initialized with FixedPaperTradingSystem")
    
    def execute_trade(self, symbol, side, quantity, order_type, signal_data=None):
        """Execute a trade
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            quantity: Order quantity
            order_type: Order type (LIMIT or MARKET)
            signal_data: Signal data that triggered the trade (optional)
            
        Returns:
            dict: Trade result
        """
        logger.info(f"Executing trade: {side} {quantity} {symbol} {order_type}")
        
        try:
            # Create order
            price = None
            if order_type == 'LIMIT':
                # Get current price for limit orders
                price = self.paper_trading.get_current_price(symbol)
            
            order_id = self.paper_trading.create_order(symbol, side, order_type, quantity, price)
            
            if not order_id:
                logger.error(f"Failed to create order for {side} {quantity} {symbol}")
                return {"success": False, "error": "Failed to create order"}
            
            # Get order details
            order = self.paper_trading.orders.get(order_id, {})
            
            # Return trade result
            return {
                "success": True,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": order.get('price'),
                "type": order_type,
                "status": order.get('status'),
                "timestamp": order.get('timestamp'),
                "signal_data": signal_data
            }
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_positions(self):
        """Get current positions
        
        Returns:
            list: List of positions
        """
        logger.info("Getting positions")
        
        try:
            # Get positions from underlying system
            positions = self.paper_trading.positions
            
            # Convert to list
            position_list = list(positions.values())
            
            logger.info(f"Retrieved {len(position_list)} positions")
            return position_list
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_balance(self):
        """Get current balance
        
        Returns:
            dict: Balance dictionary
        """
        logger.info("Getting balance")
        
        try:
            # Get balance from underlying system
            balance = self.paper_trading.balance
            
            logger.info(f"Retrieved balance: {balance}")
            return balance
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return {}
    
    def get_orders(self, symbol=None, status=None):
        """Get orders
        
        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (optional)
            
        Returns:
            list: List of orders
        """
        logger.info(f"Getting orders for {symbol if symbol else 'all symbols'}")
        
        try:
            # Get orders from underlying system
            orders = self.paper_trading.orders
            
            # Filter orders
            filtered_orders = []
            for order_id, order in orders.items():
                if symbol and order.get('symbol') != symbol:
                    continue
                if status and order.get('status') != status:
                    continue
                filtered_orders.append(order)
            
            logger.info(f"Retrieved {len(filtered_orders)} orders")
            return filtered_orders
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def start(self):
        """Start paper trading system
        
        Returns:
            bool: Whether the start was successful
        """
        logger.info("Starting paper trading system")
        
        try:
            # Start underlying system
            self.paper_trading.start()
            
            logger.info("Paper trading system started")
            return True
        except Exception as e:
            logger.error(f"Error starting paper trading system: {str(e)}")
            return False
    
    def stop(self):
        """Stop paper trading system
        
        Returns:
            bool: Whether the stop was successful
        """
        logger.info("Stopping paper trading system")
        
        try:
            # Stop underlying system
            self.paper_trading.stop()
            
            logger.info("Paper trading system stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping paper trading system: {str(e)}")
            return False

# Expose PaperTradingSystem as PaperTrading for the integrated pipeline
PaperTrading = PaperTradingSystem
logger.info("PaperTradingSystem exposed as PaperTrading for integrated pipeline")

# Make PaperTradingSystem available when importing from fixed_paper_trading
sys.modules['fixed_paper_trading'].PaperTradingSystem = PaperTradingSystem
logger.info("PaperTradingSystem class added to fixed_paper_trading module")

if __name__ == "__main__":
    # Test the fix
    try:
        from fixed_paper_trading import PaperTradingSystem
        
        print("PaperTradingSystem import successful")
        
        # Create instance
        paper_trading = PaperTradingSystem()
        
        print("PaperTradingSystem instance created")
        
        # Test trade execution
        print("Testing trade execution for BTCUSDT...")
        result = paper_trading.execute_trade("BTCUSDT", "BUY", 0.001, "MARKET")
        
        print(f"Trade execution result: {result}")
        
        # Get positions
        positions = paper_trading.get_positions()
        print(f"Current positions: {positions}")
        
        # Test PaperTrading alias
        print("\nTesting PaperTrading alias...")
        paper_trading_alias = PaperTrading()
        print("PaperTrading instance created")
        
    except Exception as e:
        print(f"Error testing PaperTradingSystem: {str(e)}")
