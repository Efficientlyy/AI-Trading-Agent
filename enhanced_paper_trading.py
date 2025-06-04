#!/usr/bin/env python
"""
Enhanced Paper Trading System

This module extends the PaperTradingSystem class with additional functionality
for comprehensive balance reporting and improved order execution.
"""

import time
import json
import logging
import os
import uuid
from datetime import datetime
from threading import RLock
from paper_trading import PaperTradingSystem
from optimized_mexc_client import OptimizedMEXCClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_paper_trading")

class EnhancedPaperTradingSystem(PaperTradingSystem):
    """Enhanced paper trading system with improved balance reporting and order execution"""
    
    def get_balances(self):
        """Get all asset balances as a dictionary"""
        with self.lock:
            # Return a copy of the balances dictionary to prevent external modification
            return dict(self.balances)
    
    def create_order(self, symbol, side, quantity, price=None):
        """Create an order (wrapper for place_order for API consistency)"""
        order_type = "LIMIT" if price is not None else "MARKET"
        return self.place_order(symbol, side, order_type, quantity, price)
    
    def execute_order(self, order_id):
        """Execute a paper trading order by order ID"""
        with self.lock:
            try:
                # Validate order ID
                if not order_id or not isinstance(order_id, str):
                    logger.error(f"Invalid order ID: {order_id}")
                    return None
                
                # Check if order exists
                if order_id not in self.open_orders:
                    logger.warning(f"Order not found: {order_id}")
                    return None
                
                # Get order details
                order = self.open_orders[order_id]
                
                # Validate order structure
                if not isinstance(order, dict):
                    logger.error(f"Invalid order structure: {order}")
                    return None
                
                # Extract order details with validation
                symbol = order.get("symbol")
                side = order.get("side")
                order_type = order.get("type")
                quantity = order.get("quantity")
                price = order.get("price")
                
                if not symbol or not side or not order_type or not quantity:
                    logger.error(f"Missing required order fields: {order}")
                    return None
                
                # Get current market price for execution
                current_price = self._get_current_price(symbol, side)
                if current_price is None:
                    logger.error(f"Failed to get current price for {symbol}")
                    return None
                
                # For LIMIT orders, check if price conditions are met
                if order_type == "LIMIT":
                    if side == "BUY" and current_price > price:
                        logger.info(f"LIMIT BUY order price condition not met: {price} < {current_price}")
                        return None
                    elif side == "SELL" and current_price < price:
                        logger.info(f"LIMIT SELL order price condition not met: {price} > {current_price}")
                        return None
                
                # Execute the order at the appropriate price
                execution_price = price if order_type == "LIMIT" else current_price
                
                # Update balances based on order side
                base_asset, quote_asset = self._get_assets_from_symbol(symbol)
                if not base_asset or not quote_asset:
                    logger.error(f"Failed to extract assets from symbol: {symbol}")
                    return None
                
                # Calculate executed amounts
                executed_base_qty = quantity
                executed_quote_qty = quantity * execution_price
                
                if side == "BUY":
                    # Add base asset
                    self.balances[base_asset] = self.balances.get(base_asset, 0.0) + executed_base_qty
                    # Quote asset already reserved during order placement
                elif side == "SELL":
                    # Add quote asset
                    self.balances[quote_asset] = self.balances.get(quote_asset, 0.0) + executed_quote_qty
                    # Base asset already reserved during order placement
                
                # Create trade record
                trade = {
                    "tradeId": str(uuid.uuid4()),
                    "orderId": order_id,
                    "symbol": symbol,
                    "side": side,
                    "price": execution_price,
                    "quantity": quantity,
                    "quoteQty": executed_quote_qty,
                    "commission": 0.0,  # Zero fee for paper trading
                    "commissionAsset": quote_asset,
                    "timestamp": int(time.time() * 1000)
                }
                
                # Update order status
                order["status"] = "FILLED"
                order["executedQty"] = quantity
                order["cummulativeQuoteQty"] = executed_quote_qty
                order["fills"] = [trade]
                
                # Move order from open to history
                self.order_history.append(order)
                del self.open_orders[order_id]
                
                # Add trade to history
                self.trade_history.append(trade)
                
                # Save state
                self._save_state()
                
                logger.info(f"Paper order executed: {side} {quantity} {symbol} @ {execution_price}")
                return trade
            except Exception as e:
                logger.error(f"Error executing paper order: {str(e)}")
                return None
    
    def cancel_order(self, order_id):
        """Cancel a paper trading order by order ID"""
        with self.lock:
            try:
                # Validate order ID
                if not order_id or not isinstance(order_id, str):
                    logger.error(f"Invalid order ID: {order_id}")
                    return None
                
                # Check if order exists
                if order_id not in self.open_orders:
                    logger.warning(f"Order not found: {order_id}")
                    return None
                
                # Get order details
                order = self.open_orders[order_id]
                
                # Validate order structure
                if not isinstance(order, dict):
                    logger.error(f"Invalid order structure: {order}")
                    return None
                
                # Extract order details with validation
                symbol = order.get("symbol")
                side = order.get("side")
                quantity = order.get("quantity")
                price = order.get("price")
                
                if not symbol or not side or not quantity or not price:
                    logger.error(f"Missing required order fields: {order}")
                    return None
                
                # Return reserved funds
                base_asset, quote_asset = self._get_assets_from_symbol(symbol)
                if not base_asset or not quote_asset:
                    logger.error(f"Failed to extract assets from symbol: {symbol}")
                    return None
                
                if side == "BUY":
                    # Return quote asset
                    reserved_amount = quantity * price
                    self.balances[quote_asset] = self.balances.get(quote_asset, 0.0) + reserved_amount
                elif side == "SELL":
                    # Return base asset
                    self.balances[base_asset] = self.balances.get(base_asset, 0.0) + quantity
                
                # Update order status
                order["status"] = "CANCELED"
                
                # Move order from open to history
                self.order_history.append(order)
                del self.open_orders[order_id]
                
                # Save state
                self._save_state()
                
                logger.info(f"Paper order canceled: {order_id}")
                return order
            except Exception as e:
                logger.error(f"Error canceling paper order: {str(e)}")
                return None
    
    def _get_assets_from_symbol(self, symbol):
        """Extract base and quote assets from symbol"""
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                logger.error(f"Invalid symbol: {symbol}")
                return None, None
            
            # Get pair configuration
            pair_config = self.config.get_trading_pair_config(symbol)
            if not pair_config:
                logger.warning(f"Invalid symbol: {symbol}")
                return None, None
            
            # Extract assets
            base_asset = pair_config.get("base_asset")
            quote_asset = pair_config.get("quote_asset")
            
            return base_asset, quote_asset
        except Exception as e:
            logger.error(f"Error extracting assets from symbol: {str(e)}")
            return None, None

# Example usage
if __name__ == "__main__":
    client = OptimizedMEXCClient()
    paper = EnhancedPaperTradingSystem(client)
    
    # Print initial balances
    print(f"Initial balances: {paper.get_balances()}")
    
    # Place a test order
    order = paper.create_order("BTCUSDC", "BUY", 0.001, 105000.0)
    if order:
        print(f"Order placed: {order}")
        
        # Execute the order
        order_id = order["orderId"]
        result = paper.execute_order(order_id)
        if result:
            print(f"Order executed: {result}")
            print(f"Updated balances: {paper.get_balances()}")
        else:
            print("Order execution failed")
    else:
        print("Order placement failed")
