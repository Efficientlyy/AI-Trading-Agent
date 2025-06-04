#!/usr/bin/env python
"""
Flash Trading Integration Module

This module integrates the optimized MEXC client, paper trading system,
and signal generation engine for flash trading with zero-fee pairs.
"""

import time
import logging
import argparse
import json
import os
from datetime import datetime
from optimized_mexc_client import OptimizedMEXCClient
from paper_trading import PaperTradingSystem
from flash_trading_config import FlashTradingConfig
from flash_trading_signals import FlashTradingSignals as SignalGenerator, MarketState

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
logger = logging.getLogger("flash_trading")

class FlashTradingSystem:
    """Flash Trading System integrating all components"""
    
    def __init__(self, env_path=None, config_path=None):
        """Initialize the flash trading system"""
        # Load configuration
        self.config = FlashTradingConfig(config_path)
        
        # Create API client
        self.client = OptimizedMEXCClient(env_path=env_path)
        
        # Create paper trading system
        self.paper_trading = PaperTradingSystem(self.client, self.config)
        
        # Create signal generator with shared client instance
        self.signal_generator = SignalGenerator(client_instance=self.client)
        
        # Running state
        self.running = False
        self.start_time = None
        self.stats = {
            "signals_generated": 0,
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_profit_loss": 0.0
        }
    
    def start(self):
        """Start the flash trading system"""
        if self.running:
            logger.warning("Flash trading system already running")
            return False
        
        # Get enabled trading pairs
        trading_pairs = self.config.get_enabled_trading_pairs()
        if not trading_pairs:
            logger.error("No enabled trading pairs found")
            return False
        
        # Extract symbols
        symbols = [pair["symbol"] for pair in trading_pairs]
        
        # Start signal generator
        self.signal_generator.start(symbols)
        
        # Set running state
        self.running = True
        self.start_time = time.time()
        
        logger.info(f"Flash trading system started for symbols: {symbols}")
        return True
    
    def stop(self):
        """Stop the flash trading system"""
        if not self.running:
            logger.warning("Flash trading system not running")
            return False
        
        # Stop signal generator
        self.signal_generator.stop()
        
        # Set running state
        self.running = False
        
        logger.info("Flash trading system stopped")
        return True
    
    @handle_api_error
    def process_signals_and_execute(self):
        """Process signals and execute trades using paper trading"""
        # DIAGNOSTIC: Log entry point
        logger.debug("Entering process_signals_and_execute")
        
        if not self.running:
            logger.warning("Flash trading system not running")
            return False
        
        # Get enabled trading pairs with robust error handling
        try:
            trading_pairs = self.config.get_enabled_trading_pairs()
            if not validate_api_response(trading_pairs, list):
                logger.warning("No valid trading pairs found in configuration")
                return False
                
            if len(trading_pairs) == 0:
                logger.warning("Empty trading pairs list in configuration")
                return False
        except Exception as e:
            log_exception(e, "get_enabled_trading_pairs")
            return False
        
        # Process each trading pair with validation
        for pair_config in trading_pairs:
            try:
                # Validate pair configuration with error handling utilities
                if not validate_api_response(pair_config, dict, ["symbol"]):
                    logger.error(f"Invalid pair configuration format: {type(pair_config)}, expected dict with symbol")
                    continue
                    
                symbol = safe_get(pair_config, "symbol")
                if not symbol or not isinstance(symbol, str):
                    logger.error(f"Invalid symbol in pair configuration: {symbol}")
                    continue
                
                # Get recent signals with robust validation and diagnostics
                try:
                    # DIAGNOSTIC: Log before API call
                    logger.debug(f"Calling get_recent_signals for {symbol}")
                    
                    signals = self.signal_generator.get_recent_signals(10)
                    
                    # DIAGNOSTIC: Log raw response
                    logger.debug(f"Raw signals response for {symbol}: {type(signals)}, count: {len(signals) if isinstance(signals, list) else 'N/A'}")
                    
                    if signals is None:
                        logger.error(f"CRITICAL: Null signals response for {symbol}")
                        signals = []
                    elif not validate_api_response(signals, list):
                        logger.warning(f"Failed to get valid signals for {symbol}")
                        signals = []
                except Exception as e:
                    log_exception(e, f"get_recent_signals for {symbol}")
                    signals = []
                
                # Filter signals for current symbol with validation
                filtered_signals = []
                for s in signals:
                    if isinstance(s, dict) and safe_get(s, "symbol") == symbol:
                        filtered_signals.append(s)
                signals = filtered_signals
                
                if not signals:
                    logger.debug(f"No signals found for {symbol}")
                    continue
                
                # Make trading decision with enhanced validation and diagnostics
                try:
                    # DIAGNOSTIC: Log before making trading decision
                    logger.debug(f"Calling make_trading_decision for {symbol} with {len(signals)} signals")
                    
                    decision = self.signal_generator.make_trading_decision(symbol, signals)
                    
                    # DIAGNOSTIC: Log raw decision
                    logger.debug(f"Raw decision for {symbol}: {type(decision)}, value: {decision}")
                    
                    if decision is None:
                        logger.error(f"CRITICAL: Null decision response for {symbol}")
                        continue
                        
                    # Validate decision object before execution with error handling utilities
                    if not validate_api_response(decision, dict, ["symbol", "side", "order_type", "size"]):
                        logger.error(f"Invalid or incomplete decision object: {decision}")
                        continue
                    
                    # Verify symbol match
                    if safe_get(decision, "symbol") != symbol:
                        logger.error(f"Decision symbol mismatch: {safe_get(decision, 'symbol')} vs {symbol}")
                        continue
                    
                    # Validate field types
                    side = safe_get(decision, "side")
                    order_type = safe_get(decision, "order_type")
                    size = safe_get(decision, "size")
                    
                    if not isinstance(side, str) or side not in ["BUY", "SELL"]:
                        logger.error(f"Invalid side in decision: {side}")
                        continue
                        
                    if not isinstance(order_type, str) or order_type not in ["LIMIT", "MARKET"]:
                        logger.error(f"Invalid order_type in decision: {order_type}")
                        continue
                        
                    if not isinstance(size, (int, float, str)) or parse_float_safely(size, 0) <= 0:
                        logger.error(f"Invalid size in decision: {size}")
                        continue
                    
                    # Execute with paper trading and diagnostics
                    logger.debug(f"Calling _execute_paper_trading_decision for {symbol} with decision: {decision}")
                    
                    result = self._execute_paper_trading_decision(decision)
                    
                    # DIAGNOSTIC: Log execution result
                    logger.debug(f"Paper trading execution result for {symbol}: {type(result)}, value: {result}")
                    
                    if not result:
                        logger.warning(f"Failed to execute paper trading decision: {decision}")
                except Exception as e:
                    log_exception(e, f"make_trading_decision for {symbol}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing trading pair {pair_config}: {str(e)}")
                continue
        
        # Process open paper trading orders with error handling
        try:
            self.paper_trading.process_open_orders()
        except Exception as e:
            logger.error(f"Error processing open orders: {str(e)}")
        
        return True
    
    @handle_api_error
    def _execute_paper_trading_decision(self, decision):
        """Execute trading decision using paper trading"""
        try:
            # Validate decision object with error handling utilities
            if not validate_api_response(decision, dict, ["symbol", "size"]):
                logger.error(f"Invalid or incomplete decision object: {decision}")
                return None
                
            # Extract and validate required fields with robust error handling
            try:
                # Get symbol with validation
                symbol = safe_get(decision, "symbol")
                if not symbol or not isinstance(symbol, str):
                    logger.error(f"Invalid symbol in decision: {symbol}")
                    return None
                    
                # Support both "side" and legacy "action" fields with validation
                side = safe_get(decision, "side", safe_get(decision, "action"))
                if not side or side not in ["BUY", "SELL"]:
                    logger.error(f"Invalid side in decision: {side}")
                    return None
                    
                # Get order type with validation
                order_type = safe_get(decision, "order_type")
                if not order_type or order_type not in ["LIMIT", "MARKET"]:
                    logger.error(f"Invalid order_type in decision: {order_type}")
                    return None
                    
                # Get quantity with validation
                quantity = safe_get(decision, "size")
                if not quantity or not isinstance(quantity, (int, float, str)) or parse_float_safely(quantity, 0) <= 0:
                    logger.error(f"Invalid quantity in decision: {quantity}")
                    return None
                    
                # Get price with validation for LIMIT orders
                price = safe_get(decision, "price")
                if order_type == "LIMIT" and (price is None or parse_float_safely(price, 0) <= 0):
                    logger.error(f"Invalid price for LIMIT order: {price}")
                    return None
                    
                # Get time in force with validation
                time_in_force = safe_get(decision, "time_in_force", "GTC")
                if order_type == "LIMIT" and time_in_force not in ["GTC", "IOC", "FOK"]:
                    logger.error(f"Invalid time_in_force in decision: {time_in_force}")
                    return None
                    
            except Exception as e:
                log_exception(e, "validate_decision_fields")
                return None
            
            # Place paper trading order with validation
            try:
                order = self.paper_trading.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    time_in_force=time_in_force
                )
            except Exception as e:
                log_exception(e, "place_paper_trading_order")
                return None
            
            # Validate order response with error handling utilities
            if not validate_api_response(order, dict, ["orderId", "symbol", "side", "status"]):
                logger.error(f"Invalid or incomplete order response: {order}")
                return None
                
            # Update statistics
            self.stats["orders_placed"] = self.stats.get("orders_placed", 0) + 1
            
            # Log order with safe access
            order_id = safe_get(order, "orderId", "unknown")
            logger.info(f"Paper order placed: {side} {quantity} {symbol} @ {price}, order ID: {order_id}")
            
            return order
        except Exception as e:
            log_exception(e, "_execute_paper_trading_decision")
            return None
    
    def run_for_duration(self, duration_seconds, update_interval=1.0):
        """Run the flash trading system for a specified duration"""
        # Start the system
        if not self.start():
            return False
        
        try:
            # Run for specified duration
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time and self.running:
                # Process signals and execute trades
                self.process_signals_and_execute()
                
                # Sleep for update interval
                time.sleep(update_interval)
                
                # Print status every 5 seconds
                elapsed = time.time() - self.start_time
                if int(elapsed) % 5 == 0:
                    self._print_status()
        
        except KeyboardInterrupt:
            logger.info("Flash trading interrupted by user")
        
        finally:
            # Stop the system
            self.stop()
            
            # Print final status
            self._print_final_status()
        
        return True
    
    @handle_api_error
    def _print_status(self):
        """Print current status"""
        try:
            # Get account information with validation
            account = self.paper_trading.get_account()
            if not validate_api_response(account, dict, ["balances"]):
                logger.warning("Invalid account response format")
                account = {"balances": []}
            
            # Get open orders with validation
            open_orders = self.paper_trading.get_open_orders()
            if not isinstance(open_orders, list):
                logger.warning("Invalid open orders response format")
                open_orders = []
            
            # Print status with safe access
            print("\n--- Flash Trading Status ---")
            print("Balances:")
            balances = safe_get(account, "balances", [])
            for balance in balances:
                if isinstance(balance, dict):
                    asset = safe_get(balance, "asset", "")
                    free = parse_float_safely(safe_get(balance, "free"), 0.0)
                    if asset and free > 0:
                        print(f"  {asset}: {free}")
            
            print("\nOpen Orders:")
            for order in open_orders:
                if isinstance(order, dict):
                    symbol = safe_get(order, "symbol", "")
                    side = safe_get(order, "side", "")
                    quantity = safe_get(order, "quantity", "")
                    price = safe_get(order, "price", "")
                    if symbol and side:
                        print(f"  {symbol} {side} {quantity} @ {price}")
            
            print("\nStatistics:")
            signals = getattr(self.signal_generator, "stats", {}).get("signals_generated", 0)
            orders = safe_get(self.stats, "orders_placed", 0)
            uptime = time.time() - (self.start_time or time.time())
            print(f"  Signals: {signals}")
            print(f"  Orders: {orders}")
            print(f"  Uptime: {uptime:.1f}s")
        except Exception as e:
            log_exception(e, "_print_status")
    
    @handle_api_error
    def _print_final_status(self):
        """Print final status"""
        try:
            # Get account information with validation
            account = self.paper_trading.get_account()
            if not validate_api_response(account, dict, ["balances"]):
                logger.warning("Invalid account response format")
                account = {"balances": []}
            
            # Print status with safe access
            print("\n=== Flash Trading Final Status ===")
            print("Balances:")
            balances = safe_get(account, "balances", [])
            for balance in balances:
                if isinstance(balance, dict):
                    asset = safe_get(balance, "asset", "")
                    free = parse_float_safely(safe_get(balance, "free"), 0.0)
                    if asset and free > 0:
                        print(f"  {asset}: {free}")
            
            print("\nStatistics:")
            # Fix type mismatch - ensure we're accessing a dictionary, not an object
            if hasattr(self.signal_generator, 'stats') and isinstance(self.signal_generator.stats, dict):
                signals = safe_get(self.signal_generator.stats, "signals_generated", 0)
            else:
                signals = 0
                logger.debug("Signal generator stats not available as dictionary")
            
            orders = safe_get(self.stats, "orders_placed", 0)
            uptime = time.time() - (self.start_time or time.time())
            print(f"  Signals Generated: {signals}")
            print(f"  Orders Placed: {orders}")
            print(f"  Uptime: {uptime:.1f}s")
        except Exception as e:
            log_exception(e, "_print_final_status")
    
    @handle_api_error
    def save_state(self, filename="flash_trading_state.json"):
        """Save current state to file"""
        try:
            # Create state with validation
            state = {
                "timestamp": int(time.time() * 1000),
                "stats": self.stats.copy() if isinstance(self.stats, dict) else {},
                "uptime": time.time() - (self.start_time or time.time())
            }
            
            # Get signals with validation
            try:
                signals = self.signal_generator.get_recent_signals(100)
                if isinstance(signals, list):
                    state["signals"] = signals
                else:
                    logger.warning(f"Invalid signals format: {type(signals)}, expected list")
                    state["signals"] = []
            except Exception as e:
                log_exception(e, "get_recent_signals")
                state["signals"] = []
            
            # Save to file with error handling
            try:
                with open(filename, "w") as f:
                    json.dump(state, f, indent=2)
                
                logger.info(f"State saved to {filename}")
            except (IOError, OSError) as e:
                logger.error(f"Error saving state to file: {str(e)}")
        except Exception as e:
            log_exception(e, "save_state")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flash Trading System')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--config', default="flash_trading_config.json", help='Path to config file')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--reset', action='store_true', help='Reset paper trading system')
    
    args = parser.parse_args()
    
    # Create flash trading system
    flash_trading = FlashTradingSystem(args.env, args.config)
    
    # Reset paper trading if requested
    if args.reset:
        flash_trading.paper_trading.reset()
        print("Paper trading system reset to initial state")
    
    # Run for specified duration
    print(f"Running flash trading system for {args.duration} seconds...")
    flash_trading.run_for_duration(args.duration)
    
    # Save state
    flash_trading.save_state()
