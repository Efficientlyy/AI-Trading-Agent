#!/usr/bin/env python
"""
Enhanced Signal Processor for Trading-Agent System

This module provides an enhanced signal processing pipeline with improved
logging, validation, and reliability for the signal-to-order flow.
"""

import os
import sys
import json
import time
import random
import logging
import threading
from queue import Queue
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from optimized_mexc_client import OptimizedMEXCClient
from paper_trading import PaperTradingSystem
from paper_trading_extension import EnhancedPaperTradingSystem
from flash_trading_signals import FlashTradingSignals
from trading_session_manager import TradingSessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("signal_processor.log"),
        logging.StreamHandler()
    ]
)

# Create a filter to add signal_id to log records
class SignalIDFilter(logging.Filter):
    """Filter that adds signal_id to log records"""
    
    def __init__(self, name=''):
        super().__init__(name)
        self.signal_id = "NONE"
    
    def filter(self, record):
        if not hasattr(record, 'signal_id'):
            record.signal_id = self.signal_id
        return True

# Add filter to root logger
signal_id_filter = SignalIDFilter()
logging.getLogger().addFilter(signal_id_filter)

class EnhancedSignalProcessor:
    """Enhanced signal processor with improved logging and validation"""
    
    def __init__(self, config=None):
        """Initialize enhanced signal processor
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.signal_queue = Queue()
        self.order_queue = Queue()
        self.logger = logging.getLogger("signal_processor")
        self.signal_counter = 0
        self.client = OptimizedMEXCClient()
        self.paper_trading = EnhancedPaperTradingSystem(self.client)
        self.setup_logging()
        
        self.logger.info("Enhanced signal processor initialized")
    
    def setup_logging(self):
        """Set up detailed logging for signal processing"""
        # Create file handler for signal processor logs
        handler = logging.FileHandler("signal_processor.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(signal_id)s] - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger if not already added
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith("signal_processor.log") 
                  for h in self.logger.handlers):
            self.logger.addHandler(handler)
        
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Signal processor logging configured")
    
    def process_signal(self, signal):
        """Process a trading signal with enhanced logging and validation
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            bool: True if signal was processed successfully, False otherwise
        """
        # Assign unique ID to signal for tracking
        signal_id = f"SIG-{int(time.time())}-{self.signal_counter}"
        self.signal_counter += 1
        signal['id'] = signal_id
        
        # Set signal ID for logging
        signal_id_filter.signal_id = signal_id
        
        # Log signal receipt
        self.logger.info(f"Received signal: {signal['type']} from {signal['source']} with strength {signal['strength']:.4f}")
        
        # Validate signal
        if not self.validate_signal(signal):
            self.logger.info(f"Signal rejected by validation")
            return False
        
        # Process signal
        try:
            self.logger.info(f"Processing signal")
            order = self.create_order_from_signal(signal)
            if order:
                self.logger.info(f"Created order: {order['orderId']}")
                self.order_queue.put(order)
                return True
            else:
                self.logger.warning(f"Failed to create order from signal")
                return False
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            return False
        finally:
            # Reset signal ID for logging
            signal_id_filter.signal_id = "NONE"
    
    def validate_signal(self, signal):
        """Validate signal quality and relevance
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            bool: True if signal is valid, False otherwise
        """
        # Check required fields
        required_fields = ["type", "source", "strength", "timestamp", "price", "symbol"]
        for field in required_fields:
            if field not in signal:
                self.logger.warning(f"Signal missing required field: {field}")
                return False
        
        # Check signal strength
        min_strength = self.config.get('min_signal_strength', 0.5)
        if signal['strength'] < min_strength:
            self.logger.info(f"Signal strength {signal['strength']:.4f} below threshold {min_strength:.4f}")
            return False
        
        # Check signal recency
        max_age = self.config.get('max_signal_age_ms', 5000)
        signal_age = time.time() * 1000 - signal['timestamp']
        if signal_age > max_age:
            self.logger.info(f"Signal age {signal_age:.0f}ms exceeds maximum {max_age}ms")
            return False
        
        # Check market conditions
        if not self.check_market_conditions(signal):
            self.logger.info(f"Signal rejected due to market conditions")
            return False
        
        self.logger.info(f"Signal passed validation")
        return True
    
    def check_market_conditions(self, signal):
        """Check if market conditions are suitable for the signal
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            bool: True if market conditions are suitable, False otherwise
        """
        try:
            # Get current market data
            symbol = signal['symbol']
            ticker = self.client.get_ticker(symbol)
            
            # Check if market is active (has recent trades)
            if 'lastPrice' not in ticker:
                self.logger.warning(f"No recent price data for {symbol}")
                return False
            
            # Check if spread is reasonable
            if 'askPrice' in ticker and 'bidPrice' in ticker:
                ask_price = float(ticker['askPrice'])
                bid_price = float(ticker['bidPrice'])
                spread_pct = (ask_price - bid_price) / ((ask_price + bid_price) / 2) * 100
                
                max_spread = self.config.get('max_spread_pct', 1.0)
                if spread_pct > max_spread:
                    self.logger.info(f"Spread {spread_pct:.2f}% exceeds maximum {max_spread:.2f}%")
                    return False
            
            # Check if price has moved significantly since signal generation
            if 'lastPrice' in ticker:
                current_price = float(ticker['lastPrice'])
                signal_price = signal['price']
                price_change_pct = abs(current_price - signal_price) / signal_price * 100
                
                max_price_change = self.config.get('max_price_change_pct', 0.5)
                if price_change_pct > max_price_change:
                    self.logger.info(f"Price changed {price_change_pct:.2f}% since signal generation, exceeds maximum {max_price_change:.2f}%")
                    return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {str(e)}")
            return False
    
    def create_order_from_signal(self, signal):
        """Create order from validated signal with retry logic
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            dict: Order dictionary or None if creation failed
        """
        max_retries = self.config.get('order_creation_retries', 3)
        retry_delay = self.config.get('order_creation_retry_delay_ms', 500) / 1000
        
        for attempt in range(max_retries):
            try:
                # Calculate position size and order price
                quantity = self.calculate_position_size(signal)
                price = self.calculate_order_price(signal)
                
                # Create order parameters
                symbol = signal['symbol']
                side = signal['type']  # BUY or SELL
                order_type = "LIMIT"
                
                self.logger.info(f"Creating {side} order for {quantity} {symbol} at {price}")
                
                # Place order
                order = self.paper_trading.place_order(symbol, side, order_type, quantity, price)
                
                if order:
                    # Add signal ID to order for tracking
                    order['signal_id'] = signal['id']
                    self.logger.info(f"Order created successfully: {order['orderId']}")
                    return order
                else:
                    self.logger.warning(f"Order placement returned None")
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying order placement (attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        self.logger.error(f"All order placement attempts failed")
                        return None
            
            except Exception as e:
                self.logger.error(f"Error creating order (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying after error")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"All order creation attempts failed")
                    return None
        
        return None
    
    def calculate_position_size(self, signal):
        """Calculate appropriate position size based on signal and account
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            float: Position size
        """
        try:
            # Get account information
            account = self.paper_trading.get_account_info()
            
            # Get base currency from symbol (e.g., BTC from BTCUSDC)
            symbol = signal['symbol']
            base_currency = symbol.split('/')[0] if '/' in symbol else symbol[:-4]
            quote_currency = symbol.split('/')[1] if '/' in symbol else symbol[-4:]
            
            # Get available balance
            available_balance = 0
            if quote_currency in account['balances']:
                available_balance = account['balances'][quote_currency]
            
            # Calculate position size based on signal strength and risk
            risk_pct = self.config.get('risk_per_trade_pct', 1.0)
            max_position_pct = self.config.get('max_position_pct', 5.0)
            
            # Adjust risk based on signal strength
            adjusted_risk_pct = risk_pct * signal['strength']
            
            # Calculate position size in quote currency
            position_size_quote = available_balance * (adjusted_risk_pct / 100)
            
            # Convert to base currency
            price = signal['price']
            position_size_base = position_size_quote / price
            
            # Apply minimum and maximum constraints
            min_position = self.config.get('min_position_size', 0.001)
            max_position = available_balance * (max_position_pct / 100) / price
            
            position_size = max(min_position, min(position_size_base, max_position))
            
            # Round to appropriate precision
            precision = self.config.get('position_precision', 3)
            position_size = round(position_size, precision)
            
            self.logger.info(f"Calculated position size: {position_size} {base_currency} (risk: {adjusted_risk_pct:.2f}%)")
            return position_size
        
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return self.config.get('default_position_size', 0.001)
    
    def calculate_order_price(self, signal):
        """Calculate appropriate order price based on signal
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            float: Order price
        """
        try:
            # Get current market data
            symbol = signal['symbol']
            ticker = self.client.get_ticker(symbol)
            
            # Get current bid/ask prices
            if 'askPrice' in ticker and 'bidPrice' in ticker:
                ask_price = float(ticker['askPrice'])
                bid_price = float(ticker['bidPrice'])
                
                # Calculate price based on signal type
                if signal['type'] == "BUY":
                    # For buy orders, use a price slightly above bid
                    price_factor = self.config.get('buy_price_factor', 1.001)
                    price = bid_price * price_factor
                else:
                    # For sell orders, use a price slightly below ask
                    price_factor = self.config.get('sell_price_factor', 0.999)
                    price = ask_price * price_factor
                
                self.logger.info(f"Calculated order price: {price} (factor: {price_factor})")
                return price
            else:
                # Fall back to signal price if bid/ask not available
                self.logger.warning(f"Bid/ask prices not available, using signal price: {signal['price']}")
                return signal['price']
        
        except Exception as e:
            self.logger.error(f"Error calculating order price: {str(e)}")
            return signal['price']
    
    def add_signal(self, signal):
        """Add a signal to the processing queue
        
        Args:
            signal: Trading signal dictionary
        """
        self.signal_queue.put(signal)
        self.logger.debug(f"Added signal to queue: {signal['type']} from {signal['source']}")


class SignalOrderIntegration:
    """Integration module connecting signals to order execution"""
    
    def __init__(self, config=None):
        """Initialize signal-order integration
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.signal_processor = EnhancedSignalProcessor(config)
        self.running = False
        self.logger = logging.getLogger("signal_order_integration")
        
        self.logger.info("Signal-order integration initialized")
    
    def start(self):
        """Start the integration pipeline"""
        self.running = True
        self.logger.info("Starting signal-to-order integration pipeline")
        
        # Start processing threads
        self.signal_thread = threading.Thread(target=self.process_signals)
        self.order_thread = threading.Thread(target=self.process_orders)
        
        self.signal_thread.daemon = True
        self.order_thread.daemon = True
        
        self.signal_thread.start()
        self.order_thread.start()
        
        self.logger.info("Signal and order processing threads started")
    
    def stop(self):
        """Stop the integration pipeline"""
        self.running = False
        self.logger.info("Stopping signal-to-order integration pipeline")
        
        # Wait for threads to terminate
        if hasattr(self, 'signal_thread') and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5.0)
        
        if hasattr(self, 'order_thread') and self.order_thread.is_alive():
            self.order_thread.join(timeout=5.0)
        
        self.logger.info("Signal-to-order integration pipeline stopped")
    
    def process_signals(self):
        """Process signals from the signal queue"""
        self.logger.info("Signal processing thread started")
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Send heartbeat log periodically
                current_time = time.time()
                if current_time - last_heartbeat > 60:
                    self.logger.debug(f"Signal processing thread heartbeat")
                    last_heartbeat = current_time
                
                # Process signals from queue
                if not self.signal_processor.signal_queue.empty():
                    signal = self.signal_processor.signal_queue.get(timeout=0.1)
                    self.signal_processor.process_signal(signal)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in signal processing thread: {str(e)}")
                time.sleep(1)  # Prevent tight loop on persistent errors
        
        self.logger.info("Signal processing thread stopped")
    
    def process_orders(self):
        """Process orders from the order queue"""
        self.logger.info("Order processing thread started")
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Send heartbeat log periodically
                current_time = time.time()
                if current_time - last_heartbeat > 60:
                    self.logger.debug(f"Order processing thread heartbeat")
                    last_heartbeat = current_time
                
                # Process orders from queue
                if not self.signal_processor.order_queue.empty():
                    order = self.signal_processor.order_queue.get(timeout=0.1)
                    self.execute_order(order)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in order processing thread: {str(e)}")
                time.sleep(1)  # Prevent tight loop on persistent errors
        
        self.logger.info("Order processing thread stopped")
    
    def execute_order(self, order):
        """Execute an order
        
        Args:
            order: Order dictionary
        """
        try:
            # Log order execution attempt
            self.logger.info(f"Executing order: {order['orderId']} ({order['side']} {order['quantity']} {order['symbol']} @ {order['price']})")
            
            # Execute order
            result = self.signal_processor.paper_trading.execute_order(order['orderId'])
            
            if result:
                self.logger.info(f"Order executed successfully: {order['orderId']}")
            else:
                self.logger.warning(f"Order execution returned None: {order['orderId']}")
        except Exception as e:
            self.logger.error(f"Error executing order {order['orderId']}: {str(e)}")
    
    def add_signal(self, signal):
        """Add a signal to the processing queue
        
        Args:
            signal: Trading signal dictionary
        """
        self.signal_processor.add_signal(signal)


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = {
        'min_signal_strength': 0.5,
        'max_signal_age_ms': 5000,
        'order_creation_retries': 3,
        'order_creation_retry_delay_ms': 500,
        'risk_per_trade_pct': 1.0,
        'max_position_pct': 5.0,
        'min_position_size': 0.001,
        'position_precision': 3,
        'default_position_size': 0.001,
        'buy_price_factor': 1.001,
        'sell_price_factor': 0.999,
        'max_spread_pct': 1.0,
        'max_price_change_pct': 0.5
    }
    
    # Create integration
    integration = SignalOrderIntegration(config)
    
    # Start integration
    integration.start()
    
    # Create test signal
    test_signal = {
        'type': 'BUY',
        'source': 'order_imbalance',
        'strength': 0.75,
        'timestamp': int(time.time() * 1000),
        'price': 105000.0,
        'symbol': 'BTCUSDC',
        'session': 'US'
    }
    
    # Add signal to queue
    integration.add_signal(test_signal)
    
    # Run for a while
    try:
        print("Running signal-order integration for 30 seconds...")
        time.sleep(30)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop integration
        integration.stop()
        print("Signal-order integration stopped")
