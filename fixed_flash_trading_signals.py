#!/usr/bin/env python
"""
Refactored Flash Trading Signals with Fixed Error Handling

This module provides signal generation and decision making for flash trading,
with strict dependency injection for the client interface and fixed error handling.
"""

import time
import logging
import json
import os
import uuid
import numpy as np
from datetime import datetime, timezone
from threading import Thread, Event, RLock
from collections import deque
from trading_session_manager import TradingSessionManager
# Import fixed error handling utilities
from fixed_error_handling_utils import safe_get, safe_get_nested, validate_api_response, log_exception, parse_float_safely

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_trading_signals")

class MarketState:
    """Represents the current state of a market for signal generation"""
    
    def __init__(self, symbol):
        """Initialize market state for a symbol"""
        self.symbol = symbol
        self.timestamp = int(time.time() * 1000)
        
        # Order book state
        self.bids = []
        self.asks = []
        self.bid_price = None
        self.ask_price = None
        self.mid_price = None
        self.spread = None
        self.spread_bps = None
        self.order_imbalance = 0.0
        
        # Price history
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.timestamp_history = deque(maxlen=100)
        
        # Derived metrics
        self.momentum = 0.0
        self.volatility = 0.0
        self.trend = 0.0
        
        # Trading metrics
        self.last_trade_price = None
        self.last_trade_side = None
        self.last_trade_time = None
    
    def update_order_book(self, bids, asks):
        """Update order book state"""
        if not bids or not asks:
            return False
        
        self.bids = bids
        self.asks = asks
        self.timestamp = int(time.time() * 1000)
        
        # Update prices
        self.bid_price = float(bids[0][0])
        self.ask_price = float(asks[0][0])
        self.mid_price = (self.bid_price + self.ask_price) / 2
        self.spread = self.ask_price - self.bid_price
        self.spread_bps = (self.spread / self.mid_price) * 10000  # Basis points
        
        # Calculate order book imbalance
        bid_volume = sum(float(bid[1]) for bid in bids[:5])
        ask_volume = sum(float(ask[1]) for ask in asks[:5])
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            self.order_imbalance = (bid_volume - ask_volume) / total_volume
        
        # Update price history
        self.price_history.append(self.mid_price)
        self.timestamp_history.append(self.timestamp)
        
        # Calculate derived metrics if we have enough history
        if len(self.price_history) >= 10:
            self._calculate_derived_metrics()
        
        return True
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from price history"""
        # Convert to numpy array for calculations
        prices = np.array(list(self.price_history))
        
        # Calculate momentum (rate of change)
        if len(prices) >= 10:
            self.momentum = (prices[-1] - prices[-10]) / prices[-10]
        
        # Calculate volatility (standard deviation of returns)
        if len(prices) >= 6:  # Need at least 6 prices for 5 returns
            try:
                # Get price differences (n-1 elements)
                price_diffs = np.diff(prices[-5:])
                
                # Get denominator prices (must be same length as price_diffs)
                denominator_prices = prices[-6:-1]
                
                # Ensure both arrays have the same shape
                min_length = min(len(price_diffs), len(denominator_prices))
                price_diffs = price_diffs[:min_length]
                denominator_prices = denominator_prices[:min_length]
                
                # Calculate returns with validated shapes
                returns = price_diffs / denominator_prices
                
                # Calculate volatility
                if len(returns) > 0:
                    self.volatility = np.std(returns) * np.sqrt(len(returns))
                else:
                    self.volatility = 0.0
            except Exception as e:
                logger.error(f"Error calculating volatility: {str(e)}")
                self.volatility = 0.0
        
        # Calculate trend (simple moving average direction)
        if len(prices) >= 20:
            sma_short = np.mean(prices[-5:])
            sma_long = np.mean(prices[-20:])
            self.trend = sma_short - sma_long

# Custom decorator for API error handling
def handle_api_error(func):
    """
    Decorator for handling API errors with consistent patterns.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get function name for context
            context = func.__name__
            logger.error(f"Error in {context}: {str(e)}")
            
            # Return appropriate default based on function name
            if context.startswith("get_"):
                return {}
            elif context == "start":
                return False
            elif context == "stop":
                return False
            elif context == "generate_signals":
                return []
            else:
                return None
    
    return wrapper

class FlashTradingSignals:
    """Signal generation and decision making for flash trading"""
    
    def __init__(self, client_instance):
        """Initialize flash trading signals
        
        Args:
            client_instance: Client instance that must provide get_order_book method
        """
        # Validate client instance
        if client_instance is None:
            raise ValueError("Client instance is required for FlashTradingSignals")
        
        # Ensure client has get_order_book method
        if not hasattr(client_instance, 'get_order_book') and hasattr(client_instance, 'get_orderbook'):
            # Add adapter method if client has get_orderbook but not get_order_book
            logger.info("Adding get_order_book adapter method to client")
            
            def get_order_book(client_self, symbol, limit=20):
                """Adapter method that calls get_orderbook"""
                return client_self.get_orderbook(symbol, limit)
            
            # Add the method to the client instance
            import types
            client_instance.get_order_book = types.MethodType(get_order_book, client_instance)
        elif not hasattr(client_instance, 'get_order_book'):
            raise ValueError("Client instance must provide get_order_book method")
        
        # Store client instance
        self.api_client = client_instance
        logger.info(f"Using provided client instance of type {type(client_instance).__name__} for SignalGenerator")
        
        # Initialize session manager
        self.session_manager = TradingSessionManager()
        
        # Signal history
        self.signals = []
        self.max_signals = 1000
        
        # Market state cache
        self.market_states = {}
        
        # Thread safety for client access
        self.client_lock = RLock()
        
        # Thread safety for market state updates
        self.market_state_lock = RLock()
        
        # Configuration dictionary for compatibility with flash_trading.py
        self.config = {
            "imbalance_threshold": 0.2,
            "momentum_threshold": 0.005,
            "volatility_threshold": 0.002,
            "min_signal_strength": 0.1,
            "position_size": 0.1
        }
        
        # Thread management for compatibility with flash_trading.py
        self.running = False
        self.symbols = []
        self.update_thread = None
        self.stop_event = Event()
    
    @handle_api_error
    def start(self, symbols):
        """Start signal generation for specified symbols
        
        Args:
            symbols: List of trading pair symbols (e.g., ['BTCUSDC', 'ETHUSDC'])
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Signal generator already running")
            return False
        
        # Store symbols
        self.symbols = symbols
        
        # Initialize market states
        for symbol in symbols:
            if symbol not in self.market_states:
                self.market_states[symbol] = MarketState(symbol)
        
        # Set running flag
        self.running = True
        
        # Start market state update thread
        self.stop_event.clear()
        self.update_thread = Thread(target=self._update_market_states_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info(f"Signal generator started for symbols: {symbols}")
        return True
    
    @handle_api_error
    def stop(self):
        """Stop signal generation
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Signal generator not running")
            return False
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for thread to stop
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        # Reset running flag
        self.running = False
        
        logger.info("Signal generator stopped")
        return True
    
    def _update_market_states_loop(self):
        """Update market states in a loop"""
        logger.info("Market state update loop started")
        
        try:
            while not self.stop_event.is_set():
                # Update market states for all symbols
                for symbol in self.symbols:
                    try:
                        self._update_market_state(symbol)
                    except Exception as e:
                        logger.error(f"Error in _update_market_state for {symbol}: {str(e)}")
                
                # Sleep to avoid excessive API calls
                time.sleep(1.0)
        except Exception as e:
            logger.error(f"Error in market state update loop: {str(e)}")
        
        logger.info("Market state update loop stopped")
    
    def _update_market_state(self, symbol):
        """Update market state for a symbol
        
        Args:
            symbol: Symbol to update market state for
            
        Returns:
            bool: True if update successful, False otherwise
        """
        # Get market state
        with self.market_state_lock:
            if symbol not in self.market_states:
                self.market_states[symbol] = MarketState(symbol)
            
            market_state = self.market_states[symbol]
        
        # Get order book
        with self.client_lock:
            order_book = self.api_client.get_order_book(symbol)
        
        # Validate order book - use fixed validation that doesn't use isinstance with dynamic types
        if not order_book or not isinstance(order_book, dict) or 'bids' not in order_book or 'asks' not in order_book:
            logger.warning(f"Invalid order book for {symbol}")
            return False
        
        # Update market state
        with self.market_state_lock:
            success = market_state.update_order_book(order_book['bids'], order_book['asks'])
            
            if not success:
                logger.warning(f"Failed to update market state for {symbol}")
                return False
        
        return True
    
    @handle_api_error
    def generate_signals(self, symbol):
        """Generate trading signals for a symbol
        
        Args:
            symbol: Symbol to generate signals for
            
        Returns:
            list: List of trading signals
        """
        # Check if running
        if not self.running:
            logger.warning("Signal generator not running")
            return []
        
        # Get market state
        with self.market_state_lock:
            if symbol not in self.market_states:
                logger.warning(f"No market state for {symbol}")
                return []
            
            market_state = self.market_states[symbol]
        
        # Check if market state is initialized
        if market_state.bid_price is None or market_state.ask_price is None:
            logger.warning(f"Market state not initialized for {symbol}")
            return []
        
        # Get current session
        current_session = self.session_manager.get_current_session()
        
        # Generate signals based on market state and current session
        signals = []
        
        # Order book imbalance signal
        if abs(market_state.order_imbalance) > self.config["imbalance_threshold"]:
            signal_type = "BUY" if market_state.order_imbalance > 0 else "SELL"
            signal_strength = abs(market_state.order_imbalance)
            
            # Adjust signal strength based on session
            if current_session == "ASIAN":
                signal_strength *= 0.8  # Reduce strength during Asian session
            elif current_session == "EUROPEAN":
                signal_strength *= 1.0  # Normal strength during European session
            elif current_session == "AMERICAN":
                signal_strength *= 1.2  # Increase strength during American session
            
            # Create signal if strength exceeds threshold
            if signal_strength >= self.config["min_signal_strength"]:
                signal = {
                    "id": str(uuid.uuid4()),
                    "symbol": symbol,
                    "type": signal_type,
                    "price": market_state.mid_price,
                    "strength": signal_strength,
                    "source": "order_book_imbalance",
                    "timestamp": market_state.timestamp
                }
                
                signals.append(signal)
        
        # Momentum signal
        if abs(market_state.momentum) > self.config["momentum_threshold"]:
            signal_type = "BUY" if market_state.momentum > 0 else "SELL"
            signal_strength = abs(market_state.momentum) / 0.01  # Normalize to 0-1 range
            
            # Adjust signal strength based on session
            if current_session == "ASIAN":
                signal_strength *= 0.7  # Reduce strength during Asian session
            elif current_session == "EUROPEAN":
                signal_strength *= 1.1  # Increase strength during European session
            elif current_session == "AMERICAN":
                signal_strength *= 1.2  # Increase strength during American session
            
            # Create signal if strength exceeds threshold
            if signal_strength >= self.config["min_signal_strength"]:
                signal = {
                    "id": str(uuid.uuid4()),
                    "symbol": symbol,
                    "type": signal_type,
                    "price": market_state.mid_price,
                    "strength": signal_strength,
                    "source": "momentum",
                    "timestamp": market_state.timestamp
                }
                
                signals.append(signal)
        
        # Store signals
        if signals:
            self.signals.extend(signals)
            
            # Trim signals if needed
            if len(self.signals) > self.max_signals:
                self.signals = self.signals[-self.max_signals:]
        
        return signals
