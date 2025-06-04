"""
Flash Trading Signals with Session Awareness

This module provides signal generation and decision making for flash trading,
with dynamic adaptation based on the current global trading session.
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
from optimized_mexc_client import OptimizedMEXCClient
from trading_session_manager import TradingSessionManager
from error_handling_utils import safe_get, safe_get_nested, validate_api_response, handle_api_error, log_exception, parse_float_safely

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

class FlashTradingSignals:
    """Signal generation and decision making for flash trading"""
    
    def __init__(self, client_instance=None, api_key=None, api_secret=None, env_path=None):
        """Initialize flash trading signals
        
        Args:
            client_instance: Existing OptimizedMEXCClient instance to use (preferred)
            api_key: API key for MEXC (used only if client_instance is None)
            api_secret: API secret for MEXC (used only if client_instance is None)
            env_path: Path to .env file (used only if client_instance is None)
        """
        # Use existing client instance if provided, otherwise create new one
        if client_instance is not None and isinstance(client_instance, OptimizedMEXCClient):
            self.api_client = client_instance
            logger.info("Using provided client instance for SignalGenerator")
        else:
            # Initialize API client with direct credentials
            self.api_client = OptimizedMEXCClient(api_key, api_secret, env_path)
            logger.info("Created new client instance for SignalGenerator")
        
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
            
        if not symbols or not isinstance(symbols, list):
            logger.error(f"Invalid symbols list: {symbols}")
            return False
            
        # Store symbols
        self.symbols = symbols
        
        # Reset stop event
        self.stop_event = Event()
        
        # Start update thread
        self.update_thread = Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Set running state
        self.running = True
        
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
            
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
            
        # Set running state
        self.running = False
        
        logger.info("Signal generator stopped")
        return True
        
    @handle_api_error
    def _update_loop(self):
        """Background thread for updating market states"""
        try:
            logger.info("Market state update loop started")
            
            while not self.stop_event.is_set():
                try:
                    # Update market state for each symbol
                    for symbol in self.symbols:
                        if self.stop_event.is_set():
                            break
                            
                        try:
                            self._update_market_state(symbol)
                        except Exception as e:
                            log_exception(e, f"_update_market_state for {symbol}")
                            
                    # Sleep for a short interval
                    self.stop_event.wait(0.5)
                    
                except Exception as e:
                    log_exception(e, "_update_loop iteration")
                    # Sleep before retrying
                    self.stop_event.wait(1.0)
                    
        except Exception as e:
            log_exception(e, "_update_loop")
        finally:
            logger.info("Market state update loop stopped")
    @handle_api_error
    def _update_market_state(self, symbol):
        """Update market state for a symbol with thread safety and robust validation
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # DIAGNOSTIC: Log before API call
        logger.debug(f"Updating market state for {symbol}")
        
        try:
            # Thread-safe client access
            with self.client_lock:
                # Get order book with diagnostics
                logger.debug(f"Requesting order book for {symbol}")
                order_book = self.api_client.get_order_book(symbol, limit=20)
                logger.debug(f"Order book response type: {type(order_book)}")
            
            # Validate order book structure with diagnostics
            if order_book is None:
                logger.error(f"CRITICAL: Null order book response for {symbol}")
                return False
                
            if not isinstance(order_book, dict):
                logger.error(f"Invalid order book response type for {symbol}: {type(order_book)}")
                return False
                
            if 'bids' not in order_book or 'asks' not in order_book:
                logger.error(f"Missing bids or asks in order book for {symbol}")
                return False
            
            # Safe access with validation
            bids = safe_get(order_book, "bids", [])
            asks = safe_get(order_book, "asks", [])
            
            if not bids or not asks:
                logger.warning(f"Empty bids or asks in order book for {symbol}")
                return False
            
            # Validate bid/ask structure with diagnostics
            try:
                # Validate at least one valid bid and ask
                if len(bids) == 0 or len(asks) == 0:
                    logger.warning(f"No bids or asks available for {symbol}")
                    return False
                
                # Validate bid/ask format
                if not isinstance(bids[0], list) or len(bids[0]) < 2:
                    logger.error(f"Invalid bid format for {symbol}: {bids[0]}")
                    return False
                
                if not isinstance(asks[0], list) or len(asks[0]) < 2:
                    logger.error(f"Invalid ask format for {symbol}: {asks[0]}")
                    return False
                
                # Thread-safe market state update
                with self.market_state_lock:
                    # Create market state if it doesn't exist
                    if symbol not in self.market_states:
                        self.market_states[symbol] = MarketState(symbol)
                    
                    # Update market state
                    return self.market_states[symbol].update_order_book(bids, asks)
            except Exception as e:
                log_exception(e, f"_update_market_state validation for {symbol}")
                return False
                
        except Exception as e:
            log_exception(e, f"_update_market_state for {symbol}")
            return False
    
    @handle_api_error
    def generate_signals(self, symbol):
        """Generate trading signals for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            
        Returns:
            list: List of trading signals
        """
        # DIAGNOSTIC: Log before generating signals
        logger.debug(f"Generating signals for {symbol}")
        
        try:
            # Update market state
            if not self._update_market_state(symbol):
                logger.warning(f"Failed to update market state for {symbol}")
                return []
            
            # Get current trading session
            current_session = self.session_manager.get_current_session_name()
            
            # Thread-safe market state access
            with self.market_state_lock:
                if symbol not in self.market_states:
                    logger.warning(f"No market state available for {symbol}")
                    return []
                
                market_state = self.market_states[symbol]
            
            # Get session-specific parameters
            session_params = self.session_manager.get_session_parameters(current_session)
            
            # Extract thresholds from session parameters with safe defaults
            imbalance_threshold = safe_get(session_params, "imbalance_threshold", 0.2)
            momentum_threshold = safe_get(session_params, "momentum_threshold", 0.005)
            volatility_threshold = safe_get(session_params, "volatility_threshold", 0.002)
            
            signals = []
            
            # Order imbalance signal
            if abs(market_state.order_imbalance) > imbalance_threshold:
                signal_type = "BUY" if market_state.order_imbalance > 0 else "SELL"
                signals.append({
                    "type": signal_type,
                    "source": "order_imbalance",
                    "strength": abs(market_state.order_imbalance),
                    "timestamp": int(time.time() * 1000),
                    "price": market_state.mid_price,
                    "symbol": symbol,
                    "session": current_session
                })
            
            # Momentum signal
            normalized_momentum = market_state.momentum / market_state.mid_price
            if abs(normalized_momentum) > momentum_threshold:
                signal_type = "BUY" if normalized_momentum > 0 else "SELL"
                signals.append({
                    "type": signal_type,
                    "source": "momentum",
                    "strength": abs(normalized_momentum),
                    "timestamp": int(time.time() * 1000),
                    "price": market_state.mid_price,
                    "symbol": symbol,
                    "session": current_session
                })
            
            # Store signals
            if signals:
                self.signals.extend(signals)
                
                # Trim signals if needed
                if len(self.signals) > self.max_signals:
                    self.signals = self.signals[-self.max_signals:]
            
            return signals
            
        except Exception as e:
            log_exception(e, f"generate_signals for {symbol}")
            return []
    
    @handle_api_error
    def generate_signal(self, symbol, candles_short=None, candles_long=None):
        """Generate a single trading signal for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDC')
            candles_short: Short-term candles for technical analysis
            candles_long: Long-term candles for technical analysis
            
        Returns:
            dict: Trading signal or None if no signal
        """
        # DIAGNOSTIC: Log before generating signal
        logger.debug(f"Generating signal for {symbol}")
        
        try:
            # Update market state
            api_symbol = symbol.replace('/', '')
            if not self._update_market_state(api_symbol):
                logger.warning(f"Failed to update market state for {symbol}")
                return None
            
            # Get current trading session
            current_session = self.session_manager.get_current_session_name()
            
            # Thread-safe market state access
            with self.market_state_lock:
                if api_symbol not in self.market_states:
                    logger.warning(f"No market state available for {symbol}")
                    return None
                
                market_state = self.market_states[api_symbol]
            
            # Get session-specific parameters
            session_params = self.session_manager.get_session_parameters(current_session)
            
            # Extract thresholds from session parameters with safe defaults
            imbalance_threshold = safe_get(session_params, "imbalance_threshold", 0.2)
            momentum_threshold = safe_get(session_params, "momentum_threshold", 0.005)
            volatility_threshold = safe_get(session_params, "volatility_threshold", 0.002)
            
            # Determine signal direction
            signal_strength = 0.0
            signal_direction = None
            signal_reason = []
            
            # Order imbalance signal
            if abs(market_state.order_imbalance) > imbalance_threshold:
                direction = "BUY" if market_state.order_imbalance > 0 else "SELL"
                strength = abs(market_state.order_imbalance)
                
                if signal_direction is None:
                    signal_direction = direction
                    signal_strength = strength
                    signal_reason.append("order_imbalance")
                elif signal_direction == direction:
                    signal_strength += strength
                    signal_reason.append("order_imbalance")
            
            # Momentum signal
            normalized_momentum = market_state.momentum / market_state.mid_price
            if abs(normalized_momentum) > momentum_threshold:
                direction = "BUY" if normalized_momentum > 0 else "SELL"
                strength = abs(normalized_momentum) / momentum_threshold
                
                if signal_direction is None:
                    signal_direction = direction
                    signal_strength = strength
                    signal_reason.append("momentum")
                elif signal_direction == direction:
                    signal_strength += strength
                    signal_reason.append("momentum")
            
            # Technical analysis from candles
            if candles_short and candles_long:
                # Simple moving average crossover
                if len(candles_short) >= 10 and len(candles_long) >= 20:
                    try:
                        # Calculate short-term SMA (10 periods)
                        short_prices = [candle["close"] for candle in candles_short[-10:]]
                        short_sma = sum(short_prices) / len(short_prices)
                        
                        # Calculate long-term SMA (20 periods)
                        long_prices = [candle["close"] for candle in candles_long[-20:]]
                        long_sma = sum(long_prices) / len(long_prices)
                        
                        # SMA crossover signal
                        if short_sma > long_sma:
                            direction = "BUY"
                            strength = (short_sma / long_sma - 1.0) * 10.0
                        else:
                            direction = "SELL"
                            strength = (long_sma / short_sma - 1.0) * 10.0
                        
                        if signal_direction is None:
                            signal_direction = direction
                            signal_strength = strength
                            signal_reason.append("sma_crossover")
                        elif signal_direction == direction:
                            signal_strength += strength
                            signal_reason.append("sma_crossover")
                    except Exception as e:
                        log_exception(e, "Technical analysis calculation")
            
            # Create signal if direction is determined and strength is sufficient
            if signal_direction and signal_strength >= self.config["min_signal_strength"]:
                signal_id = str(uuid.uuid4())
                
                signal = {
                    "id": signal_id,
                    "symbol": symbol,
                    "direction": signal_direction,
                    "strength": signal_strength,
                    "price": market_state.mid_price,
                    "timestamp": int(time.time() * 1000),
                    "session": current_session,
                    "reason": signal_reason
                }
                
                # Store signal
                self.signals.append(signal)
                
                # Trim signals if needed
                if len(self.signals) > self.max_signals:
                    self.signals = self.signals[-self.max_signals:]
                
                logger.info(f"Generated {signal_direction} signal for {symbol} with strength {signal_strength:.4f}")
                return signal
            
            return None
            
        except Exception as e:
            log_exception(e, f"generate_signal for {symbol}")
            return None

# Alias FlashTradingSignals as SignalGenerator for compatibility with existing code
SignalGenerator = FlashTradingSignals
