#!/usr/bin/env python
"""
Enhanced Flash Trading Signals with Technical Indicators

This module extends the original flash_trading_signals.py with advanced technical indicators,
multi-timeframe analysis, and dynamic thresholding for improved signal generation.
"""

import time
import logging
import json
import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from threading import Thread, Event, RLock
from collections import deque

# Import mock client for testing
from mock_exchange_client import MockExchangeClient

# Try to import real client, but don't fail if not available
try:
    from optimized_mexc_client import OptimizedMEXCClient
    REAL_CLIENT_AVAILABLE = True
except ImportError:
    REAL_CLIENT_AVAILABLE = False
    logging.warning("OptimizedMEXCClient not available, using MockExchangeClient for all operations")

try:
    from trading_session_manager import TradingSessionManager
    from indicators import TechnicalIndicators
    from error_handling_utils import safe_get, safe_get_nested, validate_api_response, handle_api_error, log_exception, parse_float_safely
except ImportError:
    # Create minimal versions for testing
    class TradingSessionManager:
        def get_current_session(self):
            return "EUROPE"
    
    class TechnicalIndicators:
        @staticmethod
        def calculate_all_indicators(market_data):
            return {
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_hist": 0.0,
                "bollinger_upper": market_data['close'][-1] * 1.02,
                "bollinger_middle": market_data['close'][-1],
                "bollinger_lower": market_data['close'][-1] * 0.98,
                "atr": market_data['close'][-1] * 0.01
            }
    
    def handle_api_error(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"API error in {func.__name__}: {str(e)}")
                return None
        return wrapper
    
    def safe_get(data, key, default=None):
        return data.get(key, default) if data else default
    
    def safe_get_nested(data, keys, default=None):
        for key in keys:
            if not data or key not in data:
                return default
            data = data[key]
        return data
    
    def validate_api_response(response):
        return True
    
    def log_exception(e, context=""):
        logging.error(f"Exception in {context}: {str(e)}")
    
    def parse_float_safely(value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_flash_trading_signals")

class EnhancedMarketState:
    """Enhanced market state with technical indicators and multi-timeframe support"""
    
    def __init__(self, symbol):
        """Initialize enhanced market state for a symbol"""
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
        
        # Price history (multi-timeframe)
        self.price_history = {
            '1m': deque(maxlen=300),    # 5 hours of 1-minute data
            '5m': deque(maxlen=288),    # 24 hours of 5-minute data
            '15m': deque(maxlen=192),   # 48 hours of 15-minute data
            '1h': deque(maxlen=168)     # 7 days of 1-hour data
        }
        self.volume_history = {
            '1m': deque(maxlen=300),
            '5m': deque(maxlen=288),
            '15m': deque(maxlen=192),
            '1h': deque(maxlen=168)
        }
        self.timestamp_history = {
            '1m': deque(maxlen=300),
            '5m': deque(maxlen=288),
            '15m': deque(maxlen=192),
            '1h': deque(maxlen=168)
        }
        
        # High/Low price history for ATR calculation
        self.high_price_history = {
            '1m': deque(maxlen=300),
            '5m': deque(maxlen=288),
            '15m': deque(maxlen=192),
            '1h': deque(maxlen=168)
        }
        self.low_price_history = {
            '1m': deque(maxlen=300),
            '5m': deque(maxlen=288),
            '15m': deque(maxlen=192),
            '1h': deque(maxlen=168)
        }
        
        # Technical indicators
        self.indicators = {
            '1m': {},
            '5m': {},
            '15m': {},
            '1h': {}
        }
        
        # Derived metrics
        self.momentum = 0.0
        self.volatility = 0.0
        self.trend = 0.0
        
        # Trading metrics
        self.last_trade_price = None
        self.last_trade_side = None
        self.last_trade_time = None
        
        # Liquidity metrics
        self.bid_liquidity = 0.0
        self.ask_liquidity = 0.0
        self.slippage_estimate = 0.0
        
        # Timeframe management
        self.last_candle_close = {
            '1m': 0,
            '5m': 0,
            '15m': 0,
            '1h': 0
        }
    
    def update_order_book(self, bids, asks):
        """Update order book state and calculate liquidity metrics"""
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
        
        # Calculate liquidity metrics
        self._calculate_liquidity_metrics(bids, asks)
        
        # Update price history for all timeframes
        self._update_price_history()
        
        # Calculate technical indicators for all timeframes
        self._calculate_technical_indicators()
        
        # Calculate derived metrics if we have enough history
        if len(self.price_history['1m']) >= 10:
            self._calculate_derived_metrics()
        
        return True
    
    def _calculate_liquidity_metrics(self, bids, asks):
        """Calculate liquidity and slippage metrics from order book"""
        try:
            # Calculate total liquidity within 1% of mid price
            price_range = self.mid_price * 0.01  # 1% range
            
            # Bid liquidity (sum of volumes for bids within 1% of mid price)
            self.bid_liquidity = sum(
                float(bid[1]) for bid in bids 
                if float(bid[0]) >= self.mid_price - price_range
            )
            
            # Ask liquidity (sum of volumes for asks within 1% of mid price)
            self.ask_liquidity = sum(
                float(ask[1]) for ask in asks 
                if float(ask[0]) <= self.mid_price + price_range
            )
            
            # Estimate slippage for a standard order size (0.1 BTC/ETH)
            standard_order_size = 0.1
            
            # Simulate market buy
            buy_slippage = self._estimate_slippage(asks, standard_order_size, "buy")
            
            # Simulate market sell
            sell_slippage = self._estimate_slippage(bids, standard_order_size, "sell")
            
            # Average slippage
            self.slippage_estimate = (buy_slippage + sell_slippage) / 2
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {str(e)}")
    
    def _estimate_slippage(self, orders, size, side):
        """Estimate slippage for a given order size and side"""
        try:
            remaining_size = size
            executed_value = 0.0
            
            for order in orders:
                price = float(order[0])
                volume = float(order[1])
                
                if remaining_size <= 0:
                    break
                
                executed_volume = min(remaining_size, volume)
                executed_value += executed_volume * price
                remaining_size -= executed_volume
            
            # If we couldn't fill the entire order
            if remaining_size > 0:
                return 0.05  # Assume 5% slippage for unfillable orders
            
            # Calculate average execution price
            avg_price = executed_value / size
            
            # Calculate slippage in basis points
            if side == "buy":
                slippage_bps = (avg_price - self.ask_price) / self.ask_price * 10000
            else:
                slippage_bps = (self.bid_price - avg_price) / self.bid_price * 10000
            
            return max(0, slippage_bps)
            
        except Exception as e:
            logger.error(f"Error estimating slippage: {str(e)}")
            return 0.0
    
    def _update_price_history(self):
        """Update price history for all timeframes"""
        current_time = self.timestamp
        
        # Always update 1m data
        self._update_timeframe_data('1m', current_time)
        
        # Update other timeframes if their candle has closed
        if self._is_candle_closed('5m', current_time):
            self._update_timeframe_data('5m', current_time)
            
        if self._is_candle_closed('15m', current_time):
            self._update_timeframe_data('15m', current_time)
            
        if self._is_candle_closed('1h', current_time):
            self._update_timeframe_data('1h', current_time)
    
    def _is_candle_closed(self, timeframe, current_time):
        """Check if a candle has closed for the given timeframe"""
        minutes_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60}
        minutes = minutes_map.get(timeframe, 1)
        
        # Convert to minutes since epoch
        current_minutes = current_time // (60 * 1000)
        last_update_minutes = self.last_candle_close[timeframe] // (60 * 1000)
        
        # Check if we've crossed a candle boundary
        return (current_minutes // minutes) > (last_update_minutes // minutes)
    
    def _update_timeframe_data(self, timeframe, current_time):
        """Update price history for a specific timeframe"""
        self.price_history[timeframe].append(self.mid_price)
        self.timestamp_history[timeframe].append(current_time)
        
        # Use bid/ask as proxy for high/low in absence of actual candle data
        self.high_price_history[timeframe].append(self.ask_price)
        self.low_price_history[timeframe].append(self.bid_price)
        
        # Update last candle close time
        self.last_candle_close[timeframe] = current_time
    
    def _calculate_technical_indicators(self):
        """Calculate technical indicators for all timeframes"""
        for timeframe in self.price_history.keys():
            if len(self.price_history[timeframe]) < 20:
                continue  # Not enough data
                
            # Convert deques to numpy arrays
            prices = np.array(list(self.price_history[timeframe]))
            highs = np.array(list(self.high_price_history[timeframe]))
            lows = np.array(list(self.low_price_history[timeframe]))
            
            # Create market data dictionary
            market_data = {
                'close': prices,
                'high': highs,
                'low': lows,
                'timestamp': np.array(list(self.timestamp_history[timeframe]))
            }
            
            # Calculate all indicators
            self.indicators[timeframe] = TechnicalIndicators.calculate_all_indicators(market_data)
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from price history"""
        # Use 1-minute timeframe for these calculations
        prices = np.array(list(self.price_history['1m']))
        
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

class EnhancedFlashTradingSignals:
    """Enhanced signal generation with technical indicators and multi-timeframe analysis"""
    
    def __init__(self, client_instance=None, pattern_recognition=None, api_key=None, api_secret=None, env_path=None):
        """Initialize enhanced flash trading signals
        
        Args:
            client_instance: Existing client instance to use (preferred)
            pattern_recognition: Pattern recognition service instance (optional)
            api_key: API key for exchange (used only if client_instance is None)
            api_secret: API secret for exchange (used only if client_instance is None)
            env_path: Path to .env file (used only if client_instance is None)
        """
        # Use existing client instance if provided, otherwise create new one
        if client_instance is not None:
            self.api_client = client_instance
            logger.info("Using provided client instance for EnhancedSignalGenerator")
        else:
            # Try to use real client if available, otherwise use mock
            if REAL_CLIENT_AVAILABLE:
                try:
                    # Initialize API client with direct credentials
                    self.api_client = OptimizedMEXCClient(api_key, api_secret, env_path)
                    logger.info("Created new OptimizedMEXCClient instance for EnhancedSignalGenerator")
                except Exception as e:
                    logger.warning(f"Failed to create OptimizedMEXCClient: {str(e)}")
                    logger.info("Falling back to MockExchangeClient")
                    self.api_client = MockExchangeClient()
            else:
                # Use mock client
                logger.info("Using MockExchangeClient for EnhancedSignalGenerator")
                self.api_client = MockExchangeClient()
        
        # Store pattern recognition service if provided
        self.pattern_recognition = pattern_recognition
        
        # Initialize session manager
        self.session_manager = TradingSessionManager()
        
        # Initialize market state cache
        self.market_states = {}
        self.market_state_lock = RLock()
        
        # Initialize signal cache
        self.signal_cache = {}
        self.signal_cache_ttl = {}
        self.signal_cache_lock = RLock()
        
        # Initialize signal thresholds
        self.signal_thresholds = {
            'momentum': 0.005,  # 0.5% price change
            'volatility': 0.01,  # 1% standard deviation
            'spread': 0.0015,   # 15 basis points
            'imbalance': 0.2,   # 20% order book imbalance
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_surge': 2.0  # 2x average volume
        }
        
        # Initialize background thread for market data updates
        self.update_thread = None
        self.stop_event = Event()
        self.update_interval = 5  # seconds
        
        # Start background thread if using real client
        if not isinstance(self.api_client, MockExchangeClient):
            self._start_background_updates()
    
    def _start_background_updates(self):
        """Start background thread for market data updates"""
        if self.update_thread is not None and self.update_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.update_thread = Thread(target=self._background_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("Started background market data update thread")
    
    def _stop_background_updates(self):
        """Stop background thread for market data updates"""
        if self.update_thread is None or not self.update_thread.is_alive():
            return
        
        self.stop_event.set()
        self.update_thread.join(timeout=10)
        logger.info("Stopped background market data update thread")
    
    def _background_update_loop(self):
        """Background loop for updating market data"""
        symbols = ["BTC/USDC", "ETH/USDT"]  # Default symbols to monitor
        
        while not self.stop_event.is_set():
            try:
                # Update market data for each symbol
                for symbol in symbols:
                    self._update_market_state(symbol)
                    
                    # Don't overwhelm the API
                    time.sleep(1)
                    
                    # Check if we should stop
                    if self.stop_event.is_set():
                        break
                
                # Wait for next update cycle
                self.stop_event.wait(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in background update loop: {str(e)}")
                time.sleep(self.update_interval)
    
    def _update_market_state(self, symbol):
        """Update market state for a symbol"""
        try:
            # Get order book
            symbol_formatted = symbol.replace("/", "")
            order_book = self.api_client.get_order_book(symbol_formatted)
            
            if not order_book or not validate_api_response(order_book):
                logger.warning(f"Invalid order book response for {symbol}")
                return False
            
            # Get bids and asks
            bids = safe_get(order_book, 'bids', [])
            asks = safe_get(order_book, 'asks', [])
            
            if not bids or not asks:
                logger.warning(f"Empty order book for {symbol}")
                return False
            
            # Get or create market state
            with self.market_state_lock:
                if symbol not in self.market_states:
                    self.market_states[symbol] = EnhancedMarketState(symbol)
                
                # Update market state
                return self.market_states[symbol].update_order_book(bids, asks)
                
        except Exception as e:
            logger.error(f"Error updating market state for {symbol}: {str(e)}")
            return False
    
    def get_signals(self, symbol, timeframe='5m', limit=200, max_signals=None):
        """Get trading signals for a symbol and timeframe
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDC")
            timeframe: Timeframe (e.g., "1m", "5m", "15m", "1h")
            limit: Number of candles to fetch
            max_signals: Maximum number of signals to return
            
        Returns:
            list: Trading signals
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            signals = self._get_cached_signals(cache_key)
            
            if signals is not None:
                logger.info(f"Using cached signals for {symbol} {timeframe}")
                return signals[:max_signals] if max_signals else signals
            
            # Get candles
            symbol_formatted = symbol.replace("/", "")
            candles = self.api_client.get_klines(
                symbol=symbol_formatted,
                interval=timeframe,
                limit=limit
            )
            
            if not candles:
                logger.warning(f"No candles returned for {symbol} {timeframe}")
                return []
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            
            # Generate signals
            signals = []
            
            # Add technical indicator signals
            signals.extend(self._generate_technical_signals(df, symbol, timeframe))
            
            # Add pattern recognition signals if available
            if self.pattern_recognition:
                try:
                    pattern_signals = self.pattern_recognition.detect_patterns(
                        df, symbol, timeframe, max_patterns=max_signals
                    )
                    if pattern_signals:
                        signals.extend(pattern_signals)
                        logger.info(f"Added {len(pattern_signals)} pattern signals for {symbol} {timeframe}")
                except Exception as e:
                    logger.error(f"Error generating pattern signals: {str(e)}")
            
            # Add order book signals
            order_book_signals = self._generate_order_book_signals(symbol)
            if order_book_signals:
                signals.extend(order_book_signals)
            
            # Cache signals
            self._cache_signals(cache_key, signals)
            
            # Limit number of signals if requested
            if max_signals and len(signals) > max_signals:
                return signals[:max_signals]
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {str(e)}")
            return []
    
    def _generate_technical_signals(self, df, symbol, timeframe):
        """Generate signals based on technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            list: Technical indicator signals
        """
        signals = []
        
        try:
            # Calculate technical indicators
            df = self._calculate_indicators(df)
            
            # RSI signals
            self._add_rsi_signals(df, signals, symbol, timeframe)
            
            # MACD signals
            self._add_macd_signals(df, signals, symbol, timeframe)
            
            # Bollinger Band signals
            self._add_bollinger_signals(df, signals, symbol, timeframe)
            
            # Volume signals
            self._add_volume_signals(df, signals, symbol, timeframe)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating technical signals: {str(e)}")
            return []
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators for DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with indicators
        """
        try:
            # Create market data dictionary
            market_data = {
                'close': df['close'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'timestamp': df['timestamp'].values
            }
            
            # Calculate indicators
            indicators = TechnicalIndicators.calculate_all_indicators(market_data)
            
            # Add indicators to DataFrame
            for indicator, values in indicators.items():
                if isinstance(values, np.ndarray) and len(values) == len(df):
                    df[indicator] = values
                elif isinstance(values, (int, float)):
                    df[indicator] = values
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def _add_rsi_signals(self, df, signals, symbol, timeframe):
        """Add RSI signals to signal list
        
        Args:
            df: DataFrame with indicators
            signals: Signal list to append to
            symbol: Trading symbol
            timeframe: Timeframe
        """
        if 'rsi' not in df.columns:
            return
        
        try:
            # Get latest RSI value
            latest_rsi = df['rsi'].iloc[-1]
            
            # Check for oversold condition
            if latest_rsi < self.signal_thresholds['rsi_oversold']:
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'technical',
                    'subtype': 'rsi_oversold',
                    'value': float(latest_rsi),
                    'threshold': self.signal_thresholds['rsi_oversold'],
                    'direction': 'buy',
                    'confidence': 0.7,
                    'metadata': {
                        'indicator': 'rsi',
                        'current_price': float(df['close'].iloc[-1])
                    }
                })
            
            # Check for overbought condition
            elif latest_rsi > self.signal_thresholds['rsi_overbought']:
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'technical',
                    'subtype': 'rsi_overbought',
                    'value': float(latest_rsi),
                    'threshold': self.signal_thresholds['rsi_overbought'],
                    'direction': 'sell',
                    'confidence': 0.7,
                    'metadata': {
                        'indicator': 'rsi',
                        'current_price': float(df['close'].iloc[-1])
                    }
                })
                
        except Exception as e:
            logger.error(f"Error adding RSI signals: {str(e)}")
    
    def _add_macd_signals(self, df, signals, symbol, timeframe):
        """Add MACD signals to signal list
        
        Args:
            df: DataFrame with indicators
            signals: Signal list to append to
            symbol: Trading symbol
            timeframe: Timeframe
        """
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return
        
        try:
            # Get latest values
            latest_macd = df['macd'].iloc[-1]
            latest_signal = df['macd_signal'].iloc[-1]
            prev_macd = df['macd'].iloc[-2] if len(df) > 1 else 0
            prev_signal = df['macd_signal'].iloc[-2] if len(df) > 1 else 0
            
            # Check for bullish crossover
            if prev_macd < prev_signal and latest_macd > latest_signal:
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'technical',
                    'subtype': 'macd_bullish_crossover',
                    'value': float(latest_macd - latest_signal),
                    'threshold': 0,
                    'direction': 'buy',
                    'confidence': 0.65,
                    'metadata': {
                        'indicator': 'macd',
                        'macd': float(latest_macd),
                        'signal': float(latest_signal),
                        'current_price': float(df['close'].iloc[-1])
                    }
                })
            
            # Check for bearish crossover
            elif prev_macd > prev_signal and latest_macd < latest_signal:
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'technical',
                    'subtype': 'macd_bearish_crossover',
                    'value': float(latest_macd - latest_signal),
                    'threshold': 0,
                    'direction': 'sell',
                    'confidence': 0.65,
                    'metadata': {
                        'indicator': 'macd',
                        'macd': float(latest_macd),
                        'signal': float(latest_signal),
                        'current_price': float(df['close'].iloc[-1])
                    }
                })
                
        except Exception as e:
            logger.error(f"Error adding MACD signals: {str(e)}")
    
    def _add_bollinger_signals(self, df, signals, symbol, timeframe):
        """Add Bollinger Band signals to signal list
        
        Args:
            df: DataFrame with indicators
            signals: Signal list to append to
            symbol: Trading symbol
            timeframe: Timeframe
        """
        if 'bollinger_upper' not in df.columns or 'bollinger_lower' not in df.columns:
            return
        
        try:
            # Get latest values
            latest_close = df['close'].iloc[-1]
            latest_upper = df['bollinger_upper'].iloc[-1]
            latest_lower = df['bollinger_lower'].iloc[-1]
            
            # Check for price above upper band
            if latest_close > latest_upper:
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'technical',
                    'subtype': 'bollinger_upper_break',
                    'value': float(latest_close),
                    'threshold': float(latest_upper),
                    'direction': 'sell',
                    'confidence': 0.6,
                    'metadata': {
                        'indicator': 'bollinger',
                        'upper_band': float(latest_upper),
                        'lower_band': float(latest_lower),
                        'current_price': float(latest_close)
                    }
                })
            
            # Check for price below lower band
            elif latest_close < latest_lower:
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'technical',
                    'subtype': 'bollinger_lower_break',
                    'value': float(latest_close),
                    'threshold': float(latest_lower),
                    'direction': 'buy',
                    'confidence': 0.6,
                    'metadata': {
                        'indicator': 'bollinger',
                        'upper_band': float(latest_upper),
                        'lower_band': float(latest_lower),
                        'current_price': float(latest_close)
                    }
                })
                
        except Exception as e:
            logger.error(f"Error adding Bollinger signals: {str(e)}")
    
    def _add_volume_signals(self, df, signals, symbol, timeframe):
        """Add volume signals to signal list
        
        Args:
            df: DataFrame with indicators
            signals: Signal list to append to
            symbol: Trading symbol
            timeframe: Timeframe
        """
        if 'volume' not in df.columns or len(df) < 20:
            return
        
        try:
            # Calculate average volume
            avg_volume = df['volume'].iloc[-20:-1].mean()
            latest_volume = df['volume'].iloc[-1]
            latest_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            
            # Check for volume surge
            if latest_volume > avg_volume * self.signal_thresholds['volume_surge']:
                # Determine direction based on price movement
                direction = 'buy' if latest_close > prev_close else 'sell'
                
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'technical',
                    'subtype': 'volume_surge',
                    'value': float(latest_volume / avg_volume),
                    'threshold': self.signal_thresholds['volume_surge'],
                    'direction': direction,
                    'confidence': 0.55,
                    'metadata': {
                        'indicator': 'volume',
                        'current_volume': float(latest_volume),
                        'avg_volume': float(avg_volume),
                        'current_price': float(latest_close),
                        'price_change': float((latest_close - prev_close) / prev_close)
                    }
                })
                
        except Exception as e:
            logger.error(f"Error adding volume signals: {str(e)}")
    
    def _generate_order_book_signals(self, symbol):
        """Generate signals based on order book
        
        Args:
            symbol: Trading symbol
            
        Returns:
            list: Order book signals
        """
        signals = []
        
        try:
            # Get market state
            with self.market_state_lock:
                if symbol not in self.market_states:
                    return []
                
                market_state = self.market_states[symbol]
            
            # Check for significant order imbalance
            if abs(market_state.order_imbalance) > self.signal_thresholds['imbalance']:
                direction = 'buy' if market_state.order_imbalance > 0 else 'sell'
                
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': '1m',  # Order book signals are short-term
                    'type': 'order_book',
                    'subtype': 'order_imbalance',
                    'value': float(market_state.order_imbalance),
                    'threshold': self.signal_thresholds['imbalance'],
                    'direction': direction,
                    'confidence': 0.5 + min(0.3, abs(market_state.order_imbalance) * 0.5),
                    'metadata': {
                        'bid_price': float(market_state.bid_price),
                        'ask_price': float(market_state.ask_price),
                        'spread_bps': float(market_state.spread_bps),
                        'bid_liquidity': float(market_state.bid_liquidity),
                        'ask_liquidity': float(market_state.ask_liquidity)
                    }
                })
            
            # Check for tight spread
            if market_state.spread_bps < self.signal_thresholds['spread']:
                signals.append({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'timeframe': '1m',
                    'type': 'order_book',
                    'subtype': 'tight_spread',
                    'value': float(market_state.spread_bps),
                    'threshold': self.signal_thresholds['spread'],
                    'direction': 'neutral',
                    'confidence': 0.4,
                    'metadata': {
                        'bid_price': float(market_state.bid_price),
                        'ask_price': float(market_state.ask_price),
                        'spread': float(market_state.spread),
                        'mid_price': float(market_state.mid_price)
                    }
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating order book signals: {str(e)}")
            return []
    
    def _get_cached_signals(self, cache_key, ttl=60):
        """Get cached signals
        
        Args:
            cache_key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            list: Cached signals or None if not found or expired
        """
        with self.signal_cache_lock:
            if cache_key in self.signal_cache:
                # Check if cache is expired
                if time.time() - self.signal_cache_ttl.get(cache_key, 0) < ttl:
                    return self.signal_cache[cache_key]
                
                # Remove expired cache
                del self.signal_cache[cache_key]
                del self.signal_cache_ttl[cache_key]
        
        return None
    
    def _cache_signals(self, cache_key, signals):
        """Cache signals
        
        Args:
            cache_key: Cache key
            signals: Signals to cache
        """
        with self.signal_cache_lock:
            self.signal_cache[cache_key] = signals
            self.signal_cache_ttl[cache_key] = time.time()
    
    def __del__(self):
        """Clean up resources"""
        self._stop_background_updates()
