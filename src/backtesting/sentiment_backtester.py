"""
Advanced Sentiment Strategy Backtester.

This module provides a comprehensive backtesting framework specifically designed for
sentiment-based trading strategies. It extends the modular backtester with sentiment-specific
functionality, including multi-source sentiment data integration, trend analysis,
and performance metrics tailored for sentiment strategies.
"""

import asyncio
import datetime
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.data.sentiment_collector import SentimentCollector
from src.utils.config_manager import ConfigManager

# Import the modular backtester components
from examples.modular_backtester.backtester import StrategyBacktester
from examples.modular_backtester.models import Position, Signal, TradeAction

logger = logging.getLogger(__name__)

class SentimentEvent:
    """Represents a sentiment event with source, value, and timestamp."""
    
    def __init__(self, 
                timestamp: datetime.datetime,
                source: str,
                symbol: str,
                sentiment_value: float,
                confidence: float = 1.0,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a sentiment event.
        
        Args:
            timestamp: Event timestamp
            source: Sentiment source (e.g., "fear_greed", "news")
            symbol: Trading symbol (e.g., "BTC", "ETH")
            sentiment_value: Normalized sentiment value (0-1)
            confidence: Confidence in the sentiment value (0-1)
            metadata: Additional metadata for the event
        """
        self.timestamp = timestamp
        self.source = source
        self.symbol = symbol
        self.sentiment_value = sentiment_value
        self.confidence = confidence
        self.metadata = metadata or {}
        
    def __str__(self):
        return (f"SentimentEvent(source={self.source}, symbol={self.symbol}, "
                f"value={self.sentiment_value:.2f}, confidence={self.confidence:.2f}, "
                f"timestamp={self.timestamp})")
    
    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> 'SentimentEvent':
        """Create a sentiment event from a DataFrame row."""
        metadata = {}
        for col in row.index:
            if col not in ['timestamp', 'source', 'symbol', 'sentiment_value', 'confidence']:
                metadata[col] = row[col]
        
        return cls(
            timestamp=row['timestamp'],
            source=row['source'],
            symbol=row['symbol'],
            sentiment_value=row['sentiment_value'],
            confidence=row.get('confidence', 1.0),
            metadata=metadata
        )


class SentimentStrategy:
    """Base class for sentiment-based strategies to be used with the backtester."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.symbol = config.get('symbol', 'BTC-USD')
        self.sources = config.get('sources', ['fear_greed', 'news', 'social_media', 'onchain'])
        self.source_weights = config.get('source_weights', {})
        self.sentiment_threshold_buy = config.get('sentiment_threshold_buy', 0.7)
        self.sentiment_threshold_sell = config.get('sentiment_threshold_sell', 0.3)
        self.contrarian = config.get('contrarian', False)
        self.position_size = config.get('position_size', 1.0)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = config.get('take_profit_pct', 0.10)
        self.max_positions = config.get('max_positions', 1)
        
        # Initialize state
        self.current_position = 0
        self.entry_price = 0
        self.current_sentiment = {}
        self.last_signal = None
        self.signals = []
        
    def process_candle(self, candle: pd.Series, sentiment_events: List[SentimentEvent]) -> Optional[Signal]:
        """
        Process a price candle and sentiment events to generate trading signals.
        
        Args:
            candle: Price data candle with OHLCV data
            sentiment_events: List of sentiment events for this candle period
            
        Returns:
            Trading signal or None
        """
        # Update current sentiment with new events
        for event in sentiment_events:
            self.current_sentiment[event.source] = event.sentiment_value
        
        # If no sentiment data yet, return None
        if not self.current_sentiment:
            return None
        
        # Calculate weighted average sentiment
        weighted_sentiment = self._calculate_weighted_sentiment()
        
        # Generate signal based on sentiment
        signal = self._generate_signal(weighted_sentiment, candle)
        
        if signal:
            self.last_signal = signal
            self.signals.append({
                'timestamp': candle.name,
                'signal': signal.action.name,
                'sentiment': weighted_sentiment,
                'price': candle['close']
            })
            
            # Update position tracking
            if signal.action == TradeAction.BUY:
                self.current_position += signal.quantity
                self.entry_price = candle['close']
            elif signal.action == TradeAction.SELL:
                self.current_position -= signal.quantity
        
        return signal
    
    def _calculate_weighted_sentiment(self) -> float:
        """
        Calculate weighted average sentiment from multiple sources.
        
        Returns:
            Weighted sentiment value (0-1)
        """
        # If weights not specified for all sources, use equal weights
        if not all(source in self.source_weights for source in self.current_sentiment.keys()):
            weights = {source: 1.0 / len(self.current_sentiment) for source in self.current_sentiment.keys()}
        else:
            weights = {source: self.source_weights.get(source, 1.0) for source in self.current_sentiment.keys()}
            
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        weights = {source: weight / weight_sum for source, weight in weights.items()}
        
        # Calculate weighted average
        weighted_sum = sum(self.current_sentiment[source] * weights[source] 
                          for source in self.current_sentiment.keys())
        
        return weighted_sum
    
    def _generate_signal(self, sentiment: float, candle: pd.Series) -> Optional[Signal]:
        """
        Generate trading signal based on sentiment and price data.
        
        Args:
            sentiment: Current weighted sentiment value
            candle: Price data for the current period
            
        Returns:
            Trading signal or None
        """
        # Invert sentiment for contrarian strategy
        if self.contrarian:
            sentiment = 1 - sentiment
        
        # Check for stop loss or take profit if in a position
        if self.current_position > 0:
            # Long position
            pnl_pct = (candle['close'] - self.entry_price) / self.entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return Signal(
                    timestamp=candle.name,
                    symbol=self.symbol,
                    action=TradeAction.SELL,
                    quantity=self.current_position,
                    price=candle['close'],
                    reason="Stop Loss"
                )
            
            if pnl_pct >= self.take_profit_pct:
                return Signal(
                    timestamp=candle.name,
                    symbol=self.symbol,
                    action=TradeAction.SELL,
                    quantity=self.current_position,
                    price=candle['close'],
                    reason="Take Profit"
                )
        
        elif self.current_position < 0:
            # Short position
            pnl_pct = (self.entry_price - candle['close']) / self.entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return Signal(
                    timestamp=candle.name,
                    symbol=self.symbol,
                    action=TradeAction.BUY,
                    quantity=-self.current_position,
                    price=candle['close'],
                    reason="Stop Loss"
                )
            
            if pnl_pct >= self.take_profit_pct:
                return Signal(
                    timestamp=candle.name,
                    symbol=self.symbol,
                    action=TradeAction.BUY,
                    quantity=-self.current_position,
                    price=candle['close'],
                    reason="Take Profit"
                )
                
        # Check for new position entry based on sentiment
        if sentiment >= self.sentiment_threshold_buy and self.current_position == 0:
            return Signal(
                timestamp=candle.name,
                symbol=self.symbol,
                action=TradeAction.BUY,
                quantity=self.position_size,
                price=candle['close'],
                reason=f"Sentiment: {sentiment:.2f}"
            )
        
        elif sentiment <= self.sentiment_threshold_sell and self.current_position == 0:
            return Signal(
                timestamp=candle.name,
                symbol=self.symbol,
                action=TradeAction.SELL,
                quantity=self.position_size,
                price=candle['close'],
                reason=f"Sentiment: {sentiment:.2f}"
            )
        
        # No signal
        return None


class AdvancedSentimentStrategy(SentimentStrategy):
    """
    Advanced sentiment-based strategy with trend analysis and adaptive parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced sentiment strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Advanced parameters
        self.trend_window = config.get('trend_window', 14)
        self.sentiment_history = []
        self.trend_strength_threshold = config.get('trend_strength_threshold', 0.3)
        self.use_adaptive_thresholds = config.get('use_adaptive_thresholds', True)
        self.technical_confirmation = config.get('technical_confirmation', True)
        self.regime_thresholds = config.get('regime_thresholds', {
            'low_volatility': {
                'sentiment_threshold_buy': 0.65,
                'sentiment_threshold_sell': 0.35,
                'position_size': 1.0
            },
            'high_volatility': {
                'sentiment_threshold_buy': 0.8,
                'sentiment_threshold_sell': 0.2,
                'position_size': 0.5
            }
        })
        
        # Technical indicator parameters
        self.rsi_window = config.get('rsi_window', 14)
        self.rsi_values = []
        self.volatility_window = config.get('volatility_window', 20)
        self.volatility_values = []
        
    def process_candle(self, candle: pd.Series, sentiment_events: List[SentimentEvent]) -> Optional[Signal]:
        """
        Process a price candle and sentiment events to generate trading signals.
        
        Args:
            candle: Price data candle with OHLCV data
            sentiment_events: List of sentiment events for this candle period
            
        Returns:
            Trading signal or None
        """
        # Update current sentiment with new events
        for event in sentiment_events:
            self.current_sentiment[event.source] = event.sentiment_value
        
        # If no sentiment data yet, return None
        if not self.current_sentiment:
            return None
        
        # Calculate weighted average sentiment
        weighted_sentiment = self._calculate_weighted_sentiment()
        self.sentiment_history.append(weighted_sentiment)
        
        # Calculate technical indicators
        self._update_technical_indicators(candle)
        
        # Detect market regime based on volatility
        current_regime = self._detect_market_regime()
        
        # Adapt parameters based on market regime if enabled
        if self.use_adaptive_thresholds and current_regime in self.regime_thresholds:
            regime_params = self.regime_thresholds[current_regime]
            self.sentiment_threshold_buy = regime_params.get('sentiment_threshold_buy', self.sentiment_threshold_buy)
            self.sentiment_threshold_sell = regime_params.get('sentiment_threshold_sell', self.sentiment_threshold_sell)
            self.position_size = regime_params.get('position_size', self.position_size)
        
        # Analyze sentiment trend
        trend_direction, trend_strength = self._analyze_sentiment_trend()
        
        # Generate signal based on sentiment, trend, and technical confirmation
        signal = self._generate_advanced_signal(weighted_sentiment, trend_direction, trend_strength, candle)
        
        if signal:
            self.last_signal = signal
            self.signals.append({
                'timestamp': candle.name,
                'signal': signal.action.name,
                'sentiment': weighted_sentiment,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'regime': current_regime,
                'price': candle['close']
            })
            
            # Update position tracking
            if signal.action == TradeAction.BUY:
                self.current_position += signal.quantity
                self.entry_price = candle['close']
            elif signal.action == TradeAction.SELL:
                self.current_position -= signal.quantity
        
        return signal
    
    def _update_technical_indicators(self, candle: pd.Series):
        """
        Update technical indicators based on price data.
        
        Args:
            candle: Price data for the current period
        """
        # Calculate RSI (simplified)
        close_prices = [c['close'] for c in self.rsi_values[-self.rsi_window:]] if self.rsi_values else []
        close_prices.append(candle['close'])
        self.rsi_values.append(candle)
        
        if len(close_prices) >= self.rsi_window:
            # Trim history to window size
            self.rsi_values = self.rsi_values[-self.rsi_window:]
            
        # Calculate volatility (using close prices)
        self.volatility_values.append(candle)
        if len(self.volatility_values) > self.volatility_window:
            self.volatility_values.pop(0)
    
    def _detect_market_regime(self) -> str:
        """
        Detect current market regime based on volatility.
        
        Returns:
            Market regime identifier
        """
        if len(self.volatility_values) < self.volatility_window:
            return 'normal'
        
        # Calculate volatility as standard deviation of returns
        closes = np.array([candle['close'] for candle in self.volatility_values])
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)
        
        # Determine regime based on volatility
        if volatility < 0.01:  # Low volatility threshold
            return 'low_volatility'
        elif volatility > 0.03:  # High volatility threshold
            return 'high_volatility'
        else:
            return 'normal'
    
    def _analyze_sentiment_trend(self) -> Tuple[str, float]:
        """
        Analyze the trend in sentiment values.
        
        Returns:
            Tuple of (trend_direction, trend_strength)
            where trend_direction is 'up', 'down', or 'sideways'
            and trend_strength is a float between 0 and 1
        """
        if len(self.sentiment_history) < self.trend_window:
            return 'sideways', 0.0
        
        # Get sentiment values for the trend window
        sentiment_window = self.sentiment_history[-self.trend_window:]
        
        # Calculate linear regression of sentiment
        x = np.arange(len(sentiment_window))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, sentiment_window)
        
        # Determine trend direction based on slope
        if slope > 0.001:  # Small positive threshold
            trend_direction = 'up'
        elif slope < -0.001:  # Small negative threshold
            trend_direction = 'down'
        else:
            trend_direction = 'sideways'
        
        # Use R-squared as trend strength
        trend_strength = r_value ** 2
        
        return trend_direction, trend_strength
    
    def _generate_advanced_signal(self, 
                                sentiment: float, 
                                trend_direction: str, 
                                trend_strength: float, 
                                candle: pd.Series) -> Optional[Signal]:
        """
        Generate trading signal based on sentiment, trend, and technical indicators.
        
        Args:
            sentiment: Current weighted sentiment value
            trend_direction: Direction of sentiment trend ('up', 'down', 'sideways')
            trend_strength: Strength of the sentiment trend (0-1)
            candle: Price data for the current period
            
        Returns:
            Trading signal or None
        """
        # Check for stop loss or take profit if in a position (reuse parent logic)
        if self.current_position > 0:
            # Long position
            pnl_pct = (candle['close'] - self.entry_price) / self.entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return Signal(
                    timestamp=candle.name,
                    symbol=self.symbol,
                    action=TradeAction.SELL,
                    quantity=self.current_position,
                    price=candle['close'],
                    reason="Stop Loss"
                )
            
            if pnl_pct >= self.take_profit_pct:
                return Signal(
                    timestamp=candle.name,
                    symbol=self.symbol,
                    action=TradeAction.SELL,
                    quantity=self.current_position,
                    price=candle['close'],
                    reason="Take Profit"
                )
        
        elif self.current_position < 0:
            # Short position (reuse parent logic)
            pnl_pct = (self.entry_price - candle['close']) / self.entry_price
            
            if pnl_pct <= -self.stop_loss_pct:
                return Signal(
                    timestamp=candle.name,
                    symbol=self.symbol,
                    action=TradeAction.BUY,
                    quantity=-self.current_position,
                    price=candle['close'],
                    reason="Stop Loss"
                )
            
            if pnl_pct >= self.take_profit_pct:
                return Signal(
                    timestamp=candle.name,
                    symbol=self.symbol,
                    action=TradeAction.BUY,
                    quantity=-self.current_position,
                    price=candle['close'],
                    reason="Take Profit"
                )
        
        # Invert sentiment for contrarian strategy
        if self.contrarian:
            sentiment = 1 - sentiment
        
        # Only consider strong trends that meet the threshold
        strong_trend = trend_strength >= self.trend_strength_threshold
        
        # Calculate RSI if we have enough data (simplified calculation)
        rsi = self._calculate_rsi() if len(self.rsi_values) >= self.rsi_window else 50
        
        # Check for technical confirmation if enabled
        technical_confirmed = True
        if self.technical_confirmation:
            if sentiment >= self.sentiment_threshold_buy:
                # For buy signals, RSI should not be overbought
                technical_confirmed = rsi < 70
            elif sentiment <= self.sentiment_threshold_sell:
                # For sell signals, RSI should not be oversold
                technical_confirmed = rsi > 30
        
        # Generate buy signal
        if (sentiment >= self.sentiment_threshold_buy and 
            self.current_position == 0 and
            technical_confirmed and
            (not strong_trend or trend_direction == 'up')):
            
            return Signal(
                timestamp=candle.name,
                symbol=self.symbol,
                action=TradeAction.BUY,
                quantity=self.position_size,
                price=candle['close'],
                reason=f"Sentiment: {sentiment:.2f}, Trend: {trend_direction} ({trend_strength:.2f})"
            )
        
        # Generate sell signal
        elif (sentiment <= self.sentiment_threshold_sell and 
              self.current_position == 0 and
              technical_confirmed and
              (not strong_trend or trend_direction == 'down')):
            
            return Signal(
                timestamp=candle.name,
                symbol=self.symbol,
                action=TradeAction.SELL,
                quantity=self.position_size,
                price=candle['close'],
                reason=f"Sentiment: {sentiment:.2f}, Trend: {trend_direction} ({trend_strength:.2f})"
            )
        
        # No signal
        return None
    
    def _calculate_rsi(self) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        Returns:
            RSI value (0-100)
        """
        if len(self.rsi_values) < self.rsi_window:
            return 50  # Default to neutral
        
        close_prices = [candle['close'] for candle in self.rsi_values]
        deltas = np.diff(close_prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class SentimentBacktester:
    """
    Comprehensive backtester for sentiment-based trading strategies.
    Integrates sentiment data with price data for strategy evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment backtester.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.symbol = config.get('symbol', 'BTC-USD')
        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date', datetime.datetime.now())
        self.sources = config.get('sources', ['fear_greed', 'news', 'social_media', 'onchain'])
        self.price_data_path = config.get('price_data_path', f"data/historical/{self.symbol}_1h.csv")
        self.strategy_class = self._get_strategy_class(config.get('strategy', 'SentimentStrategy'))
        self.strategy_config = config.get('strategy_config', {})
        self.initial_capital = config.get('initial_capital', 10000)
        
        # Initialize collector for historical sentiment data
        self.sentiment_collector = SentimentCollector()
        
        # Format dates if needed
        if isinstance(self.start_date, str):
            self.start_date = datetime.datetime.fromisoformat(self.start_date.replace('Z', '+00:00'))
        if isinstance(self.end_date, str):
            self.end_date = datetime.datetime.fromisoformat(self.end_date.replace('Z', '+00:00'))
    
    def _get_strategy_class(self, strategy_name: str) -> type:
        """
        Get strategy class by name.
        
        Args:
            strategy_name: Name of the strategy class
            
        Returns:
            Strategy class
        """
        if strategy_name == 'SentimentStrategy':
            return SentimentStrategy
        elif strategy_name == 'AdvancedSentimentStrategy':
            return AdvancedSentimentStrategy
        else:
            raise ValueError(f"Unknown strategy class: {strategy_name}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load historical price and sentiment data.
        
        Returns:
            Tuple of (price_data, sentiment_data)
        """
        # Load price data
        price_data = pd.read_csv(self.price_data_path)
        
        # Convert timestamp column to datetime
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        price_data.set_index('timestamp', inplace=True)
        
        # Filter to date range
        price_data = price_data[(price_data.index >= self.start_date) & 
                               (price_data.index <= self.end_date)]
        
        # Load sentiment data from each source
        sentiment_dfs = []
        for source in self.sources:
            try:
                df = self.sentiment_collector.load_historical_data(
                    source=source,
                    symbol=self.symbol.split('-')[0],  # Extract base currency
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                sentiment_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {source} sentiment data: {e}")
        
        if not sentiment_dfs:
            raise ValueError("No sentiment data available for backtesting")
        
        # Combine all sentiment data
        sentiment_data = pd.concat(sentiment_dfs)
        sentiment_data.set_index('timestamp', inplace=True)
        
        return price_data, sentiment_data
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the backtest with the configured strategy and data.
        
        Returns:
            Dictionary of backtest results
        """
        # Load data
        price_data, sentiment_data = self.load_data()
        
        # Initialize strategy
        strategy = self.strategy_class(self.strategy_config)
        
        # Initialize model backtester
        backtester = StrategyBacktester(
            initial_capital=self.initial_capital,
            commission_rate=self.config.get('commission_rate', 0.001)
        )
        
        # Prepare for backtest
        signals = []
        positions = []
        equity_curve = [self.initial_capital]
        current_equity = self.initial_capital
        
        # Run through each candle
        for timestamp, candle in price_data.iterrows():
            # Get sentiment events for this candle
            candle_sentiment = sentiment_data[
                (sentiment_data.index >= timestamp - pd.Timedelta(hours=1)) & 
                (sentiment_data.index <= timestamp)
            ]
            
            # Convert sentiment data to events
            sentiment_events = []
            for _, row in candle_sentiment.iterrows():
                # Add timestamp and symbol to row if not present
                row_with_defaults = row.copy()
                if 'timestamp' not in row_with_defaults:
                    row_with_defaults['timestamp'] = row.name
                if 'symbol' not in row_with_defaults:
                    row_with_defaults['symbol'] = self.symbol.split('-')[0]
                
                sentiment_events.append(SentimentEvent.from_dataframe_row(row_with_defaults))
            
            # Process candle with the strategy
            signal = strategy.process_candle(candle, sentiment_events)
            
            if signal:
                signals.append(signal)
                
                # Execute trade and update position
                trade_result = backtester.execute_signal(signal, candle)
                positions.append(trade_result)
                
                # Update equity
                current_equity = backtester.get_current_equity(candle['close'])
                
            equity_curve.append(current_equity)
        
        # Calculate performance metrics
        metrics = backtester.calculate_metrics()
        
        # Add sentiment-specific metrics
        metrics.update(self._calculate_sentiment_metrics(sentiment_data, price_data, signals))
        
        # Prepare results
        results = {
            'metrics': metrics,
            'equity_curve': pd.Series(equity_curve, index=[price_data.index[0]] + list(price_data.index)),
            'signals': signals,
            'positions': positions,
            'strategy': strategy,
            'config': self.config
        }
        
        return results
    
    def _calculate_sentiment_metrics(self, 
                                    sentiment_data: pd.DataFrame, 
                                    price_data: pd.DataFrame,
                                    signals: List[Signal]) -> Dict[str, float]:
        """
        Calculate sentiment-specific performance metrics.
        
        Args:
            sentiment_data: Sentiment data used in backtest
            price_data: Price data used in backtest
            signals: Trading signals generated during backtest
            
        Returns:
            Dictionary of sentiment-specific metrics
        """
        metrics = {}
        
        # If no signals, return empty metrics
        if not signals:
            return metrics
        
        # Signal correlation with sentiment
        signal_times = [signal.timestamp for signal in signals]
        signal_values = [1 if signal.action == TradeAction.BUY else -1 if signal.action == TradeAction.SELL else 0 
                        for signal in signals]
        
        signal_df = pd.DataFrame({
            'signal': signal_values
        }, index=signal_times)
        
        # Resample sentiment data to same frequency as price data
        freq = pd.infer_freq(price_data.index)
        if freq:
            resampled_sentiment = sentiment_data.resample(freq).mean()
        else:
            # If can't infer frequency, use daily
            resampled_sentiment = sentiment_data.resample('D').mean()
        
        # Calculate metrics per source
        for source in sentiment_data['source'].unique():
            source_sentiment = resampled_sentiment[resampled_sentiment['source'] == source]['sentiment_value']
            
            if not source_sentiment.empty and not signal_df.empty:
                # Align indexes
                aligned_sentiment = source_sentiment.reindex(signal_df.index, method='ffill')
                
                # Calculate correlation if we have enough data points
                if len(aligned_sentiment) >= 2:
                    corr = signal_df['signal'].corr(aligned_sentiment)
                    metrics[f'signal_sentiment_corr_{source}'] = corr
        
        # Calculate profit ratio by sentiment level (binned)
        profit_by_sentiment = {}
        for signal, next_signal in zip(signals[:-1], signals[1:]):
            # Skip if not a complete trade
            if (signal.action == TradeAction.BUY and next_signal.action != TradeAction.SELL) or \
               (signal.action == TradeAction.SELL and next_signal.action != TradeAction.BUY):
                continue
            
            # Calculate profit
            if signal.action == TradeAction.BUY:
                profit_pct = (next_signal.price - signal.price) / signal.price
            else:
                profit_pct = (signal.price - next_signal.price) / signal.price
            
            # Get sentiment at signal time
            signal_sentiment = 0
            sentiment_at_signal = sentiment_data[sentiment_data.index <= signal.timestamp]
            if not sentiment_at_signal.empty:
                # Get the most recent sentiment value
                signal_sentiment = sentiment_at_signal.iloc[-1]['sentiment_value']
            
            # Bin sentiment
            sentiment_bin = round(signal_sentiment * 10) / 10  # Round to nearest 0.1
            
            if sentiment_bin not in profit_by_sentiment:
                profit_by_sentiment[sentiment_bin] = []
            
            profit_by_sentiment[sentiment_bin].append(profit_pct)
        
        # Calculate average profit by sentiment bin
        for bin_value, profits in profit_by_sentiment.items():
            metrics[f'avg_profit_sentiment_{bin_value:.1f}'] = sum(profits) / len(profits) if profits else 0
        
        return metrics
    
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive backtest report.
        
        Args:
            results: Backtest results from run_backtest()
            output_path: Path to save the report (optional)
            
        Returns:
            Report text
        """
        metrics = results['metrics']
        equity_curve = results['equity_curve']
        signals = results['signals']
        
        # Generate report text
        report = []
        report.append("=" * 50)
        report.append(f"SENTIMENT STRATEGY BACKTEST REPORT: {self.symbol}")
        report.append("=" * 50)
        report.append(f"Period: {self.start_date} to {self.end_date}")
        report.append(f"Strategy: {self.strategy_class.__name__}")
        report.append(f"Initial Capital: ${self.initial_capital:.2f}")
        report.append(f"Final Equity: ${equity_curve.iloc[-1]:.2f}")
        report.append(f"Total Return: {(equity_curve.iloc[-1] / self.initial_capital - 1) * 100:.2f}%")
        report.append("-" * 50)
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 50)
        
        # Add standard metrics
        for metric, value in metrics.items():
            if metric.startswith('signal_sentiment_corr_') or metric.startswith('avg_profit_sentiment_'):
                continue  # These will be displayed in their own sections
            if isinstance(value, float):
                report.append(f"{metric}: {value:.4f}")
            else:
                report.append(f"{metric}: {value}")
        
        # Add sentiment correlation metrics
        report.append("-" * 50)
        report.append("SENTIMENT-SIGNAL CORRELATION:")
        report.append("-" * 50)
        for metric, value in metrics.items():
            if metric.startswith('signal_sentiment_corr_'):
                source = metric.replace('signal_sentiment_corr_', '')
                report.append(f"{source}: {value:.4f}")
        
        # Add profit by sentiment
        report.append("-" * 50)
        report.append("PROFIT BY SENTIMENT LEVEL:")
        report.append("-" * 50)
        sentiment_bins = sorted([float(m.replace('avg_profit_sentiment_', '')) 
                                for m in metrics if m.startswith('avg_profit_sentiment_')])
        
        for bin_value in sentiment_bins:
            metric = f'avg_profit_sentiment_{bin_value:.1f}'
            if metric in metrics:
                report.append(f"Sentiment {bin_value:.1f}: {metrics[metric]*100:.2f}%")
        
        # Add trade summary
        report.append("-" * 50)
        report.append("TRADE SUMMARY:")
        report.append("-" * 50)
        report.append(f"Total Signals: {len(signals)}")
        
        buy_signals = sum(1 for s in signals if s.action == TradeAction.BUY)
        sell_signals = sum(1 for s in signals if s.action == TradeAction.SELL)
        report.append(f"Buy Signals: {buy_signals}")
        report.append(f"Sell Signals: {sell_signals}")
        
        report_text = "\n".join(report)
        
        # Save report if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def visualize_results(self, results: Dict[str, Any], output_path: Optional[str] = None):
        """
        Visualize backtest results with matplotlib.
        
        Args:
            results: Backtest results from run_backtest()
            output_path: Path to save the visualization (optional)
        """
        equity_curve = results['equity_curve']
        signals = results['signals']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(equity_curve.index, equity_curve.values, label='Equity')
        ax1.set_title(f'Sentiment Strategy Backtest Results - {self.symbol}')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot buy and sell signals if available
        if signals:
            buy_signals = [s for s in signals if s.action == TradeAction.BUY]
            sell_signals = [s for s in signals if s.action == TradeAction.SELL]
            
            if buy_signals:
                buy_x = [s.timestamp for s in buy_signals]
                buy_y = [equity_curve.loc[s.timestamp] if s.timestamp in equity_curve.index else None 
                         for s in buy_signals]
                buy_y = [y for y in buy_y if y is not None]
                
                if buy_x and buy_y and len(buy_x) == len(buy_y):
                    ax1.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy Signal')
            
            if sell_signals:
                sell_x = [s.timestamp for s in sell_signals]
                sell_y = [equity_curve.loc[s.timestamp] if s.timestamp in equity_curve.index else None 
                          for s in sell_signals]
                sell_y = [y for y in sell_y if y is not None]
                
                if sell_x and sell_y and len(sell_x) == len(sell_y):
                    ax1.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell Signal')
        
        # Load price data for the bottom chart
        try:
            price_data = pd.read_csv(self.price_data_path)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            price_data.set_index('timestamp', inplace=True)
            
            # Filter to date range
            price_data = price_data[(price_data.index >= self.start_date) & 
                                  (price_data.index <= self.end_date)]
            
            # Plot price
            ax2.plot(price_data.index, price_data['close'], color='blue', label='Price')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price ($)')
            ax2.grid(True)
            ax2.legend()
            
            # Add buy/sell markers to price chart too
            if signals:
                buy_signals = [s for s in signals if s.action == TradeAction.BUY]
                sell_signals = [s for s in signals if s.action == TradeAction.SELL]
                
                if buy_signals:
                    buy_x = [s.timestamp for s in buy_signals]
                    buy_y = [s.price for s in buy_signals]
                    ax2.scatter(buy_x, buy_y, color='green', marker='^', s=100)
                
                if sell_signals:
                    sell_x = [s.timestamp for s in sell_signals]
                    sell_y = [s.price for s in sell_signals]
                    ax2.scatter(sell_x, sell_y, color='red', marker='v', s=100)
                    
        except Exception as e:
            logger.warning(f"Failed to plot price data: {e}")
        
        plt.tight_layout()
        
        # Save figure if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Saved visualization to {output_path}")
        
        plt.show()

    def run_parameter_optimization(self, 
                                  param_grid: Dict[str, List[Any]],
                                  metric: str = 'sharpe_ratio',
                                  report: bool = True) -> Dict[str, Any]:
        """
        Run parameter optimization using grid search.
        
        Args:
            param_grid: Dictionary of parameter names and possible values
            metric: Metric to optimize for
            report: Whether to print optimization report
            
        Returns:
            Dictionary with best parameters and results
        """
        best_score = -float('inf') if 'sharpe' in metric.lower() or 'return' in metric.lower() else float('inf')
        best_params = None
        best_results = None
        
        logger.info(f"Starting parameter optimization with {param_grid}")
        
        # Generate all parameter combinations
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        results = []
        
        # Run backtest for each parameter combination
        for i, combination in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Update strategy config
            strategy_config = self.strategy_config.copy()
            strategy_config.update(params)
            self.strategy_config = strategy_config
            
            logger.info(f"Testing parameters {i+1}/{len(param_combinations)}: {params}")
            
            # Run backtest
            try:
                backtest_results = self.run_backtest()
                
                # Extract optimization metric
                if metric in backtest_results['metrics']:
                    score = backtest_results['metrics'][metric]
                    
                    result = {
                        'params': params,
                        'score': score,
                        'metrics': backtest_results['metrics']
                    }
                    
                    results.append(result)
                    
                    # Update best parameters if better
                    if ('sharpe' in metric.lower() or 'return' in metric.lower()):
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_results = backtest_results
                    else:
                        if score < best_score:
                            best_score = score
                            best_params = params
                            best_results = backtest_results
                    
                    logger.info(f"Parameters {params} - {metric}: {score}")
                
                else:
                    logger.warning(f"Metric {metric} not found in backtest results")
            
            except Exception as e:
                logger.error(f"Error in parameter optimization: {e}")
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=('sharpe' in metric.lower() or 'return' in metric.lower()))
        
        # Print optimization report
        if report:
            print("\n" + "=" * 50)
            print("PARAMETER OPTIMIZATION RESULTS")
            print("=" * 50)
            print(f"Optimizing for: {metric}")
            
            for i, result in enumerate(results[:10]):  # Show top 10
                print(f"\nRank {i+1}:")
                print(f"Parameters: {result['params']}")
                print(f"{metric}: {result['score']}")
                print("Other metrics:")
                for m, val in result['metrics'].items():
                    if m != metric:
                        print(f"  {m}: {val}")
        
        # Return best results
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'all_results': results
        }


def main():
    """Example usage of the sentiment backtester."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Define configuration
    config = {
        'symbol': 'BTC-USD',
        'start_date': '2023-01-01',
        'end_date': '2023-03-31',
        'sources': ['fear_greed'],
        'price_data_path': 'data/historical/BTC-USD_1h.csv',
        'strategy': 'AdvancedSentimentStrategy',
        'strategy_config': {
            'sentiment_threshold_buy': 0.7,
            'sentiment_threshold_sell': 0.3,
            'contrarian': False,
            'position_size': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'trend_window': 14,
            'technical_confirmation': True
        },
        'initial_capital': 10000,
        'commission_rate': 0.001
    }
    
    # Initialize backtester
    backtester = SentimentBacktester(config)
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Generate report
    report = backtester.generate_report(results, 'reports/sentiment_backtest_report.txt')
    print(report)
    
    # Visualize results
    backtester.visualize_results(results, 'reports/sentiment_backtest_plot.png')
    
    # Run parameter optimization
    param_grid = {
        'sentiment_threshold_buy': [0.6, 0.7, 0.8],
        'sentiment_threshold_sell': [0.2, 0.3, 0.4],
        'trend_window': [7, 14, 21]
    }
    
    optimization_results = backtester.run_parameter_optimization(param_grid)
    print(f"\nBest parameters: {optimization_results['best_params']}")
    print(f"Best score: {optimization_results['best_score']}")

if __name__ == "__main__":
    main()