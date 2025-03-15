#!/usr/bin/env python
"""
Example script demonstrating a simplified sentiment strategy backtester.
This implements a standalone backtester without relying on complex dependencies.
"""

import sys
import os

# Add the project root to the Python path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
import itertools
import random
from src.optimization.genetic_optimizer import GeneticOptimizer
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Simplified dataclasses for backtesting


class SimpleSentimentEvent:
    """A simplified representation of a sentiment event."""
    
    def __init__(
        self,
        source: str,
        symbol: str,
        sentiment_value: float,
        sentiment_direction: str,
        confidence: float,
        timestamp: datetime
    ):
        """Initialize a sentiment event.
        
        Args:
            source: Source of the sentiment (e.g., "twitter", "news")
            symbol: Trading symbol (e.g., "BTC/USDT")
            sentiment_value: Sentiment value in range [-1, 1] 
            sentiment_direction: Direction ("bullish", "bearish", "neutral")
            confidence: Confidence level [0, 1]
            timestamp: Event timestamp
        """
        self.source = source
        self.symbol = symbol
        self.sentiment_value = sentiment_value
        self.sentiment_direction = sentiment_direction
        self.confidence = confidence
        self.timestamp = timestamp


class SimpleSentimentBacktester:
    """A simplified backtester for the sentiment strategy."""
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        sentiment_data: List[SimpleSentimentEvent],
        sentiment_bull_threshold: float = 0.6,
        sentiment_bear_threshold: float = 0.4,
        min_confidence: float = 0.7,
        contrarian_mode: bool = False,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.1,
        transaction_cost: float = 0.0,
        initial_capital: float = 1000.0,
        position_sizing: str = 'fixed',  # 'fixed', 'volatility', 'kelly'
        risk_per_trade: float = 0.02,    # For volatility-based sizing
        volatility_window: int = 20,     # For volatility calculation
        kelly_fraction: float = 0.5      # Kelly criterion scaling factor
    ):
        """
        Initialize the backtester.
        
        Args:
            price_data: DataFrame with price data (should have 'timestamp' and 'price' columns)
            sentiment_data: List of sentiment events
            sentiment_bull_threshold: Threshold for bullish sentiment (higher means more bullish)
            sentiment_bear_threshold: Threshold for bearish sentiment (lower means more bearish)
            min_confidence: Minimum confidence level for sentiment events
            contrarian_mode: If True, go against the sentiment signal
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            transaction_cost: Transaction cost as percentage of position size
            initial_capital: Initial capital for the backtest
            position_sizing: Method for position sizing ('fixed', 'volatility', 'kelly')
            risk_per_trade: Risk per trade as percentage of capital for volatility-based sizing
            volatility_window: Window for volatility calculation
            kelly_fraction: Fraction of full Kelly to use (0.5 = half Kelly)
        """
        # Store inputs
        self.price_data = price_data
        self.sentiment_data = sentiment_data
        self.sentiment_bull_threshold = sentiment_bull_threshold
        self.sentiment_bear_threshold = sentiment_bear_threshold
        self.min_confidence = min_confidence
        self.contrarian_mode = contrarian_mode
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.volatility_window = volatility_window
        self.kelly_fraction = kelly_fraction
        
        # Initialize state
        self.position = None
        self.equity = initial_capital
        self.equity_history = [initial_capital]
        self.equity_timestamps = [self.price_data['timestamp'].iloc[0]]
        self.trades = []
        self.signals = []  # Add back the signals list
        
        # Calculate daily returns and volatility if needed
        if position_sizing in ['volatility', 'kelly']:
            self.price_data['return'] = self.price_data['close'].pct_change()
            self.price_data['volatility'] = self.price_data['return'].rolling(
                window=volatility_window).std()
            
            # Forward fill volatility to avoid NaN values
            self.price_data['volatility'] = self.price_data['volatility'].fillna(
                method='ffill').fillna(0.01)  # Default to 1% if no data
                
        # Store equity curve attribute for plotting
        self.equity_curve = None
    
    def run_backtest(self):
        """Run the backtest.
        
        Returns:
            Dict with backtest results
        """
        logger.info("Starting backtest")
        
        # Combine price and sentiment data for chronological processing
        events = []
        
        # Add price data
        for _, row in self.price_data.iterrows():
            events.append({
                'type': 'price',
                'timestamp': row['timestamp'],
                'data': {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'] if 'volume' in row else 0
                }
            })
        
        # Add sentiment data
        for event in self.sentiment_data:
            events.append({
                'type': 'sentiment',
                'timestamp': event.timestamp,
                'data': event
            })
        
        # Sort events by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        # Process events
        for event in events:
            if event['type'] == 'price':
                # Process price update
                candle = {
                    'timestamp': event['timestamp'],
                    'open': event['data']['open'],
                    'high': event['data']['high'],
                    'low': event['data']['low'],
                    'close': event['data']['close'],
                    'volume': event['data']['volume']
                }
                self._process_price_update(candle)
                
                # Update equity curve
                current_equity = self._calculate_equity(candle['close'])
                self.equity_history.append(current_equity)
                self.equity_timestamps.append(event['timestamp'])
            
            elif event['type'] == 'sentiment':
                # Process sentiment update
                self._process_sentiment(event['data'])
        
        # Liquidate any final position at the last price
        if self.position:
            final_price = self.price_data['close'].iloc[-1]
            self._close_position(final_price, self.price_data['timestamp'].iloc[-1], "end_of_backtest")
        
        # Create equity curve DataFrame for visualization
        self.equity_curve = pd.DataFrame({
            'timestamp': self.equity_timestamps,
            'equity': self.equity_history
        })
        
        # Calculate and return performance metrics
        results = self._calculate_performance_metrics()
        logger.info(f"Backtest completed with {len(self.trades)} trades")
        
        return results
    
    def _process_price_update(self, candle):
        """Process a price update.
        
        Args:
            candle: The price candle data
        """
        # Check for stop loss or take profit
        if self.position and self.position['entry_price'] > 0:
            current_price = candle['close']
            
            # Check stop loss for long positions
            if self.position['direction'] == "long" and current_price <= self.position['entry_price'] * (1 - self.stop_loss_pct):
                self._close_position(current_price, candle['timestamp'], "stop_loss")
                
            # Check take profit for long positions
            elif self.position['direction'] == "long" and current_price >= self.position['entry_price'] * (1 + self.take_profit_pct):
                self._close_position(current_price, candle['timestamp'], "take_profit")
                
            # Check stop loss for short positions
            elif self.position['direction'] == "short" and current_price >= self.position['entry_price'] * (1 + self.stop_loss_pct):
                self._close_position(current_price, candle['timestamp'], "stop_loss")
                
            # Check take profit for short positions
            elif self.position['direction'] == "short" and current_price <= self.position['entry_price'] * (1 - self.take_profit_pct):
                self._close_position(current_price, candle['timestamp'], "take_profit")
    
    def _process_sentiment(self, event: SimpleSentimentEvent):
        """Process a sentiment event.
        
        Args:
            event: The sentiment event
        """
        # Skip events that don't meet our minimum confidence
        if event.confidence < self.min_confidence:
            return
            
        # Get the current price (use the last price in our data)
        # Find the closest price before this sentiment timestamp
        prices_before = self.price_data[self.price_data['timestamp'] <= event.timestamp]
        if len(prices_before) == 0:
            logger.warning(f"No price data before sentiment event at {event.timestamp}")
            return
            
        current_price = prices_before['close'].iloc[-1]
        
        # Determine signal direction based on sentiment
        signal_direction = None
        
        # Generate trading signal based on sentiment thresholds
        if event.sentiment_value >= self.sentiment_bull_threshold:
            signal_direction = "short" if self.contrarian_mode else "long"
        elif event.sentiment_value <= self.sentiment_bear_threshold:
            signal_direction = "long" if self.contrarian_mode else "short"
            
        # No signal if sentiment is in the neutral zone
        if signal_direction is None:
            return
            
        # Record the signal
        signal = {
            'timestamp': event.timestamp,
            'sentiment_value': event.sentiment_value,
            'direction': signal_direction,
            'price': current_price,
            'confidence': event.confidence
        }
        self.signals.append(signal)
        
        # Execute the trade if there's a change in position
        if self.position is None or signal_direction != self.position['direction']:
            # Close existing position if any
            if self.position:
                self._close_position(current_price, event.timestamp, "signal")
                
            # Open new position
            self._open_position(signal_direction, current_price, event.timestamp, event.source)
    
    def _find_closest_price_idx(self, timestamp: datetime) -> Optional[int]:
        """Find the index of the closest price data to the given timestamp.
        
        Args:
            timestamp: The timestamp to find the closest price data for
            
        Returns:
            The index of the closest price data, or None if no data is found
        """
        # Find prices before this timestamp
        prices_before = self.price_data[self.price_data['timestamp'] <= timestamp]
        
        if len(prices_before) == 0:
            logger.warning(f"No price data before sentiment event at {timestamp}")
            return None
            
        # Return the index of the most recent price data
        return prices_before.index[-1]
    
    def _open_position(self, direction: str, price: float, timestamp: datetime, source: str):
        """Open a new trading position.
        
        Args:
            direction: Trading direction ("long" or "short")
            price: Entry price
            timestamp: Entry timestamp
            source: Signal source
        """
        # Calculate position size
        position_size = self._calculate_position_size(price, self.equity)
        
        self.position = {
            'direction': direction,
            'entry_price': price,
            'entry_time': timestamp,
            'size': position_size
        }
        
        logger.info(f"Opening {direction} position at {price} ({timestamp}), size={position_size:.2f}")
        
        # Record the trade
        self.trades.append({
            'type': 'entry',
            'direction': direction,
            'price': price,
            'timestamp': timestamp,
            'size': position_size,
            'source': source
        })
    
    def _close_position(self, price: float, timestamp: datetime, reason: str):
        """Close the current trading position.
        
        Args:
            price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing ("signal", "stop_loss", "take_profit", "end_of_backtest")
        """
        if not self.position:
            return
            
        # Calculate P&L
        if self.position['direction'] == "long":
            pnl = (price / self.position['entry_price'] - 1) * self.position['size']
        else:  # short
            pnl = (self.position['entry_price'] / price - 1) * self.position['size']
            
        # Apply transaction cost
        pnl -= self.position['size'] * self.transaction_cost * 2
        
        # Update equity
        self.equity += pnl
        
        logger.info(f"Closing {self.position['direction']} position at {price} ({timestamp}), PnL={pnl:.2f}, reason={reason}")
        
        # Record the trade
        self.trades.append({
            'type': 'exit',
            'direction': self.position['direction'],
            'entry_price': self.position['entry_price'],
            'exit_price': price,
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'pnl': pnl,
            'pnl_percent': pnl / self.position['size'] * 100 if self.position['size'] > 0 else 0,
            'reason': reason
        })
        
        # Reset position data
        self.position = None
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity based on capital and any open position.
        
        Args:
            current_price: Current price
            
        Returns:
            Current equity value
        """
        equity = self.equity
        
        # Add unrealized P&L if there's an open position
        if self.position and self.position['entry_price'] > 0:
            if self.position['direction'] == "long":
                unrealized_pnl = (current_price / self.position['entry_price'] - 1) * self.position['size']
            else:  # short
                unrealized_pnl = (self.position['entry_price'] / current_price - 1) * self.position['size']
                
            equity += unrealized_pnl
            
        return equity
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # If no trades, return empty results
        if len(self.trades) == 0:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'avg_trade_return': 0.0,
                'sharpe_ratio': 0.0,
                'equity_curve': self.equity_curve
            }
            
        # Calculate trade metrics
        completed_trades = [t for t in self.trades if 'exit_time' in t]
        total_trades = len(completed_trades)
        win_trades = [t for t in completed_trades if t['pnl'] > 0]
        loss_trades = [t for t in completed_trades if t['pnl'] <= 0]
        
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor (gross profit / gross loss)
        gross_profit = sum(t['pnl'] for t in win_trades) if win_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in loss_trades)) if loss_trades else 1  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate total return
        initial_capital = self.initial_capital
        final_equity = self.equity_history[-1]
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = self.equity_history[0]
        
        for equity in self.equity_history:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate average trade return
        avg_trade_return = sum(t['pnl_percent'] for t in completed_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (if we have enough data)
        if len(self.equity_history) > 1:
            equity_returns = np.diff(self.equity_history) / self.equity_history[:-1]
            sharpe_ratio = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if np.std(equity_returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        return {
            'total_return': total_return,
            'win_rate': win_rate * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'avg_trade_return': avg_trade_return,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': self.equity_curve
        }
    
    def _calculate_position_size(self, current_price, current_equity):
        """
        Calculate position size based on the selected position sizing method.
        
        Args:
            current_price: Current price of the asset
            current_equity: Current equity in the account
            
        Returns:
            Position size in units of the asset
        """
        # Set aside a fraction for transaction costs
        available_equity = current_equity * (1 - self.transaction_cost)
        
        if self.position_sizing == 'fixed':
            # Use 95% of equity for position
            return 0.95 * available_equity / current_price
        
        elif self.position_sizing == 'volatility':
            # Get current volatility
            current_volatility = self.price_data['volatility'].iloc[-1]
            
            # Calculate position size based on risk and volatility
            # Position size = (Risk amount) / (Asset volatility * price)
            risk_amount = current_equity * self.risk_per_trade
            position_size = risk_amount / (current_volatility * current_price)
            
            # Cap at 95% of equity
            max_size = 0.95 * available_equity / current_price
            return min(position_size, max_size)
        
        elif self.position_sizing == 'kelly':
            # Simplified Kelly criterion
            # Using historical win rate and average win/loss ratio
            if not self.trades:
                return 0.5 * available_equity / current_price
            
            # Make sure to use profit_pct instead of pnl (handle different trade dictionary structure)
            wins = [t for t in self.trades if t.get('profit_pct', 0) > 0]
            losses = [t for t in self.trades if t.get('profit_pct', 0) <= 0]
            
            if not losses:
                win_rate = 1.0
                win_loss_ratio = 1.0
            else:
                win_rate = len(wins) / len(self.trades)
                avg_win = sum(t.get('profit_pct', 0) for t in wins) / len(wins) if wins else 0
                avg_loss = abs(sum(t.get('profit_pct', 0) for t in losses) / len(losses)) if losses else 1
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            
            # Kelly formula: f = (p * b - q) / b
            # where p = win probability, q = loss probability (1-p), b = win/loss ratio
            kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
            
            # Apply fraction of Kelly and cap at 95% of equity
            adjusted_kelly = max(0, kelly_pct * self.kelly_fraction)
            position_size = adjusted_kelly * available_equity / current_price
            
            max_size = 0.95 * available_equity / current_price
            return min(position_size, max_size)
        
        else:
            # Default to fixed sizing
            return 0.95 * available_equity / current_price
    
    def plot_results(self):
        """Plot the backtest results."""
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot price and trades on the first subplot
        ax1.plot(self.price_data.index, self.price_data['close'], label='Price', color='black', alpha=0.7)
        
        # Convert trades list to DataFrame if it's not already and if we have trades
        if self.trades and isinstance(self.trades, list):
            trades_df = pd.DataFrame(self.trades)
        elif isinstance(self.trades, pd.DataFrame):
            trades_df = self.trades
        else:
            trades_df = pd.DataFrame()  # Empty DataFrame if no trades
            
        # Plot entry and exit points if we have trades
        if not trades_df.empty:
            # Check if required columns exist
            trade_columns = trades_df.columns.tolist()
            
            # Long entries
            if 'direction' in trade_columns and 'type' in trade_columns:
                long_entries = trades_df[(trades_df['direction'] == 'long') & (trades_df['type'] == 'entry')]
                if not long_entries.empty and 'timestamp' in trade_columns and 'price' in trade_columns:
                    ax1.scatter(long_entries['timestamp'], long_entries['price'], 
                               color='green', marker='^', s=100, label='Long Entry')
            
                # Short entries
                short_entries = trades_df[(trades_df['direction'] == 'short') & (trades_df['type'] == 'entry')]
                if not short_entries.empty and 'timestamp' in trade_columns and 'price' in trade_columns:
                    ax1.scatter(short_entries['timestamp'], short_entries['price'], 
                               color='red', marker='v', s=100, label='Short Entry')
            
                # Exits
                exits = trades_df[trades_df['type'] == 'exit']
                if not exits.empty and 'exit_time' in trade_columns and 'exit_price' in trade_columns:
                    ax1.scatter(exits['exit_time'], exits['exit_price'], 
                               color='black', marker='x', s=100, label='Exit')
        
        ax1.set_title('Price and Trades')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot equity curve on the second subplot
        df_equity = pd.DataFrame(self.equity_history, columns=['equity'])
        df_equity['timestamp'] = self.equity_timestamps
        df_equity.set_index('timestamp', inplace=True)
        
        ax2.plot(df_equity.index, df_equity['equity'], label='Equity', color='blue')
        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Equity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Store equity curve for easy access
        self.equity_curve = df_equity


def generate_sample_price_data(days: int = 180) -> pd.DataFrame:
    """Generate sample price data for backtesting.
    
    Args:
        days: Number of days of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    # Start date
    start_date = datetime.now() - timedelta(days=days)
    
    # Generate dates (daily data)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate price data with a random walk
    close_prices = [5000.0]  # Start price
    for i in range(1, days):
        # Random daily return between -3% and 3%
        daily_return = np.random.normal(0.0005, 0.02)  # Slight upward bias
        close_prices.append(close_prices[-1] * (1 + daily_return))
    
    # Generate OHLC data based on close prices
    opens = [close_prices[0]]
    for i in range(1, days):
        opens.append(close_prices[i-1] * (1 + np.random.normal(0, 0.005)))
    
    highs = []
    lows = []
    for i in range(days):
        high_pct = np.random.uniform(0, 0.02)
        low_pct = np.random.uniform(0, 0.02)
        highs.append(max(opens[i], close_prices[i]) * (1 + high_pct))
        lows.append(min(opens[i], close_prices[i]) * (1 - low_pct))
    
    # Generate volumes
    volumes = [np.random.uniform(500, 2000) for _ in range(days)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })
    
    return df


def generate_sample_sentiment_data(price_data: pd.DataFrame, 
                                  sources: List[str] = None) -> List[SimpleSentimentEvent]:
    """Generate sample sentiment data for backtesting.
    
    This creates sentiment data that loosely correlates with price movements.
    
    Args:
        price_data: The price data to correlate with
        sources: List of sentiment sources to generate
        
    Returns:
        List of SimpleSentimentEvent objects
    """
    if sources is None:
        sources = ["twitter", "news", "reddit"]
    
    symbol = "BTC/USDT"  # Default symbol
    events = []
    
    # Parameters for each source
    source_params = {
        "twitter": {
            "frequency": 8,    # events per day
            "lag": 0,          # days lag behind price
            "correlation": 0.6, # correlation with price
            "noise": 0.4,      # random noise
            "confidence_mean": 0.7,
            "confidence_std": 0.15
        },
        "news": {
            "frequency": 4,
            "lag": 1,
            "correlation": 0.7,
            "noise": 0.3,
            "confidence_mean": 0.8,
            "confidence_std": 0.1
        },
        "reddit": {
            "frequency": 6,
            "lag": 2,
            "correlation": 0.5,
            "noise": 0.5,
            "confidence_mean": 0.6,
            "confidence_std": 0.2
        }
    }
    
    # Calculate daily price changes
    price_data['pct_change'] = price_data['close'].pct_change()
    
    # Generate sentiment for each source
    for source in sources:
        params = source_params[source]
        
        # Determine the frequency of updates (in days)
        update_freq_days = 1 / params["frequency"]
        
        # Generate sentiment events
        for i in range(0, len(price_data), max(1, int(update_freq_days))):
            # Get the price change data with lag
            lag_idx = max(0, i - params["lag"])
            if lag_idx >= len(price_data):
                continue
                
            # Base sentiment on price change, with correlation factor and noise
            price_change = price_data['pct_change'].iloc[lag_idx]
            if pd.isna(price_change):
                price_change = 0
            
            # Scale price change to a sentiment value between -1 and 1
            # Typical daily returns are ±2%, so scale by 50 for a full range
            price_sentiment = min(1.0, max(-1.0, price_change * 50))
            
            # Apply correlation and noise
            sentiment_value = (
                price_sentiment * params["correlation"] +
                np.random.uniform(-1, 1) * params["noise"]
            )
            
            # Ensure value is in range [-1, 1]
            sentiment_value = min(1.0, max(-1.0, sentiment_value))
            
            # Determine direction
            direction = "neutral"
            if sentiment_value > 0.2:
                direction = "bullish"
            elif sentiment_value < -0.2:
                direction = "bearish"
            
            # Generate confidence
            confidence = min(1.0, max(0.1, 
                             np.random.normal(params["confidence_mean"], params["confidence_std"])))
            
            # Create sentiment event
            event = SimpleSentimentEvent(
                source=source,
                symbol=symbol,
                sentiment_value=sentiment_value,
                sentiment_direction=direction,
                confidence=confidence,
                timestamp=price_data['timestamp'].iloc[i],
            )
            
            events.append(event)
    
    # Sort events by timestamp
    events.sort(key=lambda x: x.timestamp)
    
    return events


def generate_synthetic_data(days=100, volatility=0.02, starting_price=10000, sentiment_events_per_day=5):
    """Generate synthetic price and sentiment data for backtesting.
    
    Args:
        days: Number of days of data to generate
        volatility: Daily price volatility
        starting_price: Starting price
        sentiment_events_per_day: Average number of sentiment events per day
        
    Returns:
        Tuple of (price_df, sentiment_events)
    """
    # Generate daily timestamps
    timestamps = [(datetime.now() - timedelta(days=days-i)).replace(hour=0, minute=0, second=0, microsecond=0) 
                  for i in range(days+1)]
    
    # Generate random price movements
    returns = np.random.normal(0, volatility, days)
    prices = starting_price * np.cumprod(1 + returns)
    prices = np.insert(prices, 0, starting_price)  # Insert starting price
    
    # Create price dataframe with OHLCV data
    price_data = []
    for i in range(len(timestamps)):
        # Skip the first day (use it as history only)
        if i == 0:
            continue
            
        # Get current and previous close
        current_close = prices[i]
        prev_close = prices[i-1]
        
        # Generate intraday volatility
        intraday_vol = volatility * prev_close * 0.5
        
        # Generate random OHLC data around the close
        high = current_close + abs(np.random.normal(0, intraday_vol))
        low = current_close - abs(np.random.normal(0, intraday_vol))
        open_price = prev_close + np.random.normal(0, intraday_vol)
        
        # Ensure high is highest and low is lowest
        high = max(high, current_close, open_price)
        low = min(low, current_close, open_price)
        
        # Add some random volume
        volume = np.random.randint(1000, 10000)
        
        # Add to price data
        price_data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': current_close,
            'volume': volume
        })
    
    # Create price dataframe
    price_df = pd.DataFrame(price_data)
    
    # Generate sentiment events
    sentiment_events = []
    
    # Total number of events
    total_events = int(days * sentiment_events_per_day)
    
    # Generate random timestamps throughout the period
    event_timestamps = [timestamps[0] + timedelta(
                            days=np.random.uniform(1, days-1),
                            hours=np.random.randint(0, 24),
                            minutes=np.random.randint(0, 60)
                        ) for _ in range(total_events)]
    event_timestamps.sort()
    
    # Generate events with some correlation to price movements
    for i, timestamp in enumerate(event_timestamps):
        # Find the next price point after this timestamp
        next_price_idx = next((i for i, row in enumerate(price_data) 
                              if row['timestamp'] > timestamp), len(price_data) - 1)
        
        if next_price_idx > 0:
            # Get returns between this point and a few steps ahead
            future_idx = min(next_price_idx + 5, len(price_data) - 1)
            future_return = (price_data[future_idx]['close'] / price_data[next_price_idx-1]['close']) - 1
            
            # Create correlated sentiment (with some noise)
            sentiment_direction = 1 if future_return > 0 else -1
            sentiment_noise = np.random.normal(0, 0.3)  # Add noise
            
            # 70% correlation, 30% noise
            sentiment_value = 0.7 * sentiment_direction + 0.3 * sentiment_noise
            
            # Clip to [-1, 1] range
            sentiment_value = max(min(sentiment_value, 1.0), -1.0)
            
            # Randomize confidence
            confidence = np.random.uniform(0.5, 1.0)
            
            # Create sentiment event
            event = SimpleSentimentEvent(
                source="synthetic",
                symbol="BTC/USDT",
                sentiment_value=sentiment_value,
                sentiment_direction="bullish" if sentiment_value > 0 else "bearish",
                confidence=confidence,
                timestamp=timestamp
            )
            
            sentiment_events.append(event)
    
    return price_df, sentiment_events


def optimize_parameters(price_data, sentiment_data, param_ranges):
    """Optimize strategy parameters by running multiple backtests.
    
    Args:
        price_data: DataFrame with historical price data
        sentiment_data: List of sentiment events
        param_ranges: Dictionary with parameter ranges to test
        
    Returns:
        Dictionary with best parameters and results
    """
    logger.info("Starting parameter optimization")
    
    # Track best results
    best_return = -float('inf')
    best_params = {}
    best_results = {}
    all_results = []
    
    # Generate parameter combinations
    param_combinations = []
    
    # Helper function to generate all combinations
    def generate_combinations(params, current_combo=None, idx=0):
        if current_combo is None:
            current_combo = {}
            
        if idx >= len(params):
            param_combinations.append(current_combo.copy())
            return
            
        param_name = list(params.keys())[idx]
        param_values = params[param_name]
        
        for value in param_values:
            current_combo[param_name] = value
            generate_combinations(params, current_combo, idx + 1)
    
    # Generate all parameter combinations
    generate_combinations(param_ranges)
    
    total_combinations = len(param_combinations)
    logger.info(f"Testing {total_combinations} parameter combinations")
    
    # Run backtest for each combination
    for i, params in enumerate(param_combinations):
        logger.info(f"Testing combination {i+1}/{total_combinations}: {params}")
        
        # Create backtester with these parameters
        backtester = SimpleSentimentBacktester(
            price_data=price_data,
            sentiment_data=sentiment_data,
            **params
        )
        
        # Run backtest
        results = backtester.run_backtest()
        
        # Store all results for analysis
        combination_results = params.copy()
        combination_results.update({
            'total_return': results['total_return'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'max_drawdown': results['max_drawdown'],
            'total_trades': results['total_trades']
        })
        all_results.append(combination_results)
        
        # Check if this is the best result so far
        # We prioritize return, but could use other metrics
        if results['total_return'] > best_return:
            best_return = results['total_return']
            best_params = params.copy()
            best_results = results.copy()
            logger.info(f"New best return: {best_return:.2%}")
    
    # Create DataFrame with all results for analysis
    results_df = pd.DataFrame(all_results)
    
    return {
        'best_params': best_params,
        'best_results': best_results,
        'all_results': results_df
    }


def visualize_optimization_results(results_df):
    """Visualize optimization results.
    
    Args:
        results_df: DataFrame with optimization results
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sentiment Strategy Parameter Optimization Results', fontsize=16)
    
    # Plot 1: Total Return vs Bull/Bear Thresholds
    scatter = axes[0, 0].scatter(
        results_df['sentiment_bull_threshold'], 
        results_df['sentiment_bear_threshold'],
        c=results_df['total_return'],
        cmap='viridis',
        s=100
    )
    axes[0, 0].set_xlabel('Bull Threshold')
    axes[0, 0].set_ylabel('Bear Threshold')
    axes[0, 0].set_title('Return vs Thresholds')
    fig.colorbar(scatter, ax=axes[0, 0], label='Total Return')
    
    # Plot 2: Return by Contrarian Mode
    contrarian_results = results_df.groupby('contrarian_mode')['total_return'].mean().reset_index()
    axes[0, 1].bar(
        contrarian_results['contrarian_mode'].map({True: 'Contrarian', False: 'Trend-Following'}), 
        contrarian_results['total_return']
    )
    axes[0, 1].set_xlabel('Strategy Mode')
    axes[0, 1].set_ylabel('Average Return')
    axes[0, 1].set_title('Return by Strategy Mode')
    
    # Plot 3: Top 10 Parameter Combinations
    top_results = results_df.sort_values('total_return', ascending=False).head(10)
    top_results = top_results.reset_index()
    axes[1, 0].bar(
        top_results.index,
        top_results['total_return']
    )
    axes[1, 0].set_xlabel('Combination Rank')
    axes[1, 0].set_ylabel('Total Return')
    axes[1, 0].set_title('Top 10 Parameter Combinations')
    
    # Plot 4: Win Rate vs Profit Factor scatter
    scatter = axes[1, 1].scatter(
        results_df['win_rate'],
        results_df['profit_factor'],
        c=results_df['total_return'],
        cmap='viridis',
        s=100
    )
    axes[1, 1].set_xlabel('Win Rate')
    axes[1, 1].set_ylabel('Profit Factor')
    axes[1, 1].set_title('Win Rate vs Profit Factor')
    fig.colorbar(scatter, ax=axes[1, 1], label='Total Return')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def compare_strategies(price_data, sentiment_data, strategies):
    """
    Compare multiple trading strategies using the same price and sentiment data.
    
    Args:
        price_data: DataFrame with price data
        sentiment_data: List of sentiment events
        strategies: List of strategy configurations with name and params
        
    Returns:
        Dictionary with strategy results and equity curves
    """
    # Store results for each strategy
    results = {}
    equity_curves = {}
    
    # Run backtest for each strategy configuration
    for strategy in strategies:
        strategy_name = strategy['name']
        params = strategy['params']
        
        print(f"\nRunning backtest for strategy: {strategy_name}")
        
        # Create backtester with these parameters
        backtester = SimpleSentimentBacktester(
            price_data=price_data,
            sentiment_data=sentiment_data,
            **params
        )
        
        # Run backtest
        strategy_results = backtester.run_backtest()
        
        # Store results
        results[strategy_name] = strategy_results
        
        # Store equity curve if available
        if 'equity_curve' in strategy_results:
            equity_curves[strategy_name] = strategy_results['equity_curve']
    
    # Print comparison table
    print("\nStrategy Comparison Results:")
    print(f"{'Strategy':<15} {'Return':<10} {'Win Rate':<10} {'Profit Factor':<15} {'Max DD':<10} {'Trades':<10}")
    print("-" * 70)
    
    for strategy_name, metrics in results.items():
        print(f"{strategy_name:<15} {metrics['total_return']:.2f}% {metrics['win_rate']:.2f}% {metrics['profit_factor']:.2f} {metrics['max_drawdown']:.2f}% {metrics['total_trades']}")
    
    # Create combined results
    combined_results = {
        'metrics': results,
        'equity_curves': equity_curves
    }
    
    return combined_results

def plot_results_comparison(results):
    """
    Plot equity curves for multiple strategies.
    
    Args:
        results: Dictionary with strategy results including equity curves
    """
    # Extract equity curves
    equity_curves = results['equity_curves']
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    
    for strategy_name, equity_curve in equity_curves.items():
        # Get strategy metrics
        metrics = results['metrics'][strategy_name]
        return_pct = metrics['total_return']
        
        # Plot equity curve with strategy name and return in legend
        plt.plot(equity_curve['timestamp'], 
                 equity_curve['equity'], 
                 label=f"{strategy_name} ({return_pct:.2f}%)")
    
    plt.title("Strategy Comparison - Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_strategy_comparison():
    """Run comparison of different strategy configurations."""
    print("\nComparing different strategy configurations")
    
    # Generate synthetic data
    price_data, sentiment_data = generate_synthetic_data(days=180)
    
    # Define strategies to compare
    strategies = [
        {
            'name': 'Standard',
            'params': {
                'sentiment_bull_threshold': 0.6,
                'sentiment_bear_threshold': 0.4,
                'min_confidence': 0.7,
                'contrarian_mode': False,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1,
                'transaction_cost': 0.001,
                'position_sizing': 'fixed'
            }
        },
        {
            'name': 'Contrarian',
            'params': {
                'sentiment_bull_threshold': 0.6,
                'sentiment_bear_threshold': 0.4,
                'min_confidence': 0.7,
                'contrarian_mode': True,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1,
                'transaction_cost': 0.001,
                'position_sizing': 'fixed'
            }
        },
        {
            'name': 'Volatility-Sized',
            'params': {
                'sentiment_bull_threshold': 0.6,
                'sentiment_bear_threshold': 0.4,
                'min_confidence': 0.7,
                'contrarian_mode': False,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1,
                'transaction_cost': 0.001,
                'position_sizing': 'volatility',
                'risk_per_trade': 0.02,
                'volatility_window': 20
            }
        },
        {
            'name': 'Kelly-Sized',
            'params': {
                'sentiment_bull_threshold': 0.6,
                'sentiment_bear_threshold': 0.4,
                'min_confidence': 0.7,
                'contrarian_mode': False,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.1,
                'transaction_cost': 0.001,
                'position_sizing': 'kelly',
                'kelly_fraction': 0.5
            }
        }
    ]
    
    # Run backtest for each strategy
    results = compare_strategies(price_data, sentiment_data, strategies)
    
    # Visualize the results
    plot_results_comparison(results)
    
    return results


def run_realistic_backtest():
    """Run a more realistic backtest with transaction costs and market biases"""
    print("\nRunning realistic backtest with transaction costs and market biases\n")
    
    # Generate synthetic data with more realistic market patterns
    price_data, sentiment_data = generate_synthetic_data(days=180)
    
    # 1. First run with transaction costs (0.1%)
    print("Running backtest with transaction costs (0.1%)")
    
    # Initialize backtester with parameters matching the class constructor
    backtester = SimpleSentimentBacktester(
        price_data=price_data,
        sentiment_data=sentiment_data,
        sentiment_bull_threshold=0.2,
        sentiment_bear_threshold=-0.2,
        min_confidence=0.3,
        contrarian_mode=False,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        transaction_cost=0.001,  # 0.1% transaction cost
        initial_capital=1000.0,
        position_sizing='volatility',
        risk_per_trade=0.02,
        volatility_window=20,
        kelly_fraction=0.5
    )
    
    # Run the backtest
    results = backtester.run_backtest()
    
    # Print results
    print("\nBacktest Results with Transaction Costs:")
    for key, value in results.items():
        if key != 'equity_curve' and key != 'trades':
            print(f"{key}: {value:.2f}")

    print(f"equity_curve: {results['equity_curve']}")
    
    # Plot the equity curve with our new plotting function
    plot_equity_curve(results)
    
    return results


def run_simple_backtest():
    """Run a simple backtest of the sentiment strategy."""
    # Generate sample data
    price_data = generate_sample_price_data(days=180)
    sentiment_data = generate_sample_sentiment_data(price_data)
    
    # Create backtester
    backtester = SimpleSentimentBacktester(
        price_data=price_data,
        sentiment_data=sentiment_data,
        sentiment_bull_threshold=0.6,
        sentiment_bear_threshold=0.4,
        min_confidence=0.7,
        contrarian_mode=False,
        position_sizing='kelly',  # Use Kelly criterion-based sizing
        kelly_fraction=0.5,       # Half Kelly
        risk_per_trade=0.02,      # 2% risk per trade
        volatility_window=20,     # 20-day window for volatility calculation
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        transaction_cost=0.001
    )
    
    # Run the backtest
    results = backtester.run_backtest()
    
    # Print results
    print("\n=== Backtest Results ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profitable Trades: {results['profitable_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Average Profit: ${results['avg_profit']:.2f}")
    print(f"Average Loss: ${results['avg_loss']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    
    # Plot results
    backtester.plot_results()
    
    return results

def run_parameter_optimization():
    """Run parameter optimization for the sentiment strategy."""
    print("\nRunning parameter optimization...")
    
    # Generate synthetic data
    price_data, sentiment_data = generate_synthetic_data(days=180)
    
    # Define parameter ranges to test
    param_ranges = {
        'sentiment_bull_threshold': [0.4, 0.5, 0.6, 0.7],
        'sentiment_bear_threshold': [0.3, 0.4, 0.5, 0.6],
        'min_confidence': [0.6, 0.7, 0.8],
        'contrarian_mode': [True, False],
        'stop_loss_pct': [0.03, 0.05, 0.07],
        'take_profit_pct': [0.05, 0.10, 0.15],
        'transaction_cost': [0.001]  # Include transaction costs
    }
    
    # Run optimization
    results = optimize_parameters(price_data, sentiment_data, param_ranges)
    
    # Print best parameters
    print("\nBest Parameters:")
    for key, value in results['best_params'].items():
        if isinstance(value, (float, int)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Print best results
    print("\nBest Results:")
    for key, value in results['best_results'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    return results


def run_advanced_optimization():
    """
    Run advanced parameter optimization using genetic algorithms.
    
    This function uses genetic algorithms to find optimal parameter combinations
    for the sentiment trading strategy, providing better performance than simple
    grid search by efficiently exploring the parameter space.
    """
    print("\nStarting advanced parameter optimization with genetic algorithms...")
    
    # Generate synthetic data for optimization
    price_data, sentiment_data = generate_synthetic_data(days=180)
    
    # Define parameter bounds for optimization
    param_bounds = {
        'sentiment_bull_threshold': (0.1, 0.9),
        'sentiment_bear_threshold': (-0.9, -0.1),
        'stop_loss_pct': (0.02, 0.1),            # Test different stop loss levels
        'take_profit_pct': (0.05, 0.2),          # Test different take profit levels
        'min_confidence': (0.2, 0.8),            # Test different confidence thresholds
        'position_sizing': ['fixed', 'volatility', 'kelly'],  # Test different position sizing methods
        'risk_per_trade': (0.01, 0.05),          # Test different risk per trade values
        'contrarian_mode': [True, False]         # Test contrarian vs. trend-following
    }
    
    # Define fitness function for the genetic algorithm
    def fitness_function(params):
        # Add fixed parameters that aren't being optimized
        fixed_params = {
            'transaction_cost': 0.001,
            'initial_capital': 10000.0,
            'volatility_window': 20,
            'kelly_fraction': 0.5
        }
        
        # Handle position_sizing as a string choice
        if 'position_sizing' in params and params['position_sizing'] in ['fixed', 'volatility', 'kelly']:
            position_sizing = params['position_sizing']
        else:
            position_sizing = 'fixed'  # Default
        
        # Initialize backtester with the generated data and current parameters
        backtester = SimpleSentimentBacktester(
            price_data=price_data,
            sentiment_data=sentiment_data,
            position_sizing=position_sizing,
            **{k: v for k, v in params.items() if k != 'position_sizing'},
            **fixed_params
        )
        
        # Run backtest and get results
        results = backtester.run_backtest()
        
        # Calculate a combined fitness score based on multiple metrics
        # Prioritize total return with consideration of Sharpe ratio and profit factor
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        profit_factor = results.get('profit_factor', 1)
        max_drawdown = results.get('max_drawdown', 1)
        win_rate = results.get('win_rate', 0)
        
        # Penalize extreme drawdowns
        drawdown_penalty = max(0, max_drawdown - 0.2) * 5
        
        # Calculate fitness (higher is better)
        # We weight the various factors differently
        fitness = (
            total_return * 3.0 +          # Strongly prioritize return
            sharpe_ratio * 100.0 +        # Good risk-adjusted return
            profit_factor * 50.0 +        # Reward high profit to loss ratio
            win_rate * 1.0 -              # Slightly reward higher win rates
            drawdown_penalty              # Penalize extreme drawdowns
        )
        
        return fitness
    
    # Create genetic optimizer
    optimizer = GeneticOptimizer(
        param_bounds=param_bounds,
        fitness_function=fitness_function,
        population_size=20,
        generations=15,
        crossover_rate=0.7,
        mutation_rate=0.2,
        elite_ratio=0.1,
        maximize=True,
        parallel=False  # Set to True for faster optimization if your system supports it
    )
    
    # Run optimization
    results = optimizer.optimize(verbose=True)
    
    # Display results
    print("\nOptimization Results:")
    print("Best Parameter Set:")
    for param, value in results['best_params'].items():
        if isinstance(value, (float, int)):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")
    
    print(f"Best Fitness Score: {results['best_fitness']:.4f}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    
    # Now run a backtest with the optimized parameters
    print("\nRunning backtest with optimized parameters...")
    
    # Test the optimized parameters - normal mode
    normal_params = results['best_params'].copy()
    normal_params['contrarian_mode'] = False
    normal_params['transaction_cost'] = 0.001
    
    # Create backtester with optimized parameters
    normal_backtester = SimpleSentimentBacktester(
        price_data=price_data,
        sentiment_data=sentiment_data,
        **normal_params
    )
    
    # Run backtest
    normal_results = normal_backtester.run_backtest()
    
    # Now test in contrarian mode
    contrarian_params = results['best_params'].copy()
    contrarian_params['contrarian_mode'] = True
    contrarian_params['transaction_cost'] = 0.001
    
    # Create backtester with optimized parameters in contrarian mode
    contrarian_backtester = SimpleSentimentBacktester(
        price_data=price_data,
        sentiment_data=sentiment_data,
        **contrarian_params
    )
    
    # Run backtest
    contrarian_results = contrarian_backtester.run_backtest()
    
    # Compare results
    print("\nPerformance Comparison:")
    print(f"{'Strategy':<15} {'Return':<10} {'Win Rate':<10} {'Profit Factor':<15} {'Max DD':<10} {'Trades':<10}")
    print("-" * 70)
    
    print(f"{'Optimized':<15} {normal_results['total_return']:.2f}% {normal_results['win_rate']:.2f}% "
          f"{normal_results['profit_factor']:.2f} {normal_results['max_drawdown']:.2f}% {normal_results['total_trades']}")
    
    print(f"{'Opt+Contrarian':<15} {contrarian_results['total_return']:.2f}% {contrarian_results['win_rate']:.2f}% "
          f"{contrarian_results['profit_factor']:.2f} {contrarian_results['max_drawdown']:.2f}% {contrarian_results['total_trades']}")
    
    # Plot optimization progress
    print("\nPlotting optimization progress...")
    optimizer.plot_progress()
    
    # Plot equity curves for comparison
    plt.figure(figsize=(12, 6))
    if hasattr(normal_backtester, 'equity_curve') and normal_backtester.equity_curve is not None:
        plt.plot(normal_backtester.equity_curve.index, 
                 normal_backtester.equity_curve['equity'], 
                 label=f"Optimized ({normal_results['total_return']:.2f}%)")
    
    if hasattr(contrarian_backtester, 'equity_curve') and contrarian_backtester.equity_curve is not None:
        plt.plot(contrarian_backtester.equity_curve.index, 
                 contrarian_backtester.equity_curve['equity'], 
                 label=f"Optimized+Contrarian ({contrarian_results['total_return']:.2f}%)")
    
    plt.title("Strategy Comparison - Equity Curves with Optimized Parameters")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Return the best parameters and results
    return {
        'best_params': results['best_params'],
        'normal_results': normal_results,
        'contrarian_results': contrarian_results
    }


def plot_equity_curve(results):
    """
    Plot the equity curve and other relevant metrics from backtest results.
    
    Args:
        results (dict): Dictionary containing backtest results
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter
    import pandas as pd
    
    equity_curve = results['equity_curve']
    
    # Convert timestamps to datetime objects if they aren't already
    if not isinstance(equity_curve['timestamp'].iloc[0], pd.Timestamp):
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
    
    # Calculate daily returns for additional plots
    equity_curve['daily_return'] = equity_curve['equity'].pct_change()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve on the first subplot
    axes[0].plot(equity_curve['timestamp'], equity_curve['equity'], linewidth=2)
    axes[0].set_title('Equity Curve', fontsize=14)
    axes[0].set_ylabel('Equity ($)', fontsize=12)
    axes[0].grid(True)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    # Annotate key metrics on the plot
    metrics_text = f"Total Return: {results['total_return']*100:.2f}%\n"
    metrics_text += f"Win Rate: {results['win_rate']:.2f}%\n"
    metrics_text += f"Profit Factor: {results['profit_factor']:.2f}\n"
    metrics_text += f"Max Drawdown: {results['max_drawdown']*100:.2f}%\n"
    metrics_text += f"Sharpe Ratio: {results['sharpe_ratio']:.2f}"
    
    # Position the text box in figure coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.02, 0.97, metrics_text, transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
    
    # Plot daily returns on the second subplot
    daily_returns = equity_curve['daily_return'].dropna()
    axes[1].bar(equity_curve['timestamp'].iloc[1:], daily_returns, width=0.6, alpha=0.6, color='green', 
               label='Daily Returns')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Daily Returns', fontsize=14)
    axes[1].set_ylabel('Return (%)', fontsize=12)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run individual components or the full example
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Backtesting Framework')
    parser.add_argument('--component', type=str, choices=['backtest', 'optimize', 'compare', 'all'], 
                        default='all', help='Component to run (default: all)')
    
    args = parser.parse_args()
    
    if args.component == 'backtest' or args.component == 'all':
        # Run a realistic backtest with transaction costs and market biases
        run_realistic_backtest()
    
    if args.component == 'optimize' or args.component == 'all':
        # Run advanced optimization with genetic algorithms
        run_advanced_optimization()
    
    if args.component == 'compare' or args.component == 'all':
        # Run comparison of different strategy configurations
        run_strategy_comparison()
    
    print("\nThe sentiment backtesting framework now includes:")
    print("1. Realistic market conditions with transaction costs")
    print("2. Synthetic data generation for testing")
    print("3. Parameter optimization with both grid search and genetic algorithms")
    print("4. Strategy comparison capabilities")
    print("5. Advanced position sizing methods (fixed, volatility-based, Kelly)")
    print("6. Visualization of backtest results and equity curves")
