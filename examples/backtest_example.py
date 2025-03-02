#!/usr/bin/env python
"""
Example script demonstrating the high-performance backtesting engine
with a simple moving average crossover strategy.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our modules
from src.backtesting import (
    BacktestEngine, 
    TimeFrame, 
    OrderSide,
    create_backtest_engine
)
from src.rust_bridge import is_rust_available, SMA, MACrossover

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def generate_sample_data(days: int = 180) -> pd.DataFrame:
    """
    Generate sample price data for backtesting.
    This simulates a trending market with some noise.
    
    Args:
        days: Number of days of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    # Start date
    start_date = datetime.now() - timedelta(days=days)
    
    # Generate dates (hours)
    hours = days * 24
    dates = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Initial price
    price = 10000.0
    
    # Generate a price series with trend and noise
    prices = []
    for i in range(hours):
        # Add a trend component (sine wave with 30-day period)
        trend = np.sin(i / (24 * 30) * 2 * np.pi) * 1000
        
        # Add random noise
        noise = np.random.normal(0, 100)
        
        # Update price with trend and noise
        price = max(1000, price + trend * 0.01 + noise)
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, date in enumerate(dates):
        price = prices[i]
        high = price * (1 + np.random.uniform(0, 0.01))
        low = price * (1 - np.random.uniform(0, 0.01))
        open_price = price * (1 + np.random.uniform(-0.005, 0.005))
        close = price
        volume = np.random.uniform(10, 100)
        
        data.append({
            "timestamp": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })
    
    return pd.DataFrame(data)

class MovingAverageCrossoverStrategy:
    """
    Simple moving average crossover strategy.
    Goes long when fast MA crosses above slow MA,
    and goes short when fast MA crosses below slow MA.
    """
    
    def __init__(self, engine: BacktestEngine, symbol: str, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize the strategy.
        
        Args:
            engine: The backtest engine
            symbol: The symbol to trade
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
        """
        self.engine = engine
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Initialize the crossover detector
        self.crossover = MACrossover(fast_period, slow_period)
        
        # Track the current position
        self.position = 0
        
        # Track order IDs
        self.orders = []
        
        # Track performance metrics
        self.trades = []
        self.equity_curve = []
    
    def on_candle(self, candle: Dict[str, Any]) -> None:
        """
        Process a new candle.
        
        Args:
            candle: The candle data
        """
        # Update the crossover detector
        signal = self.crossover.update(candle["close"])
        
        # Check for signals
        if signal == "bullish" and self.position <= 0:
            # Close any existing short position
            if self.position < 0:
                logger.info(f"Closing short position at {candle['close']}")
                self.engine.submit_market_order(self.symbol, OrderSide.BUY, abs(self.position))
                self.position = 0
            
            # Open a long position
            logger.info(f"Opening long position at {candle['close']}")
            order_id = self.engine.submit_market_order(self.symbol, OrderSide.BUY, 1.0)
            self.orders.append(order_id)
            self.position = 1
            
        elif signal == "bearish" and self.position >= 0:
            # Close any existing long position
            if self.position > 0:
                logger.info(f"Closing long position at {candle['close']}")
                self.engine.submit_market_order(self.symbol, OrderSide.SELL, self.position)
                self.position = 0
            
            # Open a short position
            logger.info(f"Opening short position at {candle['close']}")
            order_id = self.engine.submit_market_order(self.symbol, OrderSide.SELL, 1.0)
            self.orders.append(order_id)
            self.position = -1

def run_backtest():
    """Run a backtest with the MA crossover strategy."""
    # Generate sample data
    logger.info("Generating sample data...")
    data = generate_sample_data(days=180)
    
    # Create the backtest engine
    symbol = "BTCUSDT"
    start_time = data["timestamp"].min().timestamp()
    end_time = data["timestamp"].max().timestamp()
    
    logger.info(f"Creating backtest engine (Rust available: {is_rust_available()})...")
    engine = create_backtest_engine(
        initial_balance=10000.0,
        symbols=[symbol],
        start_time=start_time,
        end_time=end_time,
        mode="candles",
        commission_rate=0.001,  # 0.1%
        slippage=0.0005,  # 0.05%
        enable_fractional_sizing=True
    )
    
    # Initialize the strategy
    strategy = MovingAverageCrossoverStrategy(engine, symbol, fast_period=20, slow_period=50)
    
    # Run the backtest
    logger.info("Running backtest...")
    start_time = datetime.now()
    
    # Process each candle
    for _, candle in data.iterrows():
        # First, process the candle in the engine
        engine.process_candle(
            symbol=symbol,
            timestamp=candle["timestamp"].timestamp(),
            open_price=candle["open"],
            high=candle["high"],
            low=candle["low"],
            close=candle["close"],
            volume=candle["volume"],
            timeframe=TimeFrame.HOUR_1
        )
        
        # Then, update the strategy
        strategy.on_candle(candle)
    
    # Get the backtest results
    results = engine.run()
    
    # Print the results
    logger.info(f"Backtest completed in {datetime.now() - start_time}")
    logger.info(f"Initial balance: ${results.initial_balance:.2f}")
    logger.info(f"Final balance: ${results.final_balance:.2f}")
    logger.info(f"Profit/Loss: ${results.final_balance - results.initial_balance:.2f} ({(results.final_balance / results.initial_balance - 1) * 100:.2f}%)")
    logger.info(f"Total trades: {results.total_trades}")
    logger.info(f"Win rate: {results.win_rate:.2f}%")
    logger.info(f"Max drawdown: {results.max_drawdown_pct:.2f}%")
    
    if results.sharpe_ratio is not None:
        logger.info(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
    
    if results.profit_factor is not None:
        logger.info(f"Profit factor: {results.profit_factor:.2f}")
    
    # Plot the results
    engine.plot_stats(title="MA Crossover Strategy Backtest Results")

if __name__ == "__main__":
    run_backtest() 