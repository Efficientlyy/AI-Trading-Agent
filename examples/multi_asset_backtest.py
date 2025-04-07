"""
Multi-Asset Backtesting Example for AI Trading Agent.

This example demonstrates how to use the backtesting framework with multiple assets
and a moving average crossover strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_acquisition.mock_provider import MockDataProvider
from src.strategies.ma_crossover_strategy import MACrossoverStrategy
from src.backtesting.backtester import Backtester
from src.trading_engine.enums import OrderSide
from src.common import logger

# Check if Rust backtester is available
try:
    from src.backtesting import RustBacktester
    RUST_AVAILABLE = True
    logger.info("Rust backtester is available")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("Rust backtester is not available, using Python implementation")


def run_backtest(use_rust=False):
    """
    Run a multi-asset backtest with MA Crossover strategy.
    
    Args:
        use_rust: Whether to use the Rust backtester
    """
    # Create mock data provider
    data_provider = MockDataProvider()
    
    # Define symbols
    symbols = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD']
    
    # Generate mock data for multiple assets
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    # Get historical data
    data = {}
    for symbol in symbols:
        try:
            df = data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            data[symbol] = df
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
    
    # Create strategy
    try:
        strategy = MACrossoverStrategy(
            symbols=symbols,
            fast_period=10,
            slow_period=30,
            risk_pct=0.02,
            max_position_pct=0.2
        )
        print("Strategy created successfully")
    except Exception as e:
        print(f"Error creating strategy: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create backtester
    try:
        if use_rust and RUST_AVAILABLE:
            backtester = RustBacktester(
                data=data,
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage=0.001,
                enable_fractional=True
            )
            logger.info("Using Rust backtester")
        else:
            backtester = Backtester(
                data=data,
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage=0.001,
                enable_fractional=True
            )
            logger.info("Using Python backtester")
        print("Backtester created successfully")
    except Exception as e:
        print(f"Error creating backtester: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run backtest
    try:
        print("Running backtest...")
        results = backtester.run(strategy.generate_signals)
        print("Backtest completed successfully")
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Print results
    print_results(results)
    
    # Plot results
    plot_results(results, data)
    
    return results


def print_results(results):
    """
    Print backtest results.
    
    Args:
        results: Backtest results
    """
    if isinstance(results, dict):
        # RustBacktester returns a dict
        metrics = results['metrics']
        portfolio_history = results['portfolio_history']
        trade_history = results['trade_history']
    else:
        # Python Backtester returns a PerformanceMetrics object
        metrics = results
        portfolio_history = None
        trade_history = None
    
    print("\n=== Backtest Results ===")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Annualized Return: {metrics.annualized_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Max Drawdown Duration: {metrics.max_drawdown_duration} days")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Avg Profit per Trade: {metrics.avg_profit_per_trade:.2f}")
    print(f"Avg Loss per Trade: {metrics.avg_loss_per_trade:.2f}")
    print(f"Avg Profit/Loss Ratio: {metrics.avg_profit_loss_ratio:.2f}")
    
    # Print trade summary by symbol
    if trade_history:
        print("\n=== Trade Summary by Symbol ===")
        symbols = set(trade.symbol for trade in trade_history)
        
        for symbol in symbols:
            symbol_trades = [t for t in trade_history if t.symbol == symbol]
            # For simplicity, we'll just count the number of trades
            # In a real implementation, we would track entry and exit prices to calculate win rate
            print(f"{symbol}: {len(symbol_trades)} trades")


def plot_results(results, data):
    """
    Plot backtest results.
    
    Args:
        results: Backtest results
        data: Market data
    """
    if isinstance(results, dict):
        # RustBacktester returns a dict
        portfolio_history = results['portfolio_history']
        equity_curve = pd.Series(
            [snapshot['total_value'] for snapshot in portfolio_history],
            index=[snapshot['timestamp'] for snapshot in portfolio_history]
        )
    else:
        # Python Backtester returns a PerformanceMetrics object
        equity_curve = results.equity_curve
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot equity curve
    axs[0].plot(equity_curve.index, equity_curve.values, label='Portfolio Value')
    axs[0].set_title('Portfolio Value Over Time')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Value ($)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot asset prices (normalized)
    for symbol, df in data.items():
        # Normalize prices to start at 1.0
        normalized_prices = df['close'] / df['close'].iloc[0]
        axs[1].plot(df.index, normalized_prices, label=symbol)
    
    axs[1].set_title('Asset Prices (Normalized)')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Normalized Price')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.show()


def compare_backtester_performance():
    """
    Compare performance between Python and Rust backtester implementations.
    """
    if not RUST_AVAILABLE:
        print("Rust backtester is not available for comparison")
        return
    
    try:
        import time
        
        # Create mock data provider
        data_provider = MockDataProvider()
        
        # Define symbols
        symbols = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD']
        
        # Generate mock data for multiple assets
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        # Get historical data
        data = {}
        for symbol in symbols:
            df = data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            data[symbol] = df
        
        # Create strategy
        strategy = MACrossoverStrategy(
            symbols=symbols,
            fast_period=10,
            slow_period=30,
            risk_pct=0.02,
            max_position_pct=0.2
        )
        
        # Time Python backtester
        start_time = time.time()
        backtester = Backtester(
            data=data,
            initial_capital=10000.0,
            commission_rate=0.001,
            slippage=0.001,
            enable_fractional=True
        )
        backtester.run(strategy.generate_signals)
        python_time = time.time() - start_time
        
        # Time Rust backtester
        start_time = time.time()
        backtester = RustBacktester(
            data=data,
            initial_capital=10000.0,
            commission_rate=0.001,
            slippage=0.001,
            enable_fractional=True
        )
        backtester.run(strategy.generate_signals)
        rust_time = time.time() - start_time
        
        # Print results
        print("\n=== Backtester Performance Comparison ===")
        print(f"Python Backtester: {python_time:.4f} seconds")
        print(f"Rust Backtester: {rust_time:.4f} seconds")
        print(f"Speedup: {python_time / rust_time:.2f}x")
    except Exception as e:
        print(f"Error in performance comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # Run backtest with Python implementation only
        results = run_backtest(use_rust=False)
        
        # Skip Rust comparison for now until we fix the Rust integration
        # if RUST_AVAILABLE:
        #     compare_backtester_performance()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
