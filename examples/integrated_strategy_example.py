"""
Integrated Strategy Example

This script demonstrates how to use the IntegratedStrategyManager with multiple
trading strategies to generate combined trading signals.

The example includes:
1. Setting up various strategy types (market regime, ML, portfolio)
2. Configuring the IntegratedStrategyManager
3. Fetching historical data
4. Generating and analyzing combined signals
5. Visualizing the results
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading_agent.agent.integrated_manager import IntegratedStrategyManager
from ai_trading_agent.agent.market_regime import MarketRegimeStrategy, MarketRegimeType
from ai_trading_agent.agent.ml_strategy import MLStrategy
from ai_trading_agent.agent.portfolio_strategy import PortfolioStrategy
# We'll use a simple data manager class for this example
from ai_trading_agent.common import logger

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def fetch_sample_data(symbols, start_date, end_date):
    """
    Fetch or generate sample data for demonstration.
    
    In a real application, this would fetch data from an exchange or data provider.
    For this example, we generate synthetic data.
    
    Args:
        symbols: List of symbols to generate data for
        start_date: Start date for the data
        end_date: End date for the data
        
    Returns:
        Dictionary mapping symbols to DataFrames with OHLCV data
    """
    data = {}
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for symbol in symbols:
        # Create a price series with some trend and noise
        base_price = 100 if symbol == 'BTC' else 50 if symbol == 'ETH' else 20
        
        # Add some market regimes
        prices = []
        n_days = len(date_range)
        
        # First third: uptrend
        trend1 = np.linspace(0, 30, n_days // 3)
        # Middle third: ranging
        trend2 = np.linspace(30, 30, n_days // 3)
        # Last third: downtrend
        trend3 = np.linspace(30, 0, n_days - (2 * (n_days // 3)))
        
        trend = np.concatenate([trend1, trend2, trend3])
        
        # Add some volatility changes
        volatility = np.ones(n_days)
        volatility[n_days//3:(2*n_days)//3] *= 2  # Higher volatility in the middle
        
        # Generate prices
        for i in range(n_days):
            noise = np.random.normal(0, 3 * volatility[i])
            prices.append(base_price + trend[i] + noise)
        
        # Create a DataFrame with OHLCV data
        df = pd.DataFrame({
            'open': [p * 0.99 for p in prices],
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(date_range))
        }, index=date_range)
        
        data[symbol] = df
    
    return data


def setup_strategies():
    """
    Set up and configure the trading strategies.
    
    Returns:
        Tuple of (market_regime_strategy, ml_strategy, portfolio_strategy)
    """
    # Market Regime Strategy
    market_regime_config = {
        'name': 'MarketRegimeStrategy',
        'regime_detector_config': {
            'lookback_window': 20,
            'volatility_window': 10,
            'trend_threshold': 0.5,
            'range_threshold': 0.3,
            'breakout_threshold': 2.0,
            'volatility_threshold': 0.2
        },
        'signal_mapping': {
            MarketRegimeType.TRENDING_UP.value: 0.8,    # Strong buy in uptrend
            MarketRegimeType.TRENDING_DOWN.value: -0.8, # Strong sell in downtrend
            MarketRegimeType.RANGING.value: 0.0,        # Neutral in ranging market
            MarketRegimeType.VOLATILE.value: -0.2,      # Slight sell in volatile markets
            MarketRegimeType.CALM.value: 0.2,           # Slight buy in calm markets
            MarketRegimeType.BREAKOUT.value: 0.5,       # Moderate buy on breakouts
            MarketRegimeType.REVERSAL.value: -0.5,      # Moderate sell on reversals
            MarketRegimeType.UNKNOWN.value: 0.0         # Neutral when regime is unknown
        }
    }
    market_regime_strategy = MarketRegimeStrategy(market_regime_config)
    
    # ML Strategy
    ml_config = {
        'name': 'MLStrategy',
        'model_type': 'random_forest',
        'prediction_horizon': 1,
        'training_lookback': 50,
        'retrain_frequency': 20,
        'confidence_threshold': 0.6,
        'features': [
            "rsi", "macd", "bb_position", "price_change", "volatility", "trend_strength"
        ]
    }
    ml_strategy = MLStrategy(ml_config)
    
    # Portfolio Strategy
    portfolio_config = {
        'name': 'PortfolioStrategy',
        'optimization_method': 'mean_variance',
        'risk_aversion': 2.0,
        'rebalance_threshold': 0.05,
        'max_position_size': 0.3,
        'lookback_window': 60,
        'target_volatility': 0.15,
        'use_shrinkage': True
    }
    portfolio_strategy = PortfolioStrategy(portfolio_config)
    
    return market_regime_strategy, ml_strategy, portfolio_strategy


def setup_integrated_manager(data_manager):
    """
    Set up and configure the IntegratedStrategyManager.
    
    Args:
        data_manager: DataManager instance
        
    Returns:
        Configured IntegratedStrategyManager
    """
    # Configuration for the integrated manager
    config = {
        'name': 'MainStrategyManager',
        'aggregation_method': 'weighted_average',  # Default method
        'strategy_weights': {
            'MarketRegimeStrategy': 0.3,
            'MLStrategy': 0.4,
            'PortfolioStrategy': 0.3
        },
        'use_meta_learner': True,  # Dynamically select best method
        'apply_risk_management': True,
        'apply_signal_filtering': True,
        'min_confidence_threshold': 0.3,
        'min_signal_strength': 0.1,
        'performance_history': {
            'weighted_average': {'sharpe_ratio': 1.2, 'win_rate': 0.6, 'profit_factor': 1.5, 'max_drawdown': 0.1},
            'dynamic_contextual': {'sharpe_ratio': 1.5, 'win_rate': 0.65, 'profit_factor': 1.8, 'max_drawdown': 0.08},
            'rule_based': {'sharpe_ratio': 1.0, 'win_rate': 0.55, 'profit_factor': 1.3, 'max_drawdown': 0.12},
            'majority_vote': {'sharpe_ratio': 1.1, 'win_rate': 0.58, 'profit_factor': 1.4, 'max_drawdown': 0.11}
        }
    }
    
    # Create the integrated strategy manager
    manager = IntegratedStrategyManager(config, data_manager)
    
    # Add strategies
    market_regime_strategy, ml_strategy, portfolio_strategy = setup_strategies()
    manager.add_strategy('MarketRegimeStrategy', market_regime_strategy)
    manager.add_strategy('MLStrategy', ml_strategy)
    manager.add_strategy('PortfolioStrategy', portfolio_strategy)
    
    return manager


def analyze_signals(signals_over_time, prices):
    """
    Analyze the generated signals and calculate performance metrics.
    
    Args:
        signals_over_time: Dictionary mapping dates to signal dictionaries
        prices: Dictionary mapping symbols to price DataFrames
        
    Returns:
        Dictionary of performance metrics
    """
    # Extract signals for each symbol
    symbol_signals = {}
    for date, signals in signals_over_time.items():
        for symbol, signal in signals.items():
            if symbol not in symbol_signals:
                symbol_signals[symbol] = {}
            symbol_signals[symbol][date] = signal
    
    # Calculate performance metrics for each symbol
    metrics = {}
    for symbol, signals in symbol_signals.items():
        if symbol not in prices:
            continue
            
        # Create a DataFrame with signals and prices
        signal_df = pd.DataFrame({
            'signal_strength': [s['signal_strength'] for d, s in signals.items()],
            'confidence': [s['confidence_score'] for d, s in signals.items()],
            'close': [prices[symbol].loc[d, 'close'] if d in prices[symbol].index else None for d in signals.keys()]
        }, index=signals.keys())
        
        # Drop rows with missing prices
        signal_df = signal_df.dropna()
        
        if len(signal_df) < 2:
            continue
            
        # Calculate returns
        signal_df['next_return'] = signal_df['close'].pct_change().shift(-1)
        
        # Calculate strategy returns (signal * next return)
        signal_df['strategy_return'] = signal_df['signal_strength'] * signal_df['next_return']
        
        # Calculate cumulative returns
        signal_df['cum_market_return'] = (1 + signal_df['next_return']).cumprod() - 1
        signal_df['cum_strategy_return'] = (1 + signal_df['strategy_return']).cumprod() - 1
        
        # Calculate performance metrics
        total_return = signal_df['cum_strategy_return'].iloc[-1]
        sharpe_ratio = signal_df['strategy_return'].mean() / signal_df['strategy_return'].std() * np.sqrt(252) if signal_df['strategy_return'].std() > 0 else 0
        
        # Calculate win rate
        signal_df['win'] = (signal_df['strategy_return'] > 0).astype(int)
        win_rate = signal_df['win'].mean()
        
        # Calculate max drawdown
        cum_returns = (1 + signal_df['strategy_return']).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        metrics[symbol] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'signal_df': signal_df  # Keep the DataFrame for plotting
        }
    
    return metrics


def plot_results(metrics, signals_over_time):
    """
    Plot the results of the integrated strategy.
    
    Args:
        metrics: Dictionary of performance metrics
        signals_over_time: Dictionary mapping dates to signal dictionaries
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot cumulative returns for each symbol
    for symbol, metric in metrics.items():
        signal_df = metric['signal_df']
        axs[0].plot(signal_df.index, signal_df['cum_strategy_return'], label=f"{symbol} Strategy")
        axs[0].plot(signal_df.index, signal_df['cum_market_return'], label=f"{symbol} Market", linestyle='--', alpha=0.5)
    
    axs[0].set_title('Cumulative Returns')
    axs[0].set_ylabel('Return')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot signal strength over time for the first symbol
    if metrics:
        symbol = list(metrics.keys())[0]
        signal_df = metrics[symbol]['signal_df']
        
        axs[1].plot(signal_df.index, signal_df['signal_strength'], label=f"{symbol} Signal Strength")
        axs[1].set_title(f'Signal Strength for {symbol}')
        axs[1].set_ylabel('Signal Strength')
        axs[1].set_ylim(-1.1, 1.1)
        axs[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        axs[1].grid(True)
        
        # Plot confidence over time
        axs[2].plot(signal_df.index, signal_df['confidence'], label=f"{symbol} Confidence")
        axs[2].set_title(f'Signal Confidence for {symbol}')
        axs[2].set_ylabel('Confidence')
        axs[2].set_ylim(0, 1.1)
        axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print("===================")
    for symbol, metric in metrics.items():
        print(f"\n{symbol}:")
        print(f"  Total Return: {metric['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metric['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {metric['win_rate']:.2%}")
        print(f"  Max Drawdown: {metric['max_drawdown']:.2%}")
    
    # Print aggregation method usage
    aggregation_methods = {}
    for date, signals in signals_over_time.items():
        for symbol, signal in signals.items():
            if 'metadata' in signal and 'aggregation_method' in signal['metadata']:
                method = signal['metadata']['aggregation_method']
                if method not in aggregation_methods:
                    aggregation_methods[method] = 0
                aggregation_methods[method] += 1
    
    print("\nAggregation Method Usage:")
    print("========================")
    total = sum(aggregation_methods.values())
    for method, count in aggregation_methods.items():
        print(f"  {method}: {count} times ({count/total:.1%})")


def main():
    """Main function to run the integrated strategy example."""
    print("Running Integrated Strategy Example...")
    
    # Define symbols and date range
    symbols = ['BTC', 'ETH', 'ADA']
    start_date = '2023-01-01'
    end_date = '2023-03-31'
    
    # Fetch sample data
    print(f"Fetching sample data for {symbols} from {start_date} to {end_date}...")
    data = fetch_sample_data(symbols, start_date, end_date)
    
    # Create a simple data manager that returns the sample data
    class SampleDataManager:
        def __init__(self, data):
            self.data = data
            
        def get_historical_data(self, symbols=None, start_date=None, end_date=None, interval=None):
            if symbols:
                return {s: self.data[s] for s in symbols if s in self.data}
            return self.data
    
    data_manager = SampleDataManager(data)
    
    # Set up the integrated strategy manager
    print("Setting up integrated strategy manager...")
    manager = setup_integrated_manager(data_manager)
    
    # Generate signals for each day in the date range
    print("Generating signals...")
    signals_over_time = {}
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for date in date_range:
        # Skip weekends for simplicity
        if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            continue
            
        # Get historical data up to this date
        historical_data = {
            symbol: df[df.index <= date] for symbol, df in data.items()
        }
        
        # Get current data (latest bar for each symbol)
        current_data = {
            symbol: df.loc[date] for symbol, df in data.items() if date in df.index
        }
        
        # Create a mock portfolio
        mock_portfolio = {
            'total_value': 10000,
            'current_exposure': 5000,  # 50% exposure
            'current_drawdown': 0.02,  # 2% drawdown
            'positions': {
                'BTC': {'value': 3000, 'quantity': 1},  # 30% in BTC
                'ETH': {'value': 2000, 'quantity': 2}   # 20% in ETH
            },
            'cash': 5000  # 50% cash
        }
        
        # Generate rich signals
        rich_signals = manager.process_data_and_generate_signals(
            current_data=current_data,
            historical_data=historical_data,
            current_portfolio=mock_portfolio,
            timestamp=date
        )
        
        signals_over_time[date] = rich_signals
    
    # Analyze the signals
    print("Analyzing signals...")
    metrics = analyze_signals(signals_over_time, data)
    
    # Plot the results
    print("Plotting results...")
    plot_results(metrics, signals_over_time)
    
    print("Example completed.")


if __name__ == "__main__":
    main()
