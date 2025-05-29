"""
Example script demonstrating how to use the advanced backtesting framework.

This script provides a complete example of backtesting the enhanced Technical Analysis
Agent components against historical market data.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add the project root to the path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

from ai_trading_agent.backtesting.advanced_backtest import AdvancedBacktester
from ai_trading_agent.data_sources.mock_data_generator import generate_mock_data, generate_mock_data_with_pattern
from ai_trading_agent.common.utils import get_logger


def prepare_multi_timeframe_data(symbols=None, days=365):
    """
    Prepare multi-timeframe data for backtesting.
    
    Args:
        symbols: List of symbols to generate data for, or None for defaults
        days: Number of days of data to generate
        
    Returns:
        Dictionary with market data in multi-timeframe format
    """
    logger = get_logger("BacktestExample")
    
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "XRP/USD"]
    
    # Generate end date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Define timeframes to generate
    timeframes = ["1d", "4h", "1h"]
    
    # Prepare data structure
    market_data = {}
    
    logger.info(f"Generating multi-timeframe data for {len(symbols)} symbols over {days} days")
    
    for symbol in symbols:
        # Generate daily data with different characteristics
        if symbol == "BTC/USD":
            # Trending market
            daily_data = generate_mock_data(
                days=days,
                volatility=0.02,
                trend_strength=0.7,
                pattern_type="trend"
            )
        elif symbol == "ETH/USD":
            # Ranging market with a specific pattern
            daily_data = generate_mock_data_with_pattern(
                pattern="double_top",
                days_before=int(days*0.6),
                days_after=int(days*0.2),
                volatility=0.025
            )
        else:
            # Volatile market
            daily_data = generate_mock_data(
                days=days,
                volatility=0.035,
                trend_strength=0.3,
                pattern_type="volatile"
            )
        
        # Set correct date range
        daily_data.index = pd.date_range(start=start_date, end=end_date, periods=len(daily_data))
        
        # Derive lower timeframes
        market_data[symbol] = {}
        market_data[symbol]["1d"] = daily_data
        
        # Generate 4-hour data (6 periods per day)
        hours_4h = days * 6
        data_4h = pd.DataFrame(index=pd.date_range(
            start=start_date, 
            periods=hours_4h,
            freq="4H"
        ))
        
        # Map 4-hour periods to daily data with added noise
        for i in range(hours_4h):
            day_idx = i // 6  # 6 four-hour periods per day
            
            if day_idx < len(daily_data):
                # Get reference daily bar
                daily_bar = daily_data.iloc[day_idx]
                
                # Add time-of-day pattern and noise
                time_of_day = i % 6
                time_factor = np.sin(time_of_day / 6 * np.pi) * 0.015
                noise = np.random.normal(0, 0.01)
                
                # Create 4h bar
                data_4h.loc[data_4h.index[i], 'open'] = daily_bar['open'] * (1 + noise + time_factor)
                data_4h.loc[data_4h.index[i], 'high'] = daily_bar['high'] * (1 + noise + time_factor + 0.002)
                data_4h.loc[data_4h.index[i], 'low'] = daily_bar['low'] * (1 + noise + time_factor - 0.002)
                data_4h.loc[data_4h.index[i], 'close'] = daily_bar['close'] * (1 + noise + time_factor)
                data_4h.loc[data_4h.index[i], 'volume'] = daily_bar['volume'] * (1 + np.random.normal(0, 0.2))
        
        market_data[symbol]["4h"] = data_4h
        
        # Generate 1-hour data (24 periods per day)
        hours_1h = days * 24
        data_1h = pd.DataFrame(index=pd.date_range(
            start=start_date, 
            periods=hours_1h,
            freq="1H"
        ))
        
        # Map 1-hour periods to daily data with added noise
        for i in range(hours_1h):
            day_idx = i // 24  # 24 one-hour periods per day
            
            if day_idx < len(daily_data):
                # Get reference daily bar
                daily_bar = daily_data.iloc[day_idx]
                
                # Add time-of-day pattern and noise
                time_of_day = i % 24
                time_factor = np.sin(time_of_day / 24 * np.pi * 2) * 0.02
                noise = np.random.normal(0, 0.015)
                
                # Create 1h bar
                data_1h.loc[data_1h.index[i], 'open'] = daily_bar['open'] * (1 + noise + time_factor)
                data_1h.loc[data_1h.index[i], 'high'] = daily_bar['high'] * (1 + noise + time_factor + 0.003)
                data_1h.loc[data_1h.index[i], 'low'] = daily_bar['low'] * (1 + noise + time_factor - 0.003)
                data_1h.loc[data_1h.index[i], 'close'] = daily_bar['close'] * (1 + noise + time_factor)
                data_1h.loc[data_1h.index[i], 'volume'] = daily_bar['volume'] * (1 + np.random.normal(0, 0.3))
        
        market_data[symbol]["1h"] = data_1h
    
    logger.info("Multi-timeframe data generation complete")
    return market_data


def run_backtest_example():
    """Run a comprehensive backtest example."""
    logger = get_logger("BacktestExample")
    logger.info("Starting advanced backtest example")
    
    # Define symbols
    symbols = ["BTC/USD", "ETH/USD", "XRP/USD"]
    
    # Prepare data
    market_data = prepare_multi_timeframe_data(symbols)
    
    # Configure the agent
    agent_config = {
        "strategies": [
            {
                "type": "MA_Cross",
                "name": "MA_Cross_Standard",
                "fast_period": 9,
                "slow_period": 21,
                "signal_period": 9,
                "confirmation_periods": 2,
                "min_divergence": 0.002,
                "max_volatility": 0.05,
                "volatility_adjustment": 0.5
            },
            {
                "type": "RSI_OB_OS",
                "name": "RSI_Standard",
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30,
                "confirmation_periods": 2,
                "buffer_zones": 5,
                "max_volatility": 0.05,
                "volatility_adjustment": 0.5
            },
            {
                "type": "Multi_TF",
                "name": "MA_Cross_Multi_TF",
                "base_strategy": "MA_Cross",
                "timeframes": ["1h", "4h", "1d"],
                "min_confirmations": 2,
                "weighting": "longer_higher",
                "base_config": {
                    "fast_period": 9,
                    "slow_period": 21,
                    "signal_period": 9,
                    "confirmation_periods": 1
                }
            }
        ],
        "indicators": [
            "sma_9",  # Use simple string identifiers 
            "sma_21",
            "ema_9",
            "ema_21",
            "rsi_14",
            "atr_14",
            "adx_14"
        ],
        "indicator_config": {
            "sma_9": {"name": "sma", "params": {"window": 9}},
            "sma_21": {"name": "sma", "params": {"window": 21}},
            "ema_9": {"name": "ema", "params": {"window": 9}},
            "ema_21": {"name": "ema", "params": {"window": 21}},
            "rsi_14": {"name": "rsi", "params": {"window": 14}},
            "atr_14": {"name": "atr", "params": {"window": 14}},
            "adx_14": {"name": "adx", "params": {"window": 14}}
        },
        "timeframes": ["1h", "4h", "1d"],
        "ml_validator": {
            "min_confidence": 0.6
        },
        "adaptive_parameters": {
            "optimization_method": "regime_based"
        },
        "indicator_config": {}  # Empty config for IndicatorEngine
    }
    
    # Configure the backtester
    backtest_config = {
        "initial_capital": 100000.0,
        "position_sizing": "fixed_percent",
        "position_size": 0.1,  # 10% of capital per trade
        "slippage_model": "fixed",
        "slippage_amount": 0.001,  # 0.1%
        "commission_model": "percentage",
        "commission_amount": 0.001,  # 0.1%
        "agent_config": agent_config
    }
    
    # Initialize backtester
    backtester = AdvancedBacktester(backtest_config)
    
    # Run backtest
    results = backtester.run_backtest(market_data, symbols)
    
    # Print summary
    summary = results.get_summary()
    logger.info("Backtest summary:")
    logger.info(f"Total trades: {summary['total_trades']}")
    logger.info(f"Total signals: {summary['total_signals']}")
    logger.info(f"Rejected signals: {summary['rejected_signals']}")
    
    if 'performance_metrics' in summary:
        metrics = summary['performance_metrics']
        logger.info(f"Initial capital: ${metrics.get('start_equity', 0):,.2f}")
        logger.info(f"Final capital: ${metrics.get('end_equity', 0):,.2f}")
        logger.info(f"Total return: {metrics.get('total_return', 0):,.2f}%")
        logger.info(f"Annualized return: {metrics.get('annualized_return', 0):,.2f}%")
        logger.info(f"Max drawdown: {metrics.get('max_drawdown', 0):,.2f}%")
        logger.info(f"Win rate: {metrics.get('win_rate', 0)*100:,.2f}%")
        logger.info(f"Profit factor: {metrics.get('profit_factor', 0):,.2f}")
        logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):,.2f}")
    
    # Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "backtest_results",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    backtester.save_results(results_dir)
    
    # Plot results
    backtester.plot_equity_curve()
    backtester.plot_trade_analysis()
    
    logger.info(f"Backtest results saved to {results_dir}")
    return backtester, results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    backtester, results = run_backtest_example()
