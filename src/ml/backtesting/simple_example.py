"""Simple example of regime-based backtesting.

This script demonstrates how to use the backtesting module to
backtest a regime-based trading strategy on SPY data.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import backtesting components
from src.ml.backtesting import (
    RegimeStrategy,
    plot_equity_curve,
    plot_drawdowns,
    plot_regime_performance,
    plot_trade_analysis,
    save_html_report
)
from src.ml.detection import RegimeDetectorFactory


def run_simple_backtest(
    symbol="SPY",
    period="5y", 
    detector_method="trend",
    detector_params=None,
    output_dir="./results"
):
    """
    Run a simple backtest of a regime-based strategy.
    
    Args:
        symbol: Ticker symbol
        period: Time period (e.g., "5y" for 5 years)
        detector_method: Regime detection method
        detector_params: Parameters for the detector
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with backtest results
    """
    print(f"Running backtest for {symbol} using {detector_method} detection...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download historical data
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate returns
    data["Returns"] = data['Close'].pct_change().fillna(0)
    
    # Default detector parameters if none provided
    if detector_params is None:
        detector_params = {
            'n_regimes': 3,
            'trend_method': 'macd'
        }
    
    # Set up the regime-based strategy
    strategy = RegimeStrategy(
        detector_method=detector_method,
        detector_params=detector_params,
        regime_rules={
            0: {'action': 'sell', 'allocation': 0.0},  # Regime 0 (bear): Sell everything
            1: {'action': 'hold', 'allocation': 0.5},  # Regime 1 (sideways): Hold half position
            2: {'action': 'buy', 'allocation': 1.0}    # Regime 2 (bull): Full position
        },
        initial_capital=10000.0,
        position_sizing='percent',
        max_position_size=1.0,
        stop_loss_pct=0.05,  # 5% stop loss
        take_profit_pct=0.2   # 20% take profit
    )
    
    # Prepare data for backtesting
    backtest_data = {
        'symbol': symbol,
        'dates': data.index.tolist(),
        'prices': data['Close'].values.tolist(),
        'returns': data['Returns'].values.tolist(),
        'volumes': data['Volume'].values.tolist(),
        'highs': data['High'].values.tolist(),
        'lows': data['Low'].values.tolist()
    }
    
    # Run backtest
    print("Running backtest...")
    results = strategy.backtest(backtest_data)
    
    # Get regimes and equity curve
    regimes = results.get('regimes', [])
    equity_curve = results.get('equity_curve', [])
    trades = results.get('trades', [])
    metrics = results.get('performance_metrics', {})
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            if 'return' in metric.lower() or 'drawdown' in metric.lower() or 'rate' in metric.lower():
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    figures = {}
    
    # Equity curve with regime background
    dates = backtest_data['dates']
    equity = np.array(equity_curve)
    benchmark = np.cumprod(1 + np.array(backtest_data['returns'])) * 10000  # Start with same capital
    
    regime_names = {
        0: "Bearish",
        1: "Neutral",
        2: "Bullish"
    }
    
    fig_equity = plot_equity_curve(
        dates=dates,
        equity=equity,
        regimes=np.array(regimes) if regimes else None,
        regime_names=regime_names,
        benchmark=benchmark,
        title=f"{symbol} Regime Strategy vs Buy & Hold",
        save_path=os.path.join(output_dir, f"{symbol}_equity_curve.png")
    )
    figures["Equity Curve"] = fig_equity
    
    # Drawdowns
    fig_drawdowns = plot_drawdowns(
        dates=dates,
        returns=np.diff(equity) / equity[:-1],
        title=f"{symbol} Strategy Drawdowns",
        save_path=os.path.join(output_dir, f"{symbol}_drawdowns.png")
    )
    figures["Drawdowns"] = fig_drawdowns
    
    # Regime performance
    if regimes:
        fig_regimes = plot_regime_performance(
            regimes=np.array(regimes),
            returns=np.array(backtest_data['returns']),
            regime_names=regime_names,
            title=f"{symbol} Performance by Market Regime",
            save_path=os.path.join(output_dir, f"{symbol}_regime_performance.png")
        )
        figures["Regime Performance"] = fig_regimes
    
    # Trade analysis
    if trades:
        fig_trades = plot_trade_analysis(
            trades=trades,
            save_path=os.path.join(output_dir, f"{symbol}_trade_analysis.png")
        )
        figures["Trade Analysis"] = fig_trades
    
    # Create HTML report
    report_path = save_html_report(
        metrics=metrics,
        trades=trades,
        figures=figures,
        output_path=os.path.join(output_dir, f"{symbol}_backtest_report.html")
    )
    print(f"\nBacktest report saved to: {report_path}")
    
    return {
        'results': results,
        'figures': figures,
        'report_path': report_path
    }


if __name__ == "__main__":
    # Run a simple backtest with default parameters
    run_simple_backtest(
        symbol="SPY",
        period="5y",
        detector_method="trend",
        output_dir="./backtest_results"
    )
    
    # Show plots
    plt.show() 