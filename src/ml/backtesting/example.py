"""Example usage of the backtesting module.

This script demonstrates how to use the backtesting module to backtest
trading strategies based on market regime detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Optional, Union, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import backtesting components
from src.ml.backtesting import (
    RegimeStrategy,
    calculate_sharpe_ratio,
    calculate_drawdowns,
    PositionSizerFactory,
    RiskManagerFactory,
    plot_equity_curve,
    plot_regime_performance,
    create_comprehensive_report
)

# Import detection components
from src.ml.detection import RegimeDetectorFactory


def download_market_data(
    symbol: str = 'SPY',
    period: str = '5y',
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Download market data using yfinance.
    
    Args:
        symbol: Ticker symbol
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
        interval: Time interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
    Returns:
        DataFrame with market data
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    
    # Ensure the data has all required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            print(f"Warning: {col} column is missing")
    
    return data


def prepare_market_data(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Prepare market data for backtesting.
    
    Args:
        df: DataFrame with market data
        symbol: Ticker symbol
        
    Returns:
        Dictionary with market data in the format expected by RegimeStrategy
    """
    # Calculate returns if not already in the DataFrame
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change().fillna(0)
    
    # Prepare data dictionary
    data = {
        'symbol': symbol,
        'dates': df.index.tolist(),
        'prices': df['Close'].values.tolist(),
        'returns': df['Returns'].values.tolist(),
        'volumes': df['Volume'].values.tolist(),
        'highs': df['High'].values.tolist(),
        'lows': df['Low'].values.tolist(),
        'opens': df['Open'].values.tolist()
    }
    
    return data


def run_backtest(
    data: Dict[str, Any],
    detector_method: str = 'trend',
    detector_params: Optional[Dict[str, Any]] = None,
    regime_rules: Optional[Dict[int, Dict[str, Any]]] = None,
    initial_capital: float = 10000.0,
    position_sizing: str = 'fixed',
    position_sizing_params: Optional[Dict[str, Any]] = None,
    risk_management: str = 'basic',
    risk_management_params: Optional[Dict[str, Any]] = None,
    lookback_window: int = 252,
    output_dir: str = './outputs'
) -> Dict[str, Any]:
    """
    Run a backtest with the specified parameters.
    
    Args:
        data: Dictionary with market data
        detector_method: Regime detection method
        detector_params: Parameters for the detector
        regime_rules: Trading rules for each regime
        initial_capital: Initial capital
        position_sizing: Position sizing method
        position_sizing_params: Parameters for position sizing
        risk_management: Risk management method
        risk_management_params: Parameters for risk management
        lookback_window: Lookback window for regime detection
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with backtest results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set default regime rules if not provided
    if regime_rules is None:
        if detector_method == 'volatility':
            # For volatility: 0=low, 1=medium, 2=high
            regime_rules = {
                0: {'action': 'buy', 'allocation': 1.0},    # Low volatility: buy
                1: {'action': 'hold', 'allocation': 0.5},   # Medium volatility: hold
                2: {'action': 'sell', 'allocation': 0.0}    # High volatility: sell
            }
        elif detector_method == 'trend':
            # For trend: 0=downtrend, 1=sideways, 2=uptrend
            regime_rules = {
                0: {'action': 'sell', 'allocation': 0.0},   # Downtrend: sell
                1: {'action': 'hold', 'allocation': 0.5},   # Sideways: hold
                2: {'action': 'buy', 'allocation': 1.0}     # Uptrend: buy
            }
        elif detector_method == 'momentum':
            # For momentum: 0=negative, 1=neutral, 2=positive
            regime_rules = {
                0: {'action': 'sell', 'allocation': 0.0},   # Negative momentum: sell
                1: {'action': 'hold', 'allocation': 0.5},   # Neutral momentum: hold
                2: {'action': 'buy', 'allocation': 1.0}     # Positive momentum: buy
            }
        elif detector_method == 'hmm':
            # For HMM: regimes are numbered 0 to n-1
            regime_rules = {
                0: {'action': 'sell', 'allocation': 0.0},   # Bear market: sell
                1: {'action': 'hold', 'allocation': 0.5},   # Transition: hold
                2: {'action': 'buy', 'allocation': 1.0}     # Bull market: buy
            }
        else:
            # Default rules
            regime_rules = {
                0: {'action': 'sell', 'allocation': 0.0},
                1: {'action': 'hold', 'allocation': 0.5},
                2: {'action': 'buy', 'allocation': 1.0}
            }
    
    # Set default position sizing parameters
    if position_sizing_params is None:
        position_sizing_params = {
            'max_position_size': 1.0
        }
    
    # Set default risk management parameters
    if risk_management_params is None:
        risk_management_params = {
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1,
            'max_holding_days': 30
        }
    
    # Create the strategy
    strategy = RegimeStrategy(
        detector_method=detector_method,
        detector_params=detector_params,
        regime_rules=regime_rules,
        initial_capital=initial_capital,
        position_sizing=position_sizing,
        position_sizing_params=position_sizing_params,
        risk_management=risk_management,
        risk_management_params=risk_management_params,
        lookback_window=lookback_window
    )
    
    # Run the backtest
    results = strategy.backtest(data)
    
    # Generate visualizations
    symbol = data.get('symbol', 'UNKNOWN')
    equity_fig = plot_equity_curve(
        dates=data['dates'], 
        equity=results['equity_curve'], 
        regimes=results['regimes'],
        title=f"{symbol} Equity Curve - {detector_method.capitalize()} Detector",
        save_path=os.path.join(output_dir, f"{symbol}_{detector_method}_equity.png")
    )
    
    # Generate regime performance visualization
    if 'regime_metrics' in results:
        regime_fig = plot_regime_performance(
            regime_metrics=results['regime_metrics'],
            title=f"{symbol} Regime Performance - {detector_method.capitalize()} Detector",
            save_path=os.path.join(output_dir, f"{symbol}_{detector_method}_regime_performance.png")
        )
    
    # Generate comprehensive report
    report_fig = create_comprehensive_report(
        dates=data['dates'],
        equity=results['equity_curve'],
        returns=results['strategy_returns'],
        trades=results['trades'],
        performance_metrics=results['performance_metrics'],
        regimes=results['regimes'],
        regime_metrics=results.get('regime_metrics'),
        title=f"{symbol} Backtest Results - {detector_method.capitalize()} Detector",
        save_path=os.path.join(output_dir, f"{symbol}_{detector_method}_report.png")
    )
    
    return results


def compare_methods(
    data: Dict[str, Any],
    methods: List[str] = ['volatility', 'trend', 'momentum', 'hmm'],
    initial_capital: float = 10000.0,
    output_dir: str = './outputs'
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different regime detection methods.
    
    Args:
        data: Dictionary with market data
        methods: List of methods to compare
        initial_capital: Initial capital
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with results for each method
    """
    results = {}
    performances = []
    
    # Run backtest for each method
    for method in methods:
        print(f"Running backtest with {method} detector...")
        result = run_backtest(
            data=data,
            detector_method=method,
            initial_capital=initial_capital,
            output_dir=output_dir
        )
        results[method] = result
        
        # Extract performance metrics
        perf = {
            'Method': method.capitalize(),
            'Total Return': result['performance_metrics']['total_return'],
            'Annual Return': result['performance_metrics']['annual_return'],
            'Sharpe Ratio': result['performance_metrics']['sharpe_ratio'],
            'Max Drawdown': result['performance_metrics']['max_drawdown'],
            'Win Rate': result['performance_metrics']['win_rate'],
            'Trades': result['performance_metrics']['num_trades']
        }
        performances.append(perf)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(performances)
    comparison_df.set_index('Method', inplace=True)
    
    # Plot comparison
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot equity curves
    ax1 = axs[0, 0]
    for method, result in results.items():
        ax1.plot(data['dates'], result['equity_curve'], label=method.capitalize())
    
    ax1.set_title('Equity Curves')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot returns
    ax2 = axs[0, 1]
    for method, result in results.items():
        cumulative_returns = np.cumprod(1 + np.array(result['strategy_returns'])) - 1
        ax2.plot(data['dates'], cumulative_returns * 100, label=method.capitalize())
    
    ax2.set_title('Cumulative Returns (%)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot Sharpe ratios
    ax3 = axs[1, 0]
    sharpes = [results[m]['performance_metrics']['sharpe_ratio'] for m in methods]
    ax3.bar(methods, sharpes, color='skyblue')
    ax3.set_title('Sharpe Ratios')
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.grid(True, alpha=0.3)
    
    # Plot max drawdowns
    ax4 = axs[1, 1]
    drawdowns = [results[m]['performance_metrics']['max_drawdown'] * 100 for m in methods]
    ax4.bar(methods, drawdowns, color='salmon')
    ax4.set_title('Maximum Drawdowns (%)')
    ax4.set_xlabel('Method')
    ax4.set_ylabel('Drawdown (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300)
    plt.show()
    
    # Print comparison table
    print("\nPerformance Comparison:")
    print(comparison_df.to_string())
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(output_dir, 'method_comparison.csv'))
    
    return results


def main():
    """Main function to demonstrate backtesting."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Download data
    print("Downloading market data...")
    symbol = 'SPY'
    data = download_market_data(symbol=symbol, period='5y')
    
    # Prepare data for backtesting
    market_data = prepare_market_data(data, symbol)
    
    # Compare methods
    print("\nComparing different regime detection methods...")
    results = compare_methods(
        data=market_data,
        methods=['volatility', 'trend', 'momentum', 'hmm'],
        initial_capital=10000.0,
        output_dir=output_dir
    )
    
    print(f"\nBacktesting complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main() 