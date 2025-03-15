#!/usr/bin/env python
"""Command-line script for running backtests.

This script provides a command-line interface for running backtests
with various regime detection methods and strategy configurations.
"""

import os
import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from .example import (
    download_market_data,
    prepare_data_for_backtest,
    run_trend_strategy_backtest,
    run_volatility_strategy_backtest,
    run_ensemble_strategy_backtest,
    compare_strategies
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run regime-based strategy backtests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data options
    parser.add_argument("--symbol", type=str, default="SPY", 
                        help="Ticker symbol")
    parser.add_argument("--period", type=str, default="5y", 
                        help="Time period (e.g., '5y', '1y', '6mo')")
    parser.add_argument("--start-date", type=str, 
                        help="Start date (YYYY-MM-DD) - overrides period if specified")
    parser.add_argument("--end-date", type=str, 
                        help="End date (YYYY-MM-DD) - overrides period if specified")
    parser.add_argument("--data-file", type=str, 
                        help="CSV file with market data (instead of downloading)")
    
    # Strategy options
    parser.add_argument("--strategy", type=str, default="compare", 
                        choices=["trend", "volatility", "momentum", "hmm", "ensemble", "compare"],
                        help="Strategy to backtest")
    parser.add_argument("--n-regimes", type=int, default=3, 
                        help="Number of regimes to detect")
    parser.add_argument("--initial-capital", type=float, default=10000.0, 
                        help="Initial capital for backtest")
    
    # Trend strategy options
    parser.add_argument("--trend-method", type=str, default="ma_crossover", 
                        choices=["ma_crossover", "price_channels", "linear_regression"],
                        help="Trend detection method")
    parser.add_argument("--fast-window", type=int, default=20, 
                        help="Fast moving average window")
    parser.add_argument("--slow-window", type=int, default=50, 
                        help="Slow moving average window")
    
    # Volatility strategy options
    parser.add_argument("--volatility-window", type=int, default=20, 
                        help="Volatility calculation window")
    parser.add_argument("--use-log-returns", action="store_true", 
                        help="Use log returns for volatility calculation")
    
    # Position sizing options
    parser.add_argument("--position-sizing", type=str, default="fixed", 
                        choices=["fixed", "percent", "kelly", "volatility"],
                        help="Position sizing method")
    parser.add_argument("--max-position-size", type=float, default=1.0, 
                        help="Maximum position size (0.0 to 1.0)")
    
    # Risk management options
    parser.add_argument("--risk-management", type=str, default="basic", 
                        choices=["basic", "trailing", "volatility"],
                        help="Risk management method")
    parser.add_argument("--stop-loss-pct", type=float, default=0.05, 
                        help="Stop loss percentage")
    parser.add_argument("--take-profit-pct", type=float, default=None, type_default=None, 
                        help="Take profit percentage")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./backtest_results", 
                        help="Output directory for results")
    parser.add_argument("--no-plots", action="store_true", 
                        help="Disable plot generation")
    parser.add_argument("--save-trades", action="store_true", 
                        help="Save trade details to CSV")
    
    return parser.parse_args()


def load_market_data(args):
    """Load market data from file or download."""
    if args.data_file and os.path.exists(args.data_file):
        print(f"Loading market data from {args.data_file}...")
        df = pd.read_csv(args.data_file)
        
        # Convert date column to datetime
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in data file: {missing_cols}")
            
            # Try to map columns by name
            col_map = {}
            for col in df.columns:
                lower_col = col.lower()
                if 'open' in lower_col:
                    col_map['Open'] = col
                elif 'high' in lower_col:
                    col_map['High'] = col
                elif 'low' in lower_col:
                    col_map['Low'] = col
                elif 'close' in lower_col or 'price' in lower_col:
                    col_map['Close'] = col
                elif 'volume' in lower_col or 'vol' in lower_col:
                    col_map['Volume'] = col
            
            # Rename columns
            df = df.rename(columns=col_map)
        
        # Calculate returns if not present
        if 'Return' not in df.columns:
            df['Return'] = df['Close'].pct_change().fillna(0)
    else:
        # Download data
        if args.start_date and args.end_date:
            # Use date range instead of period
            start_date = args.start_date
            end_date = args.end_date
            print(f"Downloading market data for {args.symbol} from {start_date} to {end_date}...")
            
            import yfinance as yf
            ticker = yf.Ticker(args.symbol)
            df = ticker.history(start=start_date, end=end_date)
        else:
            # Use period
            print(f"Downloading market data for {args.symbol} ({args.period})...")
            df = download_market_data(args.symbol, args.period)
    
    return df


def create_detector_params(args):
    """Create detector parameters based on command-line arguments."""
    if args.strategy == "trend":
        return {
            'n_regimes': args.n_regimes,
            'trend_method': args.trend_method,
            'fast_window': args.fast_window,
            'slow_window': args.slow_window
        }
    elif args.strategy == "volatility":
        return {
            'n_regimes': args.n_regimes,
            'lookback_window': args.volatility_window,
            'use_log_returns': args.use_log_returns
        }
    elif args.strategy == "momentum":
        return {
            'n_regimes': args.n_regimes,
            'lookback_window': args.volatility_window
        }
    elif args.strategy == "hmm":
        return {
            'n_regimes': args.n_regimes
        }
    elif args.strategy == "ensemble":
        return {
            'methods': ['trend', 'volatility', 'momentum'],
            'ensemble_type': 'voting',
            'voting_method': 'soft',
            'n_regimes': args.n_regimes,
            'method_params': {
                'trend': {'trend_method': args.trend_method},
                'volatility': {'lookback_window': args.volatility_window},
                'momentum': {'lookback_window': args.volatility_window}
            }
        }
    else:
        return {'n_regimes': args.n_regimes}


def create_position_sizing_params(args):
    """Create position sizing parameters based on command-line arguments."""
    if args.position_sizing == "fixed":
        return {
            'fraction': args.max_position_size
        }
    elif args.position_sizing == "percent":
        return {
            'percent': args.max_position_size
        }
    elif args.position_sizing == "kelly":
        return {
            'fraction': 0.5  # Half-Kelly for more conservative sizing
        }
    elif args.position_sizing == "volatility":
        return {
            'target_risk_pct': 0.01,
            'volatility_lookback': args.volatility_window
        }
    else:
        return {}


def create_risk_management_params(args):
    """Create risk management parameters based on command-line arguments."""
    if args.risk_management == "basic":
        return {
            'stop_loss_pct': args.stop_loss_pct,
            'take_profit_pct': args.take_profit_pct
        }
    elif args.risk_management == "trailing":
        return {
            'initial_stop_pct': args.stop_loss_pct,
            'trailing_pct': args.stop_loss_pct / 2.5  # Default trailing distance
        }
    elif args.risk_management == "volatility":
        return {
            'atr_multiplier': 2.0,
            'lookback_periods': 14
        }
    else:
        return {}


def main():
    """Run the backtest based on command-line arguments."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load market data
    df = load_market_data(args)
    
    # Save arguments for reproducibility
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, "backtest_config.json")
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # Run the appropriate strategy
    if args.strategy == "compare":
        compare_strategies(args.symbol, args.period, args.output_dir)
    else:
        # Prepare data for backtesting
        data = prepare_data_for_backtest(df, args.symbol)
        
        # Create strategy parameters
        detector_params = create_detector_params(args)
        position_sizing_params = create_position_sizing_params(args)
        risk_management_params = create_risk_management_params(args)
        
        # Run the appropriate strategy
        if args.strategy == "trend":
            run_trend_strategy_backtest(args.symbol, args.period, args.output_dir)
        elif args.strategy == "volatility":
            run_volatility_strategy_backtest(args.symbol, args.period, args.output_dir)
        elif args.strategy == "ensemble":
            run_ensemble_strategy_backtest(args.symbol, args.period, args.output_dir)
        else:
            print(f"Strategy '{args.strategy}' not implemented yet.")
    
    print(f"Backtest completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 