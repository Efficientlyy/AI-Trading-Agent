#!/usr/bin/env python
"""Run script for the backtesting example.

This script provides a command-line interface for running the backtesting example.
"""

import os
import sys
import argparse
from datetime import datetime

from .example import (
    download_market_data,
    run_trend_strategy_backtest,
    run_volatility_strategy_backtest,
    run_ensemble_strategy_backtest,
    compare_strategies
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run backtesting examples")
    
    parser.add_argument(
        "--symbols", 
        type=str, 
        default="SPY",
        help="Comma-separated list of ticker symbols (default: SPY)"
    )
    
    parser.add_argument(
        "--period", 
        type=str, 
        default="5y",
        help="Data period (e.g., 1y, 5y, max) (default: 5y)"
    )
    
    parser.add_argument(
        "--strategies", 
        type=str, 
        default="all",
        help="Comma-separated list of strategies to run (trend, volatility, ensemble, all) (default: all)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./backtest_results",
        help="Output directory for results (default: ./backtest_results)"
    )
    
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare strategies (default: False)"
    )
    
    return parser.parse_args()


def main():
    """Run the backtesting example with command-line arguments."""
    args = parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Parse strategies
    if args.strategies.lower() == "all":
        strategies = ["trend", "volatility", "ensemble"]
    else:
        strategies = [s.strip().lower() for s in args.strategies.split(",")]
        valid_strategies = ["trend", "volatility", "ensemble"]
        strategies = [s for s in strategies if s in valid_strategies]
        
        if not strategies:
            print("Error: No valid strategies specified")
            sys.exit(1)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running backtests for symbols: {', '.join(symbols)}")
    print(f"Using strategies: {', '.join(strategies)}")
    print(f"Results will be saved to: {output_dir}")
    
    # Download data and run backtests
    data_dict = {}
    results_dict = {}
    
    for symbol in symbols:
        data_dict[symbol] = download_market_data(symbol, period=args.period)
        
        symbol_dir = os.path.join(output_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        results_dict[symbol] = {}
        
        # Run selected strategy backtests
        if "trend" in strategies:
            results_dict[symbol]["trend"] = run_trend_strategy_backtest(
                data_dict[symbol], symbol_dir
            )
        
        if "volatility" in strategies:
            results_dict[symbol]["volatility"] = run_volatility_strategy_backtest(
                data_dict[symbol], symbol_dir
            )
        
        if "ensemble" in strategies:
            results_dict[symbol]["ensemble"] = run_ensemble_strategy_backtest(
                data_dict[symbol], symbol_dir
            )
        
        # Compare strategies for this symbol if multiple strategies were run
        if args.compare and len(results_dict[symbol]) > 1:
            strategy_results = list(results_dict[symbol].values())
            strategy_names = list(results_dict[symbol].keys())
            
            compare_strategies(
                strategy_results,
                [symbol] * len(strategy_results),
                [s.capitalize() for s in strategy_names],
                symbol_dir
            )
    
    # Compare best strategy across symbols if multiple symbols were used
    if args.compare and len(symbols) > 1:
        best_results = []
        best_strategy_names = []
        
        for symbol, results in results_dict.items():
            if not results:
                continue
                
            # Choose the best strategy based on Sharpe ratio
            best_strategy = max(
                results.items(),
                key=lambda x: x[1]["performance_metrics"].get("sharpe_ratio", 0)
            )
            best_results.append(best_strategy[1])
            best_strategy_names.append(f"{symbol} ({best_strategy[0].capitalize()})")
        
        if len(best_results) > 1:
            compare_strategies(
                best_results,
                symbols,
                best_strategy_names,
                output_dir
            )
    
    print(f"All backtests completed successfully! Results saved to {output_dir}")


if __name__ == "__main__":
    main() 