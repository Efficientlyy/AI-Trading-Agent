#!/usr/bin/env python
"""
Command-line script to run the market regime detection example.
"""

import argparse
import os
import sys
from .example import main as run_example
from .factory import RegimeDetectorFactory


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run market regime detection example")
    
    parser.add_argument("--symbol", type=str, default="SPY",
                        help="Ticker symbol to analyze (default: SPY)")
    
    parser.add_argument("--period", type=str, default="2y",
                        help="Time period to analyze (default: 2y)")
    
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save output files (default: current directory)")
    
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated list of methods to use (default: all)")
    
    parser.add_argument("--n-regimes", type=int, default=3,
                        help="Number of regimes to detect (default: 3)")
    
    parser.add_argument("--trend-method", type=str, default="ma_crossover",
                        help="Trend detection method: ma_crossover, adx, or slope (default: ma_crossover)")
    
    return parser.parse_args()


def main():
    """Run the example with command-line arguments."""
    args = parse_args()
    
    # Set environment variables for the example script
    os.environ["EXAMPLE_SYMBOL"] = args.symbol
    os.environ["EXAMPLE_PERIOD"] = args.period
    os.environ["EXAMPLE_OUTPUT_DIR"] = args.output_dir
    
    # Set methods to use
    if args.methods != "all":
        methods = args.methods.split(",")
        factory = RegimeDetectorFactory()
        available_methods = factory.get_available_methods()
        
        for method in methods:
            if method not in available_methods:
                print(f"Error: Unknown method '{method}'")
                print(f"Available methods: {', '.join(available_methods)}")
                sys.exit(1)
        
        os.environ["EXAMPLE_METHODS"] = args.methods
    
    # Set number of regimes
    os.environ["EXAMPLE_N_REGIMES"] = str(args.n_regimes)
    
    # Set trend method
    os.environ["EXAMPLE_TREND_METHOD"] = args.trend_method
    
    # Run the example
    run_example()


if __name__ == "__main__":
    main() 