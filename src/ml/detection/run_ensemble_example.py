#!/usr/bin/env python
"""
Command-line script to run the ensemble regime detection example.
"""

import argparse
import os
import sys
from .ensemble_example import run_ensemble_example


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run ensemble regime detection example")
    
    parser.add_argument("--symbol", type=str, default="SPY",
                        help="Ticker symbol to analyze (default: SPY)")
    
    parser.add_argument("--period", type=str, default="3y",
                        help="Time period to analyze (default: 3y)")
    
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save output files (default: current directory)")
    
    parser.add_argument("--ensemble-type", type=str, default="all",
                        choices=["all", "bagging", "boosting", "stacking"],
                        help="Ensemble technique to use (default: all)")
    
    parser.add_argument("--voting", type=str, default="soft",
                        choices=["hard", "soft"],
                        help="Voting method for ensemble (default: soft)")
    
    parser.add_argument("--n-regimes", type=int, default=3,
                        help="Number of regimes to detect (default: 3)")
    
    return parser.parse_args()


def main():
    """Run the ensemble example with command-line arguments."""
    args = parse_args()
    
    # Set environment variables for the example script
    os.environ["EXAMPLE_SYMBOL"] = args.symbol
    os.environ["EXAMPLE_PERIOD"] = args.period
    os.environ["EXAMPLE_OUTPUT_DIR"] = args.output_dir
    os.environ["EXAMPLE_ENSEMBLE_TYPE"] = args.ensemble_type
    os.environ["EXAMPLE_VOTING"] = args.voting
    os.environ["EXAMPLE_N_REGIMES"] = str(args.n_regimes)
    
    # Run the example
    run_ensemble_example()


if __name__ == "__main__":
    main() 