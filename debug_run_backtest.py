#!/usr/bin/env python
# debug_run_backtest.py

import sys
import traceback

print("Starting debug_run_backtest.py")

try:
    # Import the run_backtest module
    from scripts import run_backtest
    print("Successfully imported run_backtest")
except ImportError as e:
    print(f"ImportError: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"Other exception: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("Debug script completed.")
