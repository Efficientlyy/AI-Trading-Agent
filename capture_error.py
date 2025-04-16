#!/usr/bin/env python
# capture_error.py

import sys
import os
import traceback

# Redirect stdout and stderr to a file
error_log = open("error_output.txt", "w")
sys.stdout = error_log
sys.stderr = error_log

try:
    print("Starting error capture...")
    
    # Ensure project root is in path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")
    
    # Try importing the problematic module directly
    print("Attempting to import ai_trading_agent.trading_engine.enums...")
    from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
    print("Successfully imported enums")
    
    print("Attempting to import ai_trading_agent.trading_engine.models...")
    from ai_trading_agent.trading_engine.models import Order, Trade
    print("Successfully imported models")
    
    # Try importing the script that's failing
    print("Attempting to import scripts.run_backtest...")
    import scripts.run_backtest
    print("Successfully imported run_backtest")
    
except Exception as e:
    print(f"ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

finally:
    # Close the file and restore stdout/stderr
    error_log.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    print(f"Error capture completed. Check error_output.txt for details.")
