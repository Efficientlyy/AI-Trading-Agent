#!/usr/bin/env python
# minimal_test.py

"""
A minimal script to test imports and write results directly to a file.
"""

import os
import sys
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create a log file
with open("minimal_test_results.txt", "w") as f:
    f.write(f"Testing imports from project root: {project_root}\n\n")
    
    try:
        f.write("Importing trading_engine.enums...\n")
        from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
        f.write("✓ Successfully imported enums\n\n")
        
        f.write("Importing trading_engine.models...\n")
        from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio, Fill
        f.write("✓ Successfully imported models\n\n")
        
        f.write("Importing backtesting.performance_metrics...\n")
        from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
        f.write("✓ Successfully imported performance_metrics\n\n")
        
        f.write("Importing backtesting.backtester...\n")
        from ai_trading_agent.backtesting.backtester import Backtester
        f.write("✓ Successfully imported backtester\n\n")
        
        f.write("Importing agent.execution_handler...\n")
        from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
        f.write("✓ Successfully imported execution_handler\n\n")
        
        f.write("All imports successful!\n")
    except Exception as e:
        f.write(f"ERROR: {e}\n")
        f.write("Traceback:\n")
        traceback.print_exc(file=f)

print("Import testing completed. See minimal_test_results.txt for details.")
