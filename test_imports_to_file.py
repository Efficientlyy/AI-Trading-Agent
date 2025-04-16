#!/usr/bin/env python
# test_imports_to_file.py

"""
A script to test imports and write results to a file.
"""

import os
import sys
import traceback
from datetime import datetime

# Create a log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"import_test_{timestamp}.txt"

with open(log_file, "w") as f:
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    f.write(f"Testing imports from project root: {project_root}\n")
    f.write("-" * 50 + "\n")
    
    # Create a function to test imports
    def test_import(module_name, items=None):
        try:
            if items:
                exec(f"from {module_name} import {', '.join(items)}")
                f.write(f"✓ Successfully imported {', '.join(items)} from {module_name}\n")
                return True
            else:
                exec(f"import {module_name}")
                f.write(f"✓ Successfully imported {module_name}\n")
                return True
        except Exception as e:
            f.write(f"✗ Failed to import {module_name}: {e}\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
            f.write("\n")
            return False
    
    # Test imports one by one
    
    # Test common modules
    f.write("\nTesting common modules:\n")
    test_import("ai_trading_agent.common.logging_config", ["setup_logging"])
    
    # Test trading engine components
    f.write("\nTesting trading engine components:\n")
    test_import("ai_trading_agent.trading_engine.enums", ["OrderSide", "OrderType", "OrderStatus"])
    test_import("ai_trading_agent.trading_engine.models", ["Order", "Trade", "Position", "Portfolio", "Fill"])
    test_import("ai_trading_agent.trading_engine.portfolio_manager", ["PortfolioManager"])
    
    # Test backtesting components
    f.write("\nTesting backtesting components:\n")
    test_import("ai_trading_agent.backtesting.performance_metrics", ["calculate_metrics", "PerformanceMetrics"])
    test_import("ai_trading_agent.backtesting.backtester", ["Backtester"])
    
    # Test rust_backtester
    f.write("\nTesting rust_backtester:\n")
    rust_success = test_import("ai_trading_agent.backtesting.rust_backtester", ["RustBacktester"])
    if not rust_success:
        f.write("Note: This is expected if Rust extensions are not installed.\n")
    
    # Test agent components
    f.write("\nTesting agent components:\n")
    test_import("ai_trading_agent.agent.data_manager", ["SimpleDataManager"])
    test_import("ai_trading_agent.agent.strategy", ["SimpleStrategyManager", "SentimentStrategy"])
    test_import("ai_trading_agent.agent.risk_manager", ["SimpleRiskManager"])
    test_import("ai_trading_agent.agent.execution_handler", ["SimulatedExecutionHandler"])
    test_import("ai_trading_agent.agent.orchestrator", ["BacktestOrchestrator"])
    
    f.write("\nImport testing completed.\n")

print(f"Import testing completed. Results saved to {log_file}")
