#!/usr/bin/env python
# test_imports_minimal.py

"""
A minimal script to test imports and identify issues.
"""

import os
import sys
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create a function to test imports
def test_import(module_name, items=None):
    try:
        if items:
            exec(f"from {module_name} import {', '.join(items)}")
            print(f"✓ Successfully imported {', '.join(items)} from {module_name}")
        else:
            exec(f"import {module_name}")
            print(f"✓ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {module_name}: {e}")
        traceback.print_exc()
        return False

# Test imports one by one
print(f"Testing imports from project root: {project_root}")
print("-" * 50)

# Test common modules
print("\nTesting common modules:")
test_import("ai_trading_agent.common.logging_config", ["setup_logging"])

# Test trading engine components
print("\nTesting trading engine components:")
test_import("ai_trading_agent.trading_engine.enums", ["OrderSide", "OrderType", "OrderStatus"])
test_import("ai_trading_agent.trading_engine.models", ["Order", "Trade", "Position", "Portfolio", "Fill"])
test_import("ai_trading_agent.trading_engine.portfolio_manager", ["PortfolioManager"])

# Test backtesting components
print("\nTesting backtesting components:")
test_import("ai_trading_agent.backtesting.performance_metrics", ["calculate_metrics", "PerformanceMetrics"])
test_import("ai_trading_agent.backtesting.backtester", ["Backtester"])

# Test rust_backtester
print("\nTesting rust_backtester:")
rust_success = test_import("ai_trading_agent.backtesting.rust_backtester", ["RustBacktester"])
if not rust_success:
    print("Note: This is expected if Rust extensions are not installed.")

# Test agent components
print("\nTesting agent components:")
test_import("ai_trading_agent.agent.data_manager", ["SimpleDataManager"])
test_import("ai_trading_agent.agent.strategy", ["SimpleStrategyManager", "SentimentStrategy"])
test_import("ai_trading_agent.agent.risk_manager", ["SimpleRiskManager"])
test_import("ai_trading_agent.agent.execution_handler", ["SimulatedExecutionHandler"])
test_import("ai_trading_agent.agent.orchestrator", ["BacktestOrchestrator"])

print("\nImport testing completed.")
