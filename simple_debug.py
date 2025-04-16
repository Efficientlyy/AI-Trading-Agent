#!/usr/bin/env python
# simple_debug.py

"""
A simple script to debug import issues by attempting to import each component
individually and printing directly to console.
"""

import sys
import os
import traceback

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
print(f"Added project root to sys.path: {project_root}")

# Test imports one by one
try:
    print("\nImporting enums...")
    from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
    print("✓ Successfully imported enums")
    
    print("\nImporting models...")
    from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio, Fill
    print("✓ Successfully imported models")
    
    print("\nImporting portfolio_manager...")
    from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
    print("✓ Successfully imported portfolio_manager")
    
    print("\nImporting performance_metrics...")
    from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
    print("✓ Successfully imported performance_metrics")
    
    print("\nImporting backtester...")
    from ai_trading_agent.backtesting.backtester import Backtester
    print("✓ Successfully imported backtester")
    
    print("\nImporting rust_backtester...")
    from ai_trading_agent.backtesting.rust_backtester import RustBacktester
    print("✓ Successfully imported rust_backtester")
    
    print("\nImporting data_manager...")
    from ai_trading_agent.agent.data_manager import SimpleDataManager
    print("✓ Successfully imported data_manager")
    
    print("\nImporting strategy...")
    from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
    print("✓ Successfully imported strategy")
    
    print("\nImporting risk_manager...")
    from ai_trading_agent.agent.risk_manager import SimpleRiskManager
    print("✓ Successfully imported risk_manager")
    
    print("\nImporting execution_handler...")
    from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
    print("✓ Successfully imported execution_handler")
    
    print("\nImporting orchestrator...")
    from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
    print("✓ Successfully imported orchestrator")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTraceback:")
    traceback.print_exc()
