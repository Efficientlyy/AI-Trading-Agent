#!/usr/bin/env python
# debug_imports_detailed.py

"""
A script to debug import issues by attempting to import each component
individually and writing detailed error information to a file.
"""

import sys
import os
import traceback
from datetime import datetime

# Create a log file with timestamp to avoid overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"import_debug_{timestamp}.txt"

with open(log_file, "w") as f:
    f.write("=== IMPORT DEBUG LOG ===\n")
    f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Ensure project root is in path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    f.write(f"Added project root to sys.path: {project_root}\n\n")
    
    # Test imports one by one
    try:
        f.write("Importing enums...\n")
        from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
        f.write("✓ Successfully imported enums\n\n")
        
        f.write("Importing models...\n")
        from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio, Fill
        f.write("✓ Successfully imported models\n\n")
        
        f.write("Importing portfolio_manager...\n")
        from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
        f.write("✓ Successfully imported portfolio_manager\n\n")
        
        f.write("Importing performance_metrics...\n")
        from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
        f.write("✓ Successfully imported performance_metrics\n\n")
        
        f.write("Importing backtester...\n")
        from ai_trading_agent.backtesting.backtester import Backtester
        f.write("✓ Successfully imported backtester\n\n")
        
        f.write("Importing rust_backtester...\n")
        from ai_trading_agent.backtesting.rust_backtester import RustBacktester
        f.write("✓ Successfully imported rust_backtester\n\n")
        
        f.write("Importing data_manager...\n")
        from ai_trading_agent.agent.data_manager import SimpleDataManager
        f.write("✓ Successfully imported data_manager\n\n")
        
        f.write("Importing strategy...\n")
        from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
        f.write("✓ Successfully imported strategy\n\n")
        
        f.write("Importing risk_manager...\n")
        from ai_trading_agent.agent.risk_manager import SimpleRiskManager
        f.write("✓ Successfully imported risk_manager\n\n")
        
        f.write("Importing execution_handler...\n")
        from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
        f.write("✓ Successfully imported execution_handler\n\n")
        
        f.write("Importing orchestrator...\n")
        from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
        f.write("✓ Successfully imported orchestrator\n\n")
        
        f.write("All imports successful!\n")
        
    except Exception as e:
        f.write(f"\n❌ ERROR: {e}\n\n")
        f.write("Traceback:\n")
        traceback.print_exc(file=f)
        f.write("\n")
    
    f.write("\n=== END OF LOG ===\n")

print(f"Import testing completed. Results saved to {log_file}")
