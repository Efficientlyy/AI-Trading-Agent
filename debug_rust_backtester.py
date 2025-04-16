#!/usr/bin/env python
# debug_rust_backtester.py

"""
A focused script to debug the rust_backtester module import issues.
"""

import sys
import os
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create a log file
with open("rust_backtester_debug.log", "w") as log:
    log.write(f"Added project root to sys.path: {project_root}\n\n")
    
    try:
        # First, try importing the enums and models
        log.write("Importing enums...\n")
        from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
        log.write("✓ Successfully imported enums\n\n")
        
        log.write("Importing models...\n")
        from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio, Fill
        log.write("✓ Successfully imported models\n\n")
        
        # Try importing performance_metrics
        log.write("Importing performance_metrics...\n")
        from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
        log.write("✓ Successfully imported performance_metrics\n\n")
        
        # Try importing rust_backtester
        log.write("Importing rust_backtester...\n")
        
        # First, check if the module can be imported directly
        try:
            import ai_trading_agent.backtesting.rust_backtester
            log.write("✓ Successfully imported rust_backtester module\n")
        except Exception as e:
            log.write(f"✗ Failed to import rust_backtester module: {e}\n")
            log.write(traceback.format_exc())
            log.write("\n")
        
        # Then, try importing the RustBacktester class
        try:
            from ai_trading_agent.backtesting.rust_backtester import RustBacktester
            log.write("✓ Successfully imported RustBacktester class\n")
        except Exception as e:
            log.write(f"✗ Failed to import RustBacktester class: {e}\n")
            log.write(traceback.format_exc())
            log.write("\n")
        
        # Try importing the rust_extensions module
        try:
            log.write("Checking for rust_extensions module...\n")
            try:
                import rust_extensions
                log.write("✓ rust_extensions module is available\n")
            except ImportError:
                log.write("✗ rust_extensions module is not available\n")
                log.write("This is expected if Rust extensions are not installed\n")
        except Exception as e:
            log.write(f"Error checking rust_extensions: {e}\n")
        
        log.write("\nDebug complete.\n")
        
    except Exception as e:
        log.write(f"Unexpected error: {e}\n")
        log.write(traceback.format_exc())

print("Debug complete. Check rust_backtester_debug.log for details.")
