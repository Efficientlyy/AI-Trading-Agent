#!/usr/bin/env python
# debug_imports.py

"""
A simple script to debug import issues by attempting to import each component
individually and printing detailed error information.
"""

import sys
import os
import traceback
import importlib

# Create a log file for errors with better formatting
with open("debug_imports.log", "w") as error_log:
    # Ensure project root is in path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    error_log.write(f"Added project root to sys.path: {project_root}\n\n")
    
    # List of modules to try importing
    modules_to_test = [
        "ai_trading_agent.trading_engine.enums",
        "ai_trading_agent.trading_engine.models",
        "ai_trading_agent.trading_engine.portfolio_manager",
        "ai_trading_agent.backtesting.performance_metrics",
        "ai_trading_agent.backtesting.backtester",
        "ai_trading_agent.backtesting.rust_backtester",
        "ai_trading_agent.agent.data_manager",
        "ai_trading_agent.agent.strategy",
        "ai_trading_agent.agent.risk_manager",
        "ai_trading_agent.agent.execution_handler",
        "ai_trading_agent.agent.orchestrator"
    ]
    
    # Try importing each module
    for module_name in modules_to_test:
        error_log.write(f"Attempting to import {module_name}...\n")
        try:
            module = importlib.import_module(module_name)
            error_log.write(f"✓ Successfully imported {module_name}\n")
            
            # For key modules, try to import specific classes
            if module_name == "ai_trading_agent.trading_engine.models":
                error_log.write("  Checking specific classes in models:\n")
                for class_name in ["Order", "Trade", "Position", "Portfolio", "Fill"]:
                    try:
                        cls = getattr(module, class_name)
                        error_log.write(f"  ✓ Successfully imported {class_name} from {module_name}\n")
                    except AttributeError as e:
                        error_log.write(f"  ✗ Failed to import {class_name} from {module_name}: {e}\n")
            
            elif module_name == "ai_trading_agent.backtesting.rust_backtester":
                error_log.write("  Checking RustBacktester class:\n")
                try:
                    cls = getattr(module, "RustBacktester")
                    error_log.write(f"  ✓ Successfully imported RustBacktester from {module_name}\n")
                except AttributeError as e:
                    error_log.write(f"  ✗ Failed to import RustBacktester from {module_name}: {e}\n")
                    
            elif module_name == "ai_trading_agent.agent.execution_handler":
                error_log.write("  Checking execution handler classes:\n")
                for class_name in ["ExecutionHandlerABC", "BaseExecutionHandler", "SimulatedExecutionHandler"]:
                    try:
                        cls = getattr(module, class_name)
                        error_log.write(f"  ✓ Successfully imported {class_name} from {module_name}\n")
                    except AttributeError as e:
                        error_log.write(f"  ✗ Failed to import {class_name} from {module_name}: {e}\n")
                        
        except ImportError as e:
            error_log.write(f"✗ Failed to import {module_name}: {e}\n")
            error_log.write("Traceback:\n")
            traceback.print_exc(file=error_log)
            error_log.write("\n")
        except Exception as e:
            error_log.write(f"✗ Unexpected error importing {module_name}: {e}\n")
            error_log.write("Traceback:\n")
            traceback.print_exc(file=error_log)
            error_log.write("\n")
        
        error_log.write("\n")  # Add blank line between modules
    
    error_log.write("Import testing completed.\n")

print("Import testing completed. See debug_imports.log for detailed results.")
