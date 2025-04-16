#!/usr/bin/env python
# fix_backtest.py

"""
A script to fix the run_backtest.py file by adding proper error handling
and import diagnostics.
"""

import os
import sys
import shutil
import re

# Get the path to the run_backtest.py file
project_root = os.path.dirname(os.path.abspath(__file__))
backtest_path = os.path.join(project_root, 'scripts', 'run_backtest.py')

# Create a backup of the original file
backup_path = backtest_path + '.bak'
shutil.copy2(backtest_path, backup_path)
print(f"Created backup of run_backtest.py at {backup_path}")

# Read the content of the file
with open(backtest_path, 'r') as f:
    content = f.read()

# Add improved imports and error handling
improved_imports = """#!/usr/bin/env python
# scripts/run_backtest.py

import os
import sys
import traceback
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Set up error logging
error_log_path = "backtest_error_detailed.log"
with open(error_log_path, "w") as error_log:
    error_log.write(f"=== BACKTEST ERROR LOG ===\\n")
    error_log.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
    
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)
        error_log.write(f"Added project root to sys.path: {project_root}\\n\\n")
        
        # Import components with detailed error logging
        error_log.write("Importing common modules...\\n")
        from ai_trading_agent.common.logging_config import setup_logging
        error_log.write("✓ Successfully imported logging_config\\n\\n")
        
        error_log.write("Importing trading engine components...\\n")
        from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
        error_log.write("✓ Successfully imported enums\\n")
        
        from ai_trading_agent.trading_engine.models import Order, Trade, Position, Portfolio, Fill
        error_log.write("✓ Successfully imported models\\n")
        
        from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
        error_log.write("✓ Successfully imported portfolio_manager\\n\\n")
        
        error_log.write("Importing backtesting components...\\n")
        from ai_trading_agent.backtesting.performance_metrics import calculate_metrics, PerformanceMetrics
        error_log.write("✓ Successfully imported performance_metrics\\n")
        
        from ai_trading_agent.backtesting.backtester import Backtester
        error_log.write("✓ Successfully imported Backtester\\n\\n")
        
        # Only import RustBacktester if needed
        try:
            from ai_trading_agent.backtesting.rust_backtester import RustBacktester
            error_log.write("✓ Successfully imported RustBacktester\\n")
            RUST_AVAILABLE = True
        except ImportError as e:
            error_log.write(f"Note: RustBacktester not available: {e}\\n")
            error_log.write("This is expected if Rust extensions are not installed.\\n")
            RUST_AVAILABLE = False
        
        error_log.write("Importing agent components...\\n")
        from ai_trading_agent.agent.data_manager import SimpleDataManager
        error_log.write("✓ Successfully imported data_manager\\n")
        
        from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
        error_log.write("✓ Successfully imported strategy\\n")
        
        from ai_trading_agent.agent.risk_manager import SimpleRiskManager
        error_log.write("✓ Successfully imported risk_manager\\n")
        
        from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
        error_log.write("✓ Successfully imported execution_handler\\n")
        
        from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
        error_log.write("✓ Successfully imported orchestrator\\n\\n")
        
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Continue with the rest of the script...
"""

# Replace the imports section with our improved version
# First, find where the original imports end
import_pattern = r"(?s)^.*?import.*?(?=\n\n)"
match = re.search(import_pattern, content)

if match:
    # Replace the imports section
    new_content = re.sub(import_pattern, improved_imports, content, count=1)
    
    # Add error handling at the end of the file
    if "if __name__ == '__main__':" in new_content:
        # Add exception handling to the main block
        main_pattern = r"if __name__ == '__main__':(.*?)$"
        main_match = re.search(main_pattern, new_content, re.DOTALL)
        
        if main_match:
            main_block = main_match.group(1)
            indented_main = "\n    ".join([""] + main_block.split("\n"))
            
            error_handling = """if __name__ == '__main__':
    try:
        # Wrap the main execution in a try-except block
""" + indented_main + """
    except Exception as e:
        logger.error(f"Error in backtest execution: {e}")
        with open(error_log_path, "a") as error_log:
            error_log.write(f"\\n\\nERROR in backtest execution: {e}\\n")
            error_log.write("Traceback:\\n")
            error_log.write(traceback.format_exc())
        print(f"Error occurred. See {error_log_path} for details.")
    finally:
        with open(error_log_path, "a") as error_log:
            error_log.write(f"\\n=== END OF LOG ===\\n")
"""
            new_content = re.sub(main_pattern, error_handling, new_content, flags=re.DOTALL)
    
    # Write the modified content back to the file
    with open(backtest_path, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully updated {backtest_path} with improved error handling and imports")
else:
    print("Could not find imports section in the file. No changes made.")

print("Done.")
