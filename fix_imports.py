#!/usr/bin/env python
# fix_imports.py

"""
A script to clean up Python cache files and fix import issues.
"""

import os
import sys
import shutil
import re

def clean_pycache(directory):
    """
    Recursively remove all __pycache__ directories and .pyc files.
    """
    count = 0
    for root, dirs, files in os.walk(directory):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_dir)
                count += 1
                print(f"Removed {pycache_dir}")
            except Exception as e:
                print(f"Error removing {pycache_dir}: {e}")
        
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                try:
                    os.remove(pyc_file)
                    count += 1
                    print(f"Removed {pyc_file}")
                except Exception as e:
                    print(f"Error removing {pyc_file}: {e}")
    
    return count

def fix_performance_metrics(file_path):
    """
    Fix any issues in the performance_metrics.py file.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for the misplaced return statement
    if 'return omega' in content and 'def calculate_portfolio_diversification' in content:
        # Fix the misplaced return statement
        fixed_content = re.sub(
            r'def calculate_portfolio_diversification.*?return diversification_score\s+return omega',
            lambda m: m.group(0).replace('return omega', ''),
            content,
            flags=re.DOTALL
        )
        
        # Add the return statement to the calculate_omega_ratio function
        fixed_content = re.sub(
            r'def calculate_omega_ratio.*?omega = expected_gain / expected_loss if expected_loss != 0 else float\(\'inf\'\)',
            lambda m: m.group(0) + '\n    return omega',
            fixed_content,
            flags=re.DOTALL
        )
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        
        print(f"Fixed misplaced return statement in {file_path}")
        return True
    
    return False

def fix_backtesting_init(file_path):
    """
    Fix the __init__.py file in the backtesting package.
    """
    new_content = """\"\"\"
Backtesting module for AI Trading Agent.

This module provides tools for backtesting trading strategies.
\"\"\"

# Define __all__ without any imports initially
__all__ = ["Backtester", "calculate_metrics", "PerformanceMetrics", "RUST_AVAILABLE"]

# Set RUST_AVAILABLE to False by default
RUST_AVAILABLE = False

# Import performance metrics first as it has no dependencies on other modules
try:
    from .performance_metrics import calculate_metrics, PerformanceMetrics
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Error importing performance_metrics: {e}")
    # Create placeholder functions/classes if imports fail
    def calculate_metrics(*args, **kwargs):
        raise ImportError("calculate_metrics could not be imported")
    
    class PerformanceMetrics:
        def __init__(self, *args, **kwargs):
            raise ImportError("PerformanceMetrics could not be imported")

# Import the core backtester
try:
    from .backtester import Backtester
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Error importing Backtester: {e}")
    # Create placeholder class if import fails
    class Backtester:
        def __init__(self, *args, **kwargs):
            raise ImportError("Backtester could not be imported")

# Try to import RustBacktester, but don't fail if it's not available
try:
    # Import Rust backtester - handle gracefully if not available
    from .rust_backtester import RustBacktester
    RUST_AVAILABLE = True
    __all__.append("RustBacktester")
except ImportError as e:
    # Log the import error for debugging
    import logging
    logging.getLogger(__name__).warning(f"RustBacktester not available: {e}")
    # Create placeholder class for RustBacktester
    class RustBacktester:
        def __init__(self, *args, **kwargs):
            raise ImportError("RustBacktester is not available. Rust extensions may not be installed.")
"""
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Updated {file_path} with improved import handling")
    return True

def create_simple_test_script(directory):
    """
    Create a simple test script to verify imports are working.
    """
    test_script_path = os.path.join(directory, 'test_imports_simple.py')
    test_script_content = """#!/usr/bin/env python
# test_imports_simple.py

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Clean import paths
sys.path = list(set(sys.path))

print(f"Python path: {sys.path}")
print("\\nTesting imports...")

# Test imports
try:
    from ai_trading_agent.backtesting import Backtester, calculate_metrics, PerformanceMetrics, RUST_AVAILABLE
    print("✓ Successfully imported core backtesting components")
    
    if RUST_AVAILABLE:
        from ai_trading_agent.backtesting import RustBacktester
        print("✓ Successfully imported RustBacktester")
    else:
        print("Note: RustBacktester not available (this is expected if Rust extensions are not installed)")
    
    from ai_trading_agent.agent.data_manager import SimpleDataManager
    print("✓ Successfully imported data_manager")
    
    from ai_trading_agent.agent.strategy import SimpleStrategyManager, SentimentStrategy
    print("✓ Successfully imported strategy")
    
    from ai_trading_agent.agent.risk_manager import SimpleRiskManager
    print("✓ Successfully imported risk_manager")
    
    from ai_trading_agent.agent.execution_handler import SimulatedExecutionHandler
    print("✓ Successfully imported execution_handler")
    
    from ai_trading_agent.agent.orchestrator import BacktestOrchestrator
    print("✓ Successfully imported orchestrator")
    
    print("\\nAll imports successful!")
    
except Exception as e:
    print(f"\\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
"""
    
    with open(test_script_path, 'w') as f:
        f.write(test_script_content)
    
    print(f"Created simple test script at {test_script_path}")
    return test_script_path

if __name__ == "__main__":
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Project root: {project_root}")
    print("-" * 50)
    
    # Clean up Python cache files
    print("\nCleaning up Python cache files...")
    count = clean_pycache(project_root)
    print(f"Removed {count} cache files/directories")
    
    # Fix performance_metrics.py
    print("\nFixing performance_metrics.py...")
    performance_metrics_path = os.path.join(project_root, 'ai_trading_agent', 'backtesting', 'performance_metrics.py')
    if os.path.exists(performance_metrics_path):
        fix_performance_metrics(performance_metrics_path)
    else:
        print(f"Error: {performance_metrics_path} not found")
    
    # Fix backtesting/__init__.py
    print("\nFixing backtesting/__init__.py...")
    backtesting_init_path = os.path.join(project_root, 'ai_trading_agent', 'backtesting', '__init__.py')
    if os.path.exists(backtesting_init_path):
        fix_backtesting_init(backtesting_init_path)
    else:
        print(f"Error: {backtesting_init_path} not found")
    
    # Create a simple test script
    print("\nCreating a simple test script...")
    test_script_path = create_simple_test_script(project_root)
    
    print("\nFix completed. Run the following command to test imports:")
    print(f"python {os.path.basename(test_script_path)}")
