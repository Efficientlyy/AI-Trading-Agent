#!/usr/bin/env python
# fix_imports_comprehensive.py

"""
Comprehensive script to detect and fix all import issues in the codebase.
This script will:
1. Find all incorrect imports of enum types from models.py
2. Fix circular dependencies by restructuring imports
3. Ensure consistent import patterns throughout the codebase
4. Clean up __pycache__ directories
"""

import os
import re
import sys
import shutil
import logging
from typing import List, Dict, Tuple, Set, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("import_fixes_comprehensive.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
ENUM_NAMES = ["OrderSide", "OrderType", "OrderStatus", "PositionSide"]
MODEL_NAMES = ["Order", "Trade", "Position", "Portfolio", "Fill"]

def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in a directory and its subdirectories.
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_file_for_incorrect_imports(filepath: str) -> Tuple[bool, List[Tuple[int, str, str]]]:
    """
    Check a file for incorrect imports of enum types from models.py.
    
    Returns:
        Tuple containing:
        - Boolean indicating if incorrect imports were found
        - List of tuples (line_number, line_content, matched_pattern)
    """
    incorrect_imports = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            # Check for imports of enum types from models
            if any(f"from ai_trading_agent.trading_engine.models import {enum}" in line for enum in ENUM_NAMES):
                incorrect_imports.append((i+1, line, "enum_from_models"))
            elif "from ai_trading_agent.trading_engine.models import" in line and any(enum in line for enum in ENUM_NAMES):
                incorrect_imports.append((i+1, line, "enum_in_models_import"))
            # Check for imports of models from enums
            elif any(f"from ai_trading_agent.trading_engine.enums import {model}" in line for model in MODEL_NAMES):
                incorrect_imports.append((i+1, line, "model_from_enums"))
            elif "from ai_trading_agent.trading_engine.enums import" in line and any(model in line for model in MODEL_NAMES):
                incorrect_imports.append((i+1, line, "model_in_enums_import"))
            # Check for references to calculate_performance_metrics
            elif "calculate_performance_metrics" in line and "def calculate_performance_metrics" not in line:
                incorrect_imports.append((i+1, line, "calculate_performance_metrics"))
    
    except Exception as e:
        logger.error(f"Error checking imports in {filepath}: {e}")
        return False, []
    
    return len(incorrect_imports) > 0, incorrect_imports

def fix_incorrect_imports(filepath: str, incorrect_imports: List[Tuple[int, str, str]]) -> bool:
    """
    Fix incorrect imports in a file.
    
    Args:
        filepath: Path to the file to fix
        incorrect_imports: List of tuples (line_number, line_content, matched_pattern)
        
    Returns:
        Boolean indicating if the file was fixed
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, _, pattern in incorrect_imports:
            i = line_num - 1  # Convert to 0-based index
            
            if pattern == "enum_from_models" or pattern == "enum_in_models_import":
                # Fix imports of enum types from models
                current_line = lines[i]
                
                # Extract the imported items
                if "import " in current_line:
                    imported_items = current_line.split("import ")[1].strip()
                    # Remove commas and split by whitespace
                    imported_items = re.sub(r'[,\n]', ' ', imported_items).split()
                    
                    # Separate enums and models
                    enum_imports = [item for item in imported_items if item in ENUM_NAMES]
                    model_imports = [item for item in imported_items if item not in ENUM_NAMES]
                    
                    # Update the line
                    if enum_imports and model_imports:
                        # Need to split into two import statements
                        from_part = current_line.split("import")[0]
                        lines[i] = f"{from_part}import {', '.join(model_imports)}\n"
                        # Add a new line for enum imports
                        lines.insert(i+1, f"from ai_trading_agent.trading_engine.enums import {', '.join(enum_imports)}\n")
                    elif enum_imports:
                        # Replace with import from enums
                        lines[i] = f"from ai_trading_agent.trading_engine.enums import {', '.join(enum_imports)}\n"
            
            elif pattern == "model_from_enums" or pattern == "model_in_enums_import":
                # Fix imports of models from enums
                current_line = lines[i]
                
                # Extract the imported items
                if "import " in current_line:
                    imported_items = current_line.split("import ")[1].strip()
                    # Remove commas and split by whitespace
                    imported_items = re.sub(r'[,\n]', ' ', imported_items).split()
                    
                    # Separate enums and models
                    enum_imports = [item for item in imported_items if item in ENUM_NAMES]
                    model_imports = [item for item in imported_items if item in MODEL_NAMES]
                    
                    # Update the line
                    if enum_imports and model_imports:
                        # Need to split into two import statements
                        from_part = current_line.split("import")[0]
                        lines[i] = f"{from_part}import {', '.join(enum_imports)}\n"
                        # Add a new line for model imports
                        lines.insert(i+1, f"from ai_trading_agent.trading_engine.models import {', '.join(model_imports)}\n")
                    elif model_imports:
                        # Replace with import from models
                        lines[i] = f"from ai_trading_agent.trading_engine.models import {', '.join(model_imports)}\n"
            
            elif pattern == "calculate_performance_metrics":
                # Fix references to calculate_performance_metrics
                current_line = lines[i]
                
                # Replace calculate_performance_metrics with calculate_metrics
                lines[i] = current_line.replace("calculate_performance_metrics", "calculate_metrics")
        
        # Write the fixed content back to the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return True
    
    except Exception as e:
        logger.error(f"Error fixing imports in {filepath}: {e}")
        return False

def fix_performance_metrics_module(filepath: str) -> bool:
    """
    Fix any issues in the performance_metrics.py file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed misplaced return statement in {filepath}")
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error fixing performance_metrics module in {filepath}: {e}")
        return False

def fix_backtesting_init(filepath: str) -> bool:
    """
    Fix the __init__.py file in the backtesting package.
    """
    try:
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
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"Updated {filepath} with improved import handling")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing backtesting init in {filepath}: {e}")
        return False

def clean_pycache(directory: str) -> int:
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
                logger.info(f"Removed {pycache_dir}")
            except Exception as e:
                logger.error(f"Error removing {pycache_dir}: {e}")
        
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                try:
                    os.remove(pyc_file)
                    count += 1
                    logger.info(f"Removed {pyc_file}")
                except Exception as e:
                    logger.error(f"Error removing {pyc_file}: {e}")
    
    return count

def main():
    """
    Main function to fix all import issues in the codebase.
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Project root: {project_root}")
    
    # Define key directories
    ai_trading_agent_dir = os.path.join(project_root, 'ai_trading_agent')
    scripts_dir = os.path.join(project_root, 'scripts')
    
    # Step 1: Fix performance_metrics.py
    logger.info("\nStep 1: Fixing performance_metrics.py...")
    performance_metrics_path = os.path.join(ai_trading_agent_dir, 'backtesting', 'performance_metrics.py')
    if os.path.exists(performance_metrics_path):
        fix_performance_metrics_module(performance_metrics_path)
    else:
        logger.error(f"Error: {performance_metrics_path} not found")
    
    # Step 2: Fix backtesting/__init__.py
    logger.info("\nStep 2: Fixing backtesting/__init__.py...")
    backtesting_init_path = os.path.join(ai_trading_agent_dir, 'backtesting', '__init__.py')
    if os.path.exists(backtesting_init_path):
        fix_backtesting_init(backtesting_init_path)
    else:
        logger.error(f"Error: {backtesting_init_path} not found")
    
    # Step 3: Find and fix incorrect imports
    logger.info("\nStep 3: Finding and fixing incorrect imports...")
    python_files = find_python_files(ai_trading_agent_dir) + find_python_files(scripts_dir)
    logger.info(f"Found {len(python_files)} Python files")
    
    fixed_files = []
    for filepath in python_files:
        has_incorrect, incorrect_imports = check_file_for_incorrect_imports(filepath)
        if has_incorrect:
            logger.info(f"Found incorrect imports in {filepath}:")
            for line_num, line_content, pattern in incorrect_imports:
                logger.info(f"  Line {line_num}: {line_content.strip()} (Pattern: {pattern})")
            
            if fix_incorrect_imports(filepath, incorrect_imports):
                fixed_files.append(filepath)
    
    logger.info(f"Fixed incorrect imports in {len(fixed_files)} files")
    
    # Step 4: Clean up __pycache__ directories
    logger.info("\nStep 4: Cleaning up __pycache__ directories...")
    count = clean_pycache(project_root)
    logger.info(f"Cleaned up {count} __pycache__ directories and .pyc files")
    
    logger.info("\nImport fixes completed. Check import_fixes_comprehensive.log for details.")

if __name__ == "__main__":
    main()
