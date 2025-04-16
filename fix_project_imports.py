#!/usr/bin/env python
# fix_project_imports.py

"""
Improved script to detect and fix all import issues in the AI-Trading-Agent project.
This script focuses specifically on the ai_trading_agent directory and scripts directory,
ignoring external libraries and handling encoding errors gracefully.
"""

import os
import re
import sys
import traceback
from typing import List, Dict, Tuple, Set

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("import_fixes.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("import_fixer")

# Patterns for detecting incorrect imports
ENUM_NAMES = ['OrderSide', 'OrderType', 'OrderStatus', 'PositionSide']
MODEL_NAMES = ['Order', 'Trade', 'Position', 'Portfolio']

# Patterns for incorrect imports
INCORRECT_PATTERNS = [
    # Importing enums from models
    r'from\s+.*trading_engine\.models\s+import\s+.*(?:' + '|'.join(ENUM_NAMES) + ')',
    r'from\s+\.models\s+import\s+.*(?:' + '|'.join(ENUM_NAMES) + ')',
]

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def read_file_safely(filepath: str) -> Tuple[bool, str]:
    """Read a file safely, handling different encodings."""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return True, f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return False, ""
    
    logger.error(f"Could not read {filepath} with any of the attempted encodings")
    return False, ""

def check_file_for_incorrect_imports(filepath: str) -> Tuple[bool, List[Tuple[int, str, str]]]:
    """
    Check a file for incorrect imports.
    
    Returns:
        Tuple containing:
        - Boolean indicating if incorrect imports were found
        - List of tuples (line_number, line_content, pattern_matched)
    """
    incorrect_imports = []
    
    success, content = read_file_safely(filepath)
    if not success:
        return False, []
    
    lines = content.split('\n')
    
    # Check each pattern
    for pattern in INCORRECT_PATTERNS:
        matches = re.finditer(pattern, content)
        for match in matches:
            # Find the line number
            line_start = content[:match.start()].count('\n')
            line_content = lines[line_start]
            incorrect_imports.append((line_start + 1, line_content, pattern))
    
    return len(incorrect_imports) > 0, incorrect_imports

def fix_incorrect_imports(filepath: str, incorrect_imports: List[Tuple[int, str, str]]) -> bool:
    """
    Fix incorrect imports in a file.
    
    Args:
        filepath: Path to the file to fix
        incorrect_imports: List of tuples (line_number, line_content, pattern_matched)
        
    Returns:
        Boolean indicating if fixes were applied
    """
    success, content = read_file_safely(filepath)
    if not success:
        return False
    
    lines = content.split('\n')
    fixed = False
    
    for line_num, line_content, pattern in incorrect_imports:
        # Adjust for 0-indexed list
        idx = line_num - 1
        
        # Skip if line is already fixed or out of range
        if idx >= len(lines):
            continue
            
        current_line = lines[idx].strip()
        
        # Skip if line doesn't match (might have been modified)
        if current_line != line_content.strip():
            continue
        
        # Fix the line based on the pattern matched
        if any(enum_name in current_line for enum_name in ENUM_NAMES):
            # This is importing enums from models - fix it
            if 'from .models import' in current_line:
                # Local import within trading_engine
                enum_imports = []
                model_imports = []
                
                # Extract all imports
                imports = re.search(r'from\s+\.models\s+import\s+(.*)', current_line).group(1)
                import_items = [item.strip() for item in imports.split(',')]
                
                # Separate enum and model imports
                for item in import_items:
                    if item in ENUM_NAMES:
                        enum_imports.append(item)
                    else:
                        model_imports.append(item)
                
                # Create new import lines
                new_lines = []
                if model_imports:
                    new_lines.append(f"from .models import {', '.join(model_imports)}")
                if enum_imports:
                    new_lines.append(f"from .enums import {', '.join(enum_imports)}")
                
                # Replace the line
                lines[idx] = new_lines[0]
                if len(new_lines) > 1:
                    lines.insert(idx + 1, new_lines[1])
                
                fixed = True
                logger.info(f"Fixed local enum import in {filepath}:{line_num}")
            
            elif 'from ..trading_engine.models import' in current_line or 'from ai_trading_agent.trading_engine.models import' in current_line:
                # Import from another package
                enum_imports = []
                model_imports = []
                
                # Extract the import path and items
                if 'from ..trading_engine.models import' in current_line:
                    import_path = '..trading_engine'
                    imports = re.search(r'from\s+..trading_engine\.models\s+import\s+(.*)', current_line).group(1)
                else:
                    import_path = 'ai_trading_agent.trading_engine'
                    imports = re.search(r'from\s+ai_trading_agent\.trading_engine\.models\s+import\s+(.*)', current_line).group(1)
                
                import_items = [item.strip() for item in imports.split(',')]
                
                # Separate enum and model imports
                for item in import_items:
                    if item in ENUM_NAMES:
                        enum_imports.append(item)
                    else:
                        model_imports.append(item)
                
                # Create new import lines
                new_lines = []
                if model_imports:
                    new_lines.append(f"from {import_path}.models import {', '.join(model_imports)}")
                if enum_imports:
                    new_lines.append(f"from {import_path}.enums import {', '.join(enum_imports)}")
                
                # Replace the line
                lines[idx] = new_lines[0]
                if len(new_lines) > 1:
                    lines.insert(idx + 1, new_lines[1])
                
                fixed = True
                logger.info(f"Fixed external enum import in {filepath}:{line_num}")
    
    # Write the fixed content back to the file
    if fixed:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            logger.info(f"Fixed imports in {filepath}")
        except Exception as e:
            logger.error(f"Error writing to {filepath}: {e}")
            return False
    
    return fixed

def fix_models_py(project_root: str) -> bool:
    """Fix the models.py file to use absolute imports for enums."""
    models_path = os.path.join(project_root, 'ai_trading_agent', 'trading_engine', 'models.py')
    if not os.path.exists(models_path):
        logger.error(f"models.py not found at {models_path}")
        return False
    
    success, content = read_file_safely(models_path)
    if not success:
        return False
    
    # Replace relative import with absolute import
    if 'from .enums import' in content:
        content = content.replace(
            'from .enums import',
            'from ai_trading_agent.trading_engine.enums import'
        )
        
        try:
            with open(models_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Fixed models.py to use absolute imports")
            return True
        except Exception as e:
            logger.error(f"Error writing to {models_path}: {e}")
            return False
    
    return False

def fix_backtesting_init(project_root: str) -> bool:
    """Fix the backtesting/__init__.py file to handle imports carefully."""
    init_path = os.path.join(project_root, 'ai_trading_agent', 'backtesting', '__init__.py')
    if not os.path.exists(init_path):
        logger.error(f"backtesting/__init__.py not found at {init_path}")
        return False
    
    # Create a new version of the file with safer imports
    new_content = '''"""
Backtesting module for AI Trading Agent.

This module provides tools for backtesting trading strategies.
"""

# First import the core components
from .backtester import Backtester
from .performance_metrics import calculate_metrics, PerformanceMetrics

# Define __all__ without RustBacktester initially
__all__ = ["Backtester", "calculate_metrics", "PerformanceMetrics"]

# Import Rust backtester if available - do this after other imports to avoid circular dependencies
try:
    # Ensure trading_engine.enums is imported first to avoid circular dependencies
    from ai_trading_agent.trading_engine.enums import OrderSide, OrderType, OrderStatus
    
    # Now import RustBacktester
    from .rust_backtester import RustBacktester
    RUST_AVAILABLE = True
    __all__.append("RustBacktester")
except ImportError as e:
    RUST_AVAILABLE = False
    # Log the import error for debugging
    import logging
    logging.getLogger(__name__).warning(f"RustBacktester not available: {e}")
'''
    
    try:
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        logger.info(f"Fixed backtesting/__init__.py to handle imports carefully")
        return True
    except Exception as e:
        logger.error(f"Error writing to {init_path}: {e}")
        return False

def clean_pycache(directory: str) -> int:
    """Clean up __pycache__ directories."""
    pycache_dirs = []
    for root, dirs, _ in os.walk(directory):
        for dir in dirs:
            if dir == '__pycache__':
                pycache_dirs.append(os.path.join(root, dir))
    
    cleaned_count = 0
    for pycache_dir in pycache_dirs:
        try:
            for file in os.listdir(pycache_dir):
                file_path = os.path.join(pycache_dir, file)
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up {pycache_dir}: {e}")
    
    return cleaned_count

def main():
    """Main function to check and fix all import issues."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    ai_trading_agent_dir = os.path.join(project_root, 'ai_trading_agent')
    scripts_dir = os.path.join(project_root, 'scripts')
    
    # Step 1: Find and fix incorrect imports
    logger.info("Step 1: Finding and fixing incorrect imports...")
    
    python_files = find_python_files(ai_trading_agent_dir) + find_python_files(scripts_dir)
    fixed_files = []
    
    for filepath in python_files:
        has_incorrect, incorrect_imports = check_file_for_incorrect_imports(filepath)
        if has_incorrect:
            logger.info(f"Found incorrect imports in {filepath}:")
            for line_num, line_content, _ in incorrect_imports:
                logger.info(f"  Line {line_num}: {line_content.strip()}")
            
            if fix_incorrect_imports(filepath, incorrect_imports):
                fixed_files.append(filepath)
    
    logger.info(f"Fixed incorrect imports in {len(fixed_files)} files")
    
    # Step 2: Fix models.py to use absolute imports
    logger.info("\nStep 2: Fixing models.py to use absolute imports...")
    if fix_models_py(project_root):
        logger.info("Successfully fixed models.py")
    else:
        logger.info("No changes needed for models.py or fix failed")
    
    # Step 3: Fix backtesting/__init__.py to handle imports carefully
    logger.info("\nStep 3: Fixing backtesting/__init__.py to handle imports carefully...")
    if fix_backtesting_init(project_root):
        logger.info("Successfully fixed backtesting/__init__.py")
    else:
        logger.info("Failed to fix backtesting/__init__.py")
    
    # Step 4: Clean up __pycache__ directories
    logger.info("\nStep 4: Cleaning up __pycache__ directories...")
    cleaned_count = clean_pycache(project_root)
    logger.info(f"Cleaned up {cleaned_count} cached Python files")
    
    logger.info("\nImport fixes completed. Check import_fixes.log for details.")
    logger.info("Now try running 'python -m scripts.run_backtest' to see if the issues are resolved.")

if __name__ == "__main__":
    main()
