#!/usr/bin/env python
# fix_all_imports.py

"""
Comprehensive script to detect and fix all import issues in the codebase.
This script will:
1. Find all incorrect imports of enum types from models.py
2. Fix circular dependencies by restructuring imports
3. Ensure consistent import patterns throughout the codebase
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
    # Other patterns can be added here
]

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in a directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_file_for_incorrect_imports(filepath: str) -> Tuple[bool, List[Tuple[int, str, str]]]:
    """
    Check a file for incorrect imports.
    
    Returns:
        Tuple containing:
        - Boolean indicating if incorrect imports were found
        - List of tuples (line_number, line_content, pattern_matched)
    """
    incorrect_imports = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Check each pattern
            for pattern in INCORRECT_PATTERNS:
                matches = re.finditer(pattern, content)
                for match in matches:
                    # Find the line number
                    line_start = content[:match.start()].count('\n')
                    line_content = lines[line_start]
                    incorrect_imports.append((line_start + 1, line_content, pattern))
    except Exception as e:
        logger.error(f"Error checking {filepath}: {e}")
        traceback.print_exc()
    
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
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
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
                    lines[idx] = new_lines[0] + '\n'
                    if len(new_lines) > 1:
                        lines.insert(idx + 1, new_lines[1] + '\n')
                    
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
                    lines[idx] = new_lines[0] + '\n'
                    if len(new_lines) > 1:
                        lines.insert(idx + 1, new_lines[1] + '\n')
                    
                    fixed = True
                    logger.info(f"Fixed external enum import in {filepath}:{line_num}")
        
        # Write the fixed content back to the file
        if fixed:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            logger.info(f"Fixed imports in {filepath}")
        
        return fixed
    
    except Exception as e:
        logger.error(f"Error fixing {filepath}: {e}")
        traceback.print_exc()
        return False

def check_for_circular_imports(directory: str) -> Dict[str, Set[str]]:
    """
    Check for potential circular imports in the codebase.
    
    Returns:
        Dictionary mapping module paths to sets of imported modules
    """
    import_graph = {}
    
    python_files = find_python_files(directory)
    for filepath in python_files:
        # Get the module path
        rel_path = os.path.relpath(filepath, directory)
        module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
        
        # Skip __init__.py for now
        if module_path.endswith('__init__'):
            continue
        
        import_graph[module_path] = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Find all imports
                import_patterns = [
                    r'from\s+([\w\.]+)\s+import',  # from X import Y
                    r'import\s+([\w\.]+)'          # import X
                ]
                
                for pattern in import_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        imported_module = match.group(1)
                        # Skip standard library imports
                        if not imported_module.startswith(('ai_trading_agent', '..')):
                            continue
                        
                        # Normalize relative imports
                        if imported_module.startswith('..'):
                            # This is a relative import, normalize it
                            parts = module_path.split('.')
                            parent_module = '.'.join(parts[:-1])
                            rel_depth = imported_module.count('..')
                            parent_parts = parent_module.split('.')
                            if rel_depth <= len(parent_parts):
                                base_module = '.'.join(parent_parts[:-rel_depth])
                                rel_module = imported_module.replace('..', '', 1)
                                if rel_module.startswith('.'):
                                    rel_module = rel_module[1:]
                                if base_module and rel_module:
                                    normalized_module = f"{base_module}.{rel_module}"
                                elif base_module:
                                    normalized_module = base_module
                                else:
                                    normalized_module = rel_module
                                import_graph[module_path].add(normalized_module)
                        else:
                            import_graph[module_path].add(imported_module)
        
        except Exception as e:
            logger.error(f"Error analyzing imports in {filepath}: {e}")
    
    return import_graph

def find_circular_dependencies(import_graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Find circular dependencies in the import graph.
    
    Returns:
        List of lists, where each inner list represents a circular dependency chain
    """
    circular_deps = []
    
    def dfs(node, visited, path):
        if node in path:
            # Found a cycle
            cycle_start = path.index(node)
            circular_deps.append(path[cycle_start:] + [node])
            return
        
        if node in visited:
            return
        
        visited.add(node)
        path.append(node)
        
        if node in import_graph:
            for neighbor in import_graph[node]:
                dfs(neighbor, visited, path.copy())
    
    for node in import_graph:
        dfs(node, set(), [])
    
    return circular_deps

def fix_circular_dependencies(directory: str, circular_deps: List[List[str]]) -> None:
    """
    Fix circular dependencies by restructuring imports.
    
    This is a more complex operation that may require manual intervention,
    but we'll attempt to fix common patterns automatically.
    """
    # Focus on fixing the most common circular dependency patterns
    for cycle in circular_deps:
        logger.info(f"Attempting to fix circular dependency: {' -> '.join(cycle)}")
        
        # Check if the cycle involves trading_engine.models and trading_engine.enums
        if any('trading_engine.models' in module for module in cycle) and any('trading_engine.enums' in module for module in cycle):
            # Fix models.py to use absolute imports for enums
            models_path = os.path.join(directory, 'ai_trading_agent', 'trading_engine', 'models.py')
            if os.path.exists(models_path):
                try:
                    with open(models_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Replace relative import with absolute import
                    if 'from .enums import' in content:
                        content = content.replace(
                            'from .enums import',
                            'from ai_trading_agent.trading_engine.enums import'
                        )
                        
                        with open(models_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        logger.info(f"Fixed circular dependency in {models_path} by using absolute imports")
                except Exception as e:
                    logger.error(f"Error fixing circular dependency in {models_path}: {e}")
        
        # Check if the cycle involves backtesting and trading_engine
        if any('backtesting' in module for module in cycle) and any('trading_engine' in module for module in cycle):
            # Fix backtesting/__init__.py to import components conditionally
            init_path = os.path.join(directory, 'ai_trading_agent', 'backtesting', '__init__.py')
            if os.path.exists(init_path):
                try:
                    with open(init_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Implement a more careful import strategy
                    if 'from .rust_backtester import RustBacktester' in content:
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
                        
                        with open(init_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        logger.info(f"Fixed circular dependency in {init_path} by restructuring imports")
                except Exception as e:
                    logger.error(f"Error fixing circular dependency in {init_path}: {e}")

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
    
    # Step 2: Check for circular dependencies
    logger.info("\nStep 2: Checking for circular dependencies...")
    
    import_graph = check_for_circular_imports(project_root)
    circular_deps = find_circular_dependencies(import_graph)
    
    if circular_deps:
        logger.info(f"Found {len(circular_deps)} circular dependencies:")
        for i, cycle in enumerate(circular_deps):
            logger.info(f"  Cycle {i+1}: {' -> '.join(cycle)}")
        
        # Step 3: Fix circular dependencies
        logger.info("\nStep 3: Fixing circular dependencies...")
        fix_circular_dependencies(project_root, circular_deps)
    else:
        logger.info("No circular dependencies found")
    
    # Step 4: Clean up __pycache__ directories
    logger.info("\nStep 4: Cleaning up __pycache__ directories...")
    
    pycache_dirs = []
    for root, dirs, _ in os.walk(project_root):
        for dir in dirs:
            if dir == '__pycache__':
                pycache_dirs.append(os.path.join(root, dir))
    
    for pycache_dir in pycache_dirs:
        try:
            for file in os.listdir(pycache_dir):
                file_path = os.path.join(pycache_dir, file)
                try:
                    os.remove(file_path)
                    logger.info(f"Removed {file_path}")
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up {pycache_dir}: {e}")
    
    logger.info(f"Cleaned up {len(pycache_dirs)} __pycache__ directories")
    
    logger.info("\nImport fixes completed. Check import_fixes.log for details.")

if __name__ == "__main__":
    main()
