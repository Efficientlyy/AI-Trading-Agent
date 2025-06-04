#!/usr/bin/env python
"""
Class Name Consistency Fix Script

This script updates all references to OptimizedMEXCClient to OptimizedMEXCClient
across the codebase to ensure consistent naming and fix import errors.
"""

import os
import re
import sys
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("class_name_fix.log")
    ]
)

logger = logging.getLogger("class_name_fix")

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and subdirectories
    
    Args:
        directory: Directory to search
        
    Returns:
        list: List of Python file paths
    """
    python_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def fix_class_name_in_file(file_path: str, old_name: str, new_name: str) -> bool:
    """Fix class name references in a file
    
    Args:
        file_path: Path to the file
        old_name: Old class name
        new_name: New class name
        
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if old name exists in content
        if old_name not in content:
            return False
        
        # Replace old name with new name
        new_content = content.replace(old_name, new_name)
        
        # Write updated content back to file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Updated class name in {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing class name in {file_path}: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("Starting class name consistency fix")
    
    # Directory to search
    directory = "."
    
    # Class names to fix
    old_name = "OptimizedMEXCClient"
    new_name = "OptimizedMEXCClient"
    
    # Find all Python files
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    
    # Fix class name in each file
    modified_files = 0
    for file_path in python_files:
        if fix_class_name_in_file(file_path, old_name, new_name):
            modified_files += 1
    
    logger.info(f"Updated class name in {modified_files} files")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
