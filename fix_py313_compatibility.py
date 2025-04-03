"""
Python 3.13 Compatibility Fixer for Test Suite

This script identifies and fixes Python 3.13 compatibility issues in the test suite.
"""

import os
import re
import sys
from pathlib import Path

# Known Python 3.13 compatibility issues and their fixes
FIXES = [
    # Fix 1: Dictionary assignment syntax changes in 3.13
    (
        r"([\w\d_]+)(?:\[['\"]\w+['\"])?\s*\[['\"]\w+['\"\]]\s*=",
        lambda match: f"{match.group(0).split('[')[0]}['{match.group(0).split('[')[1].strip('[\'\"').strip('\'\"]')}'] ="
    ),
    
    # Fix 2: Update imports for removed/relocated modules
    (
        r"from\s+unittest\.mock\s+import\s+MagicMock",
        "from unittest.mock import MagicMock"  # Keep this one, it's fine
    ),
    
    # Fix 3: Fix common attribute errors in mock objects
    (
        r"([\w\d_]+)\.return_value\s*\[([^\]]+)\]\s*=",
        lambda match: f"{match.group(1)}.return_value.__getitem__.return_value = "
    ),
    
    # Fix 4: Fix for changes in collections ABCs
    (
        r"from\s+collections\s+import\s+(\w+)",
        lambda match: f"from collections.abc import {match.group(1)}" if match.group(1) in ["Mapping", "Sequence", "Iterable"] else f"from collections import {match.group(1)}"
    ),
]

def fix_file(file_path):
    """Apply Python 3.13 compatibility fixes to a file."""
    print(f"Checking {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for pattern, replacement in FIXES:
        if callable(replacement):
            # Use a function-based replacement
            content = re.sub(pattern, replacement, content)
        else:
            # Use a string-based replacement
            content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        print(f"  Applying compatibility fixes to {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def fix_directory(directory):
    """Apply Python 3.13 compatibility fixes to all Python files in a directory."""
    fixed_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_file(file_path):
                    fixed_count += 1
    
    return fixed_count

def main():
    """Run the Python 3.13 compatibility fixer."""
    print("Python 3.13 Compatibility Fixer")
    print("=" * 40)
    
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Directories to fix
    directories = [
        base_dir / "tests",
        base_dir / "src" / "analysis_agents" / "sentiment"
    ]
    
    total_fixed = 0
    for directory in directories:
        print(f"\nProcessing {directory}...")
        fixed = fix_directory(directory)
        total_fixed += fixed
        print(f"Fixed {fixed} files in {directory}")
    
    print(f"\nTotal files fixed: {total_fixed}")
    
    # Add a special fix for conftest.py which often has specific issues
    conftest_path = base_dir / "tests" / "conftest.py"
    if conftest_path.exists():
        print("\nApplying special fixes to conftest.py...")
        fixed = fix_file(conftest_path)
        if fixed:
            total_fixed += 1
    
    print("\nFix complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
