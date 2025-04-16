#!/usr/bin/env python
# find_all_enum_imports.py

import os
import re

def scan_file_for_enum_imports(filepath):
    """Scan a single Python file for all imports related to OrderSide, OrderType, etc."""
    # Patterns for incorrect imports from models
    incorrect_patterns = [
        r'from\s+.*trading_engine\.models\s+import\s+.*OrderSide',
        r'from\s+.*trading_engine\.models\s+import\s+.*OrderType',
        r'from\s+.*trading_engine\.models\s+import\s+.*OrderStatus',
        r'from\s+.*trading_engine\.models\s+import\s+.*PositionSide',
        r'from\s+\.models\s+import\s+.*OrderSide',
        r'from\s+\.models\s+import\s+.*OrderType',
        r'from\s+\.models\s+import\s+.*OrderStatus',
        r'from\s+\.models\s+import\s+.*PositionSide',
    ]
    
    # Patterns for correct imports from enums
    correct_patterns = [
        r'from\s+.*trading_engine\.enums\s+import\s+.*OrderSide',
        r'from\s+.*trading_engine\.enums\s+import\s+.*OrderType',
        r'from\s+.*trading_engine\.enums\s+import\s+.*OrderStatus',
        r'from\s+.*trading_engine\.enums\s+import\s+.*PositionSide',
        r'from\s+\.enums\s+import\s+.*OrderSide',
        r'from\s+\.enums\s+import\s+.*OrderType',
        r'from\s+\.enums\s+import\s+.*OrderStatus',
        r'from\s+\.enums\s+import\s+.*PositionSide',
    ]
    
    found_incorrect = False
    found_correct = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for incorrect patterns
            for pattern in incorrect_patterns:
                match = re.search(pattern, content)
                if match:
                    if not found_incorrect:
                        print(f"\n[INCORRECT] Found incorrect enum import in: {filepath}")
                        found_incorrect = True
                    
                    # Extract the specific line(s)
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(pattern, line):
                            print(f"  Line {i+1}: {line.strip()}")
            
            # Check for correct patterns
            for pattern in correct_patterns:
                match = re.search(pattern, content)
                if match:
                    found_correct = True
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return found_incorrect, found_correct

def scan_directory(directory):
    """Scan all Python files in a directory for enum imports"""
    incorrect_files = []
    correct_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                incorrect, correct = scan_file_for_enum_imports(filepath)
                if incorrect:
                    incorrect_files.append(filepath)
                if correct:
                    correct_files.append(filepath)
    
    return incorrect_files, correct_files

if __name__ == "__main__":
    # Check all Python files in the project
    project_dir = "ai_trading_agent"
    scripts_dir = "scripts"
    
    print(f"Scanning {project_dir} and {scripts_dir} for enum imports...")
    
    incorrect_files1, correct_files1 = scan_directory(project_dir)
    incorrect_files2, correct_files2 = scan_directory(scripts_dir)
    
    incorrect_files = incorrect_files1 + incorrect_files2
    correct_files = correct_files1 + correct_files2
    
    print("\n--- SUMMARY ---")
    if not incorrect_files:
        print("✓ No incorrect enum imports found!")
    else:
        print(f"✗ Found {len(incorrect_files)} file(s) with incorrect enum imports.")
        print("  These files are importing OrderSide/OrderType/etc. from models.py instead of enums.py.")
    
    print(f"ℹ Found {len(correct_files)} file(s) with correct enum imports from enums.py.")
