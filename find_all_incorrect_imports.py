#!/usr/bin/env python
# find_all_incorrect_imports.py

import os
import re

def scan_file_for_incorrect_imports(filepath):
    """Scan a single Python file for incorrect imports of OrderSide from models.py"""
    pattern1 = r'from\s+.*trading_engine\.models\s+import\s+.*OrderSide'
    pattern2 = r'from\s+\.models\s+import\s+.*OrderSide'
    pattern3 = r'from\s+.*models\s+import\s+.*OrderSide'
    
    found = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for the patterns
            for pattern in [pattern1, pattern2, pattern3]:
                match = re.search(pattern, content)
                if match:
                    if not found:
                        print(f"Found potential incorrect import in: {filepath}")
                        found = True
                    
                    # Extract the specific line(s)
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(pattern, line):
                            print(f"  Line {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return found

if __name__ == "__main__":
    # Check all Python files in the project
    project_dir = "ai_trading_agent"
    
    found_any = False
    
    print(f"Scanning {project_dir} recursively...")
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if scan_file_for_incorrect_imports(filepath):
                    found_any = True
    
    if not found_any:
        print("No incorrect imports found in the specified directories.")
