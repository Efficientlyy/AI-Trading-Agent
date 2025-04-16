#!/usr/bin/env python
# find_incorrect_imports.py

import os
import re

def scan_for_incorrect_imports(directory):
    """Scan all Python files for incorrect imports of OrderSide from models.py"""
    pattern1 = r'from\s+.*trading_engine\.models\s+import\s+.*OrderSide'
    pattern2 = r'from\s+\.models\s+import\s+.*OrderSide'
    
    matches = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for the patterns
                        if re.search(pattern1, content) or re.search(pattern2, content):
                            matches.append(filepath)
                            print(f"Found potential incorrect import in: {filepath}")
                            
                            # Extract the specific line(s)
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if re.search(pattern1, line) or re.search(pattern2, line):
                                    print(f"  Line {i+1}: {line.strip()}")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return matches

if __name__ == "__main__":
    project_dir = "."
    print(f"Scanning {project_dir} for incorrect imports...")
    matches = scan_for_incorrect_imports(project_dir)
    
    if not matches:
        print("No incorrect imports found.")
    else:
        print(f"\nFound {len(matches)} file(s) with potentially incorrect imports.")
        print("These files are trying to import OrderSide from models.py instead of enums.py.")
