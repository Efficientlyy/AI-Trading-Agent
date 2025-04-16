#!/usr/bin/env python
# find_incorrect_imports_focused.py

import os
import re

def scan_file_for_incorrect_imports(filepath):
    """Scan a single Python file for incorrect imports of OrderSide from models.py"""
    pattern1 = r'from\s+.*trading_engine\.models\s+import\s+.*OrderSide'
    pattern2 = r'from\s+\.models\s+import\s+.*OrderSide'
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for the patterns
            if re.search(pattern1, content) or re.search(pattern2, content):
                print(f"Found potential incorrect import in: {filepath}")
                
                # Extract the specific line(s)
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if re.search(pattern1, line) or re.search(pattern2, line):
                        print(f"  Line {i+1}: {line.strip()}")
                return True
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return False

if __name__ == "__main__":
    # Focus on specific directories where the issue is most likely to be
    directories = [
        "ai_trading_agent/backtesting",
        "ai_trading_agent/agent",
        "ai_trading_agent/trading_engine",
        "ai_trading_agent/strategies",
        "scripts"
    ]
    
    found_any = False
    
    for directory in directories:
        print(f"Scanning {directory}...")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    if scan_file_for_incorrect_imports(filepath):
                        found_any = True
    
    if not found_any:
        print("No incorrect imports found in the specified directories.")
