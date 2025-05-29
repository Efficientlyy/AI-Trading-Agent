"""
Script to fix the indentation of the logger line in indicator_engine.py.
"""

def fix_logger_indentation():
    """Fix the indentation of the logger line in indicator_engine.py."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if "INDICATOR_ENGINE __init__: Logger obtained." in line:
            # Fix the indentation for the logger line
            proper_indent = ' ' * 8  # 8 spaces for method indentation
            fixed_line = proper_indent + line.lstrip()
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed logger indentation in {file_path}")

if __name__ == "__main__":
    fix_logger_indentation()
