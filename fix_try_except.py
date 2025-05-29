"""
Script to fix the indentation in the try-except block in the indicator_engine.py file.
"""

def fix_try_except_indentation():
    """Fix the indentation in the try-except block in the indicator_engine.py file."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_try_except = False
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        if stripped_line.startswith('try:'):
            in_try_except = True
            fixed_lines.append(line)
            continue
        
        if in_try_except and (stripped_line.startswith('self.logger.critical') or 
                              stripped_line.startswith('except') or 
                              stripped_line.startswith('print')):
            # Fix the indentation for lines in the try-except block
            # Add 4 spaces after the existing indentation for the method
            proper_indent = ' ' * 12  # 8 for method + 4 for try block
            fixed_line = proper_indent + stripped_line + '\n'
            fixed_lines.append(fixed_line)
            
            if stripped_line.startswith('print'):
                in_try_except = False
            
            continue
        
        # Default case
        fixed_lines.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed try-except indentation in {file_path}")

if __name__ == "__main__":
    fix_try_except_indentation()
