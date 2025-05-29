"""
Script to fix the indentation of the calculate_all_indicators method in indicator_engine.py.
"""

import re

def fix_calculate_all_indicators_indentation():
    """Fix the indentation of the calculate_all_indicators method."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.readlines()
    
    # Find the calculate_all_indicators method
    method_line = -1
    for i, line in enumerate(content):
        if 'def calculate_all_indicators' in line:
            method_line = i
            break
    
    if method_line >= 0:
        # Fix the indentation of the method and its docstring
        fixed_content = []
        
        for i, line in enumerate(content):
            if i == method_line:
                # Ensure the method definition has proper indentation
                if not line.startswith('    '):
                    line = '    ' + line
                fixed_content.append(line)
            elif i > method_line and '"""' in line and not line.strip().endswith('"""'):
                # Beginning of docstring - ensure it has proper indentation
                if not line.startswith('        '):
                    line = '        ' + line.lstrip()
                fixed_content.append(line)
            elif i > method_line and '"""' in line and not line.strip().startswith('"""'):
                # End of docstring - ensure it has proper indentation
                if not line.startswith('        '):
                    line = '        ' + line.lstrip()
                fixed_content.append(line)
            elif i > method_line and (i <= method_line + 10 or '"""' in content[method_line + 10]):
                # Lines inside the docstring - ensure they have proper indentation
                if not line.startswith('        '):
                    line = '        ' + line.lstrip()
                fixed_content.append(line)
            else:
                fixed_content.append(line)
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as f:
            f.writelines(fixed_content)
        
        print(f"Fixed indentation of calculate_all_indicators method in {file_path}")
        return True
    else:
        print("Could not find calculate_all_indicators method in the file")
        return False

if __name__ == "__main__":
    fix_calculate_all_indicators_indentation()
