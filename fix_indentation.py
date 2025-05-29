"""
Script to fix the indentation in the indicator_engine.py file.
"""

def fix_indentation():
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_calculate_all_indicators = False
    
    for line in lines:
        if 'def calculate_all_indicators' in line:
            in_calculate_all_indicators = True
            fixed_lines.append(line)
        elif in_calculate_all_indicators and line.strip().startswith('"""'):
            # Fix the indentation of the docstring
            fixed_lines.append('        ' + line.lstrip())
        elif in_calculate_all_indicators and '"""' in line and not line.strip().startswith('"""'):
            # End of docstring
            fixed_lines.append('        ' + line.lstrip())
            in_calculate_all_indicators = False
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation()
