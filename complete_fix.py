"""
Script to comprehensively fix the indentation in the indicator_engine.py file.
This will ensure all method definitions and docstrings have proper indentation.
"""

def fix_indentation_comprehensive():
    """Fix all indentation issues in the indicator_engine.py file."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Process lines
    fixed_lines = []
    in_class = False
    in_method_def = False
    in_docstring = False
    method_indent_level = 0
    
    for i, line in enumerate(lines):
        # Detect class definition
        if line.strip().startswith('class '):
            in_class = True
            fixed_lines.append(line)
            continue
        
        # Detect method definition within class
        if in_class and line.strip().startswith('def '):
            in_method_def = True
            method_indent_level = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
        
        # Handle docstring start
        if in_method_def and '"""' in line and not in_docstring:
            in_docstring = True
            # Ensure proper indentation for docstring start
            proper_indent = ' ' * (method_indent_level + 4)
            fixed_line = proper_indent + line.lstrip()
            fixed_lines.append(fixed_line)
            continue
        
        # Handle docstring end
        elif in_method_def and in_docstring and '"""' in line and not line.strip().startswith('"""'):
            in_docstring = False
            # Ensure proper indentation for docstring end
            proper_indent = ' ' * (method_indent_level + 4)
            fixed_line = proper_indent + line.lstrip()
            fixed_lines.append(fixed_line)
            continue
        
        # Handle lines within docstring
        elif in_method_def and in_docstring:
            # Ensure proper indentation for docstring content
            proper_indent = ' ' * (method_indent_level + 4)
            fixed_line = proper_indent + line.lstrip()
            fixed_lines.append(fixed_line)
            continue
        
        # Method body lines
        elif in_method_def and not in_docstring:
            # Just add the line as is
            fixed_lines.append(line)
            continue
        
        # Default case
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"Comprehensively fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation_comprehensive()
