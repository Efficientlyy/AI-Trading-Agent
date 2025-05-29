"""
Script to fix the import statement in the indicator_engine.py file.
"""

def fix_import():
    """Fix the import statement in the indicator_engine.py file."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the incorrect import with the correct one
    corrected_content = content.replace(
        "from ai_trading_agent.utils.logging_utils import get_logger",
        "from ai_trading_agent.utils.logging import get_logger"
    )
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(corrected_content)
    
    print(f"Fixed import statement in {file_path}")
    return True

if __name__ == "__main__":
    fix_import()
