"""
Fix pattern detector issues.

This script makes a backup of the pattern_detector.py file and fixes
the issues with the metadata parameter names.
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(file_path):
    """Create a backup of the file"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def fix_metadata_parameter(file_path):
    """Fix metadata parameter in pattern detector file"""
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace metadata parameter with additional_info
    # Use regex to handle indentation and whitespace variations
    pattern = r'(\s+)metadata\s*=\s*{'
    replacement = r'\1additional_info={'
    
    new_content = re.sub(pattern, replacement, content)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    # Count replacements
    replacement_count = content.count('metadata=') + content.count('metadata =')
    print(f"Fixed {replacement_count} occurrences of metadata parameter")

def main():
    """Main function"""
    print("Fixing pattern detector issues...")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern_detector_path = os.path.join(
        base_dir, 'ai_trading_agent', 'agent', 'pattern_detector.py'
    )
    
    # Backup the file
    backup_file(pattern_detector_path)
    
    # Fix metadata parameter
    fix_metadata_parameter(pattern_detector_path)
    
    print("Pattern detector fixed. You can now run your tests.")

if __name__ == "__main__":
    main()
