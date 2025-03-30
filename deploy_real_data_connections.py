"""
Deploy Real Data Connections

This script enables real data connections in the dashboard by setting
REAL_DATA_AVAILABLE = True in the data_service.py file.
"""

import re
import sys
from pathlib import Path

def enable_real_data():
    """Enable real data connections in the dashboard."""
    # Path to data_service.py
    data_service_path = Path("src/dashboard/utils/data_service.py")
    
    # Check if file exists
    if not data_service_path.exists():
        print(f"Error: {data_service_path} not found")
        return False
    
    # Read file content
    content = data_service_path.read_text()
    
    # Check if REAL_DATA_AVAILABLE is already True
    if "REAL_DATA_AVAILABLE = True" in content:
        print("Real data connections are already enabled")
        return True
    
    # Replace REAL_DATA_AVAILABLE = False with REAL_DATA_AVAILABLE = True
    new_content = re.sub(
        r"REAL_DATA_AVAILABLE = False",
        "REAL_DATA_AVAILABLE = True",
        content
    )
    
    # Check if replacement was successful
    if new_content == content:
        print("Error: Could not find REAL_DATA_AVAILABLE = False in the file")
        return False
    
    # Write updated content back to file
    data_service_path.write_text(new_content)
    
    print("Real data connections have been enabled")
    print("REAL_DATA_AVAILABLE = True has been set in data_service.py")
    return True

def disable_real_data():
    """Disable real data connections in the dashboard."""
    # Path to data_service.py
    data_service_path = Path("src/dashboard/utils/data_service.py")
    
    # Check if file exists
    if not data_service_path.exists():
        print(f"Error: {data_service_path} not found")
        return False
    
    # Read file content
    content = data_service_path.read_text()
    
    # Check if REAL_DATA_AVAILABLE is already False
    if "REAL_DATA_AVAILABLE = False" in content:
        print("Real data connections are already disabled")
        return True
    
    # Replace REAL_DATA_AVAILABLE = True with REAL_DATA_AVAILABLE = False
    new_content = re.sub(
        r"REAL_DATA_AVAILABLE = True",
        "REAL_DATA_AVAILABLE = False",
        content
    )
    
    # Check if replacement was successful
    if new_content == content:
        print("Error: Could not find REAL_DATA_AVAILABLE = True in the file")
        return False
    
    # Write updated content back to file
    data_service_path.write_text(new_content)
    
    print("Real data connections have been disabled")
    print("REAL_DATA_AVAILABLE = False has been set in data_service.py")
    return True

def check_real_data_status():
    """Check the current status of real data connections."""
    # Path to data_service.py
    data_service_path = Path("src/dashboard/utils/data_service.py")
    
    # Check if file exists
    if not data_service_path.exists():
        print(f"Error: {data_service_path} not found")
        return None
    
    # Read file content
    content = data_service_path.read_text()
    
    # Check if REAL_DATA_AVAILABLE is True or False
    if "REAL_DATA_AVAILABLE = True" in content:
        print("Real data connections are currently ENABLED")
        return True
    elif "REAL_DATA_AVAILABLE = False" in content:
        print("Real data connections are currently DISABLED")
        return False
    else:
        print("Error: Could not determine the status of real data connections")
        return None

def print_usage():
    """Print usage information."""
    print("Usage: python deploy_real_data_connections.py [enable|disable|status]")
    print("  enable: Enable real data connections")
    print("  disable: Disable real data connections")
    print("  status: Check the current status of real data connections")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    
    # Get command
    command = sys.argv[1].lower()
    
    # Execute command
    if command == "enable":
        success = enable_real_data()
        sys.exit(0 if success else 1)
    elif command == "disable":
        success = disable_real_data()
        sys.exit(0 if success else 1)
    elif command == "status":
        status = check_real_data_status()
        sys.exit(0 if status is not None else 1)
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)
