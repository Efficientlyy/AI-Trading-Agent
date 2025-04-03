"""
Enable Real Data Connections

This script provides a command-line interface to enable or disable real data
connections in the dashboard. It modifies the configuration in the 
real_data_config.json file.

Usage:
    python enable_real_data.py [enable|disable|status]
"""

import sys
import os
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to real_data_config.json
CONFIG_PATH = Path("config/real_data_config.json")

def check_config_file():
    """Check if the config file exists and is readable/writable."""
    # Create config directory if it doesn't exist
    if not CONFIG_PATH.parent.exists():
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {CONFIG_PATH.parent}")
        except Exception as e:
            logger.error(f"Error creating directory {CONFIG_PATH.parent}: {e}")
            return False
    
    # Create config file with default values if it doesn't exist
    if not CONFIG_PATH.exists():
        try:
            default_config = {
                "enabled": False,
                "connections": {
                    "exchange_api": {
                        "enabled": True,
                        "retry_attempts": 3,
                        "timeout_seconds": 10,
                        "cache_duration_seconds": 60
                    },
                    "market_data": {
                        "enabled": True,
                        "retry_attempts": 3,
                        "timeout_seconds": 15,
                        "cache_duration_seconds": 30
                    },
                    "sentiment_api": {
                        "enabled": True,
                        "retry_attempts": 2,
                        "timeout_seconds": 20,
                        "cache_duration_seconds": 300
                    },
                    "news_api": {
                        "enabled": True,
                        "retry_attempts": 2,
                        "timeout_seconds": 20,
                        "cache_duration_seconds": 600
                    }
                },
                "fallback_strategy": {
                    "use_cached_data": True,
                    "cache_expiry_seconds": 3600,
                    "use_mock_data_on_failure": True
                },
                "error_tracking": {
                    "log_errors": True,
                    "max_consecutive_errors": 5,
                    "error_cooldown_seconds": 300
                },
                "data_validation": {
                    "validate_schema": True,
                    "validate_types": True,
                    "validate_required_fields": True
                }
            }
            
            with open(CONFIG_PATH, 'w') as f:
                json.dump(default_config, f, indent=4)
                
            logger.info(f"Created default config file: {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Error creating config file {CONFIG_PATH}: {e}")
            return False
    
    # Check if the file is readable
    if not os.access(CONFIG_PATH, os.R_OK):
        logger.error(f"Error: {CONFIG_PATH} is not readable")
        return False
    
    # Check if the file is writable
    if not os.access(CONFIG_PATH, os.W_OK):
        logger.error(f"Error: {CONFIG_PATH} is not writable")
        return False
    
    return True

def get_current_status():
    """Get the current status of real data connections."""
    if not check_config_file():
        return None
    
    try:
        # Read config file
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Return the enabled status
        return config.get('enabled', False)
    except Exception as e:
        logger.error(f"Error reading {CONFIG_PATH}: {e}")
        return None

def enable_real_data():
    """Enable real data connections."""
    if not check_config_file():
        return False
    
    try:
        # Read config file
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Check if real data is already enabled
        if config.get('enabled', False):
            logger.info("Real data connections are already enabled")
            return True
        
        # Enable real data
        config['enabled'] = True
        
        # Write updated config back to file
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info("Real data connections have been enabled")
        return True
    except Exception as e:
        logger.error(f"Error enabling real data connections: {e}")
        return False

def disable_real_data():
    """Disable real data connections."""
    if not check_config_file():
        return False
    
    try:
        # Read config file
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Check if real data is already disabled
        if not config.get('enabled', False):
            logger.info("Real data connections are already disabled")
            return True
        
        # Disable real data
        config['enabled'] = False
        
        # Write updated config back to file
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info("Real data connections have been disabled")
        return True
    except Exception as e:
        logger.error(f"Error disabling real data connections: {e}")
        return False

def print_status():
    """Print the current status of real data connections."""
    status = get_current_status()
    
    if status is True:
        print("Real data connections are currently ENABLED")
    elif status is False:
        print("Real data connections are currently DISABLED")
    else:
        print("Could not determine the status of real data connections")

def print_usage():
    """Print usage information."""
    print("Usage: python enable_real_data.py [enable|disable|status]")
    print("  enable: Enable real data connections")
    print("  disable: Disable real data connections")
    print("  status: Check the current status of real data connections")

def main():
    """Main function."""
    # Check command line arguments
    if len(sys.argv) != 2:
        print_usage()
        return 1
    
    # Get command
    command = sys.argv[1].lower()
    
    # Execute command
    if command == "enable":
        success = enable_real_data()
        return 0 if success else 1
    elif command == "disable":
        success = disable_real_data()
        return 0 if success else 1
    elif command == "status":
        print_status()
        return 0
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        return 1

if __name__ == "__main__":
    sys.exit(main())