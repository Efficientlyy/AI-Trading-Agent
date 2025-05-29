"""
Fix Data Feed Connection Utility

This script checks and fixes the data feed connection for the AI Trading Agent system.
It verifies that the MarketDataProvider is properly configured and connected.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data-feed-fixer")

# Add the project directory to the path
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

try:
    # Import the data feed manager
    from ai_trading_agent.api.data_feed_manager import data_feed_manager
except ImportError:
    logger.error("Failed to import data_feed_manager. Make sure the module exists.")
    sys.exit(1)

def check_connection():
    """Check the current status of the data feed connection."""
    logger.info("Checking data feed connection status...")
    
    try:
        status = data_feed_manager.get_status()
        logger.info(f"Current status: {status.get('status', 'Unknown')}")
        
        if status.get('status') == 'connected':
            logger.info("✅ Data feed is connected!")
            return True
        else:
            logger.warning("❌ Data feed is not connected.")
            return False
    except Exception as e:
        logger.error(f"Error checking data feed status: {e}")
        return False

def start_data_feed():
    """Start the data feed and verify connection."""
    logger.info("Starting data feed manager...")
    
    try:
        data_feed_manager.start()
        logger.info("Data feed manager started. Waiting for connection...")
        
        # Wait for the connection to establish
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Connection attempt {attempt}/{max_attempts}...")
            time.sleep(2)  # Give it time to connect
            
            status = data_feed_manager.get_status()
            if status.get('status') == 'connected':
                logger.info(f"✅ Data feed connected after {attempt} attempts!")
                return True
            
            # If we're still connecting, give it more time
            if status.get('status') == 'connecting':
                logger.info("Data feed is still connecting...")
            else:
                logger.warning(f"Data feed status: {status.get('status')}")
        
        logger.error("Failed to connect data feed after maximum attempts.")
        return False
    except Exception as e:
        logger.error(f"Error starting data feed: {e}")
        return False

def fix_data_feed():
    """Fix the data feed connection by checking and restarting if necessary."""
    logger.info("Starting data feed connection fix utility...")
    
    # 1. Check current connection
    if check_connection():
        logger.info("Data feed is already connected. No fix needed.")
        return True
    
    # 2. Try to start/restart the data feed
    logger.info("Attempting to fix data feed connection...")
    if start_data_feed():
        logger.info("Data feed connection fixed successfully!")
        
        # 3. Test with a sample price request
        try:
            logger.info("Testing connection with a sample price request...")
            price = data_feed_manager.get_current_price("BTC/USD")
            if price is not None:
                logger.info(f"✅ Success! Current BTC/USD price: {price}")
            else:
                logger.warning("Price returned None, but connection appears established.")
        except Exception as e:
            logger.error(f"Error testing price: {e}")
        
        return True
    else:
        logger.error("Failed to fix data feed connection.")
        return False

def update_system_control_api():
    """Update the system_control.py file to properly detect data feed status."""
    system_control_path = os.path.join(project_dir, "ai_trading_agent", "api", "system_control.py")
    
    if not os.path.exists(system_control_path):
        logger.error(f"System control file not found at {system_control_path}")
        return False
    
    logger.info("Updating system control API to detect data feed status...")
    
    try:
        # Read the file content
        with open(system_control_path, 'r') as file:
            content = file.read()
        
        # Check if we already have the USE_MOCK_DATA variable defined
        if "USE_MOCK_DATA = False" not in content and "USE_MOCK_DATA = True" not in content:
            # Add the USE_MOCK_DATA variable
            mock_data_line = "\n# Flag to use mock data (for development/testing)\nUSE_MOCK_DATA = False  # Set to False to use real data\n"
            import_section_end = "from pathlib import Path"
            
            if import_section_end in content:
                content = content.replace(import_section_end, import_section_end + mock_data_line)
                logger.info("Added USE_MOCK_DATA flag to system_control.py")
            else:
                logger.warning("Could not find import section in system_control.py")
        
        # Save the updated content
        with open(system_control_path, 'w') as file:
            file.write(content)
        
        logger.info("System control API updated successfully!")
        return True
    except Exception as e:
        logger.error(f"Error updating system control file: {e}")
        return False

def main():
    """Main entry point for the data feed fix utility."""
    print("\n" + "="*50)
    print(" DATA FEED CONNECTION FIX UTILITY ".center(50, "="))
    print("="*50 + "\n")
    
    # 1. Update the system control API
    update_system_control_api()
    
    # 2. Fix the data feed connection
    success = fix_data_feed()
    
    print("\n" + "="*50)
    if success:
        print(" ✅ DATA FEED CONNECTION FIXED SUCCESSFULLY ".center(50, "="))
        print("\nYou can now start the system using the start_servers.py script.")
        print("The data feed should now show as connected in the dashboard.")
    else:
        print(" ❌ DATA FEED CONNECTION FIX FAILED ".center(50, "="))
        print("\nPlease check the logs above for errors and try again.")
    print("="*50)

if __name__ == "__main__":
    main()
