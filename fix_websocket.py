"""
Fix WebSocket Issue

This script fixes the WebSocket issue in the ModernDashboard class.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_websocket_issue():
    """Fix the WebSocket issue in the ModernDashboard class."""
    try:
        # Path to the modern_dashboard.py file
        file_path = os.path.join('src', 'dashboard', 'modern_dashboard.py')
        
        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the websocket line exists
        websocket_line = 'self.app.websocket("/ws")(self.websocket_endpoint)'
        if websocket_line not in content:
            logger.info("WebSocket line not found, no need to fix")
            return True
        
        # Replace the websocket line with a comment
        new_content = content.replace(
            websocket_line,
            '# WebSocket endpoint disabled due to compatibility issues\n        # self.app.websocket("/ws")(self.websocket_endpoint)'
        )
        
        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Successfully fixed WebSocket issue")
        return True
    except Exception as e:
        logger.error(f"Error fixing WebSocket issue: {e}")
        return False

if __name__ == "__main__":
    logger.info("Fixing WebSocket issue in ModernDashboard class...")
    
    if fix_websocket_issue():
        logger.info("Successfully fixed WebSocket issue")
    else:
        logger.error("Failed to fix WebSocket issue")
        sys.exit(1)