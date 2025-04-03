"""
Apply Bitvavo Routes

This script applies the Bitvavo API routes to the ModernDashboard class.
"""

import os
import sys
import importlib
import inspect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_routes():
    """Apply the Bitvavo API routes to the ModernDashboard class."""
    try:
        # Import the ModernDashboard class
        sys.path.append(os.getcwd())
        from src.dashboard.modern_dashboard import ModernDashboard
        
        # Import the Bitvavo API handler methods
        from src.dashboard.bitvavo_api_handlers import (
            api_get_bitvavo_settings_panel,
            api_bitvavo_status,
            api_bitvavo_test_connection,
            api_bitvavo_save_credentials,
            api_bitvavo_save_settings,
            api_bitvavo_get_pairs,
            api_bitvavo_save_pairs,
            api_bitvavo_get_paper_trading,
            api_bitvavo_save_paper_trading,
            _validate_bitvavo
        )
        
        # Add the methods to the ModernDashboard class
        ModernDashboard.api_get_bitvavo_settings_panel = api_get_bitvavo_settings_panel
        ModernDashboard.api_bitvavo_status = api_bitvavo_status
        ModernDashboard.api_bitvavo_test_connection = api_bitvavo_test_connection
        ModernDashboard.api_bitvavo_save_credentials = api_bitvavo_save_credentials
        ModernDashboard.api_bitvavo_save_settings = api_bitvavo_save_settings
        ModernDashboard.api_bitvavo_get_pairs = api_bitvavo_get_pairs
        ModernDashboard.api_bitvavo_save_pairs = api_bitvavo_save_pairs
        ModernDashboard.api_bitvavo_get_paper_trading = api_bitvavo_get_paper_trading
        ModernDashboard.api_bitvavo_save_paper_trading = api_bitvavo_save_paper_trading
        ModernDashboard._validate_bitvavo = _validate_bitvavo
        
        # Add Bitvavo to the validation methods dictionary
        original_init = ModernDashboard.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call the original __init__ method
            original_init(self, *args, **kwargs)
            
            # Add Bitvavo to the validation methods dictionary if it exists
            if hasattr(self, 'validation_methods'):
                self.validation_methods['bitvavo'] = self._validate_bitvavo
                logger.info("Added Bitvavo validation method to ModernDashboard")
        
        # Replace the __init__ method with the patched version
        ModernDashboard.__init__ = patched_init
        
        # Add Bitvavo API routes to the register_routes method
        original_register_routes = ModernDashboard.register_routes
        
        def patched_register_routes(self):
            # Call the original register_routes method
            original_register_routes(self)
            
            # Add Bitvavo API routes
            self.app.route("/api/settings/bitvavo/status", methods=["GET"])(self.api_bitvavo_status)
            self.app.route("/api/settings/bitvavo/test", methods=["POST"])(self.api_bitvavo_test_connection)
            self.app.route("/api/settings/bitvavo/save", methods=["POST"])(self.api_bitvavo_save_credentials)
            self.app.route("/api/settings/bitvavo/settings", methods=["POST"])(self.api_bitvavo_save_settings)
            self.app.route("/api/settings/bitvavo/pairs", methods=["GET"])(self.api_bitvavo_get_pairs)
            self.app.route("/api/settings/bitvavo/pairs", methods=["POST"])(self.api_bitvavo_save_pairs)
            self.app.route("/api/settings/bitvavo/paper-trading", methods=["GET"])(self.api_bitvavo_get_paper_trading)
            self.app.route("/api/settings/bitvavo/paper-trading", methods=["POST"])(self.api_bitvavo_save_paper_trading)
            self.app.route("/api/templates/bitvavo_settings_panel.html", methods=["GET"])(self.api_get_bitvavo_settings_panel)
            
            logger.info("Added Bitvavo API routes to ModernDashboard")
        
        # Replace the register_routes method with the patched version
        ModernDashboard.register_routes = patched_register_routes
        
        logger.info("Successfully applied Bitvavo API routes to ModernDashboard class")
        return True
    except Exception as e:
        logger.error(f"Error applying Bitvavo API routes: {e}")
        return False

if __name__ == "__main__":
    logger.info("Applying Bitvavo API routes to ModernDashboard class...")
    
    if apply_routes():
        logger.info("Successfully applied Bitvavo API routes")
    else:
        logger.error("Failed to apply Bitvavo API routes")
        sys.exit(1)