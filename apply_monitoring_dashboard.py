#!/usr/bin/env python3
"""
Apply Monitoring Dashboard

This script applies the monitoring dashboard to the ModernDashboard class.
"""

import os
import sys
import logging
import importlib.util
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_module(module_path, module_name):
    """
    Load a module from a file path.
    
    Args:
        module_path: Path to the module file
        module_name: Name to give the module
        
    Returns:
        Module: Loaded module
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def apply_monitoring_dashboard():
    """
    Apply monitoring dashboard to ModernDashboard class.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load monitoring integration module
        monitoring_integration_path = os.path.join('src', 'dashboard', 'monitoring_integration.py')
        monitoring_integration = load_module(monitoring_integration_path, 'monitoring_integration')
        
        # Load ModernDashboard class
        modern_dashboard_path = os.path.join('src', 'dashboard', 'modern_dashboard.py')
        
        # Check if file exists
        if not os.path.exists(modern_dashboard_path):
            logger.error(f"ModernDashboard file not found: {modern_dashboard_path}")
            return False
        
        # Read file
        with open(modern_dashboard_path, 'r') as f:
            content = f.read()
        
        # Check if monitoring integration is already applied
        if 'monitoring_tab' in content:
            logger.info("Monitoring dashboard already applied")
            return True
        
        # Find the register_routes method
        register_routes_start = content.find('def register_routes(self)')
        if register_routes_start == -1:
            logger.error("register_routes method not found")
            return False
        
        # Find the end of the method
        register_routes_end = content.find('def', register_routes_start + 1)
        if register_routes_end == -1:
            register_routes_end = len(content)
        
        # Extract the method
        register_routes_method = content[register_routes_start:register_routes_end]
        
        # Find the last route registration
        last_route = register_routes_method.rfind('self.app.route')
        if last_route == -1:
            logger.error("No route registrations found")
            return False
        
        # Find the end of the last route registration
        last_route_end = register_routes_method.find(')', last_route)
        if last_route_end == -1:
            logger.error("Invalid route registration")
            return False
        
        # Insert monitoring tab route
        new_register_routes_method = register_routes_method[:last_route_end + 1] + '\n\n        # Monitoring tab\n        self.app.route("/monitoring")(self.monitoring_tab)' + register_routes_method[last_route_end + 1:]
        
        # Replace the method
        new_content = content[:register_routes_start] + new_register_routes_method + content[register_routes_end:]
        
        # Find the __init__ method
        init_start = new_content.find('def __init__(self')
        if init_start == -1:
            logger.error("__init__ method not found")
            return False
        
        # Find the end of the method
        init_end = new_content.find('def', init_start + 1)
        if init_end == -1:
            init_end = len(new_content)
        
        # Extract the method
        init_method = new_content[init_start:init_end]
        
        # Find the end of the method body
        method_body_end = init_method.rfind('\n')
        if method_body_end == -1:
            logger.error("Invalid __init__ method")
            return False
        
        # Insert monitoring API initialization
        new_init_method = init_method[:method_body_end] + '\n\n        # Initialize monitoring API\n        self.monitoring_api = None\n        self.enhanced_connector = None' + init_method[method_body_end:]
        
        # Replace the method
        new_content = new_content[:init_start] + new_init_method + new_content[init_end:]
        
        # Add monitoring tab method
        monitoring_tab_method = '''
    def monitoring_tab(self):
        """Monitoring tab route."""
        return self.render_template('monitoring_tab.html')
'''
        
        # Find a good place to insert the method
        insert_pos = new_content.rfind('\n\n    def ')
        if insert_pos == -1:
            logger.error("Could not find a place to insert monitoring_tab method")
            return False
        
        # Insert the method
        new_content = new_content[:insert_pos] + monitoring_tab_method + new_content[insert_pos:]
        
        # Find the run method
        run_start = new_content.find('def run(self')
        if run_start == -1:
            logger.error("run method not found")
            return False
        
        # Find the end of the method
        run_end = new_content.find('def', run_start + 1)
        if run_end == -1:
            run_end = len(new_content)
        
        # Extract the method
        run_method = new_content[run_start:run_end]
        
        # Find the beginning of the method body
        method_body_start = run_method.find(':')
        if method_body_start == -1:
            logger.error("Invalid run method")
            return False
        
        # Insert monitoring integration
        new_run_method = run_method[:method_body_start + 1] + '''
        # Apply monitoring integration
        try:
            from src.dashboard.monitoring_integration import apply_monitoring_integration
            apply_monitoring_integration(self)
        except Exception as e:
            logger.warning(f"Error applying monitoring integration: {e}")
            
''' + run_method[method_body_start + 1:]
        
        # Replace the method
        new_content = new_content[:run_start] + new_run_method + new_content[run_end:]
        
        # Write the updated file
        with open(modern_dashboard_path, 'w') as f:
            f.write(new_content)
        
        logger.info("Monitoring dashboard applied to ModernDashboard class")
        return True
    except Exception as e:
        logger.error(f"Error applying monitoring dashboard: {e}")
        return False

if __name__ == "__main__":
    logger.info("Applying monitoring dashboard...")
    
    if apply_monitoring_dashboard():
        logger.info("Successfully applied monitoring dashboard")
    else:
        logger.error("Failed to apply monitoring dashboard")
        sys.exit(1)