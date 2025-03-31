"""
Monitoring Integration

This module integrates the monitoring dashboard into the main application.
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_monitoring(dashboard):
    """
    Integrate monitoring dashboard into the main application.
    
    Args:
        dashboard: ModernDashboard instance
    """
    try:
        # Import monitoring API
        from src.dashboard.monitoring_api import MonitoringAPI
        
        # Create monitoring API
        monitoring_api = MonitoringAPI()
        
        # Register Flask app
        monitoring_api.register_app(dashboard.app)
        
        # Register monitoring components
        if hasattr(dashboard, 'enhanced_connector'):
            # Register enhanced connector
            monitoring_api.register_components(
                error_monitor=getattr(dashboard.enhanced_connector, 'error_monitor', None),
                rate_limit_monitor=getattr(dashboard.enhanced_connector, 'rate_limit_monitor', None),
                data_cache=getattr(dashboard.enhanced_connector, 'data_cache', None),
                connection_pool=getattr(dashboard.enhanced_connector, 'connection_pool', None),
                fallback_manager=getattr(dashboard.enhanced_connector, 'fallback_manager', None),
                enhanced_connector=dashboard.enhanced_connector
            )
        
        # Add monitoring tab route
        dashboard.app.route("/monitoring")(dashboard.monitoring_tab)
        
        # Store monitoring API
        dashboard.monitoring_api = monitoring_api
        
        logger.info("Monitoring dashboard integrated")
        return True
    except Exception as e:
        logger.error(f"Error integrating monitoring dashboard: {e}")
        return False

def add_monitoring_tab(dashboard):
    """
    Add monitoring tab to ModernDashboard class.
    
    Args:
        dashboard: ModernDashboard instance
    """
    def monitoring_tab():
        """Monitoring tab route."""
        return dashboard.render_template('monitoring_tab.html')
    
    # Add monitoring tab method
    dashboard.monitoring_tab = monitoring_tab
    
    # Add monitoring tab to navigation
    if hasattr(dashboard, 'nav_items'):
        dashboard.nav_items.append({
            'name': 'Monitoring',
            'icon': 'activity',
            'url': '/monitoring',
            'roles': ['admin', 'operator']
        })
    
    logger.info("Monitoring tab added")
    return True

def create_enhanced_connector(dashboard, api_key=None, api_secret=None):
    """
    Create enhanced Bitvavo connector.
    
    Args:
        dashboard: ModernDashboard instance
        api_key: Bitvavo API key
        api_secret: Bitvavo API secret
        
    Returns:
        EnhancedBitvavoConnector: Enhanced connector
    """
    try:
        # Import enhanced connector
        from src.execution.exchange.enhanced_bitvavo import EnhancedBitvavoConnector
        from src.monitoring.error_monitor import ErrorMonitor
        from src.monitoring.rate_limit_monitor import RateLimitMonitor
        from src.common.cache.data_cache import DataCache
        from src.common.network.fallback_manager import BitvavoFallbackManager
        from src.common.network.connection_pool import ConnectionPool
        
        # Create components
        error_monitor = ErrorMonitor()
        rate_limit_monitor = RateLimitMonitor()
        data_cache = DataCache()
        connection_pool = ConnectionPool()
        fallback_manager = BitvavoFallbackManager()
        
        # Create connector
        connector = EnhancedBitvavoConnector(
            api_key=api_key,
            api_secret=api_secret,
            connection_pool=connection_pool,
            error_monitor=error_monitor,
            rate_limit_monitor=rate_limit_monitor,
            data_cache=data_cache,
            fallback_manager=fallback_manager
        )
        
        # Initialize connector
        connector.initialize()
        
        # Store connector
        dashboard.enhanced_connector = connector
        
        logger.info("Enhanced Bitvavo connector created")
        return connector
    except Exception as e:
        logger.error(f"Error creating enhanced Bitvavo connector: {e}")
        return None

def apply_monitoring_integration(dashboard):
    """
    Apply monitoring integration to ModernDashboard.
    
    Args:
        dashboard: ModernDashboard instance
    """
    # Add monitoring tab
    add_monitoring_tab(dashboard)
    
    # Create enhanced connector
    create_enhanced_connector(dashboard)
    
    # Integrate monitoring
    integrate_monitoring(dashboard)
    
    logger.info("Monitoring integration applied")
    return True