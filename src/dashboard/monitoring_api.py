"""
Monitoring API

This module provides API endpoints for the monitoring dashboard.
"""

import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitoringAPI:
    """
    API endpoints for the monitoring dashboard.
    
    This class provides API endpoints for retrieving monitoring data,
    including rate limits, errors, cache statistics, and performance metrics.
    """
    
    def __init__(self, app=None):
        """
        Initialize the monitoring API.
        
        Args:
            app: Flask application
        """
        self.app = app
        self.error_monitor = None
        self.rate_limit_monitor = None
        self.data_cache = None
        self.connection_pool = None
        self.fallback_manager = None
        self.enhanced_connector = None
        
        # Alert history
        self.alert_history = []
        self.max_alert_history = 100
        
        # System status
        self.system_status = {
            'Bitvavo API': {
                'operational': True,
                'last_updated': datetime.now().isoformat()
            },
            'Rate Limit Monitor': {
                'operational': True,
                'last_updated': datetime.now().isoformat()
            },
            'Error Monitor': {
                'operational': True,
                'last_updated': datetime.now().isoformat()
            },
            'Data Cache': {
                'operational': True,
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Register alert callbacks
        self._register_alert_callbacks()
        
        logger.info("Monitoring API initialized")
    
    def register_app(self, app):
        """
        Register Flask application.
        
        Args:
            app: Flask application
        """
        self.app = app
        self._register_routes()
        
        logger.info("Monitoring API routes registered")
    
    def register_components(self, error_monitor=None, rate_limit_monitor=None,
                           data_cache=None, connection_pool=None,
                           fallback_manager=None, enhanced_connector=None):
        """
        Register monitoring components.
        
        Args:
            error_monitor: Error monitor
            rate_limit_monitor: Rate limit monitor
            data_cache: Data cache
            connection_pool: Connection pool
            fallback_manager: Fallback manager
            enhanced_connector: Enhanced connector
        """
        self.error_monitor = error_monitor
        self.rate_limit_monitor = rate_limit_monitor
        self.data_cache = data_cache
        self.connection_pool = connection_pool
        self.fallback_manager = fallback_manager
        self.enhanced_connector = enhanced_connector
        
        # Update system status
        self._update_system_status()
        
        logger.info("Monitoring components registered")
    
    def _register_routes(self):
        """Register API routes."""
        if not self.app:
            logger.warning("No Flask application registered")
            return
        
        # Monitoring data endpoint
        self.app.route('/api/monitoring/data')(self.get_monitoring_data)
        
        # Component-specific endpoints
        self.app.route('/api/monitoring/rate-limits')(self.get_rate_limits)
        self.app.route('/api/monitoring/errors')(self.get_errors)
        self.app.route('/api/monitoring/cache')(self.get_cache_stats)
        self.app.route('/api/monitoring/performance')(self.get_performance_stats)
        self.app.route('/api/monitoring/system')(self.get_system_status)
        self.app.route('/api/monitoring/alerts')(self.get_alerts)
        
        logger.info("Monitoring API routes registered")
    
    def _register_alert_callbacks(self):
        """Register alert callbacks."""
        # Register callbacks when components are available
        if self.error_monitor:
            self.error_monitor.register_alert_callback(self._handle_error_alert)
        
        if self.rate_limit_monitor:
            self.rate_limit_monitor.register_alert_callback(self._handle_rate_limit_alert)
    
    def _handle_error_alert(self, alert_data):
        """
        Handle error alert.
        
        Args:
            alert_data: Alert data
        """
        # Add timestamp if not present
        if 'timestamp' not in alert_data:
            alert_data['timestamp'] = datetime.now().isoformat()
        
        # Add to alert history
        self.alert_history.insert(0, alert_data)
        
        # Trim alert history
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[:self.max_alert_history]
        
        # Update system status
        self._update_system_status()
        
        logger.warning(f"Error alert: {alert_data}")
    
    def _handle_rate_limit_alert(self, alert_data):
        """
        Handle rate limit alert.
        
        Args:
            alert_data: Alert data
        """
        # Add timestamp if not present
        if 'timestamp' not in alert_data:
            alert_data['timestamp'] = datetime.now().isoformat()
        
        # Add to alert history
        self.alert_history.insert(0, alert_data)
        
        # Trim alert history
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[:self.max_alert_history]
        
        # Update system status
        self._update_system_status()
        
        logger.warning(f"Rate limit alert: {alert_data}")
    
    def _update_system_status(self):
        """Update system status."""
        # Update Bitvavo API status
        if self.enhanced_connector:
            # Check if any circuit breakers are open
            if self.error_monitor and self.error_monitor.is_circuit_open('bitvavo'):
                self.system_status['Bitvavo API'] = {
                    'operational': False,
                    'last_updated': datetime.now().isoformat()
                }
            else:
                self.system_status['Bitvavo API'] = {
                    'operational': True,
                    'last_updated': datetime.now().isoformat()
                }
        
        # Update Rate Limit Monitor status
        if self.rate_limit_monitor:
            self.system_status['Rate Limit Monitor'] = {
                'operational': True,
                'last_updated': datetime.now().isoformat()
            }
        
        # Update Error Monitor status
        if self.error_monitor:
            self.system_status['Error Monitor'] = {
                'operational': True,
                'last_updated': datetime.now().isoformat()
            }
        
        # Update Data Cache status
        if self.data_cache:
            self.system_status['Data Cache'] = {
                'operational': True,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_monitoring_data(self):
        """
        Get all monitoring data.
        
        Returns:
            Dict: Monitoring data
        """
        # Update system status
        self._update_system_status()
        
        # Collect all monitoring data
        data = {
            'rate_limits': self.get_rate_limits(raw=True),
            'errors': self.get_errors(raw=True),
            'cache': self.get_cache_stats(raw=True),
            'performance': self.get_performance_stats(raw=True),
            'system': self.get_system_status(raw=True),
            'alerts': self.get_alerts(raw=True)
        }
        
        return data
    
    def get_rate_limits(self, raw=False):
        """
        Get rate limit data.
        
        Args:
            raw: Whether to return raw data or JSON response
            
        Returns:
            Dict: Rate limit data
        """
        data = {
            'status': {},
            'history': []
        }
        
        # Get rate limit status
        if self.rate_limit_monitor:
            data['status'] = self.rate_limit_monitor.get_rate_limit_status()
        
        # Return data
        if raw:
            return data
        else:
            return json.dumps(data)
    
    def get_errors(self, raw=False):
        """
        Get error data.
        
        Args:
            raw: Whether to return raw data or JSON response
            
        Returns:
            Dict: Error data
        """
        data = {
            'error_stats': {},
            'circuit_breakers': {},
            'history': []
        }
        
        # Get error stats
        if self.error_monitor:
            data['error_stats'] = self.error_monitor.get_error_stats()
            data['circuit_breakers'] = self.error_monitor.get_circuit_breakers()
        
        # Return data
        if raw:
            return data
        else:
            return json.dumps(data)
    
    def get_cache_stats(self, raw=False):
        """
        Get cache statistics.
        
        Args:
            raw: Whether to return raw data or JSON response
            
        Returns:
            Dict: Cache statistics
        """
        data = {
            'stats': {},
            'entries': {}
        }
        
        # Get cache stats
        if self.data_cache:
            data['stats'] = self.data_cache.get_stats()
        
        # Return data
        if raw:
            return data
        else:
            return json.dumps(data)
    
    def get_performance_stats(self, raw=False):
        """
        Get performance statistics.
        
        Args:
            raw: Whether to return raw data or JSON response
            
        Returns:
            Dict: Performance statistics
        """
        data = {
            'connection_pool': {},
            'request_timing': {}
        }
        
        # Get connection pool stats
        if self.connection_pool:
            data['connection_pool'] = self.connection_pool.get_stats()
        
        # Return data
        if raw:
            return data
        else:
            return json.dumps(data)
    
    def get_system_status(self, raw=False):
        """
        Get system status.
        
        Args:
            raw: Whether to return raw data or JSON response
            
        Returns:
            Dict: System status
        """
        data = {
            'status': self.system_status
        }
        
        # Return data
        if raw:
            return data
        else:
            return json.dumps(data)
    
    def get_alerts(self, raw=False):
        """
        Get alert history.
        
        Args:
            raw: Whether to return raw data or JSON response
            
        Returns:
            Dict: Alert history
        """
        data = {
            'alerts': self.alert_history
        }
        
        # Return data
        if raw:
            return data
        else:
            return json.dumps(data)