"""
Status Reporter

This module provides a class for reporting the status of data sources,
including health, performance metrics, and error tracking.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("status_reporter")

class StatusReporter:
    """
    Reports the status of data sources, including health, performance metrics, and error tracking.
    """
    
    def __init__(self, data_service):
        """
        Initialize the status reporter.
        
        Args:
            data_service: The data service to report status for
        """
        self.data_service = data_service
        
        # Status history
        self.status_history = {}
        
        # Performance history
        self.performance_history = {}
        
        # Error history
        self.error_history = {}
        
        # Initialize status history for each data source
        self._initialize_history()
    
    def _initialize_history(self):
        """
        Initialize status history for each data source.
        """
        # Get all data sources
        data_sources = getattr(self.data_service, 'data_sources', {})
        
        # Initialize history for each data source
        for source_id in data_sources:
            if source_id not in self.status_history:
                self.status_history[source_id] = []
            
            if source_id not in self.performance_history:
                self.performance_history[source_id] = []
            
            if source_id not in self.error_history:
                self.error_history[source_id] = []
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the status of all data sources.
        
        Returns:
            Dict containing the status of all data sources
        """
        # Initialize sources dictionary
        sources = {}
        
        # Get all data sources
        data_sources = getattr(self.data_service, 'data_sources', {})
        
        # Get status for each data source
        for source_id, source in data_sources.items():
            # Get source status
            health_status = self._get_health_status(source)
            error_count = getattr(source, 'error_count', 0)
            request_count = getattr(source, 'request_count', 0)
            avg_response_time = getattr(source, 'avg_response_time', None)
            last_success_time = getattr(source, 'last_success_time', None)
            
            # Get connection details from configuration
            connection_config = self._get_connection_config(source_id)
            
            # Format response time
            response_time = self._format_response_time(avg_response_time)
            
            # Calculate uptime
            uptime = self._calculate_uptime(source_id)
            
            # Calculate error rate
            error_rate = self._calculate_error_rate(source_id)
            
            # Format last success time
            last_success = self._format_timestamp(last_success_time)
            
            # Get error history
            error_history = self._get_error_history(source_id)
            
            # Get performance history
            performance_history = self._get_performance_history(source_id)
            
            # Add source status to sources dictionary
            sources[source_id] = {
                'id': source_id,
                'health': health_status.upper(),
                'error_count': error_count,
                'request_count': request_count,
                'response_time': response_time,
                'uptime': uptime,
                'error_rate': error_rate,
                'last_success': last_success,
                'error_history': error_history,
                'performance_history': performance_history,
                
                # Connection details
                'type': connection_config.get('type', 'Unknown'),
                'endpoint': self._mask_sensitive_url(connection_config.get('endpoint', '--')),
                'retry_attempts': connection_config.get('retry_attempts', 3),
                'timeout': connection_config.get('timeout_seconds', 10),
                'cache_duration': connection_config.get('cache_duration_seconds', 60)
            }
        
        # Calculate overall system health
        healthy_sources = sum(1 for source in sources.values() if source['health'] == 'HEALTHY')
        total_sources = len(sources)
        
        if healthy_sources == total_sources:
            system_health = 'HEALTHY'
        elif healthy_sources >= total_sources * 0.7:
            system_health = 'DEGRADED'
        else:
            system_health = 'UNHEALTHY'
        
        # Calculate system-wide metrics
        avg_response_time = self._calculate_avg_response_time(sources)
        system_error_rate = self._calculate_system_error_rate(sources)
        
        # Get real data availability
        real_data_available = getattr(self.data_service, 'data_source', 'mock') == 'real'
        
        # Return system status
        return {
            'sources': sources,
            'system_health': system_health,
            'avg_response_time': avg_response_time,
            'error_rate': system_error_rate,
            'real_data_available': real_data_available,
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
    
    def test_connection(self, source_id: str) -> Dict[str, Any]:
        """
        Test a connection to a data source.
        
        Args:
            source_id: The ID of the data source to test
            
        Returns:
            Dict containing the test result
        """
        # Get all data sources
        data_sources = getattr(self.data_service, 'data_sources', {})
        
        # Check if source exists
        if source_id not in data_sources:
            return {
                'success': False,
                'message': f'Unknown data source: {source_id}'
            }
        
        # Get the source
        source = data_sources[source_id]
        
        try:
            # Test the connection
            start_time = time.time()
            
            # Check if the source has a test_connection method
            if hasattr(source, 'test_connection') and callable(source.test_connection):
                # Call the test_connection method
                result = source.test_connection()
                response_time = time.time() - start_time
                
                # Check if the test was successful
                if result.get('success', False):
                    # Add to performance history
                    self._add_performance_history(source_id, response_time)
                    
                    return {
                        'success': True,
                        'message': f'Connection successful (response time: {response_time:.2f}s)',
                        'response_time': response_time
                    }
                else:
                    # Add to error history
                    self._add_error_history(source_id, result.get('message', 'Unknown error'))
                    
                    return {
                        'success': False,
                        'message': f'Connection failed: {result.get("message", "Unknown error")}'
                    }
            else:
                # If the source doesn't have a test_connection method, try to get some data
                try:
                    # Try to get some data from the source
                    data = source.get_data('test')
                    response_time = time.time() - start_time
                    
                    # Add to performance history
                    self._add_performance_history(source_id, response_time)
                    
                    return {
                        'success': True,
                        'message': f'Connection successful (response time: {response_time:.2f}s)',
                        'response_time': response_time
                    }
                except Exception as e:
                    # Add to error history
                    self._add_error_history(source_id, str(e))
                    
                    return {
                        'success': False,
                        'message': f'Connection failed: {str(e)}'
                    }
        except Exception as e:
            # Add to error history
            self._add_error_history(source_id, str(e))
            
            return {
                'success': False,
                'message': f'Connection error: {str(e)}'
            }
    
    def reset_source_stats(self, source_id: str) -> Dict[str, Any]:
        """
        Reset statistics for a data source.
        
        Args:
            source_id: The ID of the data source to reset
            
        Returns:
            Dict containing the reset result
        """
        # Get all data sources
        data_sources = getattr(self.data_service, 'data_sources', {})
        
        # Check if source exists
        if source_id not in data_sources:
            return {
                'success': False,
                'message': f'Unknown data source: {source_id}'
            }
        
        # Get the source
        source = data_sources[source_id]
        
        try:
            # Reset source statistics
            if hasattr(source, 'reset_stats') and callable(source.reset_stats):
                source.reset_stats()
            
            # Reset history
            self.status_history[source_id] = []
            self.performance_history[source_id] = []
            self.error_history[source_id] = []
            
            return {
                'success': True,
                'message': f'Statistics for {source_id} have been reset'
            }
        except Exception as e:
            logger.error(f"Error resetting statistics for {source_id}: {e}")
            
            return {
                'success': False,
                'message': f'Failed to reset statistics: {str(e)}'
            }
    
    def reset_all_source_stats(self) -> Dict[str, Any]:
        """
        Reset statistics for all data sources.
        
        Returns:
            Dict containing the reset result
        """
        # Get all data sources
        data_sources = getattr(self.data_service, 'data_sources', {})
        
        # Track success and failures
        success_count = 0
        failure_count = 0
        failures = []
        
        # Reset stats for each source
        for source_id in data_sources:
            result = self.reset_source_stats(source_id)
            
            if result['success']:
                success_count += 1
            else:
                failure_count += 1
                failures.append(f"{source_id}: {result['message']}")
        
        # Return result
        if failure_count == 0:
            return {
                'success': True,
                'message': f'Statistics for all {success_count} sources have been reset'
            }
        else:
            return {
                'success': False,
                'message': f'Failed to reset statistics for {failure_count} out of {success_count + failure_count} sources',
                'details': failures
            }
    
    def _get_health_status(self, source) -> str:
        """
        Get the health status of a data source.
        
        Args:
            source: The data source to get the health status for
            
        Returns:
            The health status (healthy, degraded, or unhealthy)
        """
        # Check if the source has a health_status attribute
        if hasattr(source, 'health_status'):
            return source.health_status.lower()
        
        # Check if the source has an error_count attribute
        if hasattr(source, 'error_count'):
            # If there are no errors, the source is healthy
            if source.error_count == 0:
                return 'healthy'
            
            # If there are some errors, the source is degraded
            if source.error_count < 5:
                return 'degraded'
            
            # If there are many errors, the source is unhealthy
            return 'unhealthy'
        
        # Default to healthy
        return 'healthy'
    
    def _format_response_time(self, response_time) -> str:
        """
        Format a response time for display.
        
        Args:
            response_time: The response time to format
            
        Returns:
            The formatted response time
        """
        if response_time is None:
            return 'N/A'
        
        # Convert to milliseconds
        response_time_ms = response_time * 1000
        
        # Format with appropriate units
        if response_time_ms < 1:
            return f'{response_time_ms * 1000:.2f}μs'
        elif response_time_ms < 1000:
            return f'{response_time_ms:.2f}ms'
        else:
            return f'{response_time_ms / 1000:.2f}s'
    
    def _calculate_uptime(self, source_id: str) -> str:
        """
        Calculate the uptime percentage for a data source.
        
        Args:
            source_id: The ID of the data source to calculate uptime for
            
        Returns:
            The uptime percentage
        """
        # Get the status history for the source
        history = self.status_history.get(source_id, [])
        
        # If there's no history, return N/A
        if not history:
            return 'N/A'
        
        # Count the number of healthy statuses
        healthy_count = sum(1 for status in history if status == 'healthy')
        
        # Calculate the uptime percentage
        uptime_percentage = (healthy_count / len(history)) * 100
        
        return f'{uptime_percentage:.1f}%'
    
    def _calculate_error_rate(self, source_id: str) -> str:
        """
        Calculate the error rate for a data source.
        
        Args:
            source_id: The ID of the data source to calculate error rate for
            
        Returns:
            The error rate percentage
        """
        # Get the error history for the source
        errors = self.error_history.get(source_id, [])
        
        # Get the performance history for the source
        performance = self.performance_history.get(source_id, [])
        
        # If there's no history, return 0%
        if not performance:
            return '0%'
        
        # Calculate the error rate percentage
        error_rate = (len(errors) / len(performance)) * 100
        
        return f'{error_rate:.1f}%'
    
    def _format_timestamp(self, timestamp) -> str:
        """
        Format a timestamp for display.
        
        Args:
            timestamp: The timestamp to format
            
        Returns:
            The formatted timestamp
        """
        if timestamp is None:
            return 'Never'
        
        # Format the timestamp
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    def _get_error_history(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Get the error history for a data source.
        
        Args:
            source_id: The ID of the data source to get error history for
            
        Returns:
            The error history
        """
        # Get the error history for the source
        errors = self.error_history.get(source_id, [])
        
        # Return the most recent errors (up to 10)
        return errors[-10:]
    
    def _get_performance_history(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Get the performance history for a data source.
        
        Args:
            source_id: The ID of the data source to get performance history for
            
        Returns:
            The performance history
        """
        # Get the performance history for the source
        performance = self.performance_history.get(source_id, [])
        
        # Return the most recent performance data (up to 20)
        return performance[-20:]
    
    def _add_performance_history(self, source_id: str, response_time: float):
        """
        Add a performance data point to the history.
        
        Args:
            source_id: The ID of the data source to add performance data for
            response_time: The response time to add
        """
        # Get the performance history for the source
        performance = self.performance_history.get(source_id, [])
        
        # Add the performance data point
        performance.append({
            'timestamp': datetime.now().isoformat(),
            'response_time': response_time
        })
        
        # Limit the history to 100 data points
        if len(performance) > 100:
            performance = performance[-100:]
        
        # Update the performance history
        self.performance_history[source_id] = performance
        
        # Update the status history
        self._add_status_history(source_id, 'healthy')
    
    def _add_error_history(self, source_id: str, message: str):
        """
        Add an error to the history.
        
        Args:
            source_id: The ID of the data source to add error for
            message: The error message
        """
        # Get the error history for the source
        errors = self.error_history.get(source_id, [])
        
        # Add the error
        errors.append({
            'timestamp': datetime.now().isoformat(),
            'message': message
        })
        
        # Limit the history to 100 errors
        if len(errors) > 100:
            errors = errors[-100:]
        
        # Update the error history
        self.error_history[source_id] = errors
        
        # Update the status history
        self._add_status_history(source_id, 'unhealthy')
    
    def _add_status_history(self, source_id: str, status: str):
        """
        Add a status to the history.
        
        Args:
            source_id: The ID of the data source to add status for
            status: The status to add
        """
        # Get the status history for the source
        history = self.status_history.get(source_id, [])
        
        # Add the status
        history.append(status)
        
        # Limit the history to 100 statuses
        if len(history) > 100:
            history = history[-100:]
        
        # Update the status history
        self.status_history[source_id] = history
        
    def _get_connection_config(self, source_id: str) -> Dict[str, Any]:
        """
        Get the connection configuration for a data source.
        
        Args:
            source_id: The ID of the data source to get configuration for
            
        Returns:
            The connection configuration
        """
        # Try to get configuration from data service
        if hasattr(self.data_service, 'get_source_config'):
            try:
                return self.data_service.get_source_config(source_id) or {}
            except Exception as e:
                logger.warning(f"Error getting source config for {source_id}: {e}")
                return {}
        
        # Try to get configuration from settings manager
        if hasattr(self.data_service, 'settings_manager'):
            try:
                config = self.data_service.settings_manager.get_real_data_config()
                connections = config.get('connections', {})
                return connections.get(source_id, {})
            except Exception as e:
                logger.warning(f"Error getting source config from settings manager for {source_id}: {e}")
                return {}
        
        return {}
        
    def _mask_sensitive_url(self, url: str) -> str:
        """
        Mask sensitive information in a URL.
        
        Args:
            url: The URL to mask
            
        Returns:
            The masked URL
        """
        if not url or url == '--':
            return url
            
        try:
            # Check if URL contains API key or token
            import re
            
            # Mask API keys and tokens in query parameters
            masked_url = re.sub(r'(api[_-]?key|token|secret)=([^&]+)', r'\1=***', url)
            
            # Mask basic auth credentials
            masked_url = re.sub(r'(https?://)([^:]+):([^@]+)@', r'\1***:***@', masked_url)
            
            return masked_url
        except Exception:
            # If anything goes wrong, return a generic endpoint
            return 'API Endpoint'
        
    def _calculate_avg_response_time(self, sources: Dict[str, Any]) -> str:
        """
        Calculate the average response time across all sources.
        
        Args:
            sources: The sources dictionary
            
        Returns:
            The average response time
        """
        total_time = 0
        count = 0
        
        for source in sources.values():
            response_time = source.get('response_time')
            if response_time and response_time != 'N/A':
                # Extract numeric value from response time string
                try:
                    # Remove units and convert to number
                    value = float(response_time.replace('ms', '').replace('s', '').replace('μs', '').strip())
                    
                    # Convert to milliseconds if needed
                    if 's' in response_time and 'ms' not in response_time and 'μs' not in response_time:
                        value *= 1000
                    elif 'μs' in response_time:
                        value /= 1000
                        
                    total_time += value
                    count += 1
                except (ValueError, TypeError):
                    pass
        
        if count == 0:
            return 'N/A'
            
        avg_time = total_time / count
        
        # Format with appropriate units
        if avg_time < 1:
            return f'{avg_time * 1000:.2f}μs'
        elif avg_time < 1000:
            return f'{avg_time:.2f}ms'
        else:
            return f'{avg_time / 1000:.2f}s'
        
    def _calculate_system_error_rate(self, sources: Dict[str, Any]) -> str:
        """
        Calculate the system-wide error rate.
        
        Args:
            sources: The sources dictionary
            
        Returns:
            The system-wide error rate
        """
        total_errors = 0
        total_requests = 0
        
        for source in sources.values():
            total_errors += source.get('error_count', 0)
            total_requests += source.get('request_count', 0)
        
        if total_requests == 0:
            return '0%'
            
        error_rate = (total_errors / total_requests) * 100
        
        return f'{error_rate:.1f}%'