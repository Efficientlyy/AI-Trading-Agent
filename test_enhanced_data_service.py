"""
Simplified test script for the enhanced DataService implementation.

This script tests the core functionality of our enhanced DataService
without relying on the full application stack.
"""

import time
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock DataSource enum
class DataSource:
    MOCK = "mock"
    REAL = "real"

# Mock DataSourceStatus enum
class DataSourceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

# Mock MockDataGenerator class
class MockDataGenerator:
    def generate_system_health(self):
        return {
            "status": "good",
            "cpu_usage": 25.5,
            "memory_usage": 40.2,
            "disk_usage": 60.0,
            "network_latency": 15.3,
            "uptime": 3600,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_component_status(self):
        return {
            "components": [
                {
                    "name": "System Monitor",
                    "status": "running",
                    "last_update": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "load": 30.0,
                    "message": ""
                }
            ]
        }
    
    def generate_trading_performance(self):
        return {
            "equity": 105000.0,
            "daily_pnl": 5000.0,
            "daily_pnl_pct": 5.0,
            "win_rate": 65.0,
            "trades_today": 10,
            "timestamp": datetime.now().isoformat()
        }

# Simplified DataService class for testing
class EnhancedDataService:
    """
    Enhanced data service for flexible data sourcing with advanced caching
    
    This service allows the dashboard to switch between mock and real data
    sources while providing a consistent interface. It includes:
    
    - Robust error tracking for each data source
    - Advanced caching with stale-while-revalidate pattern
    - Data validation against expected schemas
    - Intelligent fallback mechanisms
    - Real-time data source status monitoring
    """
    
    # Schema definitions for data validation
    DATA_SCHEMAS = {
        'system_health': {
            'required_fields': ['status', 'cpu_usage', 'memory_usage', 'disk_usage', 'network_latency'],
            'field_types': {
                'status': str,
                'cpu_usage': float,
                'memory_usage': float,
                'disk_usage': float,
                'network_latency': float,
                'uptime': int
            }
        },
        'trading_performance': {
            'required_fields': ['equity', 'daily_pnl', 'win_rate'],
            'field_types': {
                'equity': float,
                'daily_pnl': float,
                'daily_pnl_pct': float,
                'win_rate': float,
                'trades_today': int
            }
        }
    }
    
    def __init__(self, data_source=DataSource.MOCK):
        """Initialize data service with specified source"""
        self.data_source = data_source
        self.mock_data = MockDataGenerator()
        
        # Cached data with timestamps
        self.cache = {}
        self.persistent_cache = {}
        self.cache_expiry = {
            'system_health': 5,  # 5 seconds
            'component_status': 10,  # 10 seconds
            'trading_performance': 30,  # 30 seconds
        }
        
        # Error tracking for data sources
        self.error_tracking = {}
        self.error_threshold = 3  # Number of consecutive errors to mark a source as unhealthy
        self.error_window = 300  # Time window for error rate calculation (5 minutes)
        
        # Data source health status
        self.data_source_status = {}
        self.last_health_check = {}
        
        # Initialize data source health status
        self._initialize_data_source_status()
    
    def _initialize_data_source_status(self) -> None:
        """Initialize the health status for all data sources"""
        for data_type in ['system_health', 'component_status', 'trading_performance']:
            self.data_source_status[data_type] = DataSourceStatus.UNKNOWN
            self.error_tracking[data_type] = {
                'consecutive_errors': 0,
                'error_history': [],
                'last_success': None
            }
            self.last_health_check[data_type] = time.time()
    
    def get_data(self, data_type: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get data of specified type from current source with caching
        
        This method uses the enhanced get_data_with_fallback method
        which provides better error handling, validation, and caching.
        
        Args:
            data_type: Type of data to fetch
            force_refresh: Whether to force a refresh
            
        Returns:
            Data dictionary
        """
        return self.get_data_with_fallback(data_type, force_refresh)
    
    def _get_mock_data(self, data_type: str) -> Dict[str, Any]:
        """Get mock data from generator"""
        data_generators = {
            'system_health': self.mock_data.generate_system_health,
            'component_status': self.mock_data.generate_component_status,
            'trading_performance': self.mock_data.generate_trading_performance,
        }
        
        generator = data_generators.get(data_type)
        if generator:
            return generator()
        
        # Return empty dict if no generator available
        return {}
    
    def _track_error(self, data_type: str, error: Exception) -> None:
        """
        Track an error for a specific data source
        
        Args:
            data_type: The data type that experienced an error
            error: The exception that occurred
        """
        now = time.time()
        
        # Initialize tracking if not exists
        if data_type not in self.error_tracking:
            self.error_tracking[data_type] = {
                'consecutive_errors': 0,
                'error_history': [],
                'last_success': None
            }
        
        # Update error tracking
        self.error_tracking[data_type]['consecutive_errors'] += 1
        self.error_tracking[data_type]['error_history'].append({
            'timestamp': now,
            'error': str(error),
            'type': type(error).__name__
        })
        
        # Trim error history to keep only recent errors
        self.error_tracking[data_type]['error_history'] = [
            e for e in self.error_tracking[data_type]['error_history']
            if now - e['timestamp'] < self.error_window
        ]
        
        # Update data source status based on errors
        self._update_data_source_status(data_type)
        
        # Log the error
        logger.error(f"Error in data source {data_type}: {error}")
    
    def _track_success(self, data_type: str) -> None:
        """
        Track a successful data fetch for a specific data source
        
        Args:
            data_type: The data type that was successfully fetched
        """
        now = time.time()
        
        # Initialize tracking if not exists
        if data_type not in self.error_tracking:
            self.error_tracking[data_type] = {
                'consecutive_errors': 0,
                'error_history': [],
                'last_success': None
            }
        
        # Update tracking
        self.error_tracking[data_type]['consecutive_errors'] = 0
        self.error_tracking[data_type]['last_success'] = now
        
        # Update data source status
        self._update_data_source_status(data_type)
    
    def _update_data_source_status(self, data_type: str) -> None:
        """
        Update the health status of a data source based on error tracking
        
        Args:
            data_type: The data type to update status for
        """
        if data_type not in self.error_tracking:
            self.data_source_status[data_type] = DataSourceStatus.UNKNOWN
            return
        
        tracking = self.error_tracking[data_type]
        now = time.time()
        
        # Calculate error rate
        recent_errors = len(tracking['error_history'])
        
        # If no errors and recent success, mark as healthy
        if recent_errors == 0 and tracking['last_success'] is not None:
            self.data_source_status[data_type] = DataSourceStatus.HEALTHY
            return
        
        # If consecutive errors exceed threshold, mark as unhealthy
        if tracking['consecutive_errors'] >= self.error_threshold:
            self.data_source_status[data_type] = DataSourceStatus.UNHEALTHY
            return
        
        # If some errors but not too many, mark as degraded
        if recent_errors > 0:
            self.data_source_status[data_type] = DataSourceStatus.DEGRADED
            return
        
        # Default to unknown if we can't determine status
        self.data_source_status[data_type] = DataSourceStatus.UNKNOWN
    
    def get_data_source_health(self) -> Dict[str, Any]:
        """
        Get health status for all data sources
        
        Returns:
            Dictionary with health status for each data source
        """
        result = {}
        
        for data_type in self.data_source_status.keys():
            status = self.data_source_status.get(data_type, DataSourceStatus.UNKNOWN)
            tracking = self.error_tracking.get(data_type, {
                'consecutive_errors': 0,
                'error_history': [],
                'last_success': None
            })
            
            result[data_type] = {
                'status': status.value,
                'consecutive_errors': tracking['consecutive_errors'],
                'recent_errors': len(tracking['error_history']),
                'last_success': tracking['last_success']
            }
        
        return result
    
    def _validate_data(self, data_type: str, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate data against expected schema
        
        Args:
            data_type: The type of data to validate
            data: The data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Skip validation if no schema exists
        if data_type not in self.DATA_SCHEMAS:
            return True, None
        
        schema = self.DATA_SCHEMAS[data_type]
        
        # Check required fields
        for field in schema.get('required_fields', []):
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check field types
        for field, expected_type in schema.get('field_types', {}).items():
            if field in data and not isinstance(data[field], expected_type):
                return False, f"Field {field} has wrong type: expected {expected_type.__name__}, got {type(data[field]).__name__}"
        
        return True, None
    
    def get_data_with_fallback(self, data_type: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get data with enhanced fallback mechanisms
        
        This method provides:
        - Stale-while-revalidate caching
        - Data validation
        - Error tracking
        
        Args:
            data_type: Type of data to fetch
            force_refresh: Whether to force a refresh
            
        Returns:
            Data dictionary
        """
        now = time.time()
        
        # Check if we have a valid cached item
        if not force_refresh and data_type in self.cache:
            cached_item = self.cache[data_type]
            expiry_time = self.cache_expiry.get(data_type, 30)
            
            # If cache is still fresh, return it immediately
            if now - cached_item['timestamp'] < expiry_time:
                return cached_item['data']
            
            # If cache is stale but not too old, return it and trigger async refresh
            if now - cached_item['timestamp'] < expiry_time * 3:
                # In a real async implementation, we would spawn a background task here
                # For now, we'll just log that we're using stale data
                logger.info(f"Using stale data for {data_type} while refreshing")
                
                # Schedule refresh for next request
                self.cache[data_type]['needs_refresh'] = True
                return cached_item['data']
        
        # If we need to fetch fresh data
        try:
            # Fetch fresh data (for this test, we'll just use mock data)
            data = self._get_mock_data(data_type)
            
            # Validate data
            is_valid, error_message = self._validate_data(data_type, data)
            if not is_valid:
                logger.warning(f"Invalid data for {data_type}: {error_message}")
                
                # Try to use cached data if available
                if data_type in self.cache:
                    logger.info(f"Falling back to cached data for {data_type} due to validation failure")
                    return self.cache[data_type]['data']
                
                # If no fallback available, return the invalid data anyway
                logger.warning(f"No fallback available for {data_type}, returning invalid data")
            
            # Track successful fetch
            self._track_success(data_type)
            
            # Update cache
            self.cache[data_type] = {
                'data': data,
                'timestamp': now,
                'needs_refresh': False
            }
            
            return data
            
        except Exception as e:
            # Track error
            self._track_error(data_type, e)
            
            # Try to use cached data if available
            if data_type in self.cache:
                logger.warning(f"Error fetching {data_type}, using cached data: {e}")
                return self.cache[data_type]['data']
            
            # Fall back to mock data as last resort
            logger.error(f"Error fetching {data_type}, falling back to mock data: {e}")
            return self._get_mock_data(data_type)


def test_data_service_initialization():
    """Test DataService initialization and configuration."""
    logger.info("Testing DataService initialization...")
    
    # Create a DataService instance with mock data
    service = EnhancedDataService(data_source=DataSource.MOCK)
    
    # Check if the service is properly initialized
    assert service.data_source == DataSource.MOCK
    assert hasattr(service, 'cache')
    assert hasattr(service, 'persistent_cache')
    assert hasattr(service, 'error_tracking')
    assert hasattr(service, 'data_source_status')
    
    logger.info("DataService initialization test: PASSED")
    return True

def test_caching_mechanism():
    """Test the caching mechanism with stale-while-revalidate pattern."""
    logger.info("Testing caching mechanism...")
    
    # Create a DataService instance
    service = EnhancedDataService()
    
    # Get data for the first time (should fetch fresh data)
    start_time = time.time()
    data1 = service.get_data("system_health")
    fetch_time1 = time.time() - start_time
    logger.info(f"First fetch took {fetch_time1:.4f} seconds")
    
    # Get data again immediately (should use cache)
    start_time = time.time()
    data2 = service.get_data("system_health")
    fetch_time2 = time.time() - start_time
    logger.info(f"Second fetch took {fetch_time2:.4f} seconds")
    
    # Verify that the second fetch was faster (used cache)
    assert fetch_time2 < fetch_time1
    
    # Force refresh (should fetch fresh data)
    start_time = time.time()
    data3 = service.get_data("system_health", force_refresh=True)
    fetch_time3 = time.time() - start_time
    logger.info(f"Forced refresh took {fetch_time3:.4f} seconds")
    
    # Verify that the forced refresh was slower (fetched fresh data)
    assert fetch_time3 > fetch_time2
    
    logger.info("Caching mechanism test: PASSED")
    return True

def test_error_tracking():
    """Test error tracking and data source health monitoring."""
    logger.info("Testing error tracking...")
    
    # Create a DataService instance
    service = EnhancedDataService()
    
    # Get initial health status
    initial_health = service.get_data_source_health()
    logger.info(f"Initial health status: {json.dumps(initial_health, indent=2)}")
    
    # Simulate errors for a specific data type
    data_type = "test_data_type"
    for i in range(5):
        service._track_error(data_type, Exception(f"Test error {i+1}"))
    
    # Get updated health status
    updated_health = service.get_data_source_health()
    logger.info(f"Updated health status: {json.dumps(updated_health, indent=2)}")
    
    # Verify that the data source is marked as unhealthy
    assert data_type in service.data_source_status
    assert service.data_source_status[data_type] == DataSourceStatus.UNHEALTHY
    
    # Simulate a successful fetch
    service._track_success(data_type)
    
    # Get final health status
    final_health = service.get_data_source_health()
    logger.info(f"Final health status: {json.dumps(final_health, indent=2)}")
    
    # Verify that the data source is no longer unhealthy
    assert service.data_source_status[data_type] != DataSourceStatus.UNHEALTHY
    
    logger.info("Error tracking test: PASSED")
    return True

def test_data_validation():
    """Test data validation against expected schemas."""
    logger.info("Testing data validation...")
    
    # Create a DataService instance
    service = EnhancedDataService()
    
    # Test valid data
    valid_data = {
        "status": "good",
        "cpu_usage": 25.5,
        "memory_usage": 40.2,
        "disk_usage": 60.0,
        "network_latency": 15.3,
        "uptime": 3600
    }
    
    is_valid, error_message = service._validate_data("system_health", valid_data)
    logger.info(f"Valid data validation result: {is_valid}, {error_message}")
    assert is_valid
    
    # Test invalid data (missing required field)
    invalid_data1 = {
        "status": "good",
        "cpu_usage": 25.5,
        "memory_usage": 40.2,
        # Missing disk_usage
        "network_latency": 15.3,
        "uptime": 3600
    }
    
    is_valid, error_message = service._validate_data("system_health", invalid_data1)
    logger.info(f"Invalid data (missing field) validation result: {is_valid}, {error_message}")
    assert not is_valid
    
    # Test invalid data (wrong type)
    invalid_data2 = {
        "status": "good",
        "cpu_usage": "25.5",  # String instead of float
        "memory_usage": 40.2,
        "disk_usage": 60.0,
        "network_latency": 15.3,
        "uptime": 3600
    }
    
    is_valid, error_message = service._validate_data("system_health", invalid_data2)
    logger.info(f"Invalid data (wrong type) validation result: {is_valid}, {error_message}")
    assert not is_valid
    
    logger.info("Data validation test: PASSED")
    return True

def test_enhanced_get_data():
    """Test the enhanced get_data method with fallback mechanisms."""
    logger.info("Testing enhanced get_data method...")
    
    # Create a DataService instance
    service = EnhancedDataService()
    
    # Get data using the enhanced method
    data = service.get_data("system_health")
    logger.info(f"Got data: {json.dumps(data, indent=2)}")
    
    # Verify that the data is valid
    is_valid, error_message = service._validate_data("system_health", data)
    assert is_valid
    
    logger.info("Enhanced get_data test: PASSED")
    return True

if __name__ == "__main__":
    print("\n=== Testing Enhanced DataService ===\n")
    
    # Run all tests
    tests = [
        test_data_service_initialization,
        test_caching_mechanism,
        test_error_tracking,
        test_data_validation,
        test_enhanced_get_data
    ]
    
    results = {}
    for test in tests:
        try:
            result = test()
            results[test.__name__] = "PASSED" if result else "FAILED"
        except Exception as e:
            logger.error(f"Error in {test.__name__}: {e}", exc_info=True)
            results[test.__name__] = "ERROR"
    
    # Print summary
    print("\n=== Test Results ===\n")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    # Overall result
    overall = all(result == "PASSED" for result in results.values())
    print(f"\nOverall result: {'PASSED' if overall else 'FAILED'}")