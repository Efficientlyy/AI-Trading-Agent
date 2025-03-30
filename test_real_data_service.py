"""
Test script for the enhanced DataService implementation.

This script tests the improved error handling, caching, and data validation
capabilities of the DataService class.
"""

import logging
import time
import json
from pathlib import Path
from datetime import datetime

from src.dashboard.utils.data_service import DataService, DataSource, DataSourceStatus
from src.dashboard.utils.enums import DataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_service_initialization():
    """Test DataService initialization and configuration."""
    logger.info("Testing DataService initialization...")
    
    # Create a DataService instance with mock data
    service = DataService(data_source=DataSource.MOCK)
    
    # Check if the service is properly initialized
    assert service.data_source == DataSource.MOCK
    assert hasattr(service, 'cache')
    assert hasattr(service, 'persistent_cache')
    assert hasattr(service, 'error_tracking')
    assert hasattr(service, 'data_source_status')
    
    logger.info("DataService initialization test: PASSED")
    return True

def test_data_source_switching():
    """Test switching between mock and real data sources."""
    logger.info("Testing data source switching...")
    
    # Create a DataService instance with mock data
    service = DataService(data_source=DataSource.MOCK)
    
    # Get initial status
    initial_status = service.get_data_source_status()
    logger.info(f"Initial data source status: {json.dumps(initial_status, indent=2)}")
    
    # Try to switch to real data
    switch_result = service.set_data_source(DataSource.REAL)
    logger.info(f"Switch to real data result: {json.dumps(switch_result, indent=2)}")
    
    # Get updated status
    updated_status = service.get_data_source_status()
    logger.info(f"Updated data source status: {json.dumps(updated_status, indent=2)}")
    
    # Switch back to mock data
    switch_back_result = service.set_data_source(DataSource.MOCK)
    logger.info(f"Switch back to mock data result: {json.dumps(switch_back_result, indent=2)}")
    
    logger.info("Data source switching test: PASSED")
    return True

def test_caching_mechanism():
    """Test the caching mechanism with stale-while-revalidate pattern."""
    logger.info("Testing caching mechanism...")
    
    # Create a DataService instance
    service = DataService()
    
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
    service = DataService()
    
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
    service = DataService()
    
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

def test_persistent_cache():
    """Test persistent cache for critical data."""
    logger.info("Testing persistent cache...")
    
    # Create a DataService instance
    service = DataService()
    
    # Get some data
    data = service.get_data("system_health")
    
    # Save to persistent cache
    service.persistent_cache["system_health"] = {
        "data": data,
        "timestamp": time.time()
    }
    service._save_persistent_cache()
    
    # Create a new service instance (should load from persistent cache)
    new_service = DataService()
    
    # Check if the persistent cache was loaded
    assert "system_health" in new_service.persistent_cache
    
    logger.info("Persistent cache test: PASSED")
    return True

def test_enhanced_get_data():
    """Test the enhanced get_data method with fallback mechanisms."""
    logger.info("Testing enhanced get_data method...")
    
    # Create a DataService instance
    service = DataService()
    
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
        test_data_source_switching,
        test_caching_mechanism,
        test_error_tracking,
        test_data_validation,
        test_persistent_cache,
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