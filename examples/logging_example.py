"""Example usage of the enhanced logging system.

This script demonstrates how to use the various features of the logging system.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.datetime_utils import utc_now
from src.common.logging import (
    get_logger, 
    LogOperation, 
    request_context, 
    rate_limited_log,
    timed_operation, 
    get_metrics_logger
)


def demonstrate_basic_logging():
    """Demonstrate basic logging capabilities."""
    # Get a logger for a component
    logger = get_logger("examples", "logging")
    
    # Log at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Log with additional context
    logger.info(
        "Message with context", 
        user="example_user", 
        action="demonstration"
    )
    
    # Log with datetime objects
    now = utc_now()
    logger.info(
        "Message with datetime", 
        current_time=now,
        tomorrow=now.replace(day=now.day + 1)
    )


def demonstrate_sensitive_data_masking():
    """Demonstrate masking of sensitive data in logs."""
    logger = get_logger("examples", "masking")
    
    # Log message with sensitive data
    logger.info(
        "User credentials update", 
        user="example_user",
        password="very_secret_password",  # Will be masked
        api_key="abcdef123456",  # Will be masked
        public_data="This will be visible"
    )
    
    # Log message with nested sensitive data
    logger.info(
        "Configuration loaded",
        config={
            "database": {
                "host": "localhost",
                "user": "db_user",
                "password": "db_password"  # Will be masked
            },
            "api": {
                "url": "https://api.example.com",
                "key": "api_key_value",  # Will be masked
                "timeout": 30
            }
        }
    )


def demonstrate_operation_logging():
    """Demonstrate using the LogOperation context manager."""
    logger = get_logger("examples", "operations")
    
    # Log a simple operation
    with LogOperation(logger, "simple operation"):
        # Do some work
        for i in range(1000000):
            pass
    
    # Log an operation with additional context
    with LogOperation(logger, "data processing", level="info", records=100, type="user_data"):
        # Process data
        for i in range(2000000):
            pass
    
    # Log an operation that raises an exception
    try:
        with LogOperation(logger, "failing operation"):
            # This will cause an error
            raise ValueError("Something went wrong!")
    except ValueError:
        pass  # Exception is re-raised by default, so we catch it here


def demonstrate_complex_logging():
    """Demonstrate more complex logging scenarios."""
    logger = get_logger("examples", "complex")
    
    # Log a complex data structure with mixed sensitive and non-sensitive data
    complex_data = {
        "user": {
            "id": 12345,
            "name": "John Doe",
            "email": "john@example.com",
            "session": {
                "token": "secret_session_token",  # Will be masked
                "expires": utc_now(),
                "ip": "192.168.1.1"
            }
        },
        "transactions": [
            {
                "id": "tx_123",
                "amount": 100.0,
                "card": {
                    "number": "1234-5678-9012-3456",  # Should be manually masked before logging
                    "expiry": "12/25",
                    "cvv": "123"  # Should be manually masked before logging
                }
            },
            {
                "id": "tx_456",
                "amount": 200.0,
                "api_key": "payment_api_key"  # Will be masked
            }
        ]
    }
    
    logger.info("Complex data structure", data=complex_data)


def demonstrate_request_context():
    """Demonstrate the request context tracking feature."""
    logger = get_logger("examples", "request_context")
    
    def process_request(request_id):
        # This function simulates processing a request in a new thread
        request_logger = get_logger("examples", "request_processor")
        
        with request_context(request_id):
            # Log messages in this context will include the request_id
            request_logger.info("Processing request")
            time.sleep(0.1)  # Simulate work
            request_logger.info("Request processed successfully")
    
    # Process multiple requests in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        for i in range(3):
            request_id = f"req-{i+1}"
            logger.info(f"Starting request {request_id}")
            
            # Process the request in a separate thread with the request context
            executor.submit(process_request, request_id)
    
    # Wait for all requests to complete
    time.sleep(0.5)


def demonstrate_rate_limiting():
    """Demonstrate rate-limited logging."""
    logger = get_logger("examples", "rate_limiting")
    
    logger.info("This message appears every time")
    
    # Simulate high-frequency events that should be rate-limited
    for i in range(10):
        # This will only log once per second
        rate_limited_log(logger, "info", "High-frequency event", 1.0, count=i)
        
        # But this regular log will appear every time
        logger.info(f"Regular log {i}")
        
        time.sleep(0.2)


def demonstrate_metrics_logging():
    """Demonstrate performance metrics logging."""
    logger = get_logger("examples", "metrics")
    
    # Get a metrics logger for database operations
    db_metrics = get_metrics_logger("database_operations")
    
    # Record some metrics
    db_metrics.add_value(10.5)  # 10.5ms operation
    db_metrics.add_value(15.2)  # 15.2ms operation
    db_metrics.add_value(8.7)   # 8.7ms operation
    
    # Log the metrics
    db_metrics.log_metrics()
    
    # Use the timed_operation context manager to automatically time operations
    logger.info("Starting timed operations")
    
    for i in range(3):
        with timed_operation("api_request", "examples"):
            # Simulate work with varying durations
            time.sleep(0.01 * (i + 1))
    
    # Get the metrics logger and log the results
    api_metrics = get_metrics_logger("api_request", "examples")
    api_metrics.log_metrics(reset=True)
    
    # Perform another timed operation after reset
    with timed_operation("api_request", "examples"):
        time.sleep(0.05)
    
    # Log metrics again
    api_metrics.log_metrics()


if __name__ == "__main__":
    print("\n=== Basic Logging Examples ===")
    demonstrate_basic_logging()
    
    print("\n=== Sensitive Data Masking Examples ===")
    demonstrate_sensitive_data_masking()
    
    print("\n=== Operation Logging Examples ===")
    demonstrate_operation_logging()
    
    print("\n=== Complex Logging Examples ===")
    demonstrate_complex_logging()
    
    print("\n=== Request Context Examples ===")
    demonstrate_request_context()
    
    print("\n=== Rate Limiting Examples ===")
    demonstrate_rate_limiting()
    
    print("\n=== Metrics Logging Examples ===")
    demonstrate_metrics_logging()
