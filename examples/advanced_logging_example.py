"""
Advanced Logging System Example

This example demonstrates all the advanced features of the logging system:
1. Structured logging with request context
2. Log query language
3. Log replay system
4. Health monitoring
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import structlog

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.common.health_monitoring import (
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    MetricType
)
from src.common.log_query import LogQuery
from src.common.log_replay import LogReplay
from src.common.logging import LogOperation, get_logger


def setup_environment():
    """Set up the environment for the example."""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_DIR"] = "./logs"
    
    print("Environment set up successfully.")


def generate_sample_logs():
    """Generate sample logs for demonstration."""
    logger = get_logger("example")
    print("Generating sample logs...")
    
    # Generate request logs
    for i in range(1, 6):
        request_id = f"req_{i}"
        
        with LogOperation(logger, "process_request", request_id=request_id):
            # Start request
            logger.info(
                "Request received",
                request_id=request_id,
                path=f"/api/data/{i}",
                method="GET",
                client_ip="192.168.1.1"
            )
            
            # Simulate processing
            time.sleep(random.uniform(0.1, 0.3))
            
            # Database query
            query_time = random.uniform(10, 500)
            logger.info(
                "Database query executed",
                request_id=request_id,
                query="SELECT * FROM data",
                query_time=query_time,
                rows=random.randint(1, 100)
            )
            
            # Randomly generate errors
            if i == 3:  # Make request 3 fail
                logger.error(
                    "Request failed",
                    request_id=request_id,
                    error_code="DB_TIMEOUT",
                    error_message="Database query timed out",
                    status_code=500
                )
            else:
                # Successful response
                logger.info(
                    "Request completed",
                    request_id=request_id,
                    response_time=random.uniform(50, 200),
                    status_code=200
                )
    
    # Generate system logs
    system_logger = get_logger("system")
    
    system_logger.info(
        "System started",
        version="1.0.0",
        environment="development"
    )
    
    system_logger.warning(
        "High resource usage",
        cpu_usage=85.5,
        memory_usage=70.2,
        disk_usage=65.8
    )
    
    system_logger.info(
        "Configuration loaded",
        config_file="config.json",
        settings={
            "max_connections": 100,
            "timeout": 30,
            "retry_count": 3
        }
    )
    
    print("Sample logs generated successfully.")


def demonstrate_log_query():
    """Demonstrate the log query language."""
    print("\n=== Log Query Language Demonstration ===")
    
    # Simple query
    print("\n1. Simple query for error logs:")
    query = 'level = "error"'
    log_query = LogQuery(query)
    
    results = log_query.search_directory("logs", "*.log")
    print(f"Found {len(results)} error logs:")
    for entry in results:
        print(f"  - {entry.get('message')} (request_id: {entry.get('request_id')})")
    
    # Complex query
    print("\n2. Complex query with multiple conditions:")
    query = 'level = "info" AND component = "example" AND request_id ~ "req_"'
    log_query = LogQuery(query)
    
    results = log_query.search_directory("logs", "*.log")
    print(f"Found {len(results)} matching logs")
    if results:
        print(f"  - Sample: {results[0].get('message')} (request_id: {results[0].get('request_id')})")
    else:
        print("No matching logs found.")
    
    # Query with numeric comparison
    print("\n3. Query with numeric comparison:")
    query = 'query_time > 200'
    log_query = LogQuery(query)
    
    results = log_query.search_directory("logs", "*.log")
    print(f"Found {len(results)} logs with query_time > 200ms")
    
    # Query with time range
    print("\n4. Query with time range:")
    one_hour_ago = datetime.now() - timedelta(hours=1)
    
    results = log_query.search_directory("logs", "*.log")
    print(f"Found {len(results)} logs from the last hour")


def demonstrate_log_replay():
    """Demonstrate the log replay system."""
    print("\n=== Log Replay System Demonstration ===")
    
    # Create handlers for different log levels
    info_logs = []
    error_logs = []
    
    def info_handler(entry):
        info_logs.append(entry)
        print(f"INFO: {entry.get('message')} (request_id: {entry.get('request_id')})")
    
    def error_handler(entry):
        error_logs.append(entry)
        print(f"ERROR: {entry.get('message')} (request_id: {entry.get('request_id')})")
    
    # Create replay instance
    replay = LogReplay(
        handlers={
            "info": info_handler,
            "error": error_handler
        }
    )
    
    # Replay by request ID
    print("\n1. Replaying logs for request_id 'req_3' (the one with error):")
    count = replay.replay_by_request_id("req_3")
    print(f"Replayed {count} log entries")
    
    # Replay by component
    print("\n2. Replaying logs for 'system' component:")
    info_logs.clear()
    error_logs.clear()
    
    count = replay.replay_by_component("system")
    print(f"Replayed {count} log entries")
    
    # Replay with time filter
    print("\n3. Replaying logs from the last hour:")
    info_logs.clear()
    error_logs.clear()
    
    one_hour_ago = datetime.now() - timedelta(hours=1)
    count = replay.replay_from_directory(
        "logs",
        start_time=one_hour_ago
    )
    print(f"Replayed {count} log entries")


def demonstrate_health_monitoring():
    """Demonstrate the health monitoring system."""
    print("\n=== Health Monitoring System Demonstration ===")
    
    # Create health monitor
    monitor = HealthMonitor()
    
    # Add health checks
    def check_disk_space():
        # Simulate disk space check
        free_space = random.uniform(10, 90)
        print(f"Checking disk space: {free_space:.1f}% free")
        return free_space > 20
    
    def check_api_status():
        # Simulate API check
        is_available = random.random() > 0.2
        print(f"Checking API status: {'available' if is_available else 'unavailable'}")
        return is_available
    
    def check_error_logs():
        # Check for recent error logs
        query = 'level = "error"'
        log_query = LogQuery(query)
        
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        results = log_query.search_directory(
            "logs",
            "*.log",
            start_time=five_minutes_ago
        )
        
        has_errors = len(results) > 0
        print(f"Checking recent error logs: {'errors found' if has_errors else 'no errors'}")
        return not has_errors
    
    monitor.add_check(
        HealthCheck(
            name="disk_space",
            check_func=check_disk_space,
            description="Check available disk space",
            interval=60
        )
    )
    
    monitor.add_check(
        HealthCheck(
            name="api_status",
            check_func=check_api_status,
            description="Check API availability",
            interval=30
        )
    )
    
    monitor.add_check(
        HealthCheck(
            name="error_logs",
            check_func=check_error_logs,
            description="Check for recent error logs",
            interval=60,
            dependencies=["disk_space"]  # Only check for errors if disk space is ok
        )
    )
    
    # Add metrics
    monitor.add_metric(
        name="requests.count",
        metric_type=MetricType.COUNTER,
        description="Total request count"
    )
    
    monitor.add_metric(
        name="requests.latency",
        metric_type=MetricType.HISTOGRAM,
        description="Request latency",
        thresholds={
            "warning": 200,
            "critical": 500
        }
    )
    
    # Update metrics
    for _ in range(10):
        monitor.update_metric("requests.count", 1)
        monitor.update_metric("requests.latency", random.uniform(50, 600))
    
    # Run health checks
    print("\nRunning health checks...")
    monitor._run_checks()
    
    # Get health status
    status = monitor.get_status()
    print(f"\nOverall system status: {status['status'].upper()}")
    
    print("\nHealth checks:")
    for name, check in status["checks"].items():
        print(f"  - {name}: {check['status']} ({check['description']})")
    
    print("\nMetrics:")
    for name, metric in status["metrics"].items():
        print(f"  - {name}: {metric['value']}")


def main():
    """Main function."""
    print("=== Advanced Logging System Example ===\n")
    
    # Set up environment
    setup_environment()
    
    # Generate sample logs
    generate_sample_logs()
    
    # Demonstrate log query
    demonstrate_log_query()
    
    # Demonstrate log replay
    demonstrate_log_replay()
    
    # Demonstrate health monitoring
    demonstrate_health_monitoring()
    
    print("\nExample completed successfully.")


if __name__ == "__main__":
    main()
