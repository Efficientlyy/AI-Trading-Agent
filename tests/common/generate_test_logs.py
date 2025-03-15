"""Script to generate test log data for testing the logging system."""

import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from src.common.logging import get_logger


def setup_test_environment():
    """Set up the test environment."""
    # Create test log directory
    log_dir = Path("tests/data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_DIR"] = str(log_dir)
    
    return log_dir


def generate_request_logs(logger, request_id, component, count=5):
    """Generate a sequence of logs for a single request."""
    # Common request data
    request_data = {
        "request_id": request_id,
        "component": component,
        "user_id": f"user_{random.randint(1, 1000)}",
        "ip_address": f"192.168.1.{random.randint(1, 255)}"
    }
    
    # Start request
    logger.info(
        "Request started",
        **request_data,
        path="/api/v1/data",
        method="GET"
    )
    
    # Generate random events
    for _ in range(count):
        # Random delay
        time.sleep(random.uniform(0.1, 0.5))
        
        # Random event type
        event_type = random.choice(["database_query", "cache_lookup", "api_call"])
        duration = random.uniform(10, 1000)
        
        if duration > 500:  # Slow operation
            logger.warning(
                f"Slow {event_type}",
                **request_data,
                duration=duration,
                event_type=event_type
            )
        else:
            logger.info(
                f"{event_type} completed",
                **request_data,
                duration=duration,
                event_type=event_type
            )
            
    # Randomly generate errors
    if random.random() < 0.2:  # 20% chance of error
        error_type = random.choice([
            "TimeoutError",
            "ValidationError",
            "DatabaseError",
            "AuthenticationError"
        ])
        logger.error(
            f"Request failed: {error_type}",
            **request_data,
            error_type=error_type,
            error_details="Test error details"
        )
    else:
        logger.info(
            "Request completed",
            **request_data,
            status_code=200,
            response_time=random.uniform(100, 1000)
        )


def generate_system_logs(logger, count=10):
    """Generate system-level logs."""
    components = ["scheduler", "background_worker", "cleanup_job", "monitor"]
    
    for _ in range(count):
        component = random.choice(components)
        component_logger = get_logger(component)
        
        # System metrics
        component_logger.info(
            "System metrics",
            cpu_usage=random.uniform(0, 100),
            memory_usage=random.uniform(0, 100),
            disk_usage=random.uniform(0, 100),
            network_latency=random.uniform(1, 100)
        )
        
        # Random system events
        if random.random() < 0.3:  # 30% chance of system event
            event_type = random.choice([
                "cache_cleanup",
                "connection_pool_resize",
                "config_reload",
                "memory_cleanup"
            ])
            component_logger.info(
                f"System event: {event_type}",
                event_type=event_type,
                duration=random.uniform(10, 1000)
            )
            
        time.sleep(random.uniform(0.1, 0.3))


def generate_test_data(duration_minutes=5):
    """Generate test log data for a specified duration."""
    log_dir = setup_test_environment()
    print(f"Generating test logs in {log_dir}")
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    request_id = 1000
    while datetime.now() < end_time:
        # Generate request logs
        component = random.choice(["api", "web", "mobile"])
        logger = get_logger(component)
        generate_request_logs(logger, f"req_{request_id}", component)
        request_id += 1
        
        # Generate system logs
        generate_system_logs(get_logger("system"))
        
        # Random delay between requests
        time.sleep(random.uniform(0.5, 2.0))
        
    print(f"Generated logs for {duration_minutes} minutes")
    print(f"Total requests: {request_id - 1000}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test log data")
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration in minutes to generate logs"
    )
    
    args = parser.parse_args()
    generate_test_data(args.duration)
