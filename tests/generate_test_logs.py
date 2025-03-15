"""
Generate test log entries for testing the log dashboard.

This script creates a series of log entries at different levels and from different
components to populate the log files for dashboard testing.
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.common.logging import get_logger, configure_logging
from src.common.health_monitoring import HealthMonitor

# Components that will generate logs
COMPONENTS = [
    "trading_engine", 
    "market_data", 
    "order_manager", 
    "portfolio", 
    "risk_manager",
    "strategy"
]

# Messages for different log levels with consistent formatting
DEBUG_MESSAGES = [
    "Processing batch of {0} items",
    "Received new data for {0}",
    "Cache hit ratio: {0:.2f}",
    "Connection pool size: {0}",
    "Thread {0} processing task {1}",
    "Memory usage: {0:.2f}MB",
]

INFO_MESSAGES = [
    "Successfully processed {0} items",
    "Market data update received for {0}",
    "User {0} logged in",
    "Completed task {0} in {1:.2f}ms",
    "Synced data with external system",
    "Started component initialization",
]

WARNING_MESSAGES = [
    "Slow query detected: {0}ms",
    "Retry attempt {0} for operation {1}",
    "High memory usage detected: {0}%",
    "API rate limit at {0}%",
    "Deprecation warning: {0} will be removed in version {1}",
    "Connection pool nearing capacity ({0}%)",
]

ERROR_MESSAGES = [
    "Failed to process item {0}: {1}",
    "Database connection error: {0}",
    "API request failed: {0}",
    "Exception in thread {0}: {1}",
    "Invalid configuration value for {0}: {1}",
    "Timeout waiting for resource: {0}",
]

def generate_random_logs(num_logs=100, time_span_hours=24):
    """
    Generate random log entries spread over a time period.
    
    Args:
        num_logs: Number of log entries to generate
        time_span_hours: Time span to spread logs over (in hours)
    """
    # Configure logging
    configure_logging()
    
    # Get component loggers
    loggers = {component: get_logger(component) for component in COMPONENTS}
    
    # Ensure log directory exists
    print("Starting log generation...")
    
    # Generate logs over time span
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=time_span_hours)
    
    for i in range(num_logs):
        # Select random component
        component = random.choice(COMPONENTS)
        logger = loggers[component]
        
        # Select random timestamp within time span
        progress = i / num_logs
        log_time = start_time + timedelta(seconds=progress * time_span_hours * 3600)
        
        # For testing, we'll set the timestamp in the log context
        log_context = {"timestamp": log_time.isoformat()}
        
        # Determine log level based on distribution
        level_rand = random.random()
        
        try:
            if level_rand < 0.6:  # 60% info
                message = random.choice(INFO_MESSAGES)
                if "{0}" in message:
                    if "{1}" in message:  # Has two parameters
                        message = message.format(
                            random.choice(["order", "trade", "market", "portfolio"]),
                            random.uniform(5.0, 100.0)
                        )
                    else:  # Has one parameter
                        message = message.format(random.randint(1, 1000))
                logger.info(message, **log_context)
                
            elif level_rand < 0.8:  # 20% debug
                message = random.choice(DEBUG_MESSAGES)
                if "{0}" in message:
                    if "{1}" in message:  # Has two parameters
                        message = message.format(
                            f"thread-{random.randint(1, 10)}",
                            random.randint(1000, 9999)
                        )
                    elif ":.2f" in message:  # Has float format
                        message = message.format(random.uniform(0.1, 0.99))
                    else:  # Has one parameter
                        message = message.format(random.randint(1, 100))
                logger.debug(message, **log_context)
                
            elif level_rand < 0.95:  # 15% warning
                message = random.choice(WARNING_MESSAGES)
                if "{0}" in message:
                    if "{1}" in message:  # Has two parameters
                        message = message.format(
                            random.randint(1, 5),
                            random.choice(["database", "api", "sync"])
                        )
                    elif "%" in message:  # Percentage format
                        message = message.format(random.randint(70, 95))
                    else:  # Has one parameter
                        message = message.format(random.randint(100, 500))
                logger.warning(message, **log_context)
                
            else:  # 5% error
                message = random.choice(ERROR_MESSAGES)
                if "{0}" in message:
                    if "{1}" in message:  # Has two parameters
                        message = message.format(
                            random.randint(1000, 9999),
                            random.choice([
                                "Resource not found",
                                "Permission denied",
                                "Invalid input",
                                "Internal error"
                            ])
                        )
                    else:  # Has one parameter
                        message = message.format(
                            random.choice([
                                "Connection refused",
                                "Timeout",
                                "Invalid data format",
                                "Authentication failed"
                            ])
                        )
                logger.error(message, **log_context)
            
            # Add request_id to some logs
            if random.random() < 0.3:
                request_id = f"req-{random.randint(1000, 9999)}"
                extra_context = {**log_context, "request_id": request_id}
                
                # Add some related logs with the same request_id
                related_count = random.randint(1, 3)
                for j in range(related_count):
                    related_component = random.choice(COMPONENTS)
                    related_logger = loggers[related_component]
                    
                    if random.random() < 0.7:
                        message = f"Processing step {j+1} for request {request_id}"
                        related_logger.info(message, **extra_context)
                    else:
                        message = f"Completed step {j+1} for request {request_id}"
                        related_logger.info(message, **extra_context)
        except Exception as e:
            print(f"Error generating log entry: {e}")
            print(f"Message template: {message}")
            continue
    
    print(f"Generated {num_logs} log entries across {len(COMPONENTS)} components")
    
    # Also record some system metrics
    health_monitor = HealthMonitor()
    print("Recording system metrics...")
    
    # Record CPU spikes
    for i in range(10):
        # Simulate CPU load varying over time
        cpu_value = random.randint(20, 95)
        # Use the simpler approach that doesn't require level argument
        from src.common.alerting import process_metric
        process_metric("system.cpu.usage", cpu_value)
        
        # Simulate memory usage varying over time
        memory_value = random.randint(30, 85)
        process_metric("system.memory.usage", memory_value)
    
    print("Log generation completed!")

if __name__ == "__main__":
    generate_random_logs(num_logs=500, time_span_hours=24)
