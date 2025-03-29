"""Usage Statistics Dashboard Demo.

This script demonstrates how to use the Usage Statistics Dashboard to track
LLM API usage and costs.
"""

import os
import asyncio
import random
import time
from datetime import datetime, timedelta
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("usage_statistics_demo")


async def generate_mock_usage_data():
    """Generate mock usage data for the dashboard demo.
    
    This simulates API calls to different providers and models over time.
    """
    # Import the usage tracker
    from src.analysis_agents.sentiment.usage_statistics import usage_tracker
    from src.common.events import event_bus
    
    # Initialize the usage tracker
    await usage_tracker.initialize()
    
    # Models by provider
    provider_models = {
        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "azure": ["azure-gpt-4", "azure-gpt-3.5-turbo"]
    }
    
    # Operations
    operations = ["sentiment_analysis", "event_detection", "impact_assessment"]
    
    # Generate data for the past 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    current_date = start_date
    while current_date <= end_date:
        # Number of API calls for this day varies by day of week
        weekday = current_date.weekday()
        base_calls = 50 if weekday < 5 else 20  # Less on weekends
        
        # Add some randomness and a slight upward trend
        days_passed = (current_date - start_date).days
        trend_factor = 1.0 + (days_passed / 60)  # 0.5% increase per day
        
        num_calls = int(base_calls * trend_factor * random.uniform(0.8, 1.2))
        
        logger.info(f"Generating {num_calls} mock API calls for {current_date.date().isoformat()}")
        
        # Generate calls distributed throughout the day
        for _ in range(num_calls):
            # Select random provider with weights
            provider = random.choices(
                list(provider_models.keys()),
                weights=[0.6, 0.3, 0.1],  # OpenAI used more often
                k=1
            )[0]
            
            # Select random model for this provider
            model = random.choice(provider_models[provider])
            
            # Select random operation
            operation = random.choices(
                operations,
                weights=[0.6, 0.3, 0.1],  # Sentiment analysis used more often
                k=1
            )[0]
            
            # Random token counts based on model and operation
            if "gpt-4" in model or "opus" in model:
                input_tokens = random.randint(200, 800)
                output_tokens = random.randint(100, 500)
            else:
                input_tokens = random.randint(100, 500)
                output_tokens = random.randint(50, 300)
                
            if operation == "impact_assessment":
                # Impact assessments tend to be longer
                input_tokens = int(input_tokens * 1.5)
                output_tokens = int(output_tokens * 1.5)
            
            # Random success rate (95% success)
            success = random.random() < 0.95
            
            # Random latency based on model and success
            if success:
                if "gpt-4" in model or "opus" in model:
                    latency_ms = random.uniform(800, 2000)
                else:
                    latency_ms = random.uniform(300, 1200)
            else:
                latency_ms = random.uniform(1000, 5000)
            
            # Set timestamp to this day with random hour
            hour = random.randint(8, 20)  # Business hours
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            timestamp = current_date.replace(hour=hour, minute=minute, second=second)
            
            # Create API request event
            event_data = {
                "request_id": f"mock-{timestamp.isoformat()}-{random.randint(1000, 9999)}",
                "provider": provider,
                "model": model,
                "operation": operation,
                "prompt_type": operation,
                "success": success,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "timestamp": timestamp.isoformat()
            }
            
            # Publish event as if it were a real API call
            event_bus.publish("llm_api_request", event_data)
            
            # If not successful, also publish an error event
            if not success:
                error_data = {
                    "request_id": event_data["request_id"],
                    "provider": provider,
                    "model": model,
                    "operation": operation,
                    "error": "Mock API error for demonstration",
                    "status_code": random.choice([429, 500, 502, 503]),
                    "timestamp": timestamp.isoformat()
                }
                
                event_bus.publish("llm_api_error", error_data)
        
        # Save statistics after each day
        await usage_tracker.save_statistics()
        
        # Move to next day
        current_date += timedelta(days=1)
    
    logger.info("Finished generating mock usage data")


async def run_demo():
    """Run the usage statistics dashboard demo."""
    # First, generate mock data
    await generate_mock_usage_data()
    
    # Now start the dashboard
    from src.dashboard.usage_statistics_dashboard import run_dashboard
    
    logger.info("Starting the Usage Statistics Dashboard")
    logger.info("Open your browser at http://localhost:8050/dashboard/ to view the dashboard")
    logger.info("Press Ctrl+C to stop the dashboard")
    
    # Run the dashboard
    run_dashboard(host="0.0.0.0", port=8050)


if __name__ == "__main__":
    # Check if data generation is requested
    import argparse
    
    parser = argparse.ArgumentParser(description="Usage Statistics Dashboard Demo")
    parser.add_argument("--generate-only", action="store_true", help="Only generate mock data, don't run dashboard")
    args = parser.parse_args()
    
    if args.generate_only:
        # Only generate data
        asyncio.run(generate_mock_usage_data())
    else:
        # Run the full demo
        asyncio.run(run_demo())