#!/usr/bin/env python3
"""Provider Failover Mechanism demonstration.

This script demonstrates the LLM Provider Failover System by:
1. Making calls to LLM providers
2. Simulating provider failures
3. Showing automatic failover to alternative providers
4. Demonstrating fallback response caching
"""

import asyncio
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import config
from src.common.logging import setup_logging, get_logger
from src.analysis_agents.sentiment.llm_service import LLMService
from src.analysis_agents.sentiment.provider_failover import provider_failover_manager


async def demo_basic_functionality():
    """Demonstrate basic provider failover functionality."""
    logger = get_logger("examples", "provider_failover_demo")
    logger.info("Starting basic provider failover demonstration")
    
    # Initialize LLM service
    llm_service = LLMService()
    await llm_service.initialize()
    
    # Sample text for analysis
    sample_text = """
    Bitcoin's price rose 5% today after several large institutional investors announced 
    new positions in the cryptocurrency. This follows recent positive regulatory developments 
    and growing adoption in traditional finance sectors.
    """
    
    # 1. Normal operation
    logger.info("1. Testing normal operation with primary provider...")
    result1 = await llm_service.analyze_sentiment(sample_text)
    logger.info(f"Result (normal): {json.dumps(result1, indent=2)}")
    
    # Get provider health status
    health_status = llm_service.get_provider_health_status()
    logger.info(f"Provider health status: {json.dumps(health_status, indent=2)}")
    
    # 2. Simulate provider failure
    logger.info("\n2. Simulating provider failure (OpenAI)...")
    # First, get current provider from model mapping
    provider_name = "openai"  # Default to OpenAI
    
    # Manually mark the provider as unhealthy for testing
    await provider_failover_manager._mark_provider_unhealthy(
        provider_name,
        "Simulated failure for demonstration"
    )
    
    # Check updated health status
    health_status = llm_service.get_provider_health_status()
    logger.info(f"Updated provider health status after failure: {json.dumps(health_status, indent=2)}")
    
    # 3. Make request after provider failure (should failover)
    logger.info("\n3. Testing automatic failover to alternative provider...")
    sample_text2 = """
    Ethereum developers have announced a major network upgrade that will significantly 
    reduce transaction fees and increase throughput. The upgrade is scheduled for next month 
    and is expected to address scaling concerns.
    """
    
    result2 = await llm_service.analyze_sentiment(sample_text2)
    logger.info(f"Result (after failover): {json.dumps(result2, indent=2)}")
    
    # 4. Simulate all providers failing
    logger.info("\n4. Simulating all providers failing...")
    for provider in ["openai", "anthropic", "azure"]:
        await provider_failover_manager._mark_provider_unhealthy(
            provider,
            f"Simulated failure of {provider} for demonstration"
        )
    
    # Check updated health status
    health_status = llm_service.get_provider_health_status()
    logger.info(f"Updated provider health status with all providers down: {json.dumps(health_status, indent=2)}")
    
    # 5. Make request with all providers down (should use fallback)
    logger.info("\n5. Testing fallback response with all providers down...")
    result3 = await llm_service.analyze_sentiment(sample_text)  # Using same text as first request
    logger.info(f"Result (fallback): {json.dumps(result3, indent=2)}")
    
    # 6. Simulate recovery
    logger.info("\n6. Simulating provider recovery...")
    for provider in ["openai", "anthropic", "azure"]:
        await provider_failover_manager._mark_provider_healthy(provider)
    
    # Check updated health status
    health_status = llm_service.get_provider_health_status()
    logger.info(f"Updated provider health status after recovery: {json.dumps(health_status, indent=2)}")
    
    # 7. Make request after recovery
    logger.info("\n7. Testing operation after provider recovery...")
    result4 = await llm_service.analyze_sentiment(sample_text2)  # Using same text as second request
    logger.info(f"Result (after recovery): {json.dumps(result4, indent=2)}")
    
    # Clean up
    await llm_service.close()


async def demo_circuit_breaker():
    """Demonstrate circuit breaker pattern functionality."""
    logger = get_logger("examples", "provider_failover_demo")
    logger.info("\n" + "="*40)
    logger.info("Starting circuit breaker demonstration")
    
    # Initialize LLM service
    llm_service = LLMService()
    await llm_service.initialize()
    
    # 1. First, set a short circuit breaker reset time for demo purposes
    logger.info("Setting short circuit breaker reset time for demonstration")
    provider_failover_manager.circuit_breaker_reset_time = 10  # 10 seconds
    
    # 2. Mark a provider as unhealthy
    provider_name = "openai"
    logger.info(f"Marking {provider_name} as unhealthy")
    await provider_failover_manager._mark_provider_unhealthy(
        provider_name,
        "Simulated failure for circuit breaker demo"
    )
    
    # 3. Show the circuit breaker reset time
    health_status = llm_service.get_provider_health_status()
    logger.info(f"Circuit breaker reset time: {health_status[provider_name]['circuit_breaker_reset_time']}")
    
    # 4. Wait for reset time
    logger.info(f"Waiting for circuit breaker reset ({provider_failover_manager.circuit_breaker_reset_time} seconds)...")
    await asyncio.sleep(provider_failover_manager.circuit_breaker_reset_time + 2)
    
    # 5. Show the provider is still unhealthy but ready for testing
    health_status = llm_service.get_provider_health_status()
    logger.info(f"Provider status after reset time: {health_status[provider_name]['status']}")
    
    # 6. Simulate successful ping to recover
    logger.info("Simulating successful ping...")
    provider_failover_manager.recovery_threshold = 1  # Set to 1 for demo
    await provider_failover_manager._ping_provider(provider_name)
    
    # Directly mark as healthy for demo purposes
    await provider_failover_manager._mark_provider_healthy(provider_name)
    
    # 7. Show the provider is now healthy
    health_status = llm_service.get_provider_health_status()
    logger.info(f"Provider status after successful ping: {health_status[provider_name]['status']}")
    
    # Clean up
    await llm_service.close()


async def demo_model_mapping():
    """Demonstrate model mapping and alternatives."""
    logger = get_logger("examples", "provider_failover_demo")
    logger.info("\n" + "="*40)
    logger.info("Starting model mapping demonstration")
    
    # Initialize LLM service
    llm_service = LLMService()
    await llm_service.initialize()
    
    # Show the model alternatives mapping
    logger.info("Model alternatives mapping:")
    for model, alternatives in provider_failover_manager.model_alternatives.items():
        logger.info(f"{model} → {alternatives}")
    
    # Test model routing with all providers healthy
    model_to_test = "gpt-4o"
    logger.info(f"\nTesting model routing for {model_to_test} with all providers healthy...")
    provider, model = await provider_failover_manager.select_provider_for_model(model_to_test)
    logger.info(f"Selected: {provider}/{model}")
    
    # Mark OpenAI as unhealthy
    logger.info("\nMarking OpenAI as unhealthy...")
    await provider_failover_manager._mark_provider_unhealthy("openai", "Simulated failure for model mapping demo")
    
    # Test model routing with OpenAI unhealthy
    logger.info(f"Testing model routing for {model_to_test} with OpenAI unhealthy...")
    provider, model = await provider_failover_manager.select_provider_for_model(model_to_test)
    logger.info(f"Selected: {provider}/{model}")
    
    # Mark all providers as unhealthy
    logger.info("\nMarking all providers as unhealthy...")
    for provider in ["openai", "anthropic", "azure"]:
        await provider_failover_manager._mark_provider_unhealthy(provider, f"Simulated failure of {provider}")
    
    # Test model routing with all providers unhealthy
    logger.info(f"Testing model routing for {model_to_test} with all providers unhealthy...")
    provider, model = await provider_failover_manager.select_provider_for_model(model_to_test)
    logger.info(f"Selected: {provider}/{model} (last resort)")
    
    # Restore all providers
    logger.info("\nRestoring all providers to healthy...")
    for provider in ["openai", "anthropic", "azure"]:
        await provider_failover_manager._mark_provider_healthy(provider)
    
    # Clean up
    await llm_service.close()


async def main():
    """Run the provider failover demonstration."""
    # Set up logging
    setup_logging()
    logger = get_logger("examples", "provider_failover_demo")
    logger.info("Starting Provider Failover Mechanism demonstration")
    
    # Run the demos
    await demo_basic_functionality()
    await demo_circuit_breaker()
    await demo_model_mapping()
    
    logger.info("\nProvider Failover Mechanism demonstration completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate the provider failover mechanism")
    
    args = parser.parse_args()
    
    # Run the demonstration
    asyncio.run(main())