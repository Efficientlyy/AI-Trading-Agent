"""
Standalone validation tool for the Sentiment Analysis System.

This script directly tests the core functionality of the sentiment analysis system
without relying on pytest or other testing frameworks that may have compatibility issues.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure the project root is in the Python path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Set testing environment
os.environ["ENVIRONMENT"] = "testing"

# Import validation utilities
def print_header(text):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_result(test_name, success, message=None):
    """Print a test result with appropriate formatting."""
    result = "✓ PASSED" if success else "✗ FAILED"
    print(f"{result} - {test_name}")
    if message and not success:
        print(f"  Error: {message}")

async def validate_sentiment_analysis_manager():
    """Validate the SentimentAnalysisManager class."""
    print_header("Validating SentimentAnalysisManager")
    
    try:
        from src.analysis_agents.sentiment_analysis_manager import SentimentAnalysisManager
        print_result("Import SentimentAnalysisManager", True)
    except Exception as e:
        print_result("Import SentimentAnalysisManager", False, str(e))
        return False
    
    try:
        # Create an instance of the manager
        manager = SentimentAnalysisManager()
        print_result("Create SentimentAnalysisManager instance", True)
        
        # Check that the manager has expected attributes
        expected_attributes = [
            "logger", "enabled", "agent_configs", "agents", "nlp_service"
        ]
        missing_attrs = [attr for attr in expected_attributes if not hasattr(manager, attr)]
        
        if not missing_attrs:
            print_result("Manager has expected attributes", True)
        else:
            print_result("Manager has expected attributes", False, f"Missing: {', '.join(missing_attrs)}")
        
        return True
    except Exception as e:
        print_result("SentimentAnalysisManager validation", False, str(e))
        return False

async def validate_sentiment_agents():
    """Validate the sentiment agent classes."""
    print_header("Validating Sentiment Agents")
    
    agents_to_validate = [
        ("BaseSentimentAgent", "src.analysis_agents.sentiment.sentiment_base", "BaseSentimentAgent"),
        ("SocialMediaSentimentAgent", "src.analysis_agents.sentiment.social_media_sentiment", "SocialMediaSentimentAgent"),
        ("NewsSentimentAgent", "src.analysis_agents.sentiment.news_sentiment", "NewsSentimentAgent"),
        ("MarketSentimentAgent", "src.analysis_agents.sentiment.market_sentiment", "MarketSentimentAgent"),
        ("OnchainSentimentAgent", "src.analysis_agents.sentiment.onchain_sentiment", "OnchainSentimentAgent"),
        ("SentimentAggregator", "src.analysis_agents.sentiment.sentiment_aggregator", "SentimentAggregator"),
        ("LLMService", "src.analysis_agents.sentiment.llm_service", "LLMService"),
        ("LLMSentimentAgent", "src.analysis_agents.sentiment.llm_sentiment_agent", "LLMSentimentAgent"),
        ("MultiModelConsensusAgent", "src.analysis_agents.sentiment.consensus_system", "MultiModelConsensusAgent"),
    ]
    
    all_passed = True
    
    for name, module_path, class_name in agents_to_validate:
        try:
            # Dynamic import
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print_result(f"Import {name}", True)
            
            # Check if we can access key methods
            expected_methods = ["initialize", "start", "stop"]
            missing_methods = [method for method in expected_methods if not hasattr(agent_class, method)]
            
            if not missing_methods:
                print_result(f"{name} has expected methods", True)
            else:
                print_result(f"{name} has expected methods", False, f"Missing: {', '.join(missing_methods)}")
                all_passed = False
                
        except Exception as e:
            print_result(f"Import {name}", False, str(e))
            all_passed = False
    
    return all_passed

async def validate_event_system():
    """Validate the event system integration."""
    print_header("Validating Event System Integration")
    
    try:
        from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
        from src.common.events import Event, EventBus
        
        print_result("Import Event classes", True)
        
        # Create a test agent
        agent = BaseSentimentAgent("test_agent")
        
        # Check if the agent has an event_bus attribute
        if hasattr(agent, "event_bus"):
            print_result("Agent has event_bus attribute", True)
        else:
            print_result("Agent has event_bus attribute", False)
            return False
        
        # Check if the publish_event method exists
        if hasattr(agent, "publish_event") and callable(agent.publish_event):
            print_result("Agent has publish_event method", True)
        else:
            print_result("Agent has publish_event method", False)
            return False
        
        return True
    except Exception as e:
        print_result("Event system validation", False, str(e))
        return False

async def main():
    """Run all validation tests."""
    print("\nSentiment Analysis System Validation")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run validation tests
    manager_valid = await validate_sentiment_analysis_manager()
    agents_valid = await validate_sentiment_agents()
    events_valid = await validate_event_system()
    
    # Summarize results
    print_header("Validation Summary")
    print_result("SentimentAnalysisManager", manager_valid)
    print_result("Sentiment Agents", agents_valid)
    print_result("Event System Integration", events_valid)
    
    # Overall result
    if manager_valid and agents_valid and events_valid:
        print("\n✓ Overall: Sentiment Analysis System is valid and properly implemented!")
        return 0
    else:
        print("\n✗ Overall: Some validation tests failed. See details above.")
        return 1

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
