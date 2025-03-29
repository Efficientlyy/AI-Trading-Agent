"""
Basic validation script for sentiment analysis components.

This script provides a simple way to verify that the sentiment analysis
components are properly implemented and accessible.
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_module_exists(module_path):
    """Check if a module exists and can be imported."""
    try:
        spec = importlib.util.find_spec(module_path)
        return spec is not None
    except ModuleNotFoundError:
        return False

def main():
    """Run validation checks on sentiment analysis components."""
    print("Running Sentiment Analysis Validation")
    print("=" * 60)
    
    # Check key modules
    modules_to_check = [
        "src.analysis_agents.sentiment_analysis_manager",
        "src.analysis_agents.sentiment.sentiment_base",
        "src.analysis_agents.sentiment.social_media_sentiment",
        "src.analysis_agents.sentiment.news_sentiment",
        "src.analysis_agents.sentiment.market_sentiment",
        "src.analysis_agents.sentiment.onchain_sentiment",
        "src.analysis_agents.sentiment.sentiment_aggregator",
        "src.analysis_agents.sentiment.llm_service",
        "src.analysis_agents.sentiment.llm_sentiment_agent",
        "src.analysis_agents.sentiment.consensus_system",
    ]
    
    all_passed = True
    
    print("\nChecking for required modules:")
    for module in modules_to_check:
        exists = check_module_exists(module)
        status = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {module:60} {status}")
        if not exists:
            all_passed = False
    
    # Check if modules can be imported
    if all_passed:
        print("\nAttempting to import key modules:")
        try:
            from src.analysis_agents.sentiment_analysis_manager import SentimentAnalysisManager
            print("  SentimentAnalysisManager successfully imported")
            
            from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
            print("  BaseSentimentAgent successfully imported")
            
            from src.analysis_agents.sentiment.llm_sentiment_agent import LLMSentimentAgent
            print("  LLMSentimentAgent successfully imported")
            
            print("\nAll imports successful!")
        except ImportError as e:
            print(f"  Import error: {e}")
            all_passed = False
    
    # Final result
    print("\nValidation result:")
    if all_passed:
        print("✓ All sentiment analysis components validated successfully!")
    else:
        print("✗ Validation failed. See above for details.")
        
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
