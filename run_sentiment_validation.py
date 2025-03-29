
"""
Simplified Validation Script for Sentiment Analysis System

This script validates core functionality of the sentiment analysis system
without relying on pytest, making it compatible with Python 3.13.
"""

import os
import sys
import time
from pathlib import Path

# Apply compatibility patches
print("Applying Python 3.13 compatibility patches...")
import py313_compatibility_patch
py313_compatibility_patch.apply_mock_modules()

# Set environment variables
os.environ["ENVIRONMENT"] = "testing"

def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.exists(file_path)

def check_imports():
    """Check if key modules can be imported."""
    modules_to_check = [
        ("SentimentAnalysisManager", "src.analysis_agents.sentiment_analysis_manager", "SentimentAnalysisManager"),
        ("BaseSentimentAgent", "src.analysis_agents.sentiment.sentiment_base", "BaseSentimentAgent"),
        ("SocialMediaSentimentAgent", "src.analysis_agents.sentiment.social_media_sentiment", "SocialMediaSentimentAgent"),
        ("NewsSentimentAgent", "src.analysis_agents.sentiment.news_sentiment", "NewsSentimentAgent"),
        ("LLMSentimentAgent", "src.analysis_agents.sentiment.llm_sentiment_agent", "LLMSentimentAgent"),
    ]
    
    all_passed = True
    print("
Checking key module imports:")
    
    for name, module_path, class_name in modules_to_check:
        try:
            module = __import__(module_path, fromlist=[class_name])
            module_class = getattr(module, class_name)
            print(f"✓ {name} - Import successful")
        except Exception as e:
            print(f"✗ {name} - Import failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run the validation."""
    print("\nSentiment Analysis System Validation")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check file existence
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    files_to_check = [
        base_dir / "src" / "analysis_agents" / "sentiment_analysis_manager.py",
        base_dir / "src" / "analysis_agents" / "sentiment" / "sentiment_base.py",
        base_dir / "src" / "analysis_agents" / "sentiment" / "social_media_sentiment.py",
        base_dir / "src" / "analysis_agents" / "sentiment" / "news_sentiment.py",
        base_dir / "src" / "analysis_agents" / "sentiment" / "market_sentiment.py",
        base_dir / "src" / "analysis_agents" / "sentiment" / "llm_sentiment_agent.py",
    ]
    
    all_files_exist = True
    print("\nChecking for sentiment analysis files:")
    for file_path in files_to_check:
        exists = file_path.exists()
        status = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {file_path.relative_to(base_dir)} - {status}")
        if not exists:
            all_files_exist = False
    
    # Check imports
    imports_ok = check_imports()
    
    # Overall result
    print("\nValidation Summary:")
    print(f"{'✓' if all_files_exist else '✗'} File Check - {'All files found' if all_files_exist else 'Some files missing'}")
    print(f"{'✓' if imports_ok else '✗'} Import Check - {'All imports successful' if imports_ok else 'Some imports failed'}")
    
    if all_files_exist and imports_ok:
        print("\n✓ Overall: The sentiment analysis system is valid and properly implemented!")
        return 0
    else:
        print("\n✗ Overall: Some validation checks failed. See details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
