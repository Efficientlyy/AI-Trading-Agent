"""
File existence check for sentiment analysis components.

This script checks if the necessary sentiment analysis files exist
in the expected directory structure.
"""

import os
from pathlib import Path

def main():
    """Check for the existence of sentiment analysis files."""
    print("Checking Sentiment Analysis Files")
    print("=" * 50)
    
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    src_dir = base_dir / "src"
    sentiment_dir = src_dir / "analysis_agents" / "sentiment"
    
    # Files to check
    files_to_check = [
        src_dir / "analysis_agents" / "sentiment_analysis_manager.py",
        sentiment_dir / "sentiment_base.py",
        sentiment_dir / "social_media_sentiment.py",
        sentiment_dir / "news_sentiment.py",
        sentiment_dir / "market_sentiment.py",
        sentiment_dir / "onchain_sentiment.py",
        sentiment_dir / "sentiment_aggregator.py",
        sentiment_dir / "llm_service.py",
        sentiment_dir / "llm_sentiment_agent.py",
        sentiment_dir / "consensus_system.py",
    ]
    
    all_files_exist = True
    
    # Check each file
    print("\nChecking for sentiment analysis files:")
    for file_path in files_to_check:
        exists = file_path.exists()
        status = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {file_path.relative_to(base_dir)} - {status}")
        if not exists:
            all_files_exist = False
    
    # Final result
    print("\nFile check result:")
    if all_files_exist:
        print("✓ All sentiment analysis files exist!")
    else:
        print("✗ Some files are missing. See above for details.")
    
    return 0 if all_files_exist else 1

if __name__ == "__main__":
    main()
