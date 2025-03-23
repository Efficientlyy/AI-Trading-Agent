#!/usr/bin/env python3
"""
Sentiment Analysis Implementation Verification Script

This script verifies that all necessary files for the sentiment analysis system exist.
"""

import os
import sys

def check_file(file_path):
    """Check if a file exists and print the result."""
    if os.path.isfile(file_path):
        print(f"✓ {file_path}")
        return True
    else:
        print(f"✗ {file_path}")
        return False

def main():
    """Run the verification."""
    print("=" * 80)
    print("SENTIMENT ANALYSIS VERIFICATION".center(80))
    print("=" * 80)
    
    # Determine the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Switch to base directory
    os.chdir(base_dir)
    
    print("\nChecking core implementation files...\n")
    
    # Base components
    base_components = [
        "src/analysis_agents/sentiment/sentiment_base.py",
        "src/analysis_agents/sentiment/nlp_service.py",
        "src/analysis_agents/sentiment_analysis_manager.py"
    ]
    
    # Specialized agents
    specialized_agents = [
        "src/analysis_agents/sentiment/social_media_sentiment.py",
        "src/analysis_agents/sentiment/news_sentiment.py",
        "src/analysis_agents/sentiment/market_sentiment.py",
        "src/analysis_agents/sentiment/onchain_sentiment.py",
        "src/analysis_agents/sentiment/sentiment_aggregator.py"
    ]
    
    # Special components
    special_components = [
        "src/analysis_agents/connection_engine.py",
        "src/analysis_agents/geopolitical/geopolitical_analyzer.py"
    ]
    
    # Strategies
    strategies = [
        "src/strategy/sentiment_strategy.py",
        "src/strategy/enhanced_sentiment_strategy.py"
    ]
    
    # Backtesting
    backtesting = [
        "src/backtesting/sentiment_backtester.py",
        "src/data/sentiment_collector.py"
    ]
    
    # Dashboard
    dashboard = [
        "src/dashboard/sentiment_dashboard.py",
        "dashboard_templates/sentiment_dashboard.html"
    ]
    
    # Configuration
    configuration = [
        "config/sentiment_analysis.yaml"
    ]
    
    # Documentation
    documentation = [
        "docs/SENTIMENT_ANALYSIS_IMPLEMENTATION_PLAN.md",
        "docs/SENTIMENT_ANALYSIS_SUMMARY.md",
        "docs/SENTIMENT_ANALYSIS_TESTING_PLAN.md"
    ]
    
    # Examples
    examples = [
        "examples/sentiment_analysis_demo.py",
        "examples/sentiment_backtest_example.py",
        "examples/enhanced_sentiment_strategy_demo.py",
        "examples/sentiment_real_integration_demo.py"
    ]
    
    # Check all files
    all_files = base_components + specialized_agents + special_components + strategies + backtesting + dashboard + configuration + documentation + examples
    
    results = {}
    categories = {
        "Base Components": base_components,
        "Specialized Agents": specialized_agents,
        "Special Components": special_components,
        "Strategies": strategies,
        "Backtesting": backtesting,
        "Dashboard": dashboard,
        "Configuration": configuration,
        "Documentation": documentation,
        "Examples": examples
    }
    
    # Check all files by category
    overall_result = True
    for category, files in categories.items():
        print(f"\n{category}:")
        results[category] = 0
        
        for file in files:
            if check_file(file):
                results[category] += 1
            else:
                overall_result = False
        
        # Calculate percentage
        category_pct = (results[category] / len(files)) * 100
        print(f"{results[category]}/{len(files)} files found ({category_pct:.1f}%)")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY".center(80))
    print("=" * 80)
    print("")
    
    for category, found in results.items():
        total = len(categories[category])
        pct = (found / total) * 100
        status = "COMPLETE" if found == total else "INCOMPLETE"
        print(f"{category:20}: {found}/{total} files ({pct:.1f}%) - {status}")
    
    # Overall completion
    total_files = len(all_files)
    found_files = sum(results.values())
    completion_pct = (found_files / total_files) * 100
    
    print("\n" + "=" * 80)
    print(f"OVERALL COMPLETION: {found_files}/{total_files} files ({completion_pct:.1f}%)".center(80))
    print("=" * 80)
    
    if overall_result:
        print("\nAll implementation files exist.")
        print("The sentiment analysis system appears to be properly implemented.")
    else:
        print("\nSome implementation files are missing.")
        print("The sentiment analysis system may not be fully implemented.")
    
    return 0 if overall_result else 1

if __name__ == "__main__":
    sys.exit(main())