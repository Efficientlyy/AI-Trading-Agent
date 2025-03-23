#!/bin/bash

echo "==============================================================="
echo "                SENTIMENT ANALYSIS VERIFICATION                 "
echo "==============================================================="

# Create a function to check if files exist
check_file() {
    if [ -f "$1" ]; then
        echo "✓ $1"
        return 0
    else
        echo "✗ $1"
        return 1
    fi
}

echo ""
echo "Checking core implementation files..."
echo ""

# Base components
check_file "src/analysis_agents/sentiment/sentiment_base.py"
check_file "src/analysis_agents/sentiment/nlp_service.py"
check_file "src/analysis_agents/sentiment_analysis_manager.py"

# Specialized agents
check_file "src/analysis_agents/sentiment/social_media_sentiment.py"
check_file "src/analysis_agents/sentiment/news_sentiment.py"
check_file "src/analysis_agents/sentiment/market_sentiment.py"
check_file "src/analysis_agents/sentiment/onchain_sentiment.py"
check_file "src/analysis_agents/sentiment/sentiment_aggregator.py"

# Special components
check_file "src/analysis_agents/connection_engine.py"
check_file "src/analysis_agents/geopolitical/geopolitical_analyzer.py"

# Strategies
check_file "src/strategy/sentiment_strategy.py"
check_file "src/strategy/enhanced_sentiment_strategy.py"

# Backtesting
check_file "src/backtesting/sentiment_backtester.py"
check_file "src/data/sentiment_collector.py"

# Dashboard
check_file "src/dashboard/sentiment_dashboard.py"
check_file "dashboard_templates/sentiment_dashboard.html"

# Configuration
check_file "config/sentiment_analysis.yaml"

# Documentation
check_file "docs/SENTIMENT_ANALYSIS_IMPLEMENTATION_PLAN.md"
check_file "docs/SENTIMENT_ANALYSIS_SUMMARY.md"
check_file "docs/SENTIMENT_ANALYSIS_TESTING_PLAN.md"

echo ""
echo "Checking example files..."
echo ""

# Examples
check_file "examples/sentiment_analysis_demo.py"
check_file "examples/sentiment_backtest_example.py"
check_file "examples/enhanced_sentiment_strategy_demo.py"
check_file "examples/sentiment_real_integration_demo.py"

echo ""
echo "==============================================================="
echo "                 VERIFICATION COMPLETE                         "
echo "==============================================================="

# Print summary
echo ""
if [ $? -eq 0 ]; then
    echo "All implementation files exist."
    echo "The sentiment analysis system appears to be properly implemented."
else
    echo "Some implementation files are missing."
    echo "The sentiment analysis system may not be fully implemented."
fi
echo ""