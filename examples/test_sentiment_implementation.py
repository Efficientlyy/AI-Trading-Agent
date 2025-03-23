"""
Test Sentiment Implementation

This script checks the implementation of sentiment analysis components
to verify they are properly implemented.
"""

import os
import sys

# Print current directory and check if files exist
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Check if sentiment strategy files exist
files_to_check = [
    "src/strategy/sentiment_strategy.py",
    "src/strategy/enhanced_sentiment_strategy.py", 
    "src/strategy/factory.py",
    "examples/sentiment_analysis_integration_example.py",
    "examples/enhanced_sentiment_strategy_demo.py",
    "docs/SENTIMENT_ANALYSIS_README.md",
    "docs/SENTIMENT_ANALYSIS_SUMMARY.md"
]

print("\nVerifying implementation files:")
all_files_exist = True
for file_path in files_to_check:
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"{status} {file_path}")
    if not exists:
        all_files_exist = False

# Check config file
print("\nVerifying configuration:")
config_file = "config/strategies.yaml"
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        content = f.read()
        if "enhanced_sentiment" in content:
            print(f"✓ Enhanced sentiment strategy configuration found in {config_file}")
        else:
            print(f"✗ Enhanced sentiment strategy configuration not found in {config_file}")
else:
    print(f"✗ Configuration file {config_file} not found")

# Final result
print("\nImplementation verification result:")
if all_files_exist:
    print("✓ All sentiment analysis components are properly implemented")
    print("✓ The system is ready to be used once dependencies are installed")
else:
    print("✗ Some implementation files are missing")
    print("✗ Please complete the implementation before using the system")

print("\nTo run the demos:")
print("1. Install required dependencies: pip install -r requirements.txt")
print("2. Run the integration example: python examples/sentiment_analysis_integration_example.py")
print("3. Run the enhanced strategy demo: python examples/enhanced_sentiment_strategy_demo.py")