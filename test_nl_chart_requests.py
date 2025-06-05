#!/usr/bin/env python
"""
Natural Language Chart Request Test Suite

This script tests the natural language chart request processor with various scenarios.
"""

import logging
import os
import sys
import json
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.append('/home/ubuntu/projects/Trading-Agent')

# Import the natural language processor
from natural_language_chart_processor import NaturalLanguageChartProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nl_chart_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nl_chart_test")

class NaturalLanguageChartTester:
    """Test the natural language chart request processor."""
    
    def __init__(self):
        """Initialize the tester."""
        self.processor = NaturalLanguageChartProcessor()
        self.test_cases = self._generate_test_cases()
        self.results = {}
    
    def _generate_test_cases(self) -> Dict[str, List[str]]:
        """Generate test cases for different scenarios.
        
        Returns:
            Dict[str, List[str]]: Dictionary of test cases by category
        """
        return {
            "basic_requests": [
                "Show me a BTC chart",
                "I need to see the Bitcoin price",
                "Give me the ETH chart",
                "Can I see the Solana chart?",
                "Display the BTC/USDC chart"
            ],
            "specific_chart_types": [
                "Show me a BTC candlestick chart",
                "I need to see the ETH line chart",
                "Give me the SOL volume chart",
                "Can I see the Bitcoin candles?",
                "Display the Ethereum price line"
            ],
            "specific_intervals": [
                "Show me the 1 minute BTC chart",
                "I need to see the 15 min ETH chart",
                "Give me the hourly SOL chart",
                "Can I see the daily Bitcoin chart?",
                "Display the weekly Ethereum chart"
            ],
            "with_indicators": [
                "Show me a BTC chart with SMA",
                "I need to see the ETH chart with moving average",
                "Give me the SOL chart with RSI",
                "Can I see the Bitcoin chart with bollinger bands?",
                "Display the Ethereum chart with MACD"
            ],
            "complex_requests": [
                "Show me a 15 minute BTC candlestick chart with SMA",
                "I need to see the hourly ETH line chart with bollinger bands",
                "Give me the daily SOL volume chart with RSI",
                "Can I see the 5 minute Bitcoin candles with MACD?",
                "Display the weekly Ethereum price with moving averages"
            ],
            "usdc_focus": [
                "Show me the BTCUSDC chart",
                "I need to see the ETH/USDC price",
                "Give me the SOL USDC chart",
                "Can I see the Bitcoin USDC candles?",
                "Display the Ethereum USDC line chart"
            ],
            "ambiguous_requests": [
                "Show me the crypto chart",
                "I need to see the market",
                "Give me the latest data",
                "Can I see what's happening?",
                "Display the trading information"
            ],
            "non_chart_requests": [
                "How's the weather today?",
                "What time is it?",
                "Tell me a joke",
                "Who won the game yesterday?",
                "What's the latest news?"
            ]
        }
    
    def run_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all test cases.
        
        Returns:
            Dict[str, Dict[str, Any]]: Test results by category
        """
        logger.info("Starting natural language chart request tests")
        
        for category, test_cases in self.test_cases.items():
            logger.info(f"Testing category: {category}")
            category_results = []
            
            for test_case in test_cases:
                logger.info(f"  Testing: {test_case}")
                
                # Process the request
                is_chart, params = self.processor.process_request(test_case)
                
                # Log the result
                if is_chart:
                    logger.info(f"    Detected as chart request: {params}")
                else:
                    logger.info(f"    Not detected as chart request")
                
                # Store the result
                category_results.append({
                    "request": test_case,
                    "is_chart": is_chart,
                    "params": params
                })
            
            # Store category results
            self.results[category] = category_results
        
        logger.info("Completed natural language chart request tests")
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results.
        
        Returns:
            Dict[str, Any]: Analysis of test results
        """
        if not self.results:
            logger.warning("No test results to analyze")
            return {}
        
        analysis = {
            "total_tests": 0,
            "chart_requests_detected": 0,
            "non_chart_requests_rejected": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "accuracy": 0.0,
            "category_accuracy": {}
        }
        
        # Expected results by category
        expected_chart_categories = [
            "basic_requests", 
            "specific_chart_types", 
            "specific_intervals", 
            "with_indicators", 
            "complex_requests", 
            "usdc_focus"
        ]
        expected_non_chart_categories = [
            "non_chart_requests"
        ]
        ambiguous_category = "ambiguous_requests"
        
        # Count total tests
        for category, results in self.results.items():
            analysis["total_tests"] += len(results)
        
        # Analyze by category
        for category, results in self.results.items():
            category_total = len(results)
            category_correct = 0
            
            if category in expected_chart_categories:
                # Should be detected as chart requests
                for result in results:
                    if result["is_chart"]:
                        category_correct += 1
                        analysis["chart_requests_detected"] += 1
                    else:
                        analysis["false_negatives"] += 1
            
            elif category in expected_non_chart_categories:
                # Should not be detected as chart requests
                for result in results:
                    if not result["is_chart"]:
                        category_correct += 1
                        analysis["non_chart_requests_rejected"] += 1
                    else:
                        analysis["false_positives"] += 1
            
            elif category == ambiguous_category:
                # Ambiguous requests - no clear expectation
                for result in results:
                    if result["is_chart"]:
                        logger.info(f"Ambiguous request detected as chart: {result['request']}")
                    else:
                        logger.info(f"Ambiguous request not detected as chart: {result['request']}")
            
            # Calculate category accuracy
            if category_total > 0 and category != ambiguous_category:
                category_accuracy = category_correct / category_total
                analysis["category_accuracy"][category] = {
                    "accuracy": category_accuracy,
                    "correct": category_correct,
                    "total": category_total
                }
        
        # Calculate overall accuracy (excluding ambiguous)
        total_non_ambiguous = analysis["total_tests"] - len(self.results.get(ambiguous_category, []))
        if total_non_ambiguous > 0:
            correct_predictions = analysis["chart_requests_detected"] + analysis["non_chart_requests_rejected"]
            analysis["accuracy"] = correct_predictions / total_non_ambiguous
        
        return analysis
    
    def save_results(self, filename: str = "nl_chart_test_results.json") -> None:
        """Save test results to file.
        
        Args:
            filename: Output filename
        """
        output = {
            "results": self.results,
            "analysis": self.analyze_results()
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")

def main():
    """Run the test suite."""
    # Create output directory
    os.makedirs("./test_results", exist_ok=True)
    
    # Create and run tester
    tester = NaturalLanguageChartTester()
    tester.run_tests()
    
    # Analyze results
    analysis = tester.analyze_results()
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"Total tests: {analysis['total_tests']}")
    print(f"Chart requests detected: {analysis['chart_requests_detected']}")
    print(f"Non-chart requests rejected: {analysis['non_chart_requests_rejected']}")
    print(f"False positives: {analysis['false_positives']}")
    print(f"False negatives: {analysis['false_negatives']}")
    print(f"Overall accuracy: {analysis['accuracy']:.2%}")
    
    print("\nCategory Accuracy:")
    for category, data in analysis["category_accuracy"].items():
        print(f"  {category}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    
    # Save results
    tester.save_results("./test_results/nl_chart_test_results.json")

if __name__ == "__main__":
    main()
