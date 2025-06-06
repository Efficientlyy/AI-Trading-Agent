#!/usr/bin/env python
"""
Verify Trading Signals Query Fix

This script tests the fix for trading signals query detection in the Performance Analytics module.
"""

import logging
import asyncio
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("verify_trading_signals_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def verify_trading_signals_fix():
    """Verify the fix for trading signals query detection"""
    logger.info("Starting verification of trading signals query fix")
    
    try:
        # Import the fixed module (v3)
        logger.info("Importing fixed module (v3)")
        from performance_analytics_ai_integration_fixed_v3 import PerformanceAnalyticsAIIntegration
        
        # Create an instance
        logger.info("Creating integration instance")
        integration = PerformanceAnalyticsAIIntegration()
        
        # Test the previously failing query
        test_query = "performance signals"
        logger.info(f"Testing query: '{test_query}'")
        
        # Check if it's a performance query
        is_performance = integration.is_performance_query(test_query)
        logger.info(f"Is performance query: {is_performance}")
        
        # Parse the query
        parsed = integration.parse_performance_query(test_query)
        logger.info(f"Parsed query: {parsed}")
        
        # Verify the query type is correct
        if parsed.get('query_type') == 'trading_signals':
            logger.info("✅ SUCCESS: Query correctly identified as trading_signals")
            success = True
        else:
            logger.error(f"❌ FAILURE: Query incorrectly identified as {parsed.get('query_type')}")
            success = False
        
        # Test additional trading signals queries
        additional_queries = [
            "Show me trading signals performance",
            "signals",
            "signal accuracy",
            "trading accuracy",
            "prediction accuracy"
        ]
        
        logger.info("Testing additional trading signals queries")
        additional_results = []
        
        for query in additional_queries:
            logger.info(f"Testing query: '{query}'")
            parsed = integration.parse_performance_query(query)
            result = parsed.get('query_type') == 'trading_signals'
            additional_results.append(result)
            logger.info(f"Result: {result}, Parsed as: {parsed.get('query_type')}")
        
        additional_success = all(additional_results)
        if additional_success:
            logger.info("✅ SUCCESS: All additional queries correctly identified")
        else:
            logger.error("❌ FAILURE: Some additional queries incorrectly identified")
        
        # Generate a summary report
        report = f"""# Trading Signals Query Fix Verification Report

## Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Main Test Query**: "{test_query}"
- **Main Test Result**: {"✅ PASS" if success else "❌ FAIL"}
- **Additional Tests**: {"✅ All Passed" if additional_success else "❌ Some Failed"}

## Details
- **Main Query Parsed As**: {parsed.get('query_type')}
- **Expected Query Type**: trading_signals

## Additional Queries
| Query | Result | Parsed As |
|-------|--------|-----------|
"""
        
        for i, query in enumerate(additional_queries):
            result = additional_results[i]
            parsed_type = integration.parse_performance_query(query).get('query_type')
            report += f"| {query} | {'✅ PASS' if result else '❌ FAIL'} | {parsed_type} |\n"
        
        report += f"""
## Conclusion
The trading signals query fix has {"successfully" if success and additional_success else "partially"} resolved the issue.
{"All" if success and additional_success else "Some"} test queries are now correctly identified as trading signals queries.

## Next Steps
1. {"Update the production module with the fixed version" if success else "Further investigate and fix the remaining issues"}
2. Run comprehensive user simulation tests to verify end-to-end functionality
3. Deploy the updated module to production
"""
        
        # Save the report
        report_file = "trading_signals_fix_verification_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"Verification report saved to {report_file}")
        
        return success and additional_success, report_file
    
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}", exc_info=True)
        return False, None

if __name__ == "__main__":
    try:
        # Run the verification
        result, report_file = asyncio.run(verify_trading_signals_fix())
        
        # Print the result
        if result:
            print("✅ Trading signals query fix verification PASSED")
        else:
            print("❌ Trading signals query fix verification FAILED")
        
        if report_file:
            print(f"Report saved to {report_file}")
        
        # Exit with appropriate code
        exit(0 if result else 1)
    
    except Exception as e:
        logger.error(f"Error running verification: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        exit(1)
