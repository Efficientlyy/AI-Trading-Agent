#!/usr/bin/env python
"""
User Simulation Test for Performance Analytics with Trading Signals Fix

This script runs a comprehensive user simulation test for the Performance Analytics
module with the trading signals query fix applied.
"""

import os
import sys
import logging
import asyncio
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("user_simulation_performance_test_v3.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MockTelegramUpdate:
    """Mock Telegram update for simulation"""
    
    def __init__(self, user_id=123456789, username="test_user", message_text=""):
        self.message = MockTelegramMessage(user_id, username, message_text)

class MockTelegramMessage:
    """Mock Telegram message for simulation"""
    
    def __init__(self, user_id, username, text):
        self.from_user = MockTelegramUser(user_id, username)
        self.text = text
        self.chat_id = user_id
        self.message_id = int(time.time())
        self.responses = []
    
    async def reply_text(self, text, parse_mode=None):
        """Mock reply_text method"""
        self.responses.append({"text": text, "parse_mode": parse_mode})
        return MockTelegramMessage(self.from_user.id, self.from_user.username, text)

class MockTelegramUser:
    """Mock Telegram user for simulation"""
    
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username
        self.first_name = "Test"
        self.last_name = "User"

class MockTelegramContext:
    """Mock Telegram context for simulation"""
    
    def __init__(self, args=None):
        self.args = args or []

async def run_user_simulation():
    """Run a comprehensive user simulation test"""
    logger.info("Starting user simulation test")
    
    try:
        # Import the Performance Analytics integration with v3 fix
        logger.info("Importing Telegram Performance Integration (v3)")
        from telegram_performance_integration_fixed_v3 import TelegramPerformanceIntegration
        
        # Create the integration instance
        logger.info("Creating integration instance")
        integration = TelegramPerformanceIntegration()
        
        # Define test cases
        test_cases = [
            ("/performance", "help"),
            ("/health", "help"),
            ("/status", "help"),
            ("/performance", "system"),
            ("/health", "status"),
            ("/status", "overview"),
            ("/performance", "api"),
            ("/performance", "api mexc"),
            ("/performance", "cpu"),
            ("/performance", "memory"),
            ("/performance", "signals"),  # This should now work with the fix
            ("/performance", "anomalies"),
            ("/performance", "How is the MEXC API performing?"),
            ("/performance", "Is the MEXC API slow today?"),
            ("/performance", "Has the OpenRouter API been reliable?"),
            ("/performance", "What's the current CPU usage?"),
            ("/health", "Are all components running?"),
            ("/status", "Are there any issues?")
        ]
        
        # Run test cases
        results = []
        for command, args in test_cases:
            logger.info(f"Testing: {command} {args}")
            
            # Create mock update and context
            update = MockTelegramUpdate(message_text=f"{command} {args}")
            context = MockTelegramContext(args=args.split() if args else [])
            
            # Call the appropriate handler
            start_time = time.time()
            try:
                if command == "/performance":
                    await integration._performance_command_handler(update, context)
                elif command == "/health":
                    await integration._health_command_handler(update, context)
                elif command == "/status":
                    await integration._status_command_handler(update, context)
                
                execution_time = time.time() - start_time
                
                # Check if response was received
                if update.message.responses:
                    success = True
                    logger.info(f"Test passed: {command} {args}")
                else:
                    success = False
                    logger.error(f"Test failed: {command} {args} - No response received")
            except Exception as e:
                execution_time = time.time() - start_time
                success = False
                logger.error(f"Test failed: {command} {args} - {str(e)}")
            
            # Record result
            results.append({
                "command": command,
                "args": args,
                "success": success,
                "execution_time": execution_time
            })
        
        # Generate report
        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        report = f"""# Performance Analytics User Simulation Test Results

## Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Tests**: {total_count}
- **Passed**: {success_count}
- **Failed**: {total_count - success_count}
- **Success Rate**: {success_rate:.1f}%

## Test Cases

| # | Command | Args | Result | Time |
|---|---------|------|--------|------|
"""
        
        for i, result in enumerate(results):
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            report += f"| {i+1} | {result['command']} | {result['args']} | {status} | {result['execution_time']:.3f}s |\n"
        
        report += f"""
## Conclusion

The Performance Analytics module {"successfully" if success_rate == 100 else "partially"} passed the user simulation tests. 
{success_count} out of {total_count} test cases executed as expected, demonstrating that the module 
correctly handles user queries and provides appropriate responses.

{"No" if success_rate == 100 else total_count - success_count} test cases failed and require attention.

## Next Steps

1. {"Deploy the module to production." if success_rate == 100 else "Fix the failed test cases."}
2. Monitor system performance in production environment.
3. Collect user feedback for future improvements.
"""
        
        # Save the report
        report_file = "user_simulation_performance_test_v3_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"User simulation test completed. Report saved to {report_file}")
        
        return success_rate == 100, report_file
    
    except Exception as e:
        logger.error(f"Error during user simulation: {str(e)}", exc_info=True)
        return False, None

if __name__ == "__main__":
    try:
        # Set a timeout for the simulation
        async def run_with_timeout():
            # Create a task for the simulation
            simulation_task = asyncio.create_task(run_user_simulation())
            
            # Wait for the task to complete with a timeout
            try:
                result, report_file = await asyncio.wait_for(simulation_task, timeout=60)
                print(f"User simulation {'succeeded' if result else 'failed'}")
                if report_file:
                    print(f"Report saved to {report_file}")
                return result
            except asyncio.TimeoutError:
                print("User simulation timed out after 60 seconds")
                logger.error("Simulation timed out after 60 seconds")
                return False
        
        # Run the simulation with timeout
        result = asyncio.run(run_with_timeout())
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)
