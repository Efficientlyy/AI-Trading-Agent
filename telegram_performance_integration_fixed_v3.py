#!/usr/bin/env python
"""
Telegram Performance Analytics Integration Module - V3 with Trading Signals Fix

This module integrates the Performance Analytics and System Health Monitoring
capabilities with the Telegram bot interface, enabling users to query system
performance and health through natural language commands.
"""

import logging
import re
import asyncio
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_performance_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramPerformanceIntegration:
    """
    Integrates performance analytics with the Telegram bot interface
    """
    
    def __init__(self, performance_analytics=None):
        """
        Initialize the Telegram Performance Integration
        
        Args:
            performance_analytics: The performance analytics module instance
        """
        self.performance_analytics = performance_analytics
        logger.info("Telegram Performance Integration initialized")
    
    async def handle_performance_command(self, command: str, args: str) -> str:
        """
        Handle a performance-related command from Telegram
        
        Args:
            command: The command (e.g., 'performance', 'health', 'status')
            args: Additional arguments provided with the command
            
        Returns:
            Response message to send back to the user
        """
        logger.info(f"Handling performance command: {command} with args: {args}")
        
        if not self.performance_analytics:
            try:
                # Use the fixed v3 version that handles trading signals correctly
                from performance_analytics_ai_integration_fixed_v3 import PerformanceAnalyticsAIIntegration
                self.performance_analytics = PerformanceAnalyticsAIIntegration()
                logger.info("Created new PerformanceAnalyticsAIIntegration instance (v3)")
            except ImportError:
                # Fall back to the v2 version if v3 is not available
                try:
                    from performance_analytics_ai_integration_fixed_v2 import PerformanceAnalyticsAIIntegration
                    self.performance_analytics = PerformanceAnalyticsAIIntegration()
                    logger.info("Created new PerformanceAnalyticsAIIntegration instance (v2)")
                except ImportError:
                    # Fall back to the original version if v2 is not available
                    from performance_analytics_ai_integration_fixed import PerformanceAnalyticsAIIntegration
                    self.performance_analytics = PerformanceAnalyticsAIIntegration()
                    logger.info("Created new PerformanceAnalyticsAIIntegration instance (original)")
        
        # Combine command and args to form a complete query
        full_query = f"{command} {args}".strip()
        
        # Handle help command
        if args.lower() == "help" or not args:
            return self._generate_help_message(command)
        
        # Process the query through the performance analytics module
        try:
            # Check if it's a performance query
            if self.performance_analytics.is_performance_query(full_query):
                # Generate response
                response = await self.performance_analytics.handle_performance_query(full_query)
                return response
            else:
                # Not recognized as a performance query
                return (
                    f"I don't recognize '{full_query}' as a valid performance query. "
                    f"Try using commands like:\n"
                    f"- /performance system\n"
                    f"- /performance api mexc\n"
                    f"- /performance cpu\n"
                    f"- /performance signals\n"
                    f"Or use /performance help for more information."
                )
        except Exception as e:
            logger.error(f"Error handling performance command: {str(e)}", exc_info=True)
            return (
                f"Sorry, I encountered an error while processing your performance query: {str(e)}. "
                f"Please try again or use a different query format."
            )
    
    def _generate_help_message(self, command: str) -> str:
        """
        Generate a help message for performance commands
        
        Args:
            command: The base command (e.g., 'performance', 'health')
            
        Returns:
            Help message
        """
        if command.lower() == "performance":
            return (
                "# Performance Analytics Commands\n\n"
                "Use the following commands to query system performance and health:\n\n"
                "## General Commands\n"
                "- `/performance system` - Overall system health and status\n"
                "- `/performance overview` - Complete performance overview\n\n"
                
                "## API Performance\n"
                "- `/performance api` - All API performance metrics\n"
                "- `/performance api mexc` - MEXC API performance\n"
                "- `/performance api openrouter` - OpenRouter API performance\n"
                "- `/performance api telegram` - Telegram API performance\n\n"
                
                "## Resource Usage\n"
                "- `/performance cpu` - CPU usage metrics\n"
                "- `/performance memory` - Memory usage metrics\n"
                "- `/performance disk` - Disk usage metrics\n"
                "- `/performance resources` - All resource usage metrics\n\n"
                
                "## Trading Signals\n"
                "- `/performance signals` - Trading signal accuracy metrics\n"
                "- `/performance signals btc` - BTC signal accuracy\n"
                "- `/performance signals eth` - ETH signal accuracy\n\n"
                
                "## Anomalies\n"
                "- `/performance anomalies` - Detected system anomalies\n\n"
                
                "## Timeframes\n"
                "Add timeframe to any command:\n"
                "- `/performance cpu last 30 minutes`\n"
                "- `/performance api mexc last 2 hours`\n"
                "- `/performance signals last day`\n\n"
                
                "You can also ask natural language questions like:\n"
                "- `/performance How is the MEXC API performing?`\n"
                "- `/performance What's the current CPU usage?`\n"
                "- `/performance Are there any anomalies in the system?`"
            )
        elif command.lower() == "health":
            return (
                "# System Health Commands\n\n"
                "Use the following commands to check system health:\n\n"
                "- `/health status` - Current system health status\n"
                "- `/health components` - Status of all system components\n"
                "- `/health issues` - Current system issues or warnings\n"
                "- `/health recommendations` - Recommendations for improving system health\n\n"
                
                "You can also ask natural language questions like:\n"
                "- `/health Is everything working correctly?`\n"
                "- `/health Are all components running?`\n"
                "- `/health What needs attention?`"
            )
        elif command.lower() == "status":
            return (
                "# System Status Commands\n\n"
                "Use the following commands to check system status:\n\n"
                "- `/status overview` - Overall system status\n"
                "- `/status api` - API connection status\n"
                "- `/status resources` - Resource usage status\n"
                "- `/status signals` - Trading signal status\n\n"
                
                "You can also ask natural language questions like:\n"
                "- `/status What's the current system status?`\n"
                "- `/status Are all APIs connected?`\n"
                "- `/status How are resources looking?`"
            )
        else:
            return (
                f"# {command.capitalize()} Commands\n\n"
                f"I don't have specific help for the '{command}' command. "
                f"Try using `/performance help` for information on performance analytics commands."
            )
    
    def register_handlers(self, dispatcher):
        """
        Register command handlers with the Telegram dispatcher
        
        Args:
            dispatcher: The Telegram dispatcher to register handlers with
        """
        from telegram.ext import CommandHandler, MessageHandler, filters
        
        # Register command handlers
        dispatcher.add_handler(CommandHandler("performance", self._performance_command_handler))
        dispatcher.add_handler(CommandHandler("health", self._health_command_handler))
        dispatcher.add_handler(CommandHandler("status", self._status_command_handler))
        
        logger.info("Registered performance command handlers with dispatcher")
    
    async def _performance_command_handler(self, update, context):
        """Telegram command handler for /performance command"""
        try:
            args = context.args
            args_text = ' '.join(args) if args else ''
            
            response = await self.handle_performance_command("performance", args_text)
            await update.message.reply_text(response, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in performance command handler: {str(e)}", exc_info=True)
            await update.message.reply_text(
                f"Sorry, I encountered an error while processing your performance command: {str(e)}. "
                f"Please try again or contact the administrator."
            )
    
    async def _health_command_handler(self, update, context):
        """Telegram command handler for /health command"""
        try:
            args = context.args
            args_text = ' '.join(args) if args else ''
            
            response = await self.handle_performance_command("health", args_text)
            await update.message.reply_text(response, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in health command handler: {str(e)}", exc_info=True)
            await update.message.reply_text(
                f"Sorry, I encountered an error while processing your health command: {str(e)}. "
                f"Please try again or contact the administrator."
            )
    
    async def _status_command_handler(self, update, context):
        """Telegram command handler for /status command"""
        try:
            args = context.args
            args_text = ' '.join(args) if args else ''
            
            response = await self.handle_performance_command("status", args_text)
            await update.message.reply_text(response, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in status command handler: {str(e)}", exc_info=True)
            await update.message.reply_text(
                f"Sorry, I encountered an error while processing your status command: {str(e)}. "
                f"Please try again or contact the administrator."
            )

# Example usage (for testing)
if __name__ == "__main__":
    try:
        # Try to import the fixed v3 version first
        from performance_analytics_ai_integration_fixed_v3 import PerformanceAnalyticsAIIntegration
        print("Using performance_analytics_ai_integration_fixed_v3")
    except ImportError:
        try:
            # Fall back to v2 if v3 is not available
            from performance_analytics_ai_integration_fixed_v2 import PerformanceAnalyticsAIIntegration
            print("Using performance_analytics_ai_integration_fixed_v2")
        except ImportError:
            # Fall back to the original version
            from performance_analytics_ai_integration_fixed import PerformanceAnalyticsAIIntegration
            print("Using performance_analytics_ai_integration_fixed")
    
    async def test_integration():
        try:
            # Create performance analytics module
            performance_analytics = PerformanceAnalyticsAIIntegration()
            
            # Create telegram integration
            telegram_integration = TelegramPerformanceIntegration(performance_analytics)
            
            # Test commands
            test_commands = [
                ("performance", "help"),
                ("performance", "system"),
                ("performance", "api mexc"),
                ("performance", "cpu"),
                ("performance", "signals"),  # This should now work with the fix
                ("performance", "How is the MEXC API performing?"),
                ("performance", "Is the MEXC API slow today?"),
                ("performance", "Has the OpenRouter API been reliable?"),
                ("health", "status"),
                ("status", "overview")
            ]
            
            for command, args in test_commands:
                print(f"\nTesting: /{command} {args}")
                print("-" * 50)
                response = await telegram_integration.handle_performance_command(command, args)
                print(response[:500] + "..." if len(response) > 500 else response)
                print("=" * 80)
        except Exception as e:
            print(f"Error in test integration: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    asyncio.run(test_integration())
