#!/usr/bin/env python
"""
Enhanced Mock LLM Provider for System Overseer Validation

This module provides an enhanced mock LLM provider that generates more realistic
and contextually relevant responses for validation testing.
"""

import os
import sys
import json
import time
import logging
import re
from typing import Dict, List, Any, Tuple

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import System Overseer components
from system_overseer.llm_client import LLMClient, LLMMessage, LLMResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_overseer.validation.enhanced_mock")


class EnhancedMockLLMProvider:
    """Enhanced mock LLM provider for more realistic validation testing."""
    
    def __init__(self, provider_id: str = "enhanced_mock", name: str = "Enhanced Mock Provider"):
        """Initialize enhanced mock provider.
        
        Args:
            provider_id: Provider identifier
            name: Provider name
        """
        self.provider_id = provider_id
        self.name = name
        self.calls = []
        self.models = ["gpt-3.5-turbo", "gpt-4"]
        
        # Load response templates
        self.templates = self._load_templates()
        
        # Initialize conversation memory
        self.memory = {}
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load response templates.
        
        Returns:
            dict: Response templates by category
        """
        # Default templates
        templates = {
            "greeting": [
                "Hello! I'm your Trading System Overseer. I'm here to help you manage your trading system and provide insights. How can I assist you today?",
                "Hi there! I'm ready to help with your trading needs. I can provide market updates, manage your trading pairs, or adjust system settings. What would you like to do?"
            ],
            "market_status": [
                "The crypto market is currently showing {trend} movement. Bitcoin is {btc_change}% {btc_direction} in the last 24 hours, while Ethereum is {eth_change}% {eth_direction}. The overall market sentiment appears to be {sentiment}.",
                "Based on current data, the market is trending {trend}. Bitcoin price is at {btc_price} USD, {btc_direction} by {btc_change}%. The trading volume across major exchanges is {volume_status}."
            ],
            "trading_pair": [
                "I've {action} {pair} to your active trading pairs. You now have {pair_count} active pairs: {pairs_list}.",
                "Your request to {action} {pair} has been processed. Your updated trading pairs are: {pairs_list}."
            ],
            "system_status": [
                "The trading system is currently {status}. CPU usage is at {cpu}%, memory usage at {memory}%. There are {active_trades} active trades and {pending_orders} pending orders. {issues}",
                "System status: {status}. All components are functioning normally. The system has executed {trade_count} trades in the last 24 hours with a success rate of {success_rate}%. {issues}"
            ],
            "parameter_adjustment": [
                "I've updated the {param_name} to {param_value}. This change will affect {affected_component} and may result in {outcome}.",
                "Parameter {param_name} has been set to {param_value}. This {risk_level} setting will be applied to all future trades."
            ],
            "trade_info": [
                "Your recent trades include {trade_list}. The most profitable was {best_trade} with a gain of {profit}%.",
                "In the last 24 hours, you've completed {trade_count} trades. The largest position was {largest_trade} with a volume of {volume} USD."
            ],
            "error_handling": [
                "I'm sorry, but I cannot execute that command. {reason}. Please try a different approach or let me know if you need help with something else.",
                "That operation cannot be completed due to {reason}. Would you like me to suggest an alternative?"
            ]
        }
        
        return templates
    
    def _get_template_for_message(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """Get appropriate template and variables for a message.
        
        Args:
            message: User message
            
        Returns:
            tuple: Template and variables
        """
        message = message.lower()
        
        # Check for greeting
        if re.search(r'hello|hi|hey|greetings|good (morning|afternoon|evening)', message):
            template_key = "greeting"
            variables = {}
        
        # Check for market status
        elif re.search(r'market|price|trend|bitcoin|btc|eth|crypto', message):
            template_key = "market_status"
            variables = {
                "trend": self._random_choice(["upward", "downward", "sideways", "volatile"]),
                "btc_change": self._random_range(0.1, 5.0),
                "btc_direction": self._random_choice(["up", "down"]),
                "eth_change": self._random_range(0.1, 7.0),
                "eth_direction": self._random_choice(["up", "down"]),
                "sentiment": self._random_choice(["bullish", "bearish", "neutral", "cautiously optimistic"]),
                "btc_price": self._random_range(25000, 45000),
                "volume_status": self._random_choice(["above average", "below average", "steady", "increasing"])
            }
        
        # Check for trading pair management
        elif re.search(r'add|remove|pair|trading pair', message):
            action = "added" if "add" in message else "removed"
            pair = self._extract_trading_pair(message) or "ETHUSDC"
            
            template_key = "trading_pair"
            variables = {
                "action": action,
                "pair": pair,
                "pair_count": self._random_range(1, 5),
                "pairs_list": "BTCUSDC, ETHUSDC, SOLUSDC"
            }
        
        # Check for system status
        elif re.search(r'status|system|issue|problem', message):
            template_key = "system_status"
            variables = {
                "status": self._random_choice(["online", "operational", "running normally", "in maintenance mode"]),
                "cpu": self._random_range(10, 80),
                "memory": self._random_range(20, 70),
                "active_trades": self._random_range(0, 10),
                "pending_orders": self._random_range(0, 5),
                "issues": self._random_choice([
                    "No issues detected.",
                    "There's a minor latency issue with the MEXC API.",
                    "One warning was logged regarding rate limiting."
                ]),
                "trade_count": self._random_range(10, 100),
                "success_rate": self._random_range(85, 99)
            }
        
        # Check for parameter adjustment
        elif re.search(r'change|set|adjust|parameter|level|frequency', message):
            template_key = "parameter_adjustment"
            
            # Extract parameter name and value
            param_name = "risk level" if "risk" in message else "notification frequency"
            param_value = "conservative" if "conservative" in message else "high"
            
            variables = {
                "param_name": param_name,
                "param_value": param_value,
                "affected_component": "trading strategy" if param_name == "risk level" else "notification system",
                "outcome": "more cautious trading" if param_name == "risk level" else "more frequent updates",
                "risk_level": "lower risk" if param_value == "conservative" else "higher frequency"
            }
        
        # Check for trade information
        elif re.search(r'trade|profit|performance', message):
            template_key = "trade_info"
            variables = {
                "trade_list": "BTC/USDC (buy), ETH/USDC (sell), SOL/USDC (buy)",
                "best_trade": "ETH/USDC",
                "profit": self._random_range(2.5, 15.0),
                "trade_count": self._random_range(5, 20),
                "largest_trade": "BTC/USDC",
                "volume": self._random_range(1000, 10000)
            }
        
        # Error handling
        elif re.search(r'impossible|fail|error|invalid', message):
            template_key = "error_handling"
            variables = {
                "reason": self._random_choice([
                    "the operation is not supported",
                    "insufficient permissions",
                    "the requested resource is unavailable",
                    "the system is currently in read-only mode"
                ])
            }
        
        # Default to greeting for unknown messages
        else:
            template_key = "greeting"
            variables = {}
        
        # Get template
        templates = self.templates.get(template_key, self.templates["greeting"])
        template = self._random_choice(templates)
        
        return template, variables
    
    def _random_choice(self, options: List[Any]) -> Any:
        """Choose random item from list.
        
        Args:
            options: List of options
            
        Returns:
            Any: Random choice
        """
        import random
        return random.choice(options)
    
    def _random_range(self, min_val: float, max_val: float, precision: int = 2) -> float:
        """Generate random number in range.
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            precision: Decimal precision
            
        Returns:
            float: Random number
        """
        import random
        return round(random.uniform(min_val, max_val), precision)
    
    def _extract_trading_pair(self, message: str) -> str:
        """Extract trading pair from message.
        
        Args:
            message: User message
            
        Returns:
            str: Trading pair or None
        """
        # Common trading pairs
        pairs = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "DOGEUSDC", "ADAUSDC", "DOTUSDC"]
        
        # Check for pairs in message
        for pair in pairs:
            if pair in message.upper():
                return pair
        
        return None
    
    def _format_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Format template with variables.
        
        Args:
            template: Template string
            variables: Variables for template
            
        Returns:
            str: Formatted string
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in template: {e}")
            return template
    
    def generate_response(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        options: Dict[str, Any] = None
    ) -> LLMResponse:
        """Generate enhanced mock response.
        
        Args:
            messages: Input messages
            max_tokens: Maximum tokens
            temperature: Temperature
            options: Additional options
            
        Returns:
            LLMResponse: Response
        """
        # Record call
        self.calls.append({
            "messages": [m.to_dict() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "options": options
        })
        
        # Get user message
        user_message = next((m for m in messages if m.role == "user"), None)
        
        if not user_message:
            # Create default message
            message = LLMMessage(
                role="assistant",
                content="I'm sorry, I didn't receive a user message to respond to."
            )
            
            return LLMResponse(
                message=message,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": len(message.content.split()),
                    "total_tokens": len(message.content.split())
                }
            )
        
        # Get template and variables
        template, variables = self._get_template_for_message(user_message.content)
        
        # Format response
        content = self._format_template(template, variables)
        
        # Create message
        message = LLMMessage(role="assistant", content=content)
        
        # Create response
        return LLMResponse(
            message=message,
            usage={
                "prompt_tokens": sum(len(m.content.split()) for m in messages),
                "completion_tokens": len(content.split()),
                "total_tokens": sum(len(m.content.split()) for m in messages) + len(content.split())
            }
        )
    
    def get_completion(
        self,
        messages: List[LLMMessage],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs  # Accept any additional kwargs
    ) -> LLMResponse:
        """Get completion from LLM.
        
        This is an alias for generate_response to maintain compatibility with LLMClient.
        
        Args:
            messages: Input messages
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Temperature
            **kwargs: Additional keyword arguments
            
        Returns:
            LLMResponse: Response
        """
        # Extract options from kwargs if needed
        options = kwargs.get('options', {})
        
        return self.generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            options=options
        )


# For testing
if __name__ == "__main__":
    # Create provider
    provider = EnhancedMockLLMProvider()
    
    # Test messages
    test_messages = [
        "Hello, how are you today?",
        "What's the current market status?",
        "Add ETHUSDC to my trading pairs",
        "What's the system status?",
        "Change risk level to conservative",
        "Tell me about my recent trades",
        "Execute impossible command"
    ]
    
    # Process messages
    for message_text in test_messages:
        print(f"\nUser: {message_text}")
        
        # Create message
        message = LLMMessage(role="user", content=message_text)
        
        # Get response
        response = provider.get_completion(messages=[message])
        
        # Print response
        print(f"System: {response.message.content}")
