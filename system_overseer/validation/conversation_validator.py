#!/usr/bin/env python
"""
Conversational Validation Script for System Overseer

This script validates the conversational capabilities of the System Overseer
by simulating real-world user interactions and testing the system's responses.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Tuple

# Add project root to Python path to fix import issues
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import System Overseer components
from system_overseer.module_registry import ModuleRegistry
from system_overseer.config_registry import ConfigRegistry
from system_overseer.event_bus import EventBus
from system_overseer.core import SystemCore
from system_overseer.llm_client import LLMClient, LLMMessage
from system_overseer.dialogue_manager import DialogueManager
from system_overseer.personality_system import PersonalitySystem

# Import enhanced mock provider
from system_overseer.validation.enhanced_mock_provider import EnhancedMockLLMProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("conversation_validation.log")
    ]
)
logger = logging.getLogger("system_overseer.validation")


class ConversationValidator:
    """Validates conversational capabilities of the System Overseer."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize conversation validator.
        
        Args:
            data_dir: Data directory
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create system components
        self.module_registry = ModuleRegistry()
        self.config_registry = ConfigRegistry(
            config_dir=os.path.join(data_dir, "config")
        )
        self.event_bus = EventBus()
        
        # Create system core
        self.system_core = SystemCore(
            config_registry=self.config_registry,
            event_bus=self.event_bus,
            data_dir=os.path.join(data_dir, "system")
        )
        
        # Create LLM client
        self.llm_client = LLMClient()
        
        # Create dialogue manager
        self.dialogue_manager = DialogueManager(
            llm_client=self.llm_client,
            data_dir=os.path.join(data_dir, "dialogue")
        )
        
        # Create personality system
        self.personality_system = PersonalitySystem(
            data_dir=os.path.join(data_dir, "personality")
        )
        
        # Register components with module registry
        self.module_registry.register_service("config_registry", self.config_registry)
        self.module_registry.register_service("event_bus", self.event_bus)
        self.module_registry.register_service("llm_client", self.llm_client)
        self.module_registry.register_service("dialogue_manager", self.dialogue_manager)
        self.module_registry.register_service("personality_system", self.personality_system)
        
        # Initialize system core
        self.system_core.initialize()
        
        # Test scenarios
        self.scenarios = [
            {
                "name": "Basic Greeting",
                "messages": [
                    "Hello, how are you today?",
                    "What can you help me with?"
                ],
                "expected_keywords": ["hello", "help", "assist"]
            },
            {
                "name": "Market Status Query",
                "messages": [
                    "How is the crypto market doing today?",
                    "What's the trend for Bitcoin?"
                ],
                "expected_keywords": ["market", "bitcoin", "trend", "price"]
            },
            {
                "name": "Trading Pair Management",
                "messages": [
                    "I want to add ETHUSDC to my trading pairs",
                    "Remove SOLUSDC from my active pairs"
                ],
                "expected_keywords": ["eth", "sol", "pair", "add", "remove"]
            },
            {
                "name": "System Status",
                "messages": [
                    "What's the current status of the trading system?",
                    "Are there any issues I should know about?"
                ],
                "expected_keywords": ["status", "system", "trading", "issue"]
            },
            {
                "name": "Parameter Adjustment",
                "messages": [
                    "I want to change the risk level to conservative",
                    "Set the notification frequency to high"
                ],
                "expected_keywords": ["risk", "conservative", "notification", "frequency", "set", "change"]
            },
            {
                "name": "Multi-turn Conversation",
                "messages": [
                    "Tell me about the latest trades",
                    "Which one was most profitable?",
                    "Why did that happen?"
                ],
                "expected_keywords": ["trade", "profit", "reason", "happen"]
            },
            {
                "name": "Error Handling",
                "messages": [
                    "Execute impossible command",
                    "Do something that will definitely fail"
                ],
                "expected_keywords": ["cannot", "unable", "error", "invalid"]
            }
        ]
        
        # Results
        self.results = []
    
    def register_enhanced_mock_provider(self):
        """Register enhanced mock LLM provider for testing."""
        # Create enhanced mock provider
        enhanced_mock_provider = EnhancedMockLLMProvider()
        
        # Register provider
        self.llm_client.register_provider(enhanced_mock_provider, is_default=True)
        
        logger.info("Registered enhanced mock LLM provider")
    
    def run_validation(self):
        """Run validation tests."""
        logger.info("Starting conversation validation")
        
        # Register enhanced mock provider
        self.register_enhanced_mock_provider()
        
        # Run scenarios
        for scenario in self.scenarios:
            logger.info(f"Running scenario: {scenario['name']}")
            
            # Create user ID for scenario
            user_id = f"test_user_{scenario['name'].lower().replace(' ', '_')}"
            
            # Process messages
            responses = []
            for message in scenario["messages"]:
                logger.info(f"User: {message}")
                
                # Process message
                response, context = self.dialogue_manager.process_user_message(
                    user_id=user_id,
                    message_text=message
                )
                
                # Log response
                logger.info(f"System: {response.message.content}")
                
                # Add to responses
                responses.append(response.message.content)
            
            # Check for expected keywords
            keywords_found = []
            for keyword in scenario["expected_keywords"]:
                found = False
                for response in responses:
                    if keyword.lower() in response.lower():
                        found = True
                        break
                
                keywords_found.append({
                    "keyword": keyword,
                    "found": found
                })
            
            # Calculate success rate
            total_keywords = len(scenario["expected_keywords"])
            found_keywords = sum(1 for kw in keywords_found if kw["found"])
            success_rate = found_keywords / total_keywords if total_keywords > 0 else 0
            
            # Add result
            self.results.append({
                "scenario": scenario["name"],
                "messages": scenario["messages"],
                "responses": responses,
                "keywords_found": keywords_found,
                "success_rate": success_rate
            })
            
            logger.info(f"Scenario success rate: {success_rate:.2%}")
        
        # Calculate overall success rate
        total_success = sum(result["success_rate"] for result in self.results)
        overall_success = total_success / len(self.results) if self.results else 0
        
        logger.info(f"Overall success rate: {overall_success:.2%}")
        
        # Return results
        return {
            "results": self.results,
            "overall_success": overall_success
        }
    
    def save_results(self, filename: str = "validation_results.json"):
        """Save validation results to file.
        
        Args:
            filename: Output filename
        """
        # Create results object
        results = {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.results,
            "overall_success": sum(result["success_rate"] for result in self.results) / len(self.results) if self.results else 0
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Validate System Overseer conversational capabilities")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--output", default="validation_results.json", help="Output file")
    args = parser.parse_args()
    
    # Create validator
    validator = ConversationValidator(data_dir=args.data_dir)
    
    # Run validation
    validator.run_validation()
    
    # Save results
    validator.save_results(args.output)


if __name__ == "__main__":
    main()
