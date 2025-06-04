#!/usr/bin/env python
"""
Standalone LLM Test

This script tests the LLM decision making without dependencies on other modules.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("standalone_llm_test.log")
    ]
)

logger = logging.getLogger("standalone_llm_test")

class MockLLMOverseer:
    """Mock LLM overseer for testing"""
    
    def __init__(self, use_mock_data=True):
        """Initialize mock LLM overseer
        
        Args:
            use_mock_data: Whether to use mock data
        """
        self.use_mock_data = use_mock_data
        logger.info("Mock LLM overseer initialized")
    
    def get_strategic_decision(self, symbol, market_data):
        """Get strategic decision for a symbol
        
        Args:
            symbol: Trading pair symbol
            market_data: Market data dictionary
            
        Returns:
            dict: Strategic decision
        """
        logger.info(f"Getting strategic decision for {symbol}")
        
        # Generate mock decision based on market data
        price = market_data.get('price', 0)
        momentum = market_data.get('momentum', 0)
        
        if momentum > 0.01:
            action = "BUY"
            confidence = min(0.5 + momentum * 2, 0.9)
        elif momentum < -0.01:
            action = "SELL"
            confidence = min(0.5 + abs(momentum) * 2, 0.9)
        else:
            action = "HOLD"
            confidence = 0.7
        
        decision = {
            "action": action,
            "confidence": confidence,
            "reasoning": f"Decision based on price {price} and momentum {momentum}"
        }
        
        logger.info(f"Generated strategic decision for {symbol}: {decision}")
        return decision

def test_llm_overseer():
    """Test LLM overseer"""
    logger.info("Starting standalone LLM test")
    
    try:
        # Create LLM overseer
        llm_overseer = MockLLMOverseer()
        logger.info("LLM overseer created")
        
        # Test symbols
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        # Test with mock data for each symbol
        for symbol in symbols:
            logger.info(f"Testing LLM overseer for {symbol}")
            
            # Create mock market data
            market_data = {
                'price': 65000 if symbol.startswith("BTC") else (3000 if symbol.startswith("ETH") else 150),
                'momentum': 0.02,
                'volatility': 0.01,
                'volume': 1000000,
                'timestamp': int(time.time() * 1000)
            }
            
            # Get strategic decision
            decision = llm_overseer.get_strategic_decision(symbol, market_data)
            logger.info(f"Strategic decision for {symbol}: {decision}")
            
            # Wait between symbols
            time.sleep(1)
        
        return True
    
    except Exception as e:
        logger.error(f"Error during LLM test: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    success = test_llm_overseer()
    
    # Print result
    if success:
        print("Standalone LLM test completed successfully")
    else:
        print("Standalone LLM test failed")
