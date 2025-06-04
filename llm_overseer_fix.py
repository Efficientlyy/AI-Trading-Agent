#!/usr/bin/env python
"""
LLM Overseer Fix for Trading-Agent System

This module provides a wrapper and compatibility layer for the LLMOverseer class,
ensuring it works properly in production mode with no mock data.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_overseer_fix.log")
    ]
)

logger = logging.getLogger("llm_overseer_fix")

# Import the original LLM overseer module
try:
    from llm_overseer.main import LLMOverseer as OriginalLLMOverseer
    logger.info("Successfully imported original LLMOverseer")
except ImportError as e:
    logger.error(f"Failed to import original LLMOverseer: {str(e)}")
    raise

class LLMOverseer:
    """Wrapper class for LLMOverseer to provide compatibility with production mode"""
    
    def __init__(self, use_mock_data=False, **kwargs):
        """Initialize LLM overseer
        
        Args:
            use_mock_data: Whether to use mock data (ignored in production mode)
            **kwargs: Additional arguments to pass to the original LLMOverseer
        """
        # Store use_mock_data flag but ignore it in production
        self.use_mock_data = False  # Always set to False for production mode
        
        if use_mock_data:
            logger.warning("Mock data requested but disabled in production mode")
        
        # Initialize underlying LLM overseer
        self.llm_overseer = OriginalLLMOverseer(**kwargs)
        
        logger.info("LLMOverseer wrapper initialized with production mode (no mock data)")
    
    def get_strategic_decision(self, symbol, market_data):
        """Get strategic decision for a symbol
        
        Args:
            symbol: Trading pair symbol
            market_data: Market data dictionary
            
        Returns:
            dict: Strategic decision
        """
        logger.info(f"Getting strategic decision for {symbol}")
        
        try:
            # Call underlying LLM overseer
            decision = self.llm_overseer.get_strategic_decision(symbol, market_data)
            
            logger.info(f"Generated strategic decision for {symbol}: {decision}")
            return decision
        except Exception as e:
            logger.error(f"Error getting strategic decision for {symbol}: {str(e)}")
            return {"action": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def analyze_market_context(self, symbol, timeframe="1h"):
        """Analyze market context for a symbol
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            
        Returns:
            dict: Market context analysis
        """
        logger.info(f"Analyzing market context for {symbol} on {timeframe} timeframe")
        
        try:
            # Call underlying LLM overseer
            analysis = self.llm_overseer.analyze_market_context(symbol, timeframe)
            
            logger.info(f"Generated market context analysis for {symbol}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing market context for {symbol}: {str(e)}")
            return {"market_regime": "UNKNOWN", "confidence": 0.0, "analysis": f"Error: {str(e)}"}
    
    def evaluate_signal(self, symbol, signal_data):
        """Evaluate a trading signal
        
        Args:
            symbol: Trading pair symbol
            signal_data: Signal data dictionary
            
        Returns:
            dict: Signal evaluation
        """
        logger.info(f"Evaluating signal for {symbol}")
        
        try:
            # Call underlying LLM overseer
            evaluation = self.llm_overseer.evaluate_signal(symbol, signal_data)
            
            logger.info(f"Generated signal evaluation for {symbol}: {evaluation}")
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating signal for {symbol}: {str(e)}")
            return {"valid": False, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}

# Replace LLMOverseer in llm_overseer.main module
sys.modules['llm_overseer.main'].LLMOverseer = LLMOverseer
logger.info("LLMOverseer class replaced in llm_overseer.main module")

if __name__ == "__main__":
    # Test the fix
    try:
        from llm_overseer.main import LLMOverseer
        
        print("LLMOverseer import successful")
        
        # Create instance
        llm_overseer = LLMOverseer(use_mock_data=False)
        
        print("LLMOverseer instance created")
        
        # Test strategic decision
        print("Testing strategic decision for BTCUSDT...")
        decision = llm_overseer.get_strategic_decision("BTCUSDT", {"price": 65000, "momentum": 0.02, "volatility": 0.01})
        
        print(f"Strategic decision: {decision}")
    except Exception as e:
        print(f"Error testing LLMOverseer: {str(e)}")
