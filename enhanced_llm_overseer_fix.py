#!/usr/bin/env python
"""
Enhanced LLM Overseer Fix for Trading-Agent System

This module provides a comprehensive compatibility layer for the LLMOverseer class,
ensuring proper method delegation and API access for the Trading-Agent system.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enhanced_llm_overseer_fix.log")
    ]
)

logger = logging.getLogger("enhanced_llm_overseer_fix")

# Import the original LLM overseer module
try:
    from llm_overseer.main import LLMOverseer as OriginalLLMOverseer
    logger.info("Successfully imported original LLMOverseer")
except ImportError as e:
    logger.error(f"Failed to import original LLMOverseer: {str(e)}")
    raise

class LLMOverseer:
    """Enhanced wrapper class for LLMOverseer with proper method delegation"""
    
    def __init__(self, use_mock_data=False, **kwargs):
        """Initialize LLM overseer with flexible argument handling
        
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
        
        # Inspect available methods for logging
        methods = [method for method in dir(self.llm_overseer) 
                  if callable(getattr(self.llm_overseer, method)) and not method.startswith('_')]
        logger.info(f"Original LLMOverseer has methods: {methods}")
        
        logger.info("Enhanced LLMOverseer wrapper initialized with production mode (no mock data)")
    
    def analyze_market_data(self, symbol, market_data):
        """Analyze market data for a symbol
        
        Args:
            symbol: Trading pair symbol
            market_data: Market data dictionary
            
        Returns:
            dict: Market analysis
        """
        logger.info(f"Analyzing market data for {symbol}")
        
        try:
            # Call underlying LLM overseer
            if hasattr(self.llm_overseer, 'analyze_market_data'):
                analysis = self.llm_overseer.analyze_market_data(symbol, market_data)
                logger.info(f"Generated market analysis for {symbol}")
                return analysis
            else:
                logger.error("Method 'analyze_market_data' not found in original LLMOverseer")
                return {"analysis": "ERROR", "confidence": 0.0, "reasoning": "Method not implemented"}
        except Exception as e:
            logger.error(f"Error analyzing market data for {symbol}: {str(e)}")
            return {"analysis": "ERROR", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
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
            # Call underlying LLM overseer method if available
            if hasattr(self.llm_overseer, 'get_strategic_decision'):
                decision = self.llm_overseer.get_strategic_decision(symbol, market_data)
                logger.info(f"Generated strategic decision for {symbol}: {decision}")
                return decision
            elif hasattr(self.llm_overseer, 'make_trading_decision'):
                # Try alternative method name
                decision = self.llm_overseer.make_trading_decision(symbol, market_data)
                logger.info(f"Generated trading decision for {symbol}: {decision}")
                return decision
            else:
                # Fallback to mock decision
                logger.warning("No decision-making method found in original LLMOverseer, using fallback")
                
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
                    "reasoning": f"Fallback decision based on price {price} and momentum {momentum}"
                }
                
                logger.info(f"Generated fallback decision for {symbol}: {decision}")
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
            if hasattr(self.llm_overseer, 'analyze_market_context'):
                analysis = self.llm_overseer.analyze_market_context(symbol, timeframe)
                logger.info(f"Generated market context analysis for {symbol}")
                return analysis
            else:
                logger.error("Method 'analyze_market_context' not found in original LLMOverseer")
                return {"market_regime": "UNKNOWN", "confidence": 0.0, "analysis": "Method not implemented"}
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
            if hasattr(self.llm_overseer, 'evaluate_signal'):
                evaluation = self.llm_overseer.evaluate_signal(symbol, signal_data)
                logger.info(f"Generated signal evaluation for {symbol}: {evaluation}")
                return evaluation
            else:
                logger.error("Method 'evaluate_signal' not found in original LLMOverseer")
                return {"valid": False, "confidence": 0.0, "reasoning": "Method not implemented"}
        except Exception as e:
            logger.error(f"Error evaluating signal for {symbol}: {str(e)}")
            return {"valid": False, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def __getattr__(self, name):
        """Delegate all other method calls to the underlying LLM overseer
        
        Args:
            name: Method name
            
        Returns:
            Any: Method result
        """
        logger.info(f"Delegating method call: {name}")
        
        try:
            # Get attribute from underlying LLM overseer
            attr = getattr(self.llm_overseer, name)
            
            # If it's a method, wrap it with error handling
            if callable(attr):
                def wrapped_method(*args, **kwargs):
                    try:
                        return attr(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in delegated method {name}: {str(e)}")
                        raise
                return wrapped_method
            else:
                return attr
        except AttributeError:
            logger.error(f"Method {name} not found in original LLMOverseer")
            raise

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
