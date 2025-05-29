#!/usr/bin/env python
"""
MEXC Trading Demo

This script demonstrates the integration of MEXC Spot V3 API with the AI Trading Agent.
It shows how to fetch real-time market data, run technical analysis, and execute trades.
"""

import asyncio
import logging
import time
from datetime import datetime
import pandas as pd
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("mexc_demo")

# Import AI Trading Agent components
from ai_trading_agent.data_acquisition.mexc_spot_v3_client import MexcSpotV3Client
from ai_trading_agent.data_acquisition.mexc_trading_connector import MexcTradingConnector
from ai_trading_agent.data_acquisition.market_data_provider import get_market_data_provider
from ai_trading_agent.agent.technical_analysis_agent import TechnicalAnalysisAgent, DataMode
from ai_trading_agent.config.mexc_config import MEXC_CONFIG, TRADING_PAIRS
from ai_trading_agent.agent.agent_definitions import AgentStatus


async def verify_api_keys():
    """Verify that API keys are configured correctly."""
    if not MEXC_CONFIG.get('API_KEY') or not MEXC_CONFIG.get('API_SECRET'):
        logger.warning("API keys not found in configuration. Some features may be limited.")
        return False
    
    # Test API key validity
    client = MexcSpotV3Client()
    try:
        # Get system status
        status = await client.get_system_status()
        logger.info(f"MEXC system status: {status}")
        
        # Try an authenticated endpoint
        if MEXC_CONFIG.get('API_KEY') and MEXC_CONFIG.get('API_SECRET'):
            try:
                account = await client.get_account_info()
                logger.info("API keys are valid. Authenticated successfully.")
                return True
            except Exception as e:
                logger.error(f"API key authentication failed: {e}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error connecting to MEXC API: {e}")
        return False
    finally:
        await client.close()


async def demonstrate_market_data():
    """Demonstrate fetching market data from MEXC."""
    logger.info("== MEXC Market Data Demonstration ==")
    
    # Initialize market data provider
    market_data_provider = get_market_data_provider(TRADING_PAIRS)
    await market_data_provider.initialize()
    
    # Get current prices
    prices = await market_data_provider.get_current_prices()
    logger.info(f"Current prices: {json.dumps(prices, indent=2)}")
    
    # Get historical data
    logger.info("Fetching historical data for all trading pairs...")
    for interval in ["1m", "5m", "1h"]:
        market_data = await market_data_provider.get_market_data(interval=interval, limit=10)
        for symbol, df in market_data.items():
            if not df.empty:
                logger.info(f"{symbol} - {interval} interval - Last {len(df)} candles:")
                logger.info(f"{df.tail(3)}")
    
    # Get order book
    symbol = TRADING_PAIRS[0]
    orderbook = await market_data_provider.get_orderbook(symbol, 5)
    logger.info(f"Order book for {symbol} (top 5):")
    if "bids" in orderbook and "asks" in orderbook:
        logger.info(f"Top bids: {orderbook['bids'][:5]}")
        logger.info(f"Top asks: {orderbook['asks'][:5]}")
    
    # Clean up
    await market_data_provider.close()


async def demonstrate_technical_analysis():
    """Demonstrate technical analysis using real MEXC data."""
    logger.info("== Technical Analysis with MEXC Data Demonstration ==")
    
    # Create technical analysis agent
    agent = TechnicalAnalysisAgent(
        agent_id_suffix="mexc_demo",
        name="MEXC Technical Analyzer",
        symbols=TRADING_PAIRS,
        data_mode=DataMode.REAL
    )
    
    # Process data and generate signals
    logger.info("Processing market data with technical analysis agent...")
    signals = agent.process()
    
    if signals:
        logger.info(f"Generated {len(signals)} trading signals:")
        for i, signal in enumerate(signals):
            logger.info(f"Signal {i+1}: {signal}")
    else:
        logger.info("No trading signals generated.")
    
    # Get technical state
    tech_state = agent.get_technical_state()
    logger.info("Technical Analysis State:")
    
    # Display indicators for the first symbol
    if "indicators" in tech_state and tech_state["indicators"]:
        symbol = TRADING_PAIRS[0]
        if symbol in tech_state["indicators"]:
            logger.info(f"Indicators for {symbol}:")
            indicators = tech_state["indicators"][symbol]
            # Display a few key indicators
            for indicator_name in ["rsi", "macd", "bollinger_bands", "stochastic"]:
                if indicator_name in indicators:
                    logger.info(f"  {indicator_name.upper()}: {indicators[indicator_name][-1] if len(indicators[indicator_name]) > 0 else 'N/A'}")
    
    # Display patterns
    if "patterns" in tech_state and tech_state["patterns"]:
        logger.info("Detected patterns:")
        for symbol, patterns in tech_state["patterns"].items():
            if patterns:
                logger.info(f"  {symbol}: {patterns}")
    
    # Get metrics
    metrics = agent.get_component_metrics()
    logger.info(f"Agent metrics: {json.dumps(metrics['agent'], indent=2)}")


async def demonstrate_toggling_data_source():
    """Demonstrate toggling between mock and real data."""
    logger.info("== Data Source Toggling Demonstration ==")
    
    # Create technical analysis agent with mock data initially
    agent = TechnicalAnalysisAgent(
        agent_id_suffix="toggle_demo",
        name="Data Toggle Demo Agent",
        symbols=TRADING_PAIRS,
        data_mode=DataMode.MOCK
    )
    
    # Process with mock data
    logger.info("Processing with MOCK data...")
    mock_signals = agent.process()
    logger.info(f"Generated {len(mock_signals) if mock_signals else 0} signals with mock data")
    
    # Toggle to real data
    logger.info("Toggling to REAL data...")
    data_mode = agent.toggle_data_source()
    logger.info(f"Data mode is now: {data_mode}")
    
    # Process with real data
    logger.info("Processing with REAL data...")
    real_signals = agent.process()
    logger.info(f"Generated {len(real_signals) if real_signals else 0} signals with real data")
    
    # Compare results
    logger.info("Comparison of mock vs real data processing:")
    mock_count = len(mock_signals) if mock_signals else 0
    real_count = len(real_signals) if real_signals else 0
    logger.info(f"  Mock data signals: {mock_count}")
    logger.info(f"  Real data signals: {real_count}")
    logger.info(f"  Difference: {abs(mock_count - real_count)} signals")


async def run_demo():
    """Run the complete MEXC integration demonstration."""
    logger.info("Starting MEXC Trading Integration Demo")
    logger.info(f"Using trading pairs: {TRADING_PAIRS}")
    
    # Check configuration
    api_keys_valid = await verify_api_keys()
    
    # Market data demo
    await demonstrate_market_data()
    
    # Technical analysis demo
    await demonstrate_technical_analysis()
    
    # Data source toggling demo
    await demonstrate_toggling_data_source()
    
    logger.info("MEXC Trading Integration Demo completed successfully")


if __name__ == "__main__":
    # Check for API keys file
    api_keys_path = Path(__file__).parent / 'api_keys.json'
    if not api_keys_path.exists():
        logger.warning(
            f"api_keys.json not found. Please create this file based on api_keys.json.template "
            f"and add your MEXC API keys to enable full functionality."
        )
    
    # Run the demo
    asyncio.run(run_demo())
