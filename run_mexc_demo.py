#!/usr/bin/env python
"""
MEXC Integration Demo

This script demonstrates the MEXC Spot V3 API integration with the AI Trading Agent.
It retrieves market data, performs technical analysis, and shows how to use the 
WebSocket API for real-time data.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mexc_demo")

# Ensure project is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_trading_agent.data_acquisition.mexc_spot_v3_client import MexcSpotV3Client
from ai_trading_agent.data_acquisition.mexc_trading_connector import MexcTradingConnector
from ai_trading_agent.config.mexc_config import MEXC_CONFIG, TRADING_PAIRS
from ai_trading_agent.agent.technical_analysis_agent import TechnicalAnalysisAgent, DataSource


async def check_api_keys():
    """Check if API keys are properly configured."""
    if not MEXC_CONFIG.get('API_KEY') or not MEXC_CONFIG.get('API_SECRET'):
        logger.warning("API keys not found or empty. Some functionality may be limited.")
        logger.info("Please set your API keys in api_keys.json or environment variables.")
        return False
    return True


async def fetch_and_display_market_data():
    """Fetch and display basic market data from MEXC."""
    logger.info("Initializing MEXC Spot V3 Client...")
    client = MexcSpotV3Client()
    
    try:
        # Check system status
        status = await client.get_system_status()
        logger.info(f"MEXC System Status: {status}")
        
        # Fetch ticker data for BTC/USDC
        symbol = "BTC/USDC"
        ticker = await client.get_ticker(symbol)
        logger.info(f"BTC/USDC Ticker: {json.dumps(ticker, indent=2)}")
        
        # Fetch recent klines (candlestick data)
        klines = await client.get_klines(symbol, "1h", limit=5)
        logger.info(f"Recent 1h Klines for {symbol}: {json.dumps(klines[:5], indent=2)}")
        
        # Fetch orderbook
        orderbook = await client.get_orderbook(symbol, limit=5)
        logger.info(f"Orderbook for {symbol} (top 5 entries):")
        logger.info(f"  Bids: {json.dumps(orderbook.get('bids', [])[:5], indent=2)}")
        logger.info(f"  Asks: {json.dumps(orderbook.get('asks', [])[:5], indent=2)}")
        
        logger.info("Basic market data fetching completed successfully.")
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
    finally:
        await client.close()


async def demonstrate_technical_analysis():
    """Demonstrate technical analysis using MEXC data."""
    logger.info("\nInitializing Technical Analysis Agent with MEXC data source...")
    
    # Create a technical analysis agent
    agent = TechnicalAnalysisAgent(
        agent_id_suffix="mexc_demo",
        name="MEXC Demo Agent",
        symbols=["BTC/USDC", "ETH/USDC"],
        data_source=DataSource.MEXC
    )
    
    try:
        # Get market data for BTC/USDC
        symbol = "BTC/USDC"
        interval = "1h"
        
        logger.info(f"Fetching {interval} market data for {symbol}...")
        df = await agent.get_market_data(symbol, interval, limit=100)
        
        if df is not None and not df.empty:
            logger.info(f"Successfully retrieved {len(df)} candles")
            logger.info(f"Data sample:\n{df.head(3)}")
            
            # Calculate some basic indicators
            if hasattr(agent, 'indicator_engine'):
                # Calculate RSI
                rsi_period = 14
                logger.info(f"Calculating RSI({rsi_period})...")
                if hasattr(agent.indicator_engine, 'calculate_rsi'):
                    rsi = agent.indicator_engine.calculate_rsi(df, period=rsi_period)
                    logger.info(f"Latest RSI({rsi_period}): {rsi.iloc[-1]:.2f}")
                
                # Calculate Moving Averages
                logger.info("Calculating Moving Averages...")
                if hasattr(agent.indicator_engine, 'calculate_sma'):
                    sma_20 = agent.indicator_engine.calculate_sma(df, period=20)
                    sma_50 = agent.indicator_engine.calculate_sma(df, period=50)
                    logger.info(f"Latest SMA(20): {sma_20.iloc[-1]:.2f}")
                    logger.info(f"Latest SMA(50): {sma_50.iloc[-1]:.2f}")
                    
                    # Simple trading signal based on MA crossover
                    if sma_20.iloc[-2] < sma_50.iloc[-2] and sma_20.iloc[-1] > sma_50.iloc[-1]:
                        logger.info("SIGNAL: Bullish MA crossover detected (SMA20 crossed above SMA50)")
                    elif sma_20.iloc[-2] > sma_50.iloc[-2] and sma_20.iloc[-1] < sma_50.iloc[-1]:
                        logger.info("SIGNAL: Bearish MA crossover detected (SMA20 crossed below SMA50)")
                    else:
                        logger.info("No MA crossover signal detected")
        else:
            logger.warning("Failed to retrieve market data")
            
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")


async def setup_trading_connector():
    """Demonstrate the trading connector functionality."""
    logger.info("\nInitializing MEXC Trading Connector...")
    connector = MexcTradingConnector()
    
    # Define a callback for ticker updates
    async def ticker_callback(symbol: str, data: dict):
        logger.info(f"Received ticker update for {symbol}: price={data.get('price')}, change={data.get('change')}%")
    
    try:
        # Register the callback
        connector.register_ticker_callback(ticker_callback)
        
        # Subscribe to ticker updates for BTC/USDC
        symbol = "BTC/USDC"
        logger.info(f"Subscribing to ticker updates for {symbol}...")
        await connector.subscribe_to_tickers([symbol])
        
        # Get initial ticker data
        ticker = await connector.get_ticker(symbol)
        logger.info(f"Initial {symbol} ticker: {json.dumps(ticker, indent=2)}")
        
        # Wait for a few ticker updates
        logger.info("Waiting for 30 seconds to receive real-time ticker updates...")
        await asyncio.sleep(30)
        
    except Exception as e:
        logger.error(f"Error in trading connector demo: {e}")
    finally:
        # Close the connector
        await connector.close()


async def run_demo():
    """Run the full MEXC integration demo."""
    logger.info("Starting MEXC Integration Demo")
    logger.info("=" * 80)
    
    # Check API keys
    have_keys = await check_api_keys()
    if not have_keys:
        logger.warning("Continuing without API keys, some functionality may be limited")
    
    # Part 1: Fetch basic market data
    logger.info("\nPART 1: Basic Market Data")
    logger.info("-" * 80)
    await fetch_and_display_market_data()
    
    # Part 2: Demonstrate technical analysis
    logger.info("\nPART 2: Technical Analysis")
    logger.info("-" * 80)
    await demonstrate_technical_analysis()
    
    # Part 3: Demonstrate trading connector
    logger.info("\nPART 3: Trading Connector")
    logger.info("-" * 80)
    await setup_trading_connector()
    
    logger.info("\nMEXC Integration Demo completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        # Run the demo
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        raise
