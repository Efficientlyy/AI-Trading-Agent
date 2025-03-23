"""
On-Chain Metrics Integration Demo

This example demonstrates the on-chain metrics integration in the market sentiment
analysis system, showing how to initialize and use the blockchain clients to fetch 
on-chain data from providers like Blockchain.com and Glassnode.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.analysis_agents.sentiment.blockchain_client import (
    BlockchainClient, BlockchainComClient, GlassnodeClient
)
from src.analysis_agents.sentiment.onchain_sentiment import OnchainSentimentAgent
from src.common.config import config
from src.common.events import event_bus
from src.models.events import SentimentEvent


# Event handler for sentiment events
async def sentiment_event_handler(event: SentimentEvent):
    """Handle sentiment events."""
    logger = logging.getLogger("onchain_demo")
    logger.info(f"Received sentiment event for {event.symbol}:")
    logger.info(f"  Source: {event.source}")
    logger.info(f"  Direction: {event.sentiment_direction}")
    logger.info(f"  Value: {event.sentiment_value:.2f}")
    logger.info(f"  Confidence: {event.confidence:.2f}")
    
    # Check if this is an on-chain related event
    if 'active_addresses' in event.details:
        logger.info("  On-Chain Metrics:")
        logger.info(f"  - Active Addresses: {event.details.get('active_addresses', 'N/A')}")
        logger.info(f"  - Large Transactions: {event.details.get('large_transactions_count', 'N/A')}")
        if 'exchange_reserves_change' in event.details:
            logger.info(f"  - Exchange Reserves Change: {event.details.get('exchange_reserves_change', 'N/A'):.2f}%")


async def visualize_onchain_data(asset, metrics_data):
    """Visualize on-chain metrics data.
    
    Args:
        asset: The cryptocurrency asset symbol
        metrics_data: Dictionary of metrics data time series
    """
    logger = logging.getLogger("onchain_demo")
    logger.info(f"Visualizing on-chain metrics for {asset}")
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"On-Chain Metrics for {asset}", fontsize=16)
    
    # Plot active addresses
    if 'active_addresses' in metrics_data and len(metrics_data['active_addresses']) > 0:
        ax = axs[0, 0]
        df = pd.DataFrame(metrics_data['active_addresses'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        ax.plot(df['timestamp'], df['count'], marker='o', linestyle='-', color='blue')
        ax.set_title('Active Addresses')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Add percent change annotation
        if len(df) > 1:
            change = ((df['count'].iloc[-1] / df['count'].iloc[0]) - 1) * 100
            ax.annotate(f"Change: {change:.2f}%", 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot large transactions
    if 'large_transactions' in metrics_data and len(metrics_data['large_transactions']) > 0:
        ax = axs[0, 1]
        df = pd.DataFrame(metrics_data['large_transactions'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        ax.plot(df['timestamp'], df['volume'], marker='o', linestyle='-', color='green')
        ax.set_title('Large Transaction Volume')
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis labels with appropriate scale
        if df['volume'].max() > 1_000_000_000:
            ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1_000_000_000:.1f}B')
        elif df['volume'].max() > 1_000_000:
            ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1_000_000:.1f}M')
        
        # Add percent change annotation
        if len(df) > 1:
            change = ((df['volume'].iloc[-1] / df['volume'].iloc[0]) - 1) * 100
            ax.annotate(f"Change: {change:.2f}%", 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot hash rate (for PoW coins)
    if 'hash_rate' in metrics_data and len(metrics_data['hash_rate']) > 0:
        ax = axs[1, 0]
        df = pd.DataFrame(metrics_data['hash_rate'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        ax.plot(df['timestamp'], df['hash_rate'], marker='o', linestyle='-', color='red')
        ax.set_title('Hash Rate')
        ax.set_ylabel('EH/s')
        ax.grid(True, alpha=0.3)
        
        # Add percent change annotation
        if len(df) > 1:
            change = ((df['hash_rate'].iloc[-1] / df['hash_rate'].iloc[0]) - 1) * 100
            ax.annotate(f"Change: {change:.2f}%", 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot exchange reserves
    if 'exchange_reserves' in metrics_data and len(metrics_data['exchange_reserves']) > 0:
        ax = axs[1, 1]
        df = pd.DataFrame(metrics_data['exchange_reserves'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        ax.plot(df['timestamp'], df['reserves'], marker='o', linestyle='-', color='purple')
        ax.set_title('Exchange Reserves')
        ax.set_ylabel('Amount')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis labels with appropriate scale
        if df['reserves'].max() > 1_000_000:
            ax.yaxis.set_major_formatter(lambda x, pos: f'{x/1_000_000:.1f}M')
        
        # Add percent change annotation
        if len(df) > 1:
            change = ((df['reserves'].iloc[-1] / df['reserves'].iloc[0]) - 1) * 100
            color = 'red' if change > 0 else 'green'  # Exchange outflows (negative change) are bullish
            ax.annotate(f"Change: {change:.2f}%", 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     fontsize=12,
                     color=color,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{asset}_onchain_metrics.png')
    logger.info(f"Chart saved to {asset}_onchain_metrics.png")


async def run_onchain_metrics_demo():
    """Run the on-chain metrics demo."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("onchain_demo")
    logger.info("Starting on-chain metrics demo")
    
    # Subscribe to sentiment events to monitor them
    event_bus.subscribe("sentiment_event", sentiment_event_handler)
    
    # Get API keys from config
    blockchain_com_api_key = config.get("sentiment.apis.blockchain_com.api_key", "")
    glassnode_api_key = config.get("sentiment.apis.glassnode.api_key", "")
    
    # Create blockchain client
    client = BlockchainClient(
        blockchain_com_api_key=blockchain_com_api_key,
        glassnode_api_key=glassnode_api_key
    )
    logger.info("Initialized blockchain client")
    
    try:
        # Assets to analyze
        assets = ["BTC", "ETH"]
        
        # Time periods
        time_periods = ["24h", "7d"]
        
        for asset in assets:
            logger.info(f"Fetching on-chain metrics for {asset}")
            
            # Collect metrics data for visualization
            metrics_data = {
                'active_addresses': [],
                'large_transactions': [],
                'hash_rate': [],
                'exchange_reserves': []
            }
            
            # Fetch data for different time periods
            for period in time_periods:
                # Fetch active addresses
                active_addr_data = await client.get_active_addresses(
                    asset=asset,
                    time_period=period
                )
                metrics_data['active_addresses'].append(active_addr_data)
                
                logger.info(f"Active addresses ({period}): {active_addr_data.get('count', 'N/A'):,}")
                logger.info(f"Change: {active_addr_data.get('change_percentage', 'N/A'):.2f}%")
                
                # Fetch large transactions
                large_tx_data = await client.get_large_transactions(
                    asset=asset,
                    time_period=period
                )
                metrics_data['large_transactions'].append(large_tx_data)
                
                logger.info(f"Large transactions ({period}): {large_tx_data.get('count', 'N/A'):,}")
                logger.info(f"Volume: ${large_tx_data.get('volume', 'N/A'):,.2f}")
                
                # Fetch hash rate (for PoW coins)
                hash_rate_data = await client.get_hash_rate(
                    asset=asset,
                    time_period=period
                )
                
                if hash_rate_data:
                    metrics_data['hash_rate'].append(hash_rate_data)
                    logger.info(f"Hash rate ({period}): {hash_rate_data.get('hash_rate', 'N/A'):.2f} {hash_rate_data.get('units', 'EH/s')}")
                    logger.info(f"Change: {hash_rate_data.get('change_percentage', 'N/A'):.2f}%")
                else:
                    logger.info(f"Hash rate not available for {asset}")
                
                # Fetch exchange reserves
                exchange_data = await client.get_exchange_reserves(
                    asset=asset,
                    time_period=period
                )
                metrics_data['exchange_reserves'].append(exchange_data)
                
                logger.info(f"Exchange reserves ({period}): {exchange_data.get('reserves', 'N/A'):,.2f}")
                logger.info(f"Change: {exchange_data.get('change_percentage', 'N/A'):.2f}%")
                
                logger.info("-" * 40)
            
            # Visualize metrics
            await visualize_onchain_data(asset, metrics_data)
        
        # Create and initialize onchain sentiment agent
        logger.info("Initializing On-Chain Sentiment Agent...")
        agent = OnchainSentimentAgent("onchain")
        await agent.initialize()
        
        # Start the agent
        logger.info("Starting On-Chain Sentiment Agent...")
        await agent.start()
        
        # Let it run for a while
        logger.info("Agent running, waiting for sentiment analysis...")
        await asyncio.sleep(10)
        
        # Force analysis for BTC/USDT and ETH/USDT
        logger.info("Manually triggering on-chain sentiment analysis...")
        await agent._analyze_onchain_metrics("BTC/USDT")
        await agent._analyze_onchain_metrics("ETH/USDT")
        
        # Wait for events to be processed
        await asyncio.sleep(5)
        
        logger.info("On-chain metrics demo completed")
        
    except Exception as e:
        logger.error(f"Error in on-chain metrics demo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Close clients
        if 'client' in locals():
            await client.close()
        
        # Stop agent if it was started
        if 'agent' in locals():
            logger.info("Stopping On-Chain Sentiment Agent...")
            await agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_onchain_metrics_demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")