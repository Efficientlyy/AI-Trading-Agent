"""
Fear & Greed Index Integration Demo

This example demonstrates the Fear & Greed index integration in the market sentiment
analysis system, showing how to initialize and use the FearGreedClient to fetch 
sentiment data from the Alternative.me API.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis_agents.sentiment.market_sentiment import FearGreedClient, MarketSentimentAgent
from src.common.config import config
from src.common.events import event_bus
from src.models.events import SentimentEvent


# Event handler for sentiment events
async def sentiment_event_handler(event: SentimentEvent):
    """Handle sentiment events."""
    logger = logging.getLogger("fear_greed_demo")
    logger.info(f"Received sentiment event for {event.symbol}:")
    logger.info(f"  Source: {event.source}")
    logger.info(f"  Direction: {event.sentiment_direction}")
    logger.info(f"  Value: {event.sentiment_value:.2f}")
    logger.info(f"  Confidence: {event.confidence:.2f}")
    
    # Check if this is a Fear & Greed related event
    if 'fear_greed_index' in event.details:
        logger.info(f"  Fear & Greed Index: {event.details['fear_greed_index']}")
        logger.info(f"  Classification: {event.details.get('fear_greed_classification', 'N/A')}")


async def visualize_historical_data(historical_data, days=30):
    """Visualize historical Fear & Greed data.
    
    Args:
        historical_data: List of historical Fear & Greed data points
        days: Number of days to display
    """
    logger = logging.getLogger("fear_greed_demo")
    logger.info(f"Visualizing {len(historical_data)} historical Fear & Greed data points")
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Convert timestamp strings to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp (oldest first)
    df = df.sort_values('timestamp')
    
    # Prepare plot
    plt.figure(figsize=(12, 6))
    
    # Plot Fear & Greed values
    plt.plot(df['timestamp'], df['value'], marker='o', linestyle='-', color='blue')
    
    # Add colored background based on classification
    for i, classification in enumerate(df['classification'].unique()):
        mask = df['classification'] == classification
        if mask.any():
            plt.fill_between(
                df['timestamp'], 
                0, 100, 
                where=mask.values, 
                alpha=0.2,
                label=classification.capitalize()
            )
    
    # Set title and labels
    plt.title('Cryptocurrency Fear & Greed Index', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Fear & Greed Index Value', fontsize=12)
    
    # Set y-axis limits
    plt.ylim(0, 100)
    
    # Add guidelines
    plt.axhline(y=25, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=75, color='g', linestyle='--', alpha=0.3)
    
    # Add text annotations
    plt.text(df['timestamp'].iloc[0], 10, 'Extreme Fear', color='darkred')
    plt.text(df['timestamp'].iloc[0], 35, 'Fear', color='red')
    plt.text(df['timestamp'].iloc[0], 55, 'Neutral', color='black')
    plt.text(df['timestamp'].iloc[0], 85, 'Greed/Extreme Greed', color='darkgreen')
    
    # Add legend
    plt.legend(loc='upper right')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Show plot
    plt.tight_layout()
    plt.savefig('fear_greed_index.png')
    logger.info("Fear & Greed index plot saved as 'fear_greed_index.png'")


async def run_fear_greed_demo():
    """Run the Fear & Greed index demo."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("fear_greed_demo")
    logger.info("Starting Fear & Greed index demo")
    
    # Subscribe to sentiment events to monitor them
    event_bus.subscribe("sentiment_event", sentiment_event_handler)
    
    # Create Fear & Greed client directly
    client = FearGreedClient()
    logger.info("Initialized Fear & Greed client")
    
    try:
        # Fetch current Fear & Greed index
        logger.info("Fetching current Fear & Greed index...")
        current_index = await client.get_current_index()
        
        logger.info(f"Current Fear & Greed Index: {current_index['value']}")
        logger.info(f"Classification: {current_index['classification']}")
        logger.info(f"Timestamp: {current_index['timestamp']}")
        
        # Get historical data
        logger.info("Fetching historical Fear & Greed data (last 30 days)...")
        historical_data = await client.get_historical_index(days=30)
        
        logger.info(f"Retrieved {len(historical_data)} historical data points")
        
        # Display first few historical data points
        for i, data_point in enumerate(historical_data[:5]):
            logger.info(f"  {data_point['timestamp']} - Value: {data_point['value']} ({data_point['classification']})")
        
        # Visualize historical data
        logger.info("Visualizing historical Fear & Greed data...")
        await visualize_historical_data(historical_data)
        
        # Create and initialize market sentiment agent
        logger.info("Initializing Market Sentiment Agent with Fear & Greed index...")
        agent = MarketSentimentAgent("market")
        await agent.initialize()
        
        # Start the agent
        logger.info("Starting Market Sentiment Agent...")
        await agent.start()
        
        # Let it run for a while
        logger.info("Agent running, waiting for sentiment analysis...")
        await asyncio.sleep(10)
        
        # Force analysis for BTC/USDT
        logger.info("Manually triggering market sentiment analysis for BTC/USDT...")
        await agent._analyze_market_sentiment_indicators("BTC/USDT")
        
        # Wait for events to be processed
        await asyncio.sleep(5)
        
        logger.info("Fear & Greed index demo completed")
        
    except Exception as e:
        logger.error(f"Error in Fear & Greed demo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Close client
        if 'client' in locals():
            await client.close()
        
        # Stop agent if it was started
        if 'agent' in locals():
            logger.info("Stopping Market Sentiment Agent...")
            await agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_fear_greed_demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")