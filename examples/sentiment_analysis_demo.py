"""
Sentiment Analysis System Demo

This example demonstrates the sentiment analysis system in action, 
showing how to initialize and run the sentiment analysis manager
and its various sentiment analysis agents.
"""

import asyncio
import logging
from datetime import datetime

from src.analysis_agents.sentiment_analysis_manager import SentimentAnalysisManager
from src.common.config import config
# Use standard logging configuration since setup_logging might not be available
from src.models.market_data import CandleData, TimeFrame


async def run_sentiment_demo():
    """Run the sentiment analysis system demo."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("sentiment_demo")
    logger.info("Starting sentiment analysis demo")
    
    # Create sentiment analysis manager
    manager = SentimentAnalysisManager()
    
    # Initialize and start the manager
    await manager.initialize()
    await manager.start()
    
    try:
        # Wait for the sentiment system to process some data
        logger.info("Sentiment analysis system is running...")
        logger.info("Processing sentiment from various sources for configured symbols")
        logger.info("This may take a few moments as data is collected and analyzed")
        
        # Allow the system to run for a short time
        await asyncio.sleep(10)
        
        # Display information about active agents
        agents = manager.get_all_agents()
        logger.info(f"Active sentiment agents: {len(agents)}")
        for agent in agents:
            logger.info(f"  - {agent.name}: {agent.__class__.__name__}")
        
        # Run for a specified time
        demo_runtime = 60  # seconds
        logger.info(f"Sentiment analysis system will run for {demo_runtime} seconds")
        logger.info("Check logs for sentiment events being published")
        
        # Wait for the specified time
        await asyncio.sleep(demo_runtime)
        
        # Generate some sample candle data for market analysis
        sample_candles = generate_sample_candles("BTC/USDT", 30)
        logger.info(f"Generated {len(sample_candles)} sample candles for market analysis")
        
        # Analyze market data using sentiment agents
        logger.info("Analyzing market data with sentiment agents...")
        for agent in agents:
            await agent.analyze_market_data(
                symbol="BTC/USDT",
                exchange="Binance",
                timeframe=TimeFrame("1h"),
                candles=sample_candles
            )
        
        # Force aggregation for the Aggregator agent if available
        aggregator = manager.get_agent("aggregator")
        if aggregator:
            # Check if it's specifically the SentimentAggregator type
            from src.analysis_agents.sentiment.sentiment_aggregator import SentimentAggregator
            if isinstance(aggregator, SentimentAggregator):
                logger.info("Running final sentiment aggregation...")
                # Use the public method to aggregate sentiment
                result = await aggregator.aggregate_sentiment("BTC/USDT")
                if result:
                    logger.info(f"Aggregated sentiment for BTC/USDT: {result['direction']} "
                              f"(value: {result['value']:.2f}, confidence: {result['confidence']:.2f})")
            
        logger.info("Sentiment analysis demo completed")
        
    finally:
        # Stop the manager
        logger.info("Stopping sentiment analysis system...")
        await manager.stop()
        logger.info("Sentiment analysis system stopped")


def generate_sample_candles(symbol: str, count: int) -> list[CandleData]:
    """Generate sample candle data for testing.
    
    Args:
        symbol: The trading pair symbol
        count: The number of candles to generate
        
    Returns:
        List of sample candle data
    """
    import random
    from datetime import datetime, timedelta
    
    candles = []
    now = datetime.utcnow()
    
    # Start price
    price = 50000.0 if symbol.startswith("BTC") else 3000.0
    
    for i in range(count):
        # Generate random price movement
        change_pct = random.uniform(-0.02, 0.02)
        close_price = price * (1 + change_pct)
        
        # Generate candle with some randomness
        high_price = max(price, close_price) * random.uniform(1.0, 1.01)
        low_price = min(price, close_price) * random.uniform(0.99, 1.0)
        
        # Create candle
        candle = CandleData(
            symbol=symbol,
            exchange="Binance",
            timeframe=TimeFrame("1h"),
            timestamp=now - timedelta(hours=count-i),
            open=price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=random.uniform(100, 1000)
        )
        
        candles.append(candle)
        
        # Update price for next candle
        price = close_price
        
    return candles


if __name__ == "__main__":
    asyncio.run(run_sentiment_demo())
