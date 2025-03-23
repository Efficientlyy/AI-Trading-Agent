"""
Sentiment Analysis Integration Example

This example demonstrates how to integrate and use the sentiment analysis
system with real API clients and perform sentiment-based trading.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

from src.analysis_agents.sentiment_analysis_manager import SentimentAnalysisManager
from src.common.config import config
from src.common.events import event_bus
from src.models.events import SentimentEvent
from src.models.market_data import CandleData, TimeFrame
from src.models.signals import Signal, SignalType
from src.strategy.base_strategy import Strategy


# Event handler for sentiment events
async def sentiment_event_handler(event: SentimentEvent):
    """Handle sentiment events."""
    logger = logging.getLogger("sentiment_integration")
    logger.info(f"Received sentiment event for {event.symbol}:")
    logger.info(f"  Source: {event.source}")
    logger.info(f"  Direction: {event.sentiment_direction}")
    logger.info(f"  Value: {event.sentiment_value:.2f}")
    logger.info(f"  Confidence: {event.confidence:.2f}")
    
    # Log additional details
    if event.details:
        logger.info("  Details:")
        for key, value in event.details.items():
            logger.info(f"    {key}: {value}")


class SentimentBasedStrategy(Strategy):
    """A simple strategy based on sentiment signals.
    
    This strategy generates trading signals based on sentiment
    events received from the sentiment analysis system.
    """
    
    def __init__(self, strategy_id: str = "sentiment_strategy"):
        """Initialize the sentiment-based strategy.
        
        Args:
            strategy_id: The unique identifier for this strategy
        """
        super().__init__(strategy_id)
        self.logger = logging.getLogger("sentiment_integration.strategy")
        
        # Configuration parameters
        self.sentiment_bull_threshold = config.get(
            f"strategies.{strategy_id}.sentiment_bull_threshold", 0.6)
        self.sentiment_bear_threshold = config.get(
            f"strategies.{strategy_id}.sentiment_bear_threshold", 0.4)
        self.min_confidence = config.get(
            f"strategies.{strategy_id}.min_confidence", 0.7)
        self.contrarian_mode = config.get(
            f"strategies.{strategy_id}.contrarian_mode", False)
        
        # Sentiment cache for symbols
        self.sentiment_cache = {}
        
        # Subscribe to sentiment events
        event_bus.subscribe("sentiment_event", self.on_sentiment_event)
    
    async def on_sentiment_event(self, event: SentimentEvent):
        """Handle sentiment events.
        
        Args:
            event: The sentiment event to process
        """
        # Store sentiment data
        symbol = event.symbol
        value = event.sentiment_value
        direction = event.sentiment_direction
        confidence = event.confidence
        source = event.source
        
        # Initialize sentiment cache for this symbol if needed
        if symbol not in self.sentiment_cache:
            self.sentiment_cache[symbol] = {}
        
        # Store sentiment by source
        self.sentiment_cache[symbol][source] = {
            "value": value,
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.utcnow(),
            "details": event.details
        }
        
        # Check if we should generate a trading signal
        await self.check_for_signal(symbol)
    
    async def check_for_signal(self, symbol: str):
        """Check if we should generate a trading signal.
        
        Args:
            symbol: The trading pair symbol
        """
        # Check if we have sentiment data for this symbol
        if symbol not in self.sentiment_cache:
            return
        
        # Get sentiment sources for this symbol
        sources = self.sentiment_cache[symbol]
        
        # Check if we have enough high-confidence sources
        high_confidence_sources = {
            source: data for source, data in sources.items()
            if data["confidence"] >= self.min_confidence
        }
        
        if len(high_confidence_sources) < 1:
            return
        
        # Calculate weighted sentiment value
        total_weight = sum(data["confidence"] for data in high_confidence_sources.values())
        weighted_value = sum(
            data["value"] * data["confidence"] for data in high_confidence_sources.values()
        ) / total_weight if total_weight > 0 else 0.5
        
        # Calculate overall confidence
        overall_confidence = total_weight / len(high_confidence_sources)
        
        # Determine signal direction
        if self.contrarian_mode:
            # Contrarian mode: inverse the signal
            if weighted_value <= self.sentiment_bear_threshold:
                signal_type = SignalType.LONG
                signal_direction = "bullish (contrarian)"
            elif weighted_value >= self.sentiment_bull_threshold:
                signal_type = SignalType.SHORT
                signal_direction = "bearish (contrarian)"
            else:
                return  # No signal for neutral sentiment
        else:
            # Normal mode
            if weighted_value >= self.sentiment_bull_threshold:
                signal_type = SignalType.LONG
                signal_direction = "bullish"
            elif weighted_value <= self.sentiment_bear_threshold:
                signal_type = SignalType.SHORT
                signal_direction = "bearish"
            else:
                return  # No signal for neutral sentiment
        
        # Create and publish signal
        signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            strategy=self.strategy_id,
            confidence=overall_confidence,
            metadata={
                "sentiment_value": weighted_value,
                "sources": list(high_confidence_sources.keys()),
                "contrarian_mode": self.contrarian_mode
            }
        )
        
        # Publish the signal
        await self.publish_signal(signal)
        
        self.logger.info(f"Generated {signal_direction} signal for {symbol}")
        self.logger.info(f"  Signal type: {signal_type.name}")
        self.logger.info(f"  Sentiment value: {weighted_value:.2f}")
        self.logger.info(f"  Confidence: {overall_confidence:.2f}")
        self.logger.info(f"  Sources: {', '.join(high_confidence_sources.keys())}")


async def run_sentiment_integration():
    """Run the sentiment analysis integration example."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("sentiment_integration")
    logger.info("Starting sentiment analysis integration example")
    
    # Subscribe to sentiment events to monitor them
    event_bus.subscribe("sentiment_event", sentiment_event_handler)
    
    # Create sentiment analysis manager
    manager = SentimentAnalysisManager()
    
    # Create sentiment-based strategy
    strategy = SentimentBasedStrategy()
    
    # Initialize and start components
    await manager.initialize()
    await strategy.initialize()
    await manager.start()
    await strategy.start()
    
    try:
        # Run the example for a specified time
        runtime = 300  # 5 minutes
        logger.info(f"Running sentiment integration for {runtime} seconds")
        logger.info("Monitoring sentiment events and trading signals...")
        
        # Wait for the specified time
        await asyncio.sleep(runtime)
        
        # Display information about active agents
        agents = manager.get_all_agents()
        logger.info(f"Active sentiment agents: {len(agents)}")
        for agent in agents:
            logger.info(f"  - {agent.name}")
        
        # Display sentiment cache from strategy
        logger.info("Sentiment data collected by strategy:")
        for symbol, sources in strategy.sentiment_cache.items():
            logger.info(f"  Symbol: {symbol}")
            for source, data in sources.items():
                logger.info(f"    Source: {source}")
                logger.info(f"      Value: {data['value']:.2f}")
                logger.info(f"      Direction: {data['direction']}")
                logger.info(f"      Confidence: {data['confidence']:.2f}")
                logger.info(f"      Time: {data['timestamp']}")
        
        logger.info("Sentiment integration example completed")
    finally:
        # Stop components
        await strategy.stop()
        await manager.stop()
        logger.info("Components stopped")


async def generate_and_visualize_sentiment_data():
    """Generate and visualize sample sentiment data."""
    # Create sample data
    timestamps = pd.date_range(start='2023-01-01', periods=30, freq='D')
    
    # Social media sentiment
    social_media_sentiment = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5,
                             0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7,
                             0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # News sentiment
    news_sentiment = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4,
                     0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6,
                     0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Market sentiment
    market_sentiment = [0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
                       0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4,
                       0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6]
    
    # Aggregated sentiment
    aggregated_sentiment = [(s + n + m) / 3 for s, n, m in 
                          zip(social_media_sentiment, news_sentiment, market_sentiment)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': timestamps,
        'Social Media': social_media_sentiment,
        'News': news_sentiment,
        'Market': market_sentiment,
        'Aggregated': aggregated_sentiment
    })
    
    # Set index to Date
    df.set_index('Date', inplace=True)
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    df[['Social Media', 'News', 'Market']].plot(ax=plt.gca())
    plt.title('Source Sentiment Over Time')
    plt.ylabel('Sentiment Value (0-1)')
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    df['Aggregated'].plot(ax=plt.gca(), color='green', linewidth=2)
    plt.title('Aggregated Sentiment Over Time')
    plt.ylabel('Sentiment Value (0-1)')
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    plt.axhspan(0.6, 1.0, color='green', alpha=0.2)
    plt.axhspan(0.0, 0.4, color='red', alpha=0.2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'examples/output/sentiment'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/sentiment_analysis.png')
    
    logging.getLogger("sentiment_integration").info(f"Sentiment visualization saved to {output_dir}/sentiment_analysis.png")


if __name__ == "__main__":
    # First generate and visualize sample data
    asyncio.run(generate_and_visualize_sentiment_data())
    
    # Then run the integration example
    asyncio.run(run_sentiment_integration())