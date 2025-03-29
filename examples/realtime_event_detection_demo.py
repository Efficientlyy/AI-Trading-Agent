"""
Real-Time Event Detection System Demo

This example demonstrates how to use the Real-Time Event Detection System
to identify and analyze market-moving events as they happen.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis_agents.sentiment.llm_service import LLMService
from src.analysis_agents.sentiment.consensus_system import MultiModelConsensusAgent
from src.analysis_agents.early_detection.realtime_detector import RealtimeEventDetector
from src.analysis_agents.early_detection.sentiment_integration import SentimentEventIntegration
from src.analysis_agents.early_detection.models import (
    EarlyEvent, EventCategory, SourceType, ConfidenceLevel, 
    ImpactMagnitude, ImpactDirection, ImpactTimeframe
)
from src.common.events import event_bus
from src.common.logging import setup_logging

# Set up logging
setup_logging(log_level=logging.INFO)
logger = logging.getLogger("realtime_event_demo")


class EventCollector:
    """Collects events for demonstration purposes."""
    
    def __init__(self):
        """Initialize the event collector."""
        self.collected_events = []
        self.collected_signals = []
    
    async def initialize(self):
        """Initialize event collection."""
        # Subscribe to events
        await event_bus.subscribe("RealtimeEventDetected", self._handle_event)
        await event_bus.subscribe("EarlyEventSignal", self._handle_signal)
    
    async def _handle_event(self, event):
        """Handle a detected event."""
        self.collected_events.append(event)
        logger.info(f"Collected event: {event.get('payload', {}).get('title', 'Unknown')}")
    
    async def _handle_signal(self, signal):
        """Handle a generated signal."""
        self.collected_signals.append(signal)
        logger.info(f"Collected signal: {signal.get('payload', {}).get('title', 'Unknown')}")
    
    def get_events(self):
        """Get collected events."""
        return self.collected_events
    
    def get_signals(self):
        """Get collected signals."""
        return self.collected_signals


async def demo_event_detection():
    """Demonstrate real-time event detection."""
    logger.info("Initializing real-time event detector...")
    detector = RealtimeEventDetector()
    await detector.initialize()
    await detector.start()
    
    # Set up event collector
    collector = EventCollector()
    await collector.initialize()
    
    # Example text data to analyze
    test_data = [
        {
            "text": "BREAKING: The Federal Reserve has announced an emergency rate cut of 75 basis points in response to market volatility. This is the largest cut since the financial crisis of 2008.",
            "source_name": "Financial Times",
            "source_type": "news",
            "source_url": "https://example.com/financial-times",
            "metadata": {
                "publisher": "Financial Times",
                "author": "John Smith",
                "title": "Fed Announces Emergency Rate Cut",
                "category": "monetary_policy"
            }
        },
        {
            "text": "JUST IN: Major cryptocurrency exchange has detected suspicious activity and has temporarily suspended all withdrawals. The company is investigating the issue and will update users shortly.",
            "source_name": "CryptoDaily",
            "source_type": "news",
            "source_url": "https://example.com/crypto-daily",
            "metadata": {
                "publisher": "CryptoDaily",
                "author": "Sarah Johnson",
                "title": "Major Exchange Suspends Withdrawals",
                "category": "market"
            }
        },
        {
            "text": "I just heard from a reliable source that there's going to be a major regulatory announcement regarding Bitcoin ETFs tomorrow. This could be the news we've been waiting for!",
            "source_name": "Twitter",
            "source_type": "social_media",
            "source_url": "https://twitter.com/user123",
            "metadata": {
                "platform": "Twitter",
                "user": "CryptoInsider",
                "followers": 50000,
                "verified": False
            }
        },
        {
            "text": "Our company has just completed a successful implementation of blockchain technology for supply chain tracking. This deployment demonstrates the real-world utility of distributed ledger technology.",
            "source_name": "Corporate Blog",
            "source_type": "official",
            "source_url": "https://example.com/blog",
            "metadata": {
                "organization": "Tech Inc.",
                "type": "press_release",
                "title": "Blockchain Implementation Success"
            }
        },
        {
            "text": "Market analysis: BTC is showing signs of a potential trend reversal with increasing volume and decreasing selling pressure. Several key technical indicators suggest a bullish pattern could be forming.",
            "source_name": "Trading View",
            "source_type": "financial_data",
            "source_url": "https://example.com/trading-view",
            "metadata": {
                "platform": "TradingView",
                "user": "AnalystPro",
                "followers": 25000,
                "verified": True
            }
        }
    ]
    
    # Process each test data item
    for data in test_data:
        logger.info(f"Processing text from {data['source_name']}...")
        await detector.process_text(
            text=data["text"],
            source_name=data["source_name"],
            source_type=data["source_type"],
            source_url=data["source_url"],
            metadata=data["metadata"]
        )
        
        # Pause between items to simulate real-time data flow
        await asyncio.sleep(2)
    
    # Wait for processing to complete
    logger.info("Waiting for event detection processing...")
    await asyncio.sleep(5)
    
    # Get the detected events
    events = collector.get_events()
    signals = collector.get_signals()
    
    # Display results
    logger.info(f"\nDetected {len(events)} events and {len(signals)} signals")
    
    if events:
        logger.info("\nDetected Events:")
        for i, event in enumerate(events):
            payload = event.get("payload", {})
            logger.info(f"{i+1}. {payload.get('title', 'Unknown')}")
            logger.info(f"   Category: {payload.get('category', 'unknown')}")
            logger.info(f"   Confidence: {payload.get('confidence', 0)}")
            
            # Show impact assessment if available
            impact = payload.get("impact_assessment")
            if impact:
                logger.info(f"   Impact: {impact.get('direction', 'unclear')} - Magnitude: {impact.get('magnitude', 0)}")
                logger.info(f"   Timeframe: {impact.get('timeframe', 'unknown')}")
                
                # Show affected assets
                assets = impact.get("assets", {})
                if assets:
                    logger.info(f"   Affected assets: {', '.join(assets.keys())}")
            
            logger.info("")
    
    if signals:
        logger.info("\nGenerated Signals:")
        for i, signal in enumerate(signals):
            payload = signal.get("payload", {})
            logger.info(f"{i+1}. {payload.get('title', 'Unknown')}")
            logger.info(f"   Priority: {payload.get('priority', 0)}")
            
            # Show recommended actions
            actions = payload.get("recommended_actions", [])
            if actions:
                for action in actions:
                    logger.info(f"   Action: {action.get('type', 'unknown')} - {action.get('description', '')}")
            
            logger.info("")
    
    # Stop the detector
    await detector.stop()


async def demo_sentiment_integration():
    """Demonstrate integration between sentiment analysis and event detection."""
    logger.info("Initializing sentiment integration...")
    
    # Create components
    detector = RealtimeEventDetector()
    consensus_agent = MultiModelConsensusAgent("consensus")
    
    # Initialize components
    await detector.initialize()
    await consensus_agent.initialize()
    
    # Create integration
    integration = SentimentEventIntegration(detector, consensus_agent)
    await integration.initialize()
    
    # Start components
    await detector.start()
    await consensus_agent.start()
    
    # Set up event collector
    collector = EventCollector()
    await collector.initialize()
    
    # Publish test sentiment events
    logger.info("Publishing test sentiment events...")
    
    # 1. Consensus disagreement event
    await event_bus.publish(
        event_type="sentiment_event",
        source="consensus_agent",
        payload={
            "symbol": "BTC/USDT",
            "direction": "neutral",
            "value": 0.52,
            "confidence": 0.87,
            "is_extreme": False,
            "signal_type": "consensus",
            "sources": ["social_media", "news", "market_sentiment", "llm"],
            "details": {
                "disagreement_level": 0.65,  # High disagreement
                "direction_counts": {
                    "bullish": 3,
                    "bearish": 2,
                    "neutral": 1
                },
                "source_types": ["social_media", "news", "market_sentiment", "llm"],
                "models": ["gpt-4o", "finbert", "distilbert"],
                "event_type": "consensus_shift"
            }
        }
    )
    
    # 2. Extreme sentiment event
    await event_bus.publish(
        event_type="sentiment_event",
        source="social_media_agent",
        payload={
            "symbol": "ETH/USDT",
            "direction": "bullish",
            "value": 0.89,  # Very bullish
            "confidence": 0.82,
            "is_extreme": True,
            "signal_type": "sentiment",
            "sources": ["social_media"],
            "details": {
                "explanation": "Extremely positive social media reactions to Ethereum updates",
                "key_points": ["ETH 2.0 progress", "increasing adoption", "institutional interest"],
                "text_source": "Twitter",
                "source_category": "social_media",
                "event_type": "extreme_sentiment"
            }
        }
    )
    
    # 3. Sentiment-price divergence event
    await event_bus.publish(
        event_type="sentiment_event",
        source="market_sentiment_agent",
        payload={
            "symbol": "SOL/USDT",
            "direction": "bearish",
            "value": 0.25,
            "confidence": 0.78,
            "is_extreme": False,
            "signal_type": "divergence",
            "sources": ["market_sentiment"],
            "details": {
                "sentiment_value": 0.25,
                "sentiment_direction": "bearish",
                "price_direction": "bullish",
                "price_change": 12.5,
                "divergence_type": "bearish_sentiment_vs_bullish_price"
            }
        }
    )
    
    # Wait for processing
    logger.info("Waiting for event processing...")
    await asyncio.sleep(10)
    
    # Get detected events and signals
    events = collector.get_events()
    signals = collector.get_signals()
    
    # Display results
    logger.info(f"\nDetected {len(events)} events and {len(signals)} signals from sentiment integration")
    
    if events:
        logger.info("\nDetected Events from Sentiment:")
        for i, event in enumerate(events):
            payload = event.get("payload", {})
            logger.info(f"{i+1}. {payload.get('title', 'Unknown')}")
            logger.info(f"   Category: {payload.get('category', 'unknown')}")
            logger.info(f"   Confidence: {payload.get('confidence', 0)}")
            logger.info("")
    
    # Check consensus agent for new sentiment data
    for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        consensus = consensus_agent.get_consensus(symbol)
        if consensus:
            logger.info(f"Consensus for {symbol}:")
            logger.info(f"   Direction: {consensus.get('direction')} ({consensus.get('value', 0):.2f})")
            logger.info(f"   Confidence: {consensus.get('confidence', 0):.2f}")
            logger.info(f"   Sources: {consensus.get('source_count', 0)}")
            logger.info("")
    
    # Stop components
    await detector.stop()
    await consensus_agent.stop()


async def main():
    """Run the demonstration."""
    logger.info("Starting Real-Time Event Detection Demo")
    
    # Demo basic event detection
    logger.info("\n=== Real-Time Event Detection Demo ===")
    await demo_event_detection()
    
    # Demo sentiment integration
    logger.info("\n=== Sentiment Integration Demo ===")
    await demo_sentiment_integration()
    
    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())