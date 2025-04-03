"""
Multi-Model Consensus System Demo

This example demonstrates how to use the Multi-Model Consensus System
to combine sentiment signals from multiple sources and models.
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
from src.analysis_agents.sentiment.consensus_system import ConsensusSystem, MultiModelConsensusAgent
from src.common.config import config
from src.common.logging import setup_logging

# Set up logging
setup_logging(log_level=logging.INFO)
logger = logging.getLogger("consensus_demo")


async def demo_llm_analysis():
    """Demonstrate LLM-based sentiment analysis and event detection."""
    logger.info("Initializing LLM service...")
    llm_service = LLMService()
    await llm_service.initialize()
    
    # Example texts to analyze
    texts = [
        "Bitcoin just broke through the $100,000 level for the first time ever! This is a major milestone for cryptocurrency adoption.",
        "Ethereum's Proof-of-Stake upgrade has been delayed again due to technical issues. Developers say they need more time for testing.",
        "The SEC has approved multiple Bitcoin ETF applications, opening the way for institutional investment in cryptocurrencies.",
        "The Federal Reserve announced a 0.5% interest rate increase, impacting all risk assets including cryptocurrencies.",
        "A major cryptocurrency exchange announced significant layoffs today, citing market conditions and reduced trading volumes."
    ]
    
    # Analyze sentiment
    logger.info("Analyzing sentiment with LLM...")
    sentiment_results = await llm_service.analyze_sentiment(texts)
    
    # Detect market events
    logger.info("Detecting market events with LLM...")
    event_results = await llm_service.detect_market_event(texts)
    
    # Print results
    for i, (text, sentiment, event) in enumerate(zip(texts, sentiment_results, event_results)):
        logger.info(f"\nText {i+1}: {text[:50]}...")
        logger.info(f"Sentiment: {sentiment.get('direction', 'neutral')} ({sentiment.get('value', 0.5):.2f}) - Confidence: {sentiment.get('confidence', 0):.2f}")
        
        if event.get("is_market_event", False):
            logger.info(f"Market Event Detected: {event.get('event_type')} - Severity: {event.get('severity')}")
            logger.info(f"Explanation: {event.get('explanation', '')[:100]}...")
            
            # If it's a significant event, assess the impact
            if event.get("severity", 0) >= 5:
                market_context = "Bitcoin is currently in an uptrend, trading above the 200-day moving average. Market sentiment has been positive in recent weeks with moderate volatility."
                
                logger.info("Assessing market impact...")
                impact = await llm_service.assess_market_impact(
                    event=text,
                    market_context=market_context
                )
                
                logger.info(f"Impact Direction: {impact.get('primary_impact_direction', 'neutral')}")
                logger.info(f"Impact Magnitude: {impact.get('impact_magnitude', 0):.2f}")
                logger.info(f"Affected Assets: {impact.get('affected_assets', {})}")
        else:
            logger.info("No significant market event detected")
            
    # Close the LLM service
    await llm_service.close()


async def demo_consensus_system():
    """Demonstrate the Multi-Model Consensus System."""
    logger.info("Creating consensus system...")
    
    # Create a consensus system
    consensus = ConsensusSystem("consensus")
    
    # Create simulated sentiment data from different sources and models
    sentiment_data = [
        # LLM sources
        {
            "value": 0.78,
            "direction": "bullish",
            "confidence": 0.85,
            "source_type": "llm",
            "model": "gpt-4o",
            "timestamp": datetime.utcnow() - timedelta(minutes=30),
            "text": "Bitcoin shows strong technical patterns with increasing volume."
        },
        {
            "value": 0.71,
            "direction": "bullish",
            "confidence": 0.82,
            "source_type": "llm",
            "model": "claude-3-opus",
            "timestamp": datetime.utcnow() - timedelta(minutes=45),
            "text": "Multiple metrics suggest bullish sentiment for BTC in the short term."
        },
        {
            "value": 0.65,
            "direction": "bullish",
            "confidence": 0.75,
            "source_type": "llm",
            "model": "llama-3-70b",
            "timestamp": datetime.utcnow() - timedelta(minutes=55),
            "text": "Bitcoin price action looks positive with strong support levels."
        },
        
        # Social media sentiment
        {
            "value": 0.62,
            "direction": "bullish",
            "confidence": 0.68,
            "source_type": "social_media",
            "model": "finbert",
            "timestamp": datetime.utcnow() - timedelta(hours=1),
            "platform": "Twitter"
        },
        {
            "value": 0.58,
            "direction": "bullish",
            "confidence": 0.72,
            "source_type": "social_media",
            "model": "distilbert",
            "timestamp": datetime.utcnow() - timedelta(minutes=80),
            "platform": "Reddit"
        },
        
        # News sentiment
        {
            "value": 0.67,
            "direction": "bullish",
            "confidence": 0.79,
            "source_type": "news",
            "model": "finbert",
            "timestamp": datetime.utcnow() - timedelta(hours=2),
            "source": "CoinDesk"
        },
        
        # Market sentiment
        {
            "value": 0.72,
            "direction": "bullish",
            "confidence": 0.84,
            "source_type": "market_sentiment",
            "model": "fear_greed_index",
            "timestamp": datetime.utcnow() - timedelta(hours=6)
        },
        
        # Dissenting opinion
        {
            "value": 0.32,
            "direction": "bearish",
            "confidence": 0.67,
            "source_type": "onchain",
            "model": "blockchain_analysis",
            "timestamp": datetime.utcnow() - timedelta(hours=3),
            "metric": "exchange_inflows"
        }
    ]
    
    # Compute consensus
    consensus_result = consensus.compute_consensus(sentiment_data)
    
    # Print the consensus result
    logger.info("\nConsensus Result:")
    logger.info(f"Direction: {consensus_result.get('direction')} ({consensus_result.get('value'):.2f})")
    logger.info(f"Confidence: {consensus_result.get('confidence'):.2f}")
    logger.info(f"Sources: {consensus_result.get('unique_source_types')} unique source types, {consensus_result.get('unique_models')} unique models")
    logger.info(f"Disagreement Level: {consensus_result.get('disagreement_level'):.2f}")
    
    if consensus_result.get('has_major_disagreement', False):
        logger.info("Major disagreement detected - using Bayesian aggregation")
    elif consensus_result.get('has_minor_disagreement', False):
        logger.info("Minor disagreement detected")
    
    logger.info(f"Direction Counts: {json.dumps(consensus_result.get('direction_counts', {}))}")
    
    # Demonstrate performance tracking
    logger.info("\nRecording historical performance...")
    
    # Simulate tracking performance over time
    for i, data in enumerate(sentiment_data):
        # Simulate different accuracy levels based on source
        if data["source_type"] == "llm" and data["model"] in ["gpt-4o", "claude-3-opus"]:
            accuracy = 0.85  # High accuracy for advanced LLMs
        elif data["source_type"] == "market_sentiment":
            accuracy = 0.78  # Good accuracy for market indicators
        elif data["source_type"] == "social_media":
            accuracy = 0.65  # Lower accuracy for social sentiment
        else:
            accuracy = 0.72  # Average accuracy for other sources
            
        # Add some randomness
        import random
        accuracy_with_noise = max(0.1, min(0.9, accuracy + random.uniform(-0.1, 0.1)))
        
        # Calculate simulated actual outcome (sentiment value +/- error based on accuracy)
        error = 1.0 - accuracy_with_noise
        actual_outcome = max(0.0, min(1.0, data["value"] + random.uniform(-error, error)))
        
        # Record performance
        consensus.record_performance(
            source_type=data["source_type"],
            model=data["model"],
            prediction=data["value"],
            actual_outcome=actual_outcome,
            timestamp=data["timestamp"]
        )
    
    # Get performance metrics
    performance = consensus.get_model_performance()
    
    logger.info("\nPerformance Metrics:")
    for source_model, score in sorted(performance.items(), key=lambda x: x[1], reverse=True):
        source, model = source_model.split(":", 1)
        logger.info(f"{source} ({model}): {score:.3f}")


async def demo_complete_pipeline():
    """Demonstrate the complete sentiment analysis pipeline with consensus."""
    logger.info("Initializing complete sentiment pipeline...")
    
    # Create a multi-model consensus agent
    agent = MultiModelConsensusAgent("consensus")
    await agent.initialize()
    await agent.start()
    
    # Submit sentiment from different sources
    symbols = ["BTC/USDT", "ETH/USDT"]
    
    for symbol in symbols:
        logger.info(f"\nSubmitting sentiment data for {symbol}...")
        
        # Submit sentiment from LLM source
        await agent.submit_sentiment(
            symbol=symbol,
            value=0.73,
            direction="bullish",
            confidence=0.85,
            source_type="llm",
            model="gpt-4o",
            metadata={"explanation": "Strong technical patterns and increasing volume"}
        )
        
        # Submit sentiment from social media
        await agent.submit_sentiment(
            symbol=symbol,
            value=0.68,
            direction="bullish",
            confidence=0.72,
            source_type="social_media",
            model="finbert",
            metadata={"platform": "Twitter", "post_count": 1250}
        )
        
        # Submit sentiment from news
        await agent.submit_sentiment(
            symbol=symbol,
            value=0.65,
            direction="bullish",
            confidence=0.80,
            source_type="news",
            model="finbert",
            metadata={"source": "CoinDesk", "articles": 8}
        )
        
        # Submit different sentiment from on-chain metrics
        await agent.submit_sentiment(
            symbol=symbol,
            value=0.42,
            direction="bearish",
            confidence=0.67,
            source_type="onchain",
            model="blockchain_analysis",
            metadata={"metric": "exchange_inflows", "value": "increasing"}
        )
        
        # Process consensus
        await agent._process_consensus(symbol)
        
        # Get consensus result
        consensus_result = agent.get_consensus(symbol)
        
        if consensus_result:
            logger.info(f"Consensus for {symbol}:")
            logger.info(f"Direction: {consensus_result.get('direction')} ({consensus_result.get('value'):.2f})")
            logger.info(f"Confidence: {consensus_result.get('confidence'):.2f}")
            logger.info(f"Disagreement Level: {consensus_result.get('disagreement_level'):.2f}")
            logger.info(f"Direction Counts: {json.dumps(consensus_result.get('direction_counts', {}))}")
    
    # Stop the agent
    await agent.stop()


async def main():
    """Run all the demonstration functions."""
    logger.info("Starting Multi-Model Consensus System Demo")
    
    # Demo LLM analysis
    logger.info("\n=== LLM Sentiment Analysis Demo ===")
    await demo_llm_analysis()
    
    # Demo consensus system
    logger.info("\n=== Consensus System Demo ===")
    await demo_consensus_system()
    
    # Demo complete pipeline
    logger.info("\n=== Complete Pipeline Demo ===")
    await demo_complete_pipeline()
    
    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())