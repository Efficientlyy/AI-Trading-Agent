#!/usr/bin/env python
"""
Sentiment Analysis System Test Script

This script conducts a comprehensive test of the sentiment analysis system implementation,
including all components and their integration.
"""

import asyncio
import os
import sys
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis_agents.sentiment.nlp_service import NLPService
from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.analysis_agents.sentiment.social_media_sentiment import SocialMediaSentimentAgent
from src.analysis_agents.sentiment.news_sentiment import NewsSentimentAgent
from src.analysis_agents.sentiment.market_sentiment import MarketSentimentAgent
from src.analysis_agents.sentiment.onchain_sentiment import OnchainSentimentAgent
from src.analysis_agents.sentiment.sentiment_aggregator import SentimentAggregator
from src.analysis_agents.sentiment_analysis_manager import SentimentAnalysisManager
from src.analysis_agents.connection_engine import ConnectionEngine
from src.analysis_agents.geopolitical.geopolitical_analyzer import GeopoliticalAnalyzer
from src.strategy.sentiment_strategy import SentimentStrategy
from src.strategy.enhanced_sentiment_strategy import EnhancedSentimentStrategy
from src.common.logging import setup_logging, get_logger
from src.common.config import config
from src.common.events import event_bus


# Configure logging
setup_logging(level=logging.INFO)
logger = get_logger("test_script", "sentiment_test")


class SentimentSystemTest:
    """Test class for the sentiment analysis system."""
    
    def __init__(self):
        """Initialize the test class."""
        self.logger = get_logger("test_script", "sentiment_test")
        self.results = {
            "components_test": {},
            "integration_test": {},
            "strategy_test": {}
        }
        self.received_events = []
        
        # Create test output directory
        os.makedirs("test_output", exist_ok=True)
    
    async def setup(self):
        """Set up test environment."""
        self.logger.info("Setting up test environment")
        
        # Subscribe to sentiment events for testing
        event_bus.subscribe("sentiment_event", self.handle_sentiment_event)
        
        # Create a manager instance
        self.manager = SentimentAnalysisManager()
    
    async def handle_sentiment_event(self, event):
        """Handler for sentiment events during testing."""
        self.logger.info(f"Received sentiment event: {event.source} - {event.symbol} - {event.sentiment_direction}")
        self.received_events.append({
            "source": event.source,
            "symbol": event.symbol,
            "value": event.sentiment_value,
            "direction": event.sentiment_direction,
            "confidence": event.confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "details": event.details
        })
    
    async def test_nlp_service(self):
        """Test the NLP service."""
        self.logger.info("Testing NLP service")
        
        try:
            # Create and initialize NLP service
            nlp_service = NLPService()
            await nlp_service.initialize()
            
            # Test text samples
            test_samples = [
                "Bitcoin is going to the moon! I'm so bullish on this project.",
                "The market is crashing and I think we're entering a bear market.",
                "Neutral sentiment with no strong feelings either way."
            ]
            
            # Analyze sentiment
            sentiment_scores = await nlp_service.analyze_sentiment(test_samples)
            
            # Check results
            self.logger.info(f"NLP Service test results: {sentiment_scores}")
            
            # Verify the scores are in the correct range
            all_valid = all(0 <= score <= 1 for score in sentiment_scores)
            
            # Check if bullish sample has higher score than bearish
            direction_correct = sentiment_scores[0] > sentiment_scores[1]
            
            # Store results
            self.results["components_test"]["nlp_service"] = {
                "status": "passed" if all_valid and direction_correct else "failed",
                "scores": sentiment_scores,
                "all_valid_range": all_valid,
                "direction_correct": direction_correct
            }
            
            return all_valid and direction_correct
            
        except Exception as e:
            self.logger.error(f"Error testing NLP service: {str(e)}")
            self.results["components_test"]["nlp_service"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    async def test_sentiment_agents(self):
        """Test all sentiment agents."""
        self.logger.info("Testing sentiment agents")
        
        agents_to_test = [
            ("social_media", SocialMediaSentimentAgent),
            ("news", NewsSentimentAgent),
            ("market_sentiment", MarketSentimentAgent),
            ("onchain", OnchainSentimentAgent),
            ("aggregator", SentimentAggregator)
        ]
        
        all_passed = True
        
        # Create NLP service for agents
        nlp_service = NLPService()
        await nlp_service.initialize()
        
        for agent_id, agent_class in agents_to_test:
            try:
                # Create agent
                agent = agent_class(agent_id)
                agent.set_nlp_service(nlp_service)
                
                # Override API clients for testing
                self._mock_agent_api_clients(agent)
                
                # Initialize agent
                await agent.initialize()
                
                # Add test data
                symbol = "BTC/USDT"
                sentiment_value = 0.7 if agent_id != "aggregator" else 0.6  # Slightly different for aggregator
                direction = "bullish"
                confidence = 0.8
                
                # Update sentiment cache with test data
                sentiment_shift = agent._update_sentiment_cache(
                    symbol=symbol,
                    source_type=agent_id,
                    sentiment_value=sentiment_value,
                    direction=direction,
                    confidence=confidence,
                    additional_data={"test": True}
                )
                
                # Verify cache update
                cache_updated = (
                    symbol in agent.sentiment_cache and
                    agent_id in agent.sentiment_cache[symbol] and
                    agent.sentiment_cache[symbol][agent_id]["value"] == sentiment_value
                )
                
                # Start agent briefly to publish events
                await agent.start()
                await asyncio.sleep(2)  # Allow time for events to be published
                await agent.stop()
                
                # Store results
                self.results["components_test"][agent_id] = {
                    "status": "passed" if cache_updated else "failed",
                    "cache_updated": cache_updated,
                    "sentiment_shift": sentiment_shift
                }
                
                if not cache_updated:
                    all_passed = False
                
            except Exception as e:
                self.logger.error(f"Error testing {agent_id} agent: {str(e)}")
                self.results["components_test"][agent_id] = {
                    "status": "error",
                    "error": str(e)
                }
                all_passed = False
        
        return all_passed
    
    def _mock_agent_api_clients(self, agent):
        """Mock API clients for testing agents."""
        agent_id = agent.agent_id
        
        if agent_id == "social_media":
            agent.twitter_client = self._create_mock_twitter_client()
            agent.reddit_client = self._create_mock_reddit_client()
        elif agent_id == "news":
            agent.news_api_client = self._create_mock_news_api_client()
            agent.crypto_news_client = self._create_mock_crypto_news_client()
        elif agent_id == "market_sentiment":
            agent.fear_greed_client = self._create_mock_fear_greed_client()
            agent.exchange_data_client = self._create_mock_exchange_data_client()
        elif agent_id == "onchain":
            agent.blockchain_client = self._create_mock_blockchain_client()
    
    def _create_mock_twitter_client(self):
        """Create a mock Twitter client for testing."""
        class MockTwitterClient:
            async def search_tweets(self, query, count, result_type):
                return [
                    "Bitcoin is going to the moon! I'm so bullish on this project.",
                    "Incredible news for BTC, this is going to be huge!"
                ]
        return MockTwitterClient()
    
    def _create_mock_reddit_client(self):
        """Create a mock Reddit client for testing."""
        class MockRedditClient:
            async def get_hot_posts(self, subreddit, limit, time_filter):
                return [
                    "I'm feeling really good about crypto right now.",
                    "The market looks strong today."
                ]
        return MockRedditClient()
    
    def _create_mock_news_api_client(self):
        """Create a mock News API client for testing."""
        class MockNewsApiClient:
            async def get_everything(self, q, language, sort_by, page_size):
                return [
                    {"title": "Bitcoin Surges to New Heights", "description": "Bitcoin is reaching new all-time highs."},
                    {"title": "Crypto Market Analysis", "description": "The market is showing strong bullish signals."}
                ]
        return MockNewsApiClient()
    
    def _create_mock_crypto_news_client(self):
        """Create a mock Crypto News client for testing."""
        class MockCryptoNewsClient:
            async def get_news(self, categories, keywords, limit):
                return [
                    {"title": "Bitcoin Adoption Growing", "description": "More institutions are adopting Bitcoin."},
                    {"title": "Crypto Regulations", "description": "New regulations may be positive for the market."}
                ]
        return MockCryptoNewsClient()
    
    def _create_mock_fear_greed_client(self):
        """Create a mock Fear & Greed client for testing."""
        class MockFearGreedClient:
            async def get_current_index(self):
                return {"value": 75, "classification": "Greed"}
        return MockFearGreedClient()
    
    def _create_mock_exchange_data_client(self):
        """Create a mock Exchange Data client for testing."""
        class MockExchangeDataClient:
            async def get_long_short_ratio(self, symbol):
                return {"longShortRatio": 1.5}
        return MockExchangeDataClient()
    
    def _create_mock_blockchain_client(self):
        """Create a mock Blockchain client for testing."""
        class MockBlockchainClient:
            async def get_large_transactions(self, asset, time_period):
                return {"count": 150, "volume": 5000, "average_volume": 4000}
                
            async def get_active_addresses(self, asset, time_period):
                return {"count": 950000, "change_percentage": 5.2}
                
            async def get_hash_rate(self, asset, time_period):
                return {"hash_rate": 150, "change_percentage": 3.1}
                
            async def get_exchange_reserves(self, asset, time_period):
                return {"reserves": 2100000, "change_percentage": -2.3}
        return MockBlockchainClient()
    
    async def test_manager(self):
        """Test the sentiment analysis manager."""
        self.logger.info("Testing sentiment analysis manager")
        
        try:
            # Initialize and start the manager
            await self.manager.initialize()
            await self.manager.start()
            
            # Wait for agents to process
            await asyncio.sleep(5)
            
            # Check if all agents are initialized and running
            agents_initialized = all(agent.initialized for agent in self.manager.agents.values())
            
            # Store results
            self.results["components_test"]["manager"] = {
                "status": "passed" if agents_initialized else "failed",
                "agents_initialized": agents_initialized,
                "agent_count": len(self.manager.agents)
            }
            
            # Stop the manager
            await self.manager.stop()
            
            return agents_initialized
            
        except Exception as e:
            self.logger.error(f"Error testing manager: {str(e)}")
            self.results["components_test"]["manager"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    async def test_special_analyzers(self):
        """Test the special analyzer components."""
        self.logger.info("Testing special analyzers (ConnectionEngine, GeopoliticalAnalyzer)")
        
        all_passed = True
        
        # Test GeopoliticalAnalyzer
        try:
            geo_analyzer = GeopoliticalAnalyzer()
            await geo_analyzer.initialize()
            
            # Test event impact assessment
            test_event = {
                "title": "US-China trade talks resume",
                "description": "Trade negotiations between the US and China are set to resume next week.",
                "category": "economic"
            }
            
            # Analyze the event
            event_impact = await geo_analyzer.assess_event_impact(test_event)
            
            # Verify basic functionality
            geo_analyzer_working = event_impact is not None
            
            self.results["components_test"]["geopolitical_analyzer"] = {
                "status": "passed" if geo_analyzer_working else "failed",
                "event_impact": event_impact
            }
            
            if not geo_analyzer_working:
                all_passed = False
                
        except Exception as e:
            self.logger.error(f"Error testing GeopoliticalAnalyzer: {str(e)}")
            self.results["components_test"]["geopolitical_analyzer"] = {
                "status": "error",
                "error": str(e)
            }
            all_passed = False
        
        # Test ConnectionEngine
        try:
            connection_engine = ConnectionEngine()
            await connection_engine.initialize()
            
            # Test connection detection
            test_events = [
                {
                    "id": "event1",
                    "title": "US increases tariffs on Chinese goods",
                    "description": "The US has increased tariffs on $200 billion of Chinese goods.",
                    "category": "economic"
                },
                {
                    "id": "event2",
                    "title": "China retaliates with tariffs on US goods",
                    "description": "China has announced retaliatory tariffs on $60 billion of US goods.",
                    "category": "economic"
                }
            ]
            
            # Add events to the engine
            for event in test_events:
                await connection_engine.add_event(event)
            
            # Detect connections
            connections = await connection_engine.detect_connections()
            
            # Verify basic functionality
            connection_engine_working = len(connections) > 0
            
            self.results["components_test"]["connection_engine"] = {
                "status": "passed" if connection_engine_working else "failed",
                "connections_detected": len(connections)
            }
            
            if not connection_engine_working:
                all_passed = False
                
        except Exception as e:
            self.logger.error(f"Error testing ConnectionEngine: {str(e)}")
            self.results["components_test"]["connection_engine"] = {
                "status": "error",
                "error": str(e)
            }
            all_passed = False
        
        return all_passed
    
    async def test_sentiment_strategy(self):
        """Test the sentiment strategy."""
        self.logger.info("Testing sentiment strategy")
        
        try:
            # Create and initialize strategy
            strategy = SentimentStrategy()
            await strategy.initialize()
            
            # Create test data
            symbol = "BTC/USDT"
            event = {
                "source": "social_media_sentiment",
                "symbol": symbol,
                "sentiment_value": 0.75,
                "sentiment_direction": "bullish",
                "confidence": 0.8,
                "details": {"test": True}
            }
            
            # Process the event
            await strategy._handle_sentiment_event(event)
            
            # Check if strategy has processed the data
            has_sentiment_data = (
                symbol in strategy.sentiment_data and
                "social_media" in strategy.sentiment_data[symbol]
            )
            
            # Store results
            self.results["strategy_test"]["sentiment_strategy"] = {
                "status": "passed" if has_sentiment_data else "failed",
                "has_sentiment_data": has_sentiment_data
            }
            
            await strategy.stop()
            
            return has_sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error testing sentiment strategy: {str(e)}")
            self.results["strategy_test"]["sentiment_strategy"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    async def test_enhanced_sentiment_strategy(self):
        """Test the enhanced sentiment strategy."""
        self.logger.info("Testing enhanced sentiment strategy")
        
        try:
            # Create and initialize strategy
            strategy = EnhancedSentimentStrategy()
            await strategy.initialize()
            
            # Create test data
            symbol = "BTC/USDT"
            
            # Create and process various sentiment events
            sources = ["social_media_sentiment", "news_sentiment", "market_sentiment", "onchain_sentiment"]
            
            for source in sources:
                event = {
                    "source": source,
                    "symbol": symbol,
                    "sentiment_value": 0.75,  # Bullish
                    "sentiment_direction": "bullish",
                    "confidence": 0.8,
                    "details": {"test": True}
                }
                
                # Process the event
                await strategy._handle_sentiment_event(event)
            
            # Check if strategy has processed the data
            has_complete_data = (
                symbol in strategy.sentiment_data and
                len(strategy.sentiment_data[symbol]) >= 4
            )
            
            # Store results
            self.results["strategy_test"]["enhanced_sentiment_strategy"] = {
                "status": "passed" if has_complete_data else "failed",
                "has_complete_data": has_complete_data,
                "source_count": len(strategy.sentiment_data.get(symbol, {}))
            }
            
            await strategy.stop()
            
            return has_complete_data
            
        except Exception as e:
            self.logger.error(f"Error testing enhanced sentiment strategy: {str(e)}")
            self.results["strategy_test"]["enhanced_sentiment_strategy"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    async def test_integration(self):
        """Test the integration of all sentiment components."""
        self.logger.info("Testing sentiment system integration")
        
        try:
            # Clear received events
            self.received_events = []
            
            # Create and initialize manager with all components
            manager = SentimentAnalysisManager()
            await manager.initialize()
            await manager.start()
            
            # Create enhanced sentiment strategy
            strategy = EnhancedSentimentStrategy()
            await strategy.initialize()
            await strategy.start()
            
            # Wait for components to initialize and generate events
            self.logger.info("Waiting for events to be generated...")
            await asyncio.sleep(10)  # Give time for events to be generated
            
            # Stop components
            await strategy.stop()
            await manager.stop()
            
            # Analyze results
            event_count = len(self.received_events)
            event_sources = set(event["source"] for event in self.received_events)
            
            # Verify integration
            integration_working = event_count > 0 and len(event_sources) > 0
            
            self.results["integration_test"]["overall"] = {
                "status": "passed" if integration_working else "failed",
                "event_count": event_count,
                "event_sources": list(event_sources)
            }
            
            self.logger.info(f"Received {event_count} events from {len(event_sources)} sources during integration test")
            
            return integration_working
            
        except Exception as e:
            self.logger.error(f"Error in integration test: {str(e)}")
            self.results["integration_test"]["overall"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    async def run_all_tests(self):
        """Run all tests."""
        self.logger.info("Starting comprehensive sentiment system tests")
        
        # Set up test environment
        await self.setup()
        
        # Test results
        test_results = {
            "nlp_service": await self.test_nlp_service(),
            "sentiment_agents": await self.test_sentiment_agents(),
            "manager": await self.test_manager(),
            "special_analyzers": await self.test_special_analyzers(),
            "sentiment_strategy": await self.test_sentiment_strategy(),
            "enhanced_sentiment_strategy": await self.test_enhanced_sentiment_strategy(),
            "integration": await self.test_integration()
        }
        
        # Overall result
        all_passed = all(test_results.values())
        
        # Final result
        self.results["overall"] = {
            "status": "passed" if all_passed else "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "test_results": test_results
        }
        
        # Save results to file
        self.save_results()
        
        return all_passed
    
    def save_results(self):
        """Save test results to file."""
        result_path = "test_output/sentiment_system_test_results.json"
        
        with open(result_path, "w") as f:
            json.dump(self.results, f, indent=2)
            
        self.logger.info(f"Test results saved to {result_path}")


async def main():
    """Run the sentiment system test."""
    # Display banner
    print("=" * 80)
    print("SENTIMENT ANALYSIS SYSTEM COMPREHENSIVE TEST".center(80))
    print("=" * 80)
    
    # Create and run test
    test = SentimentSystemTest()
    all_passed = await test.run_all_tests()
    
    # Display results
    print("\n" + "=" * 80)
    print("TEST RESULTS".center(80))
    print("=" * 80)
    
    # Format component test results
    print("\nComponent Tests:")
    for component, result in test.results["components_test"].items():
        status = result["status"]
        status_display = "✓" if status == "passed" else "✗" if status == "failed" else "!"
        print(f"  {status_display} {component:25} - {status.upper()}")
    
    # Format strategy test results
    print("\nStrategy Tests:")
    for strategy, result in test.results["strategy_test"].items():
        status = result["status"]
        status_display = "✓" if status == "passed" else "✗" if status == "failed" else "!"
        print(f"  {status_display} {strategy:25} - {status.upper()}")
    
    # Format integration test results
    print("\nIntegration Tests:")
    integration_result = test.results["integration_test"].get("overall", {})
    status = integration_result.get("status", "unknown")
    status_display = "✓" if status == "passed" else "✗" if status == "failed" else "!"
    print(f"  {status_display} {'Overall integration':25} - {status.upper()}")
    
    # Overall status
    overall_status = test.results["overall"]["status"]
    print("\n" + "=" * 80)
    print(f"OVERALL TEST STATUS: {overall_status.upper()}".center(80))
    print("=" * 80)
    
    print(f"\nDetailed results saved to: test_output/sentiment_system_test_results.json\n")
    
    # Return with appropriate exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)