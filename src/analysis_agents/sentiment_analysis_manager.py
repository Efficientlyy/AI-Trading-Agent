"""Sentiment analysis agent manager.

This module provides a manager for creating and coordinating various
sentiment analysis agents, such as social media, news, market, and
on-chain sentiment analyzers.
"""

import asyncio
from typing import Dict, List, Optional, Set, Any

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.analysis_agents.sentiment.social_media_sentiment import SocialMediaSentimentAgent
from src.analysis_agents.sentiment.news_sentiment import NewsSentimentAgent
from src.analysis_agents.sentiment.market_sentiment import MarketSentimentAgent
from src.analysis_agents.sentiment.onchain_sentiment import OnchainSentimentAgent
from src.analysis_agents.sentiment.sentiment_aggregator import SentimentAggregator
from src.analysis_agents.sentiment.nlp_service import NLPService
from src.common.config import config
from src.common.logging import get_logger
from src.common.component import Component


class SentimentAnalysisManager(Component):
    """Manager for sentiment analysis agents.
    
    This class creates and manages various sentiment analysis agents,
    providing a unified interface for sentiment analysis in the system.
    """
    
    def __init__(self):
        """Initialize the sentiment analysis manager."""
        super().__init__("sentiment_analysis_manager")
        self.logger = get_logger("analysis_agents", "sentiment_manager")
        
        # Load configuration
        self.enabled = config.get("analysis_agents.sentiment.enabled", True)
        self.agent_configs = config.get("analysis_agents.sentiment.agents", {})
        
        # Create NLP service
        self.nlp_service = NLPService()
        
        # Create sentiment agents
        self.agents: Dict[str, BaseSentimentAgent] = {}
        self._create_agents()
    
    def _create_agents(self) -> None:
        """Create the sentiment analysis agents based on configuration."""
        # Create social media sentiment agent
        if config.get("analysis_agents.sentiment.social_media.enabled", True):
            self.agents["social_media"] = SocialMediaSentimentAgent("social_media")
            self.logger.info("Created social media sentiment agent")
            
        # Create news sentiment agent
        if config.get("analysis_agents.sentiment.news.enabled", True):
            self.agents["news"] = NewsSentimentAgent("news")
            self.logger.info("Created news sentiment agent")
            
        # Create market sentiment agent
        if config.get("analysis_agents.sentiment.market.enabled", True):
            self.agents["market"] = MarketSentimentAgent("market")
            self.logger.info("Created market sentiment agent")
            
        # Create onchain sentiment agent
        if config.get("analysis_agents.sentiment.onchain.enabled", True):
            self.agents["onchain"] = OnchainSentimentAgent("onchain")
            self.logger.info("Created onchain sentiment agent")
            
        # Create sentiment aggregator
        if config.get("analysis_agents.sentiment.aggregator.enabled", True):
            self.agents["aggregator"] = SentimentAggregator("aggregator")
            self.logger.info("Created sentiment aggregator")
    
    async def _initialize(self) -> None:
        """Initialize all sentiment analysis agents."""
        if not self.enabled:
            self.logger.info("Sentiment analysis manager is disabled")
            return
            
        self.logger.info("Initializing sentiment analysis manager")
        
        # Initialize NLP service first
        await self.nlp_service.initialize()
        
        # Initialize all agents
        init_tasks = []
        for agent_id, agent in self.agents.items():
            # Pass NLP service to agent
            agent.set_nlp_service(self.nlp_service)
            init_tasks.append(agent.initialize())
            
        # Wait for all agents to initialize
        if init_tasks:
            await asyncio.gather(*init_tasks)
    
    async def _start(self) -> None:
        """Start all sentiment analysis agents."""
        if not self.enabled:
            return
            
        self.logger.info("Starting sentiment analysis manager")
        
        # Start all agents
        start_tasks = []
        for agent_id, agent in self.agents.items():
            start_tasks.append(agent.start())
            
        # Wait for all agents to start
        if start_tasks:
            await asyncio.gather(*start_tasks)
    
    async def _stop(self) -> None:
        """Stop all sentiment analysis agents."""
        if not self.enabled:
            return
            
        self.logger.info("Stopping sentiment analysis manager")
        
        # Stop all agents
        stop_tasks = []
        for agent_id, agent in self.agents.items():
            stop_tasks.append(agent.stop())
            
        # Wait for all agents to stop
        if stop_tasks:
            await asyncio.gather(*stop_tasks)
    
    def get_agent(self, agent_id: str) -> Optional[BaseSentimentAgent]:
        """Get a sentiment agent by ID.
        
        Args:
            agent_id: The ID of the agent to get
            
        Returns:
            The sentiment agent, or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[BaseSentimentAgent]:
        """Get all sentiment agents.
        
        Returns:
            List of all sentiment agents
        """
        return list(self.agents.values())
