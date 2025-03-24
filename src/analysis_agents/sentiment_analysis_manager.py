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
from src.analysis_agents.sentiment.llm_service import LLMService
from src.analysis_agents.sentiment.llm_sentiment_agent import LLMSentimentAgent
from src.analysis_agents.sentiment.consensus_system import MultiModelConsensusAgent
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
        
        # Create services
        self.nlp_service = NLPService()
        self.llm_service = None  # Will be initialized if LLM agents are enabled
        
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
            
        # Create LLM sentiment agent if enabled
        if config.get("analysis_agents.sentiment.llm.enabled", False):
            self.agents["llm"] = LLMSentimentAgent("llm")
            self.logger.info("Created LLM sentiment agent")
            
        # Create Multi-Model Consensus agent if enabled
        if config.get("analysis_agents.sentiment.consensus.enabled", False):
            self.agents["consensus"] = MultiModelConsensusAgent("consensus")
            self.logger.info("Created multi-model consensus agent")
            
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
        
        # Initialize LLM service if needed
        llm_enabled = config.get("analysis_agents.sentiment.llm.enabled", False)
        if llm_enabled:
            self.llm_service = LLMService()
            await self.llm_service.initialize()
            self.logger.info("Initialized LLM service")
        
        # Initialize all agents
        init_tasks = []
        for agent_id, agent in self.agents.items():
            # Pass services to agents
            agent.set_nlp_service(self.nlp_service)
            
            # If this is an LLM agent, set the LLM service
            if agent_id == "llm" and self.llm_service:
                self.agents[agent_id].llm_service = self.llm_service
            
            init_tasks.append(agent.initialize())
            
        # Wait for all agents to initialize
        if init_tasks:
            await asyncio.gather(*init_tasks)
            
        # Connect consensus agent to receive events from other agents if enabled
        if "consensus" in self.agents:
            self._connect_consensus_agent()
    
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
            
        # Close LLM service if initialized
        if self.llm_service:
            await self.llm_service.close()
            self.logger.info("Closed LLM service")
    
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
        
    def _connect_consensus_agent(self) -> None:
        """Connect other agents to send signals to the consensus agent."""
        if "consensus" not in self.agents:
            return
            
        consensus_agent = self.agents["consensus"]
        
        # Setup event listeners to forward sentiment data to consensus system
        for agent_id, agent in self.agents.items():
            # Skip the consensus agent itself to avoid circular references
            if agent_id == "consensus":
                continue
                
            # Create callback to forward sentiment to consensus
            async def forward_sentiment_to_consensus(event, agent_source=agent_id):
                # Extract relevant data from the event
                try:
                    symbol = event.symbol
                    value = event.sentiment_value
                    direction = event.direction
                    confidence = event.confidence
                    
                    # Determine source type based on agent_id
                    source_type = self._map_agent_to_source_type(agent_source)
                    
                    # Get model info if available
                    model = None
                    if "details" in event and "model" in event.details:
                        model = event.details["model"]
                    elif agent_source == "llm" and "details" in event:
                        # Try to extract model from LLM agent event
                        model = event.details.get("model", "llm")
                        
                    # Additional metadata
                    metadata = {
                        "event_type": event.event_type,
                        "is_extreme": event.is_extreme,
                        "signal_type": event.signal_type,
                        "timestamp": event.timestamp
                    }
                    
                    # Add details if available
                    if hasattr(event, "details") and event.details:
                        metadata["details"] = event.details
                    
                    # Submit to consensus system
                    await consensus_agent.submit_sentiment(
                        symbol=symbol,
                        value=value,
                        direction=direction,
                        confidence=confidence,
                        source_type=source_type,
                        model=model,
                        metadata=metadata
                    )
                except Exception as e:
                    self.logger.error(f"Error forwarding sentiment to consensus: {str(e)}")
            
            # Register the callback
            agent.event_bus.subscribe("sentiment_event", forward_sentiment_to_consensus)
            self.logger.debug(f"Connected {agent_id} agent to consensus system")
            
    def _map_agent_to_source_type(self, agent_id: str) -> str:
        """Map agent ID to source type.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Standardized source type
        """
        # Direct mappings
        if agent_id in ["social_media", "news", "market", "onchain"]:
            return agent_id
        elif agent_id == "llm":
            return "llm"
        elif agent_id == "aggregator":
            return "aggregate"
        else:
            return "other"
