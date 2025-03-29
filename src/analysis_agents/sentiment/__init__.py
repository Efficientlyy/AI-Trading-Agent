"""Sentiment analysis module.

This package contains components for analyzing sentiment data from
various sources, including social media, news, market indicators,
and on-chain metrics.
"""

from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.analysis_agents.sentiment.sentiment_aggregator import SentimentAggregator
from src.analysis_agents.sentiment.nlp_service import NLPService
from src.analysis_agents.sentiment.llm_service import LLMService
from src.analysis_agents.sentiment.llm_sentiment_agent import LLMSentimentAgent
from src.analysis_agents.sentiment.consensus_system import ConsensusSystem, MultiModelConsensusAgent

__all__ = [
    'BaseSentimentAgent',
    'SentimentAggregator',
    'NLPService',
    'LLMService',
    'LLMSentimentAgent',
    'ConsensusSystem',
    'MultiModelConsensusAgent'
]
