"""
Data collectors for the Early Event Detection System.

This package contains collectors for different data sources used in early event detection.
"""

from src.analysis_agents.early_detection.data_collectors.collector_manager import (
    BaseDataCollector,
    DataCollectorManager,
    SocialMediaCollector,
    OfficialSourceCollector
)
from src.analysis_agents.early_detection.data_collectors.twitter_collector import TwitterCollector
from src.analysis_agents.early_detection.data_collectors.reddit_collector import RedditCollector
from src.analysis_agents.early_detection.data_collectors.news_collector import NewsCollector
from src.analysis_agents.early_detection.data_collectors.financial_data_collector import FinancialDataCollector