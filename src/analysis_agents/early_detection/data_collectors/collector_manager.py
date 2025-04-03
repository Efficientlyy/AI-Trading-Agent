"""
Data Collector Manager for the Early Event Detection System.

This module manages the various data collectors used to gather information
from different sources for early event detection.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.common.config import config
from src.common.logging import get_logger
from src.analysis_agents.early_detection.models import SourceType, EventSource
from src.analysis_agents.early_detection.data_collectors.twitter_collector import TwitterCollector
from src.analysis_agents.early_detection.data_collectors.reddit_collector import RedditCollector
from src.analysis_agents.early_detection.data_collectors.news_collector import NewsCollector
from src.analysis_agents.early_detection.data_collectors.financial_data_collector import FinancialDataCollector


class BaseDataCollector:
    """Base class for data collectors."""
    
    def __init__(self, source_type: SourceType):
        """Initialize the data collector.
        
        Args:
            source_type: The type of source this collector gathers data from
        """
        self.source_type = source_type
        self.logger = get_logger("early_detection", f"collector_{source_type.value}")
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the data collector."""
        self.is_initialized = True
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from the source.
        
        Returns:
            List of collected data items
        """
        raise NotImplementedError("Subclasses must implement collect()")


class SocialMediaCollector(BaseDataCollector):
    """Collector for social media data."""
    
    def __init__(self):
        """Initialize the social media collector."""
        super().__init__(SourceType.SOCIAL_MEDIA)
        self.platforms = config.get("early_detection.data_collection.social_media.platforms", 
                                   ["twitter", "reddit"])
        
        # Initialize specific collectors
        self.twitter_collector = TwitterCollector() if "twitter" in self.platforms else None
        self.reddit_collector = RedditCollector() if "reddit" in self.platforms else None
    
    async def initialize(self):
        """Initialize the social media collector."""
        await super().initialize()
        
        self.logger.info(f"Initializing social media collector for {self.platforms}")
        
        # Initialize specific collectors
        if self.twitter_collector:
            await self.twitter_collector.initialize()
        
        if self.reddit_collector:
            await self.reddit_collector.initialize()
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from social media platforms.
        
        Returns:
            List of collected data items
        """
        self.logger.info("Collecting data from social media")
        
        collected_data = []
        
        # Create tasks for each platform
        tasks = []
        
        if self.twitter_collector:
            tasks.append(asyncio.create_task(self.twitter_collector.collect()))
        
        if self.reddit_collector:
            tasks.append(asyncio.create_task(self.reddit_collector.collect()))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Error collecting social media data: {result}")
                continue
            
            if result:
                collected_data.extend(result)
        
        self.logger.info(f"Collected {len(collected_data)} items from social media")
        return collected_data


class OfficialSourceCollector(BaseDataCollector):
    """Collector for official source data (central banks, governments, etc.)."""
    
    def __init__(self):
        """Initialize the official source collector."""
        super().__init__(SourceType.OFFICIAL)
        # This would be configured from config in a real implementation
        self.sources = config.get("early_detection.data_collection.official.sources", 
                                 ["federal_reserve", "ecb", "sec", "cftc", "us_treasury"])
    
    async def initialize(self):
        """Initialize the official source collector."""
        await super().initialize()
        # In a real implementation, this would initialize API clients
        self.logger.info(f"Initialized official source collector for {self.sources}")
    
    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from official sources.
        
        Returns:
            List of collected data items
        """
        # Mock implementation - in a real system, this would call APIs
        self.logger.info("Collecting data from official sources")
        
        collected_data = []
        
        # Add some mock data for demonstration
        for i, source in enumerate(self.sources):
            collected_data.append({
                "source": EventSource(
                    id=f"{source}_0",
                    type=SourceType.OFFICIAL,
                    name=source.replace("_", " ").title(),
                    url=f"https://{source}.gov/example/0",
                    reliability_score=0.9
                ),
                "title": f"Official statement from {source}",
                "content": f"Sample official statement about monetary policy from {source}",
                "timestamp": datetime.now(),
                "metadata": {
                    "document_type": "press_release",
                    "official": True
                }
            })
        
        self.logger.info(f"Collected {len(collected_data)} items from official sources")
        return collected_data


class DataCollectorManager:
    """Manager for data collectors."""
    
    def __init__(self):
        """Initialize the data collector manager."""
        self.logger = get_logger("early_detection", "collector_manager")
        self.collectors = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the data collector manager and all collectors."""
        self.logger.info("Initializing data collector manager")
        
        # Initialize collectors
        self.collectors = {
            SourceType.SOCIAL_MEDIA: SocialMediaCollector(),
            SourceType.NEWS: NewsCollector(),
            SourceType.OFFICIAL: OfficialSourceCollector(),
            SourceType.FINANCIAL_DATA: FinancialDataCollector()
        }
        
        # Initialize each collector
        for collector_type, collector in self.collectors.items():
            try:
                collector.initialize()
                self.logger.info(f"Initialized {collector_type.value} collector")
            except Exception as e:
                self.logger.error(f"Error initializing {collector_type.value} collector: {e}")
        
        self.is_initialized = True
        self.logger.info("Data collector manager initialized")
    
    async def collect_data(self) -> Dict[SourceType, List[Dict[str, Any]]]:
        """Collect data from all sources.
        
        Returns:
            Dictionary mapping source types to collected data
        """
        if not self.is_initialized:
            self.logger.warning("Data collector manager not initialized")
            return {}
        
        self.logger.info("Collecting data from all sources")
        
        # Create tasks for all collectors
        tasks = {}
        for source_type, collector in self.collectors.items():
            tasks[source_type] = asyncio.create_task(collector.collect())
        
        # Wait for all tasks to complete
        results = {}
        for source_type, task in tasks.items():
            try:
                results[source_type] = await task
                self.logger.info(f"Collected {len(results[source_type])} items from {source_type.value}")
            except Exception as e:
                self.logger.error(f"Error collecting data from {source_type.value}: {e}")
                results[source_type] = []
        
        # Log total collected items
        total_items = sum(len(items) for items in results.values())
        self.logger.info(f"Total collected items: {total_items}")
        
        return results