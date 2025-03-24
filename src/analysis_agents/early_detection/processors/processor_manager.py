"""
Data Processor Manager for the Early Event Detection System.

This module manages the various data processors used to transform and enhance
the raw data collected by data collectors.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.common.config import config
from src.common.logging import get_logger
from src.analysis_agents.early_detection.models import SourceType
from src.analysis_agents.sentiment.nlp_service import NLPService


class BaseProcessor:
    """Base class for data processors."""
    
    def __init__(self, processor_name: str):
        """Initialize the processor.
        
        Args:
            processor_name: Name of the processor
        """
        self.name = processor_name
        self.logger = get_logger("early_detection", f"processor_{processor_name}")
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the processor."""
        self.is_initialized = True
    
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the data.
        
        Args:
            data: Raw data to process
            
        Returns:
            Processed data
        """
        raise NotImplementedError("Subclasses must implement process()")


class DocumentProcessor(BaseProcessor):
    """Processor for text documents (news, statements, etc.)."""
    
    def __init__(self):
        """Initialize the document processor."""
        super().__init__("document")
        self.nlp_service = None
    
    async def initialize(self):
        """Initialize the document processor."""
        await super().initialize()
        
        # Initialize NLP service
        self.nlp_service = NLPService()
        await self.nlp_service.initialize()
        
        self.logger.info("Document processor initialized")
    
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process document data.
        
        Args:
            data: Raw document data
            
        Returns:
            Processed document data with enhanced features
        """
        self.logger.info(f"Processing {len(data)} documents")
        processed_data = []
        
        for item in data:
            try:
                # Extract text content
                text = ""
                if "content" in item:
                    text = item["content"]
                elif "title" in item and "content" in item:
                    text = f"{item['title']}. {item['content']}"
                
                if not text:
                    # Skip items without text content
                    continue
                
                # Create processed item with original data
                processed_item = item.copy()
                
                # Add NLP features
                processed_item["processed"] = await self._add_nlp_features(text)
                
                # Add to processed data
                processed_data.append(processed_item)
            
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
        
        self.logger.info(f"Processed {len(processed_data)} documents")
        return processed_data
    
    async def _add_nlp_features(self, text: str) -> Dict[str, Any]:
        """Add NLP features to text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of NLP features
        """
        features = {}
        
        if not self.nlp_service:
            return features
        
        try:
            # Analyze sentiment
            sentiment_scores = await self.nlp_service.analyze_sentiment([text])
            if sentiment_scores:
                features["sentiment"] = sentiment_scores[0]
            
            # Extract entities
            entities = await self.nlp_service.extract_entities(text)
            features["entities"] = entities
            
            # Extract keywords
            keywords = await self.nlp_service.extract_keywords(text)
            features["keywords"] = keywords
            
            # Add additional features in a real implementation
            # - Topic classification
            # - Named entity linking
            # - Event extraction
            # - etc.
        
        except Exception as e:
            self.logger.error(f"Error adding NLP features: {e}")
        
        return features


class SocialMediaProcessor(BaseProcessor):
    """Processor for social media data."""
    
    def __init__(self):
        """Initialize the social media processor."""
        super().__init__("social_media")
        self.nlp_service = None
    
    async def initialize(self):
        """Initialize the social media processor."""
        await super().initialize()
        
        # Initialize NLP service
        self.nlp_service = NLPService()
        await self.nlp_service.initialize()
        
        self.logger.info("Social media processor initialized")
    
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process social media data.
        
        Args:
            data: Raw social media data
            
        Returns:
            Processed social media data with enhanced features
        """
        self.logger.info(f"Processing {len(data)} social media items")
        processed_data = []
        
        for item in data:
            try:
                # Create processed item with original data
                processed_item = item.copy()
                
                # Extract text content
                text = item.get("content", "")
                if not text:
                    continue
                
                # Add NLP features
                processed_item["processed"] = await self._add_nlp_features(text)
                
                # Add social media specific features
                processed_item["processed"]["social_metrics"] = self._calculate_social_metrics(item)
                
                # Add to processed data
                processed_data.append(processed_item)
            
            except Exception as e:
                self.logger.error(f"Error processing social media item: {e}")
        
        self.logger.info(f"Processed {len(processed_data)} social media items")
        return processed_data
    
    async def _add_nlp_features(self, text: str) -> Dict[str, Any]:
        """Add NLP features to text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of NLP features
        """
        features = {}
        
        if not self.nlp_service:
            return features
        
        try:
            # Analyze sentiment
            sentiment_scores = await self.nlp_service.analyze_sentiment([text])
            if sentiment_scores:
                features["sentiment"] = sentiment_scores[0]
            
            # Extract entities
            entities = await self.nlp_service.extract_entities(text)
            features["entities"] = entities
            
            # Extract keywords
            keywords = await self.nlp_service.extract_keywords(text)
            features["keywords"] = keywords
        
        except Exception as e:
            self.logger.error(f"Error adding NLP features: {e}")
        
        return features
    
    def _calculate_social_metrics(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate social media specific metrics.
        
        Args:
            item: Social media item
            
        Returns:
            Dictionary of social metrics
        """
        metrics = {
            "influence_score": 0.0,
            "spread_potential": 0.0,
            "engagement_level": 0.0
        }
        
        # Extract metadata
        metadata = item.get("metadata", {})
        
        # Calculate metrics based on platform
        source = item.get("source", {})
        source_name = source.get("name", "").lower() if isinstance(source, dict) else ""
        
        if source_name == "twitter":
            # Calculate Twitter-specific metrics
            likes = metadata.get("likes", 0)
            retweets = metadata.get("retweets", 0)
            followers = metadata.get("followers", 0)
            
            # Simple influence score based on followers
            metrics["influence_score"] = min(1.0, followers / 100000)
            
            # Spread potential based on retweets and followers
            metrics["spread_potential"] = min(1.0, (retweets * 10) / (followers + 1))
            
            # Engagement level based on likes and retweets
            metrics["engagement_level"] = min(1.0, (likes + retweets * 2) / 1000)
        
        elif source_name == "reddit":
            # Calculate Reddit-specific metrics
            upvotes = metadata.get("upvotes", 0)
            comments = metadata.get("comments", 0)
            
            # Simple influence score based on upvotes
            metrics["influence_score"] = min(1.0, upvotes / 5000)
            
            # Spread potential based on comments
            metrics["spread_potential"] = min(1.0, comments / 500)
            
            # Engagement level based on upvotes and comments
            metrics["engagement_level"] = min(1.0, (upvotes + comments * 5) / 5000)
        
        return metrics


class FinancialDataProcessor(BaseProcessor):
    """Processor for financial market data."""
    
    def __init__(self):
        """Initialize the financial data processor."""
        super().__init__("financial_data")
    
    async def initialize(self):
        """Initialize the financial data processor."""
        await super().initialize()
        self.logger.info("Financial data processor initialized")
    
    async def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process financial market data.
        
        Args:
            data: Raw financial data
            
        Returns:
            Processed financial data with calculated indicators
        """
        self.logger.info(f"Processing {len(data)} financial data items")
        processed_data = []
        
        for item in data:
            try:
                # Create processed item with original data
                processed_item = item.copy()
                
                # Extract data
                market_data = item.get("data", {})
                data_type = item.get("data_type", "")
                
                # Add calculated features
                processed_item["processed"] = {
                    "anomalies": self._detect_anomalies(market_data, data_type),
                    "indicators": self._calculate_indicators(market_data, data_type)
                }
                
                # Add to processed data
                processed_data.append(processed_item)
            
            except Exception as e:
                self.logger.error(f"Error processing financial data: {e}")
        
        self.logger.info(f"Processed {len(processed_data)} financial data items")
        return processed_data
    
    def _detect_anomalies(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Detect anomalies in financial data.
        
        Args:
            data: Financial data
            data_type: Type of data
            
        Returns:
            Dictionary of detected anomalies
        """
        # In a real implementation, this would use statistical methods
        # For now, return mock data
        anomalies = {}
        
        # Mock anomaly detection
        for asset, value in data.items():
            # Simulate random anomalies for demonstration
            is_anomaly = False
            
            # For demo purposes, detect anomalies for specific asset-type combinations
            if asset == "BTC" and data_type == "volume":
                is_anomaly = True
                anomalies[asset] = {
                    "is_anomaly": True,
                    "score": 0.85,
                    "description": "Unusually high trading volume"
                }
            elif asset == "ETH" and data_type == "options":
                is_anomaly = True
                anomalies[asset] = {
                    "is_anomaly": True,
                    "score": 0.75,
                    "description": "Unusual options activity"
                }
        
        return anomalies
    
    def _calculate_indicators(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Calculate indicators from financial data.
        
        Args:
            data: Financial data
            data_type: Type of data
            
        Returns:
            Dictionary of calculated indicators
        """
        # In a real implementation, this would calculate technical indicators
        # For now, return mock data
        indicators = {}
        
        if data_type == "price":
            # Mock price indicators
            for asset, value in data.items():
                indicators[asset] = {
                    "momentum": 0.6,  # Mock value
                    "volatility": 0.4,  # Mock value
                    "trend": "bullish" if asset in ["BTC", "SOL"] else "bearish"
                }
        
        elif data_type == "volume":
            # Mock volume indicators
            for asset, value in data.items():
                indicators[asset] = {
                    "volume_surge": 0.3,  # Mock value
                    "buy_sell_ratio": 1.2,  # Mock value
                    "unusual_activity": asset == "BTC"
                }
        
        return indicators


class ProcessorManager:
    """Manager for data processors."""
    
    def __init__(self):
        """Initialize the processor manager."""
        self.logger = get_logger("early_detection", "processor_manager")
        self.processors = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the processor manager and all processors."""
        self.logger.info("Initializing processor manager")
        
        # Initialize processors
        self.processors = {
            "document": DocumentProcessor(),
            "social_media": SocialMediaProcessor(),
            "financial_data": FinancialDataProcessor()
        }
        
        # Initialize each processor
        for processor_name, processor in self.processors.items():
            try:
                processor.initialize()
                self.logger.info(f"Initialized {processor_name} processor")
            except Exception as e:
                self.logger.error(f"Error initializing {processor_name} processor: {e}")
        
        self.is_initialized = True
        self.logger.info("Processor manager initialized")
    
    async def process_data(self, data: Dict[SourceType, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Process data from all sources.
        
        Args:
            data: Dictionary mapping source types to collected data
            
        Returns:
            List of processed data items
        """
        if not self.is_initialized:
            self.logger.warning("Processor manager not initialized")
            return []
        
        self.logger.info("Processing data from all sources")
        
        # Create tasks for processing different types of data
        tasks = []
        
        # Process documents (news, official sources)
        documents = []
        documents.extend(data.get(SourceType.NEWS, []))
        documents.extend(data.get(SourceType.OFFICIAL, []))
        
        if documents and "document" in self.processors:
            tasks.append(asyncio.create_task(
                self.processors["document"].process(documents)
            ))
        
        # Process social media
        social_media = data.get(SourceType.SOCIAL_MEDIA, [])
        if social_media and "social_media" in self.processors:
            tasks.append(asyncio.create_task(
                self.processors["social_media"].process(social_media)
            ))
        
        # Process financial data
        financial_data = data.get(SourceType.FINANCIAL_DATA, [])
        if financial_data and "financial_data" in self.processors:
            tasks.append(asyncio.create_task(
                self.processors["financial_data"].process(financial_data)
            ))
        
        # Wait for all tasks to complete
        results = []
        for task in tasks:
            try:
                task_results = await task
                results.extend(task_results)
            except Exception as e:
                self.logger.error(f"Error in processing task: {e}")
        
        self.logger.info(f"Processed {len(results)} data items")
        return results