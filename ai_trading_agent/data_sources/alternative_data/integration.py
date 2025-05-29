"""
Alternative Data Integration Module.

This module provides a unified interface for accessing and analyzing data from
multiple alternative data sources, combining signals to produce actionable
trading insights.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set

import pandas as pd
import numpy as np

from .base import AlternativeDataConfig
from .satellite_imagery import SatelliteImageryAnalyzer
from .social_media import SocialMediaSentimentAnalyzer
from .supply_chain import SupplyChainDataAnalyzer
from ...health.monitoring import register_health_check, HealthStatus


logger = logging.getLogger(__name__)


class AlternativeDataIntegration:
    """
    Unified interface for alternative data integration.
    
    This class manages multiple alternative data sources, coordinates data
    fetching, and combines signals into unified trading insights.
    """
    
    def __init__(
        self,
        satellite_config: Optional[AlternativeDataConfig] = None,
        social_media_config: Optional[AlternativeDataConfig] = None,
        supply_chain_config: Optional[AlternativeDataConfig] = None
    ):
        """
        Initialize the alternative data integration.
        
        Args:
            satellite_config: Configuration for satellite imagery
            social_media_config: Configuration for social media sentiment
            supply_chain_config: Configuration for supply chain data
        """
        self.sources = {}
        self.last_fetch = {}
        self.signals = {}
        
        # Initialize data sources if provided
        if satellite_config:
            self.sources["satellite"] = SatelliteImageryAnalyzer(satellite_config)
        
        if social_media_config:
            self.sources["social_media"] = SocialMediaSentimentAnalyzer(social_media_config)
        
        if supply_chain_config:
            self.sources["supply_chain"] = SupplyChainDataAnalyzer(supply_chain_config)
        
        # Register with health monitoring
        register_health_check(
            component_id="alternative_data_integration",
            check_function=self.get_health_status,
            interval_seconds=600
        )
        
        logger.info(f"Initialized alternative data integration with sources: {list(self.sources.keys())}")
    
    def add_data_source(self, name: str, source: Any) -> None:
        """
        Add a new alternative data source.
        
        Args:
            name: Identifier for the data source
            source: Data source object
        """
        self.sources[name] = source
        logger.info(f"Added alternative data source: {name}")
    
    async def fetch_all_data(self, query_params: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all available sources.
        
        Args:
            query_params: Dictionary mapping source names to their query parameters
            
        Returns:
            Dictionary of source name to DataFrame of results
        """
        tasks = []
        for source_name, source in self.sources.items():
            if source_name in query_params:
                task = asyncio.create_task(
                    self._fetch_source_data(source_name, source, query_params[source_name])
                )
                tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_dict = {}
        for i, source_name in enumerate([name for name in self.sources.keys() if name in query_params]):
            result = results[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error fetching data from {source_name}: {result}")
                data_dict[source_name] = pd.DataFrame()
            else:
                data_dict[source_name] = result
                self.last_fetch[source_name] = datetime.now()
        
        return data_dict
    
    async def _fetch_source_data(
        self, 
        source_name: str, 
        source: Any, 
        query: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Fetch data from a single source with error handling.
        
        Args:
            source_name: Name of the data source
            source: Data source object
            query: Query parameters
            
        Returns:
            DataFrame of results
        """
        try:
            logger.info(f"Fetching data from {source_name} with query: {query}")
            data = await source.fetch_data(query)
            logger.info(f"Successfully fetched {len(data)} records from {source_name}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data from {source_name}: {e}")
            raise
    
    def process_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process raw data into trading signals across all sources.
        
        Args:
            data_dict: Dictionary of source name to DataFrame
            
        Returns:
            Dictionary containing aggregated signals and insights
        """
        source_signals = {}
        
        # Process signals from each source
        for source_name, data in data_dict.items():
            if data.empty:
                logger.warning(f"Empty data from {source_name}, skipping signal processing")
                continue
                
            try:
                source = self.sources[source_name]
                signal = source.process_signal(data)
                source_signals[source_name] = signal
                self.signals[source_name] = signal
                
                logger.info(
                    f"Processed signal from {source_name}: {signal['signal']} "
                    f"(strength: {signal['strength']:.2f})"
                )
            except Exception as e:
                logger.error(f"Error processing signal from {source_name}: {e}")
        
        if not source_signals:
            return {
                "signal": "neutral",
                "strength": 0,
                "source_signals": {},
                "insights": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Aggregate signals across sources
        return self._aggregate_signals(source_signals)
    
    def _aggregate_signals(self, source_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate signals from multiple sources into a unified signal.
        
        Args:
            source_signals: Dictionary of source name to signal data
            
        Returns:
            Aggregated signal
        """
        # Source weights - can be adjusted based on historical performance
        source_weights = {
            "satellite": 0.35,
            "social_media": 0.30,
            "supply_chain": 0.35
        }
        
        # Normalize weights for available sources
        available_sources = list(source_signals.keys())
        total_weight = sum(source_weights.get(source, 0.33) for source in available_sources)
        
        normalized_weights = {
            source: source_weights.get(source, 0.33) / total_weight 
            for source in available_sources
        }
        
        # Calculate weighted signal
        weighted_sum = 0
        for source, signal in source_signals.items():
            # Convert to numeric value (-1 to 1)
            if signal["signal"] == "bullish":
                numeric_signal = signal["strength"]
            elif signal["signal"] == "bearish":
                numeric_signal = -signal["strength"]
            else:
                numeric_signal = 0
                
            weighted_sum += numeric_signal * normalized_weights[source]
        
        # Determine aggregated signal direction and strength
        if weighted_sum > 0.1:
            signal = "bullish"
            strength = weighted_sum
        elif weighted_sum < -0.1:
            signal = "bearish"
            strength = abs(weighted_sum)
        else:
            signal = "neutral"
            strength = abs(weighted_sum)
        
        # Collect all insights
        all_insights = []
        for source, signal_data in source_signals.items():
            insights = signal_data.get("insights", [])
            for insight in insights:
                # Add source to each insight
                insight["source"] = source
                all_insights.append(insight)
        
        return {
            "signal": signal,
            "strength": float(strength),
            "source_signals": source_signals,
            "source_weights": normalized_weights,
            "insights": all_insights,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_recent_signals(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Get recently processed signals that are still valid.
        
        Args:
            max_age_hours: Maximum age of signals in hours
            
        Returns:
            Dictionary of recent signals by source
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        recent_signals = {}
        for source, last_time in self.last_fetch.items():
            if last_time >= cutoff_time and source in self.signals:
                recent_signals[source] = self.signals[source]
        
        return recent_signals
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all alternative data sources.
        
        Returns:
            Dictionary containing health status information
        """
        status = {
            "component": "alternative_data_integration",
            "status": "healthy",
            "sources": {},
            "last_fetch": {},
            "source_count": len(self.sources)
        }
        
        # Collect source statuses
        for source_name, source in self.sources.items():
            try:
                source_status = source.get_health_status()
                status["sources"][source_name] = source_status
                
                if source_name in self.last_fetch:
                    status["last_fetch"][source_name] = self.last_fetch[source_name].isoformat()
            except Exception as e:
                status["sources"][source_name] = {
                    "status": "error",
                    "error": str(e)
                }
                status["status"] = "degraded"
        
        # Determine overall status
        if not self.sources:
            status["status"] = "warning"
            status["message"] = "No alternative data sources configured"
            
        return status
    
    async def __aenter__(self):
        """Support for async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting async context."""
        # Close all sources if they support it
        for source in self.sources.values():
            if hasattr(source, "__aexit__"):
                await source.__aexit__(exc_type, exc_val, exc_tb)
