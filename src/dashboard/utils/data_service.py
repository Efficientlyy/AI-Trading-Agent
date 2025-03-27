"""
Data Service Module

This module provides the data service layer for the dashboard, handling
both mock and real data sources with caching capabilities.

Follows the Single Responsibility Principle by separating data handling
from visualization and UI logic.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional

from src.dashboard.utils.enums import DataSource
from src.dashboard.utils.mock_data import MockDataGenerator

# Flag to indicate if real data components are available
REAL_DATA_AVAILABLE = False

try:
    # Attempt to import real data components
    from src.common.performance import PerformanceTracker
    from src.common.system import SystemMonitor
    from src.common.logging import LogQuery
    REAL_DATA_AVAILABLE = True
except ImportError:
    # Mock implementations for standalone mode
    class LogQuery:
        """Mock Log Query class"""
        def get_logs(self, limit=100, level=None, component=None, search=None):
            return []

    class PerformanceTracker:
        """Mock Performance Tracker class"""
        def get_performance_summary(self, period="daily"):
            return {}
        
        def get_performance_metrics(self):
            return {}

    class SystemMonitor:
        """Mock System Monitor class"""
        def get_system_health(self):
            return {}


class DataService:
    """
    Data service for flexible data sourcing with caching
    
    This service allows the dashboard to switch between mock and real data
    sources while providing a consistent interface.
    """
    
    def __init__(self, data_source=DataSource.MOCK):
        """Initialize data service with specified source"""
        self.data_source = data_source
        self.mock_data = MockDataGenerator()
        
        # Cached data with timestamps
        self.cache = {}
        self.cache_expiry = {
            'system_health': 5,  # 5 seconds
            'component_status': 10,  # 10 seconds
            'trading_performance': 30,  # 30 seconds
            'market_regime': 60,  # 1 minute
            'sentiment': 60,  # 1 minute
            'risk_management': 60,  # 1 minute
            'performance_analytics': 300,  # 5 minutes
            'logs_monitoring': 10,  # 10 seconds
        }
    
    def set_data_source(self, data_source: str) -> None:
        """Set the data source to use (mock or real)"""
        if data_source not in [DataSource.MOCK, DataSource.REAL]:
            raise ValueError(f"Invalid data source: {data_source}")
        
        # Don't allow switching to real data if not available
        if data_source == DataSource.REAL and not REAL_DATA_AVAILABLE:
            raise ValueError("Real data sources are not available")
        
        self.data_source = data_source
        self.cache.clear()  # Clear cache on data source change
    
    def get_data(self, data_type: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get data of specified type from current source with caching"""
        now = time.time()
        
        # Check if we have a valid cached item
        if not force_refresh and data_type in self.cache:
            cached_item = self.cache[data_type]
            expiry_time = self.cache_expiry.get(data_type, 30)  # Default 30s expiry
            
            if now - cached_item['timestamp'] < expiry_time:
                return cached_item['data']
        
        # If no valid cache, fetch fresh data
        data = self._fetch_data(data_type)
        
        # Update cache
        self.cache[data_type] = {
            'data': data,
            'timestamp': now
        }
        
        return data
    
    def _fetch_data(self, data_type: str) -> Dict[str, Any]:
        """Fetch data from appropriate source based on type"""
        if self.data_source == DataSource.MOCK:
            return self._get_mock_data(data_type)
        else:
            return self._get_real_data(data_type)
    
    def _get_mock_data(self, data_type: str) -> Dict[str, Any]:
        """Get mock data from generator"""
        data_generators = {
            'system_health': self.mock_data.generate_system_health,
            'component_status': self.mock_data.generate_component_status,
            'trading_performance': self.mock_data.generate_trading_performance,
            'market_regime': self.mock_data.generate_market_regime_data,
        }
        
        generator = data_generators.get(data_type)
        if generator:
            return generator()
        
        # Return empty dict if no generator available
        return {}
    
    def _get_real_data(self, data_type: str) -> Dict[str, Any]:
        """Get real data from system components"""
        if not REAL_DATA_AVAILABLE:
            return self._get_mock_data(data_type)  # Fall back to mock if real unavailable
        
        # Map data types to real data functions
        data_fetchers = {
            'system_health': SystemMonitor().get_system_health,
            'trading_performance': self._get_real_performance_data,
            'logs_monitoring': self._get_real_logs_data,
        }
        
        fetcher = data_fetchers.get(data_type)
        if fetcher:
            try:
                return fetcher()
            except Exception as e:
                # Log the error and fall back to mock data
                print(f"Error fetching real data for {data_type}: {e}")
                return self._get_mock_data(data_type)
        
        # Fall back to mock data if no real data fetcher available
        return self._get_mock_data(data_type)
    
    def _get_real_performance_data(self) -> Dict[str, Any]:
        """Get real performance data from PerformanceTracker"""
        tracker = PerformanceTracker()
        summary = tracker.get_performance_summary()
        metrics = tracker.get_performance_metrics()
        
        # Combine data
        return {
            **summary,
            **metrics
        }
    
    def _get_real_logs_data(self) -> Dict[str, Any]:
        """Get real logs data from LogQuery"""
        query = LogQuery()
        logs = query.get_logs(limit=100)
        
        return {
            'logs': logs,
            'count': len(logs),
            'timestamp': datetime.now().isoformat()
        }
