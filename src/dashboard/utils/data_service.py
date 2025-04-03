"""
Data Service Module

This module provides the data service layer for the dashboard, handling
both mock and real data sources with caching capabilities.

Follows the Single Responsibility Principle by separating data handling
from visualization and UI logic.

Enhanced with:
- Robust error tracking for data sources
- Advanced caching strategy
- Data validation
- Intelligent fallback mechanisms
- Real-time data source status tracking
"""

import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from enum import Enum
from pathlib import Path

from src.dashboard.utils.enums import DataSource
from src.dashboard.utils.mock_data import MockDataGenerator

# Import real data components
from src.common.performance import PerformanceTracker, get_performance_tracker
from src.common.system import SystemMonitor, get_system_monitor
from src.common.log_query import LogQuery

# Flag to indicate if real data components are available
REAL_DATA_AVAILABLE = True

# Check if data directories exist
from pathlib import Path

DATA_DIR = Path("data")
TRADES_DIR = DATA_DIR / "trades"
LOGS_DIR = Path("logs")
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Check if we have any real data
HAS_TRADE_DATA = len(list(TRADES_DIR.glob("*.json"))) > 0
HAS_LOG_DATA = len(list(LOGS_DIR.glob("*.log*"))) > 0

# Create component availability flags
COMPONENTS_AVAILABLE = {
    "system_health": True,  # SystemMonitor always available
    "logs_monitoring": HAS_LOG_DATA,
    "trading_performance": HAS_TRADE_DATA,
    "market_regime": False,  # No real data yet
    "sentiment": False,      # No real data yet
    "risk_management": False, # No real data yet
    "performance_analytics": False # No real data yet
}

# Configure logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"Real data components available: {REAL_DATA_AVAILABLE}")
logger.info(f"Component availability: {COMPONENTS_AVAILABLE}")

# Data source health status enum
class DataSourceStatus(Enum):
    """Health status of a data source"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DataService:
    """
    Enhanced data service for flexible data sourcing with advanced caching
    
    This service allows the dashboard to switch between mock and real data
    sources while providing a consistent interface. It includes:
    
    - Robust error tracking for each data source
    - Advanced caching with stale-while-revalidate pattern
    - Data validation against expected schemas
    - Intelligent fallback mechanisms
    - Real-time data source status monitoring
    """
    
    # Schema definitions for data validation
    DATA_SCHEMAS = {
        'system_health': {
            'required_fields': ['status', 'cpu_usage', 'memory_usage', 'disk_usage', 'network_latency'],
            'field_types': {
                'status': str,
                'cpu_usage': float,
                'memory_usage': float,
                'disk_usage': float,
                'network_latency': float,
                'uptime': int
            }
        },
        'trading_performance': {
            'required_fields': ['equity', 'daily_pnl', 'win_rate'],
            'field_types': {
                'equity': float,
                'daily_pnl': float,
                'daily_pnl_pct': float,
                'win_rate': float,
                'trades_today': int
            }
        },
        'logs_monitoring': {
            'required_fields': ['logs', 'count'],
            'field_types': {
                'logs': list,
                'count': int
            }
        }
    }
    
    def __init__(self, data_source=DataSource.MOCK):
        """Initialize data service with specified source"""
        self.data_source = data_source
        self.mock_data = MockDataGenerator()
        
        # Cached data with timestamps
        self.cache = {}
        self.persistent_cache = {}
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
        
        # Error tracking for data sources
        self.error_tracking = {}
        self.error_threshold = 3  # Number of consecutive errors to mark a source as unhealthy
        self.error_window = 300  # Time window for error rate calculation (5 minutes)
        
        # Data source health status
        self.data_source_status = {}
        self.last_health_check = {}
        
        # Load persistent cache from disk
        self._load_persistent_cache()
        
        # Initialize data source health status
        self._initialize_data_source_status()
    
    def set_data_source(self, data_source: str) -> Dict[str, Any]:
        """
        Set the data source to use (mock or real)
        
        Args:
            data_source: The data source to use (MOCK or REAL)
            
        Returns:
            Dictionary with status information about the data source change
        """
        if data_source not in [DataSource.MOCK, DataSource.REAL]:
            raise ValueError(f"Invalid data source: {data_source}")
        
        # Don't allow switching to real data if not available
        if data_source == DataSource.REAL and not REAL_DATA_AVAILABLE:
            logger.error(f"Real data sources are not available. REAL_DATA_AVAILABLE={REAL_DATA_AVAILABLE}")
            return {
                "success": False,
                "message": "Real data sources are not available",
                "current_source": self.data_source,
                "available_components": {}
            }
        
        # If switching to real data, check which components have data
        available_components = {}
        if data_source == DataSource.REAL:
            available_components = COMPONENTS_AVAILABLE.copy()
            logger.info(f"Available components for real data: {available_components}")
            
            # If no components have data, don't allow switching
            if not any(available_components.values()):
                logger.error(f"No real data available for any component. COMPONENTS_AVAILABLE={COMPONENTS_AVAILABLE}")
                return {
                    "success": False,
                    "message": "No real data available for any component",
                    "current_source": self.data_source,
                    "available_components": available_components
                }
        
        # Switch data source
        old_source = self.data_source
        self.data_source = data_source
        self.cache.clear()  # Clear cache on data source change
        
        logger.info(f"Data source changed from {old_source} to {data_source}")
        
        # Return status information
        return {
            "success": True,
            "message": f"Data source changed to {data_source}",
            "previous_source": old_source,
            "current_source": self.data_source,
            "available_components": available_components
        }
        
    def get_data_source_status(self) -> Dict[str, Any]:
        """
        Get status information about the current data source
        
        Returns:
            Dictionary with status information
        """
        available_components = {}
        if self.data_source == DataSource.REAL:
            available_components = COMPONENTS_AVAILABLE.copy()
            
        return {
            "current_source": self.data_source,
            "real_data_available": REAL_DATA_AVAILABLE,
            "available_components": available_components,
            "has_trade_data": HAS_TRADE_DATA,
            "has_log_data": HAS_LOG_DATA
        }
    
    def get_data(self, data_type: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get data of specified type from current source with caching
        
        This method now uses the enhanced get_data_with_fallback method
        which provides better error handling, validation, and caching.
        
        Args:
            data_type: Type of data to fetch
            force_refresh: Whether to force a refresh
            
        Returns:
            Data dictionary
        """
        return self.get_data_with_fallback(data_type, force_refresh)
    
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
        # Check if real data is available globally
        if not REAL_DATA_AVAILABLE:
            logger.warning(f"Real data components not available, using mock data for {data_type}")
            return self._get_mock_data(data_type)
            
        # Check if this specific component has data available
        if data_type in COMPONENTS_AVAILABLE and not COMPONENTS_AVAILABLE[data_type]:
            logger.info(f"Component {data_type} has no real data available, using mock data")
            return self._get_mock_data(data_type)
        
        # Map data types to real data functions
        data_fetchers = {
            'system_health': self._get_real_system_health,
            'trading_performance': self._get_real_performance_data,
            'logs_monitoring': self._get_real_logs_data,
            'component_status': self._get_real_component_status,
            'market_regime': self._get_mock_data,  # No real data yet, use mock
            'sentiment': self._get_mock_data,      # No real data yet, use mock
            'risk_management': self._get_mock_data, # No real data yet, use mock
            'performance_analytics': self._get_mock_data, # No real data yet, use mock
        }
        
        # Get the appropriate fetcher function
        fetcher = data_fetchers.get(data_type)
        if not fetcher:
            # If no fetcher is defined, fall back to mock data
            logger.info(f"No real data fetcher for {data_type}, using mock data")
            return self._get_mock_data(data_type)
            
        # Try to fetch real data with error handling
        try:
            logger.debug(f"Fetching real data for {data_type}")
            
            # Check cache first if caching is enabled for this data type
            cache_key = f"real_{data_type}"
            if cache_key in self.cache and not self._is_cache_expired(cache_key):
                logger.debug(f"Using cached real data for {data_type}")
                return self.cache[cache_key]['data']
            
            # Fetch fresh data
            start_time = time.time()
            data = fetcher() if data_type not in ['market_regime', 'sentiment', 'risk_management', 'performance_analytics'] else fetcher(data_type)
            
            # Add metadata to indicate this is real data
            if isinstance(data, dict):
                data['data_source'] = 'real'
                data['timestamp'] = datetime.now().isoformat()
                data['fetch_time'] = time.time() - start_time
            
            # Cache the result if it's valid
            if data:
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
            
            return data
        except Exception as e:
            # Log the error and fall back to mock data
            logger.error(f"Error fetching real data for {data_type}: {e}", exc_info=True)
            
            # Try to use cached data if available, even if expired
            cache_key = f"real_{data_type}"
            if cache_key in self.cache:
                logger.warning(f"Using expired cached data for {data_type} due to fetch error")
                return self.cache[cache_key]['data']
            
            # Otherwise fall back to mock data
            logger.warning(f"Falling back to mock data for {data_type}")
            return self._get_mock_data(data_type)
    
    def _is_cache_expired(self, cache_key: str) -> bool:
        """Check if a cache entry is expired"""
        if cache_key not in self.cache:
            return True
            
        now = time.time()
        timestamp = self.cache[cache_key]['timestamp']
        data_type = cache_key.split('_', 1)[1] if '_' in cache_key else cache_key
        expiry_time = self.cache_expiry.get(data_type, 30)  # Default 30s expiry
        
        return now - timestamp >= expiry_time
        
    def _get_real_system_health(self) -> Dict[str, Any]:
        """Get real system health data from SystemMonitor"""
        # Use the singleton instance
        monitor = get_system_monitor()
        
        # Get system health data
        health_data = monitor.get_system_health()
        
        # Add additional metadata
        health_data["data_source"] = "real"
        health_data["timestamp"] = datetime.now().isoformat()
        
        return health_data
        
    def _get_real_component_status(self) -> Dict[str, Any]:
        """Get real component status data"""
        # Create component status based on real availability
        components = []
        
        # System components
        components.append({
            "name": "System Monitor",
            "status": "running",
            "last_update": datetime.now().isoformat(),
            "version": "1.0.0",
            "load": get_system_monitor()._get_cpu_usage(),
            "message": ""
        })
        
        # Log Query component
        log_status = "running" if HAS_LOG_DATA else "warning"
        log_message = "" if HAS_LOG_DATA else "No log data available"
        components.append({
            "name": "Log Query",
            "status": log_status,
            "last_update": datetime.now().isoformat(),
            "version": "1.0.0",
            "load": 0,
            "message": log_message
        })
        
        # Performance Tracker component
        perf_status = "running" if HAS_TRADE_DATA else "warning"
        perf_message = "" if HAS_TRADE_DATA else "No trade data available"
        components.append({
            "name": "Performance Tracker",
            "status": perf_status,
            "last_update": datetime.now().isoformat(),
            "version": "1.0.0",
            "load": 0,
            "message": perf_message
        })
        
        return {"components": components}
    
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
    
    # Enhanced methods for error tracking, data validation, and caching
    
    def _initialize_data_source_status(self) -> None:
        """Initialize the health status for all data sources"""
        for data_type in COMPONENTS_AVAILABLE.keys():
            self.data_source_status[data_type] = DataSourceStatus.UNKNOWN
            self.error_tracking[data_type] = {
                'consecutive_errors': 0,
                'error_history': [],
                'last_success': None
            }
            self.last_health_check[data_type] = time.time()
    
    def _track_error(self, data_type: str, error: Exception) -> None:
        """
        Track an error for a specific data source
        
        Args:
            data_type: The data type that experienced an error
            error: The exception that occurred
        """
        now = time.time()
        
        # Initialize tracking if not exists
        if data_type not in self.error_tracking:
            self.error_tracking[data_type] = {
                'consecutive_errors': 0,
                'error_history': [],
                'last_success': None
            }
        
        # Update error tracking
        self.error_tracking[data_type]['consecutive_errors'] += 1
        self.error_tracking[data_type]['error_history'].append({
            'timestamp': now,
            'error': str(error),
            'type': type(error).__name__
        })
        
        # Trim error history to keep only recent errors
        self.error_tracking[data_type]['error_history'] = [
            e for e in self.error_tracking[data_type]['error_history']
            if now - e['timestamp'] < self.error_window
        ]
        
        # Update data source status based on errors
        self._update_data_source_status(data_type)
        
        # Log the error
        logger.error(f"Error in data source {data_type}: {error}")
    
    def _track_success(self, data_type: str) -> None:
        """
        Track a successful data fetch for a specific data source
        
        Args:
            data_type: The data type that was successfully fetched
        """
        now = time.time()
        
        # Initialize tracking if not exists
        if data_type not in self.error_tracking:
            self.error_tracking[data_type] = {
                'consecutive_errors': 0,
                'error_history': [],
                'last_success': None
            }
        
        # Update tracking
        self.error_tracking[data_type]['consecutive_errors'] = 0
        self.error_tracking[data_type]['last_success'] = now
        
        # Update data source status
        self._update_data_source_status(data_type)
    
    def _update_data_source_status(self, data_type: str) -> None:
        """
        Update the health status of a data source based on error tracking
        
        Args:
            data_type: The data type to update status for
        """
        if data_type not in self.error_tracking:
            self.data_source_status[data_type] = DataSourceStatus.UNKNOWN
            return
        
        tracking = self.error_tracking[data_type]
        now = time.time()
        
        # Calculate error rate
        recent_errors = len(tracking['error_history'])
        
        # If no errors and recent success, mark as healthy
        if recent_errors == 0 and tracking['last_success'] is not None:
            self.data_source_status[data_type] = DataSourceStatus.HEALTHY
            return
        
        # If consecutive errors exceed threshold, mark as unhealthy
        if tracking['consecutive_errors'] >= self.error_threshold:
            self.data_source_status[data_type] = DataSourceStatus.UNHEALTHY
            return
        
        # If some errors but not too many, mark as degraded
        if recent_errors > 0:
            self.data_source_status[data_type] = DataSourceStatus.DEGRADED
            return
        
        # Default to unknown if we can't determine status
        self.data_source_status[data_type] = DataSourceStatus.UNKNOWN
    
    def get_data_source_health(self) -> Dict[str, Any]:
        """
        Get health status for all data sources
        
        Returns:
            Dictionary with health status for each data source
        """
        result = {}
        
        for data_type in COMPONENTS_AVAILABLE.keys():
            status = self.data_source_status.get(data_type, DataSourceStatus.UNKNOWN)
            tracking = self.error_tracking.get(data_type, {
                'consecutive_errors': 0,
                'error_history': [],
                'last_success': None
            })
            
            result[data_type] = {
                'status': status.value,
                'consecutive_errors': tracking['consecutive_errors'],
                'recent_errors': len(tracking['error_history']),
                'last_success': tracking['last_success'],
                'available': COMPONENTS_AVAILABLE.get(data_type, False)
            }
        
        return result
    
    def _validate_data(self, data_type: str, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate data against expected schema
        
        Args:
            data_type: The type of data to validate
            data: The data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Skip validation if no schema exists
        if data_type not in self.DATA_SCHEMAS:
            return True, None
        
        schema = self.DATA_SCHEMAS[data_type]
        
        # Check required fields
        for field in schema.get('required_fields', []):
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check field types
        for field, expected_type in schema.get('field_types', {}).items():
            if field in data and not isinstance(data[field], expected_type):
                return False, f"Field {field} has wrong type: expected {expected_type.__name__}, got {type(data[field]).__name__}"
        
        return True, None
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache from disk"""
        cache_file = CACHE_DIR / "data_service_cache.json"
        
        if not cache_file.exists():
            return
        
        try:
            with open(cache_file, 'r') as f:
                self.persistent_cache = json.load(f)
                logger.info(f"Loaded persistent cache with {len(self.persistent_cache)} entries")
        except Exception as e:
            logger.error(f"Error loading persistent cache: {e}")
            self.persistent_cache = {}
    
    def _save_persistent_cache(self) -> None:
        """Save persistent cache to disk"""
        cache_file = CACHE_DIR / "data_service_cache.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.persistent_cache, f)
                logger.info(f"Saved persistent cache with {len(self.persistent_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving persistent cache: {e}")
    
    def get_data_with_fallback(self, data_type: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get data with enhanced fallback mechanisms
        
        This method extends get_data with:
        - Stale-while-revalidate caching
        - Data validation
        - Error tracking
        - Persistent caching for critical data
        
        Args:
            data_type: Type of data to fetch
            force_refresh: Whether to force a refresh
            
        Returns:
            Data dictionary
        """
        now = time.time()
        
        # Check if we have a valid cached item
        if not force_refresh and data_type in self.cache:
            cached_item = self.cache[data_type]
            expiry_time = self.cache_expiry.get(data_type, 30)
            
            # If cache is still fresh, return it immediately
            if now - cached_item['timestamp'] < expiry_time:
                return cached_item['data']
            
            # If cache is stale but not too old, return it and trigger async refresh
            if now - cached_item['timestamp'] < expiry_time * 3:
                # In a real async implementation, we would spawn a background task here
                # For now, we'll just log that we're using stale data
                logger.info(f"Using stale data for {data_type} while refreshing")
                
                # Schedule refresh for next request
                self.cache[data_type]['needs_refresh'] = True
                return cached_item['data']
        
        # If we need to fetch fresh data
        try:
            # Fetch fresh data
            data = self._fetch_data(data_type)
            
            # Validate data
            is_valid, error_message = self._validate_data(data_type, data)
            if not is_valid:
                logger.warning(f"Invalid data for {data_type}: {error_message}")
                
                # Try to use cached data if available
                if data_type in self.cache:
                    logger.info(f"Falling back to cached data for {data_type} due to validation failure")
                    return self.cache[data_type]['data']
                
                # Try to use persistent cache if available
                if data_type in self.persistent_cache:
                    logger.info(f"Falling back to persistent cache for {data_type} due to validation failure")
                    return self.persistent_cache[data_type]['data']
                
                # If no fallback available, return the invalid data anyway
                logger.warning(f"No fallback available for {data_type}, returning invalid data")
            
            # Track successful fetch
            self._track_success(data_type)
            
            # Update cache
            self.cache[data_type] = {
                'data': data,
                'timestamp': now,
                'needs_refresh': False
            }
            
            # Update persistent cache for important data types
            if data_type in ['trading_performance', 'system_health']:
                self.persistent_cache[data_type] = {
                    'data': data,
                    'timestamp': now
                }
                self._save_persistent_cache()
            
            return data
            
        except Exception as e:
            # Track error
            self._track_error(data_type, e)
            
            # Try to use cached data if available
            if data_type in self.cache:
                logger.warning(f"Error fetching {data_type}, using cached data: {e}")
                return self.cache[data_type]['data']
            
            # Try to use persistent cache if available
            if data_type in self.persistent_cache:
                logger.warning(f"Error fetching {data_type}, using persistent cache: {e}")
                return self.persistent_cache[data_type]['data']
            
            # Fall back to mock data as last resort
            logger.error(f"Error fetching {data_type}, falling back to mock data: {e}")
            return self._get_mock_data(data_type)
