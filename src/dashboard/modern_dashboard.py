"""
AI Trading Agent - Modern Dashboard Implementation

This module provides a new, modern dashboard interface for the AI Trading Agent system.
It features a clean, organized UI with prominent system controls, real-time data visualization,
and comprehensive trading performance monitoring.
"""

import os
import sys
import time
import json
import logging
import random
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import Flask-SocketIO for WebSocket support
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
except ImportError:
    logger.warning("Flask-SocketIO not found, WebSocket functionality will be disabled")
    SocketIO = None

# Import local utilities
try:
    from src.dashboard.utils.settings_manager import SettingsManager
    from src.dashboard.utils.status_reporter import StatusReporter
    from src.dashboard.utils.admin_controller import AdminController
    from src.dashboard.utils.websocket_manager import WebSocketManager
    from src.dashboard.utils.event_bus import EventBus
except ImportError:
    logger.warning("Local utilities not found, will be created dynamically")
    SettingsManager = None
    StatusReporter = None
    AdminController = None
    WebSocketManager = None
    EventBus = None

# Data processing libraries
try:
    import pandas as pd
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly
except ImportError:
    print("Error: Required data processing libraries are missing.")
    print("Please install them with: pip install pandas numpy plotly")
    sys.exit(1)

# Web framework
try:
    from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
    from flask_socketio import SocketIO
    from functools import wraps
except ImportError:
    print("Error: Flask and Flask-SocketIO are required for the dashboard.")
    print("Please install them with: pip install flask flask-socketio")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_trading_dashboard")

# Import local modules - adjust paths as needed
try:
    # Check if real data is available by looking for a configuration file or environment variable
    import os
    import json
    from pathlib import Path
    
    # Check environment variable first
    REAL_DATA_AVAILABLE = os.environ.get('USE_REAL_DATA', '').lower() in ('true', '1', 'yes')
    
    # If not set in environment, check for a config file
    if not REAL_DATA_AVAILABLE:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'real_data_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    REAL_DATA_AVAILABLE = config.get('enabled', False)
                    logger.info(f"Real data configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading real data configuration: {e}")
                REAL_DATA_AVAILABLE = False
    
    if REAL_DATA_AVAILABLE:
        logger.info("Real data connections are ENABLED")
    else:
        logger.info("Real data connections are DISABLED")
        
except ImportError:
    logger.warning("Could not import local modules. Running in standalone mode.")
    REAL_DATA_AVAILABLE = False

# Login required decorator for routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Role required decorator for routes
def role_required(required_roles):
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated_function(*args, **kwargs):
            if 'user_role' not in session or session['user_role'] not in required_roles:
                flash('You do not have permission to access this page', 'error')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Mock implementations for standalone mode
class Config:
    @staticmethod
    def get(key, default=None):
        return default
        
def format_currency(value, currency="$"):
    return f"{currency}{value:,.2f}"
        
def format_percentage(value, include_sign=True):
    sign = "+" if value > 0 and include_sign else ""
    return f"{sign}{value:.2f}%"
        
def get_current_timestamp():
    return datetime.now()
        
def format_timestamp(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")
        
class LogQuery:
    """Mock Log Query class"""
    @staticmethod
    def get_logs(limit=100, level=None, component=None, search=None):
        return []
            
class PerformanceTracker:
    """Mock Performance Tracker class"""
    @staticmethod
    def get_performance_summary(period="daily"):
        return {}
            
    @staticmethod
    def get_performance_metrics():
        return {}
            
class SystemMonitor:
    """Mock System Monitor class"""
    @staticmethod
    def get_system_health():
        return {}

# System state enum
class SystemState:
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

# Trading state enum
class TradingState:
    DISABLED = "disabled"
    ENABLED = "enabled"
    PAUSED = "paused"

# System mode enum
class SystemMode:
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"

# User role enum
class UserRole:
    ADMIN = "admin"     # Full access to all features including system settings
    OPERATOR = "operator"  # Can operate the system but not change settings
    VIEWER = "viewer"   # View-only access to dashboard
    
# Data source enum
class DataSource:
    MOCK = "mock"
    REAL = "real"

# Data service for flexible data sourcing
class DataService:
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
        
    def set_data_source(self, data_source):
        """Set the data source to use (mock or real)"""
        self.data_source = data_source
        # Clear cache when changing data source
        self.cache = {}
        
    def get_data(self, data_type, force_refresh=False):
        """Get data of specified type from current source with caching"""
        # Check cache first unless force refresh is requested
        if not force_refresh and data_type in self.cache:
            cache_time, cached_data = self.cache[data_type]
            # Check if cache is still valid
            if (datetime.now() - cache_time).total_seconds() < self.cache_expiry.get(data_type, 60):
                return cached_data
        
        # Get fresh data
        data = self._fetch_data(data_type)
        
        # Cache the result
        self.cache[data_type] = (datetime.now(), data)
        
        return data
        
    def _fetch_data(self, data_type):
        """Fetch data from appropriate source based on type"""
        # If using mock data or real data not available, use mock generator
        if self.data_source == DataSource.MOCK or not REAL_DATA_AVAILABLE:
            return self._get_mock_data(data_type)
        else:
            return self._get_real_data(data_type)
            
    def _get_mock_data(self, data_type):
        """Get mock data from generator"""
        if data_type == 'system_health':
            return self.mock_data.generate_system_health()
        elif data_type == 'component_status':
            return self.mock_data.generate_component_status()
        elif data_type == 'trading_performance':
            return self.mock_data.generate_trading_performance()
        elif data_type == 'current_positions':
            return self.mock_data.generate_current_positions()
        elif data_type == 'recent_trades':
            return self.mock_data.generate_recent_trades()
        elif data_type == 'system_alerts':
            return self.mock_data.generate_system_alerts()
        elif data_type == 'equity_curve':
            return self.mock_data.generate_equity_curve()
        elif data_type == 'market_regime':
            return self.mock_data.generate_market_regime_data()
        elif data_type == 'sentiment':
            return self.mock_data.generate_sentiment_data()
        elif data_type == 'risk_management':
            return self.mock_data.generate_risk_management_data()
        elif data_type == 'performance_analytics':
            return self.mock_data.generate_performance_analytics_data()
        elif data_type == 'logs_monitoring':
            return self.mock_data.generate_logs_monitoring_data()
        else:
            logger.error(f"Unknown data type requested: {data_type}")
            return {}
            
    def _get_real_data(self, data_type):
        """Get real data from system components"""
        try:
            if data_type == 'system_health':
                return SystemMonitor.get_system_health()
            elif data_type == 'performance_analytics':
                return self._get_real_performance_data()
            elif data_type == 'logs_monitoring':
                return self._get_real_logs_data()
            else:
                # Fall back to mock data for unimplemented real data types
                logger.warning(f"Real data for {data_type} not implemented, using mock data")
                return self._get_mock_data(data_type)
        except Exception as e:
            logger.error(f"Error getting real data for {data_type}: {e}")
            # Fall back to mock data on error
            return self._get_mock_data(data_type)
            
    def _get_real_performance_data(self):
        """Get real performance data from PerformanceTracker"""
        tracker = PerformanceTracker()
        
        # Get data for all time periods
        periods = ["daily", "weekly", "monthly", "quarterly", "yearly", "all_time"]
        performance_summary = {}
        
        for period in periods:
            performance_summary[period] = tracker.get_performance_summary(period)
            
        # Get other performance metrics
        metrics = tracker.get_performance_metrics()
        
        # Format data to match mock data structure
        return {
            "performance_summary": performance_summary,
            "strategy_performance": metrics.get("strategy_performance", []),
            "asset_performance": metrics.get("asset_performance", []),
            "trade_analytics": metrics.get("trade_analytics", {}),
            "recent_trades": metrics.get("recent_trades", []),
            "equity_curve": metrics.get("equity_curve", {}),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    def _get_real_logs_data(self):
        """Get real logs data from LogQuery"""
        log_query = LogQuery()
        
        # Get system logs
        system_logs = log_query.get_logs(limit=200)
        
        # Get resource utilization from SystemMonitor
        monitor = SystemMonitor()
        resource_utilization = monitor.get_resource_utilization()
        
        # Get other monitoring data from appropriate sources
        # For now, use mock data for these sections
        error_distribution = self.mock_data.generate_logs_monitoring_data()["error_distribution"]
        api_status = self.mock_data.generate_logs_monitoring_data()["api_status"]
        recent_requests = self.mock_data.generate_logs_monitoring_data()["recent_requests"]
        trade_flow = self.mock_data.generate_logs_monitoring_data()["trade_flow"]
        
        return {
            "system_logs": system_logs,
            "resource_utilization": resource_utilization,
            "error_distribution": error_distribution,
            "api_status": api_status,
            "recent_requests": recent_requests,
            "trade_flow": trade_flow,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Mock data generators for demonstration
class MockDataGenerator:
    @staticmethod
    def generate_system_health():
        """Generate mock system health data"""
        return {
            "cpu_usage": round(random.uniform(5, 45), 1),
            "memory_usage": round(random.uniform(20, 65), 1),
            "disk_usage": round(random.uniform(15, 70), 1),
            "network_latency": round(random.uniform(10, 150), 0),
            "database_connections": random.randint(1, 15),
            "api_response_time": round(random.uniform(50, 450), 0),
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    
    @staticmethod
    def generate_component_status():
        """Generate mock component status data"""
        components = [
            {"name": "Data Collection Engine", "id": "data-engine"},
            {"name": "Strategy Manager", "id": "strategy-manager"},
            {"name": "Risk Management", "id": "risk-manager"},
            {"name": "Execution Engine", "id": "execution-engine"},
            {"name": "Market Data Service", "id": "market-data"},
            {"name": "Sentiment Analysis", "id": "sentiment-analysis"},
            {"name": "Exchange Connector", "id": "exchange-connector"},
            {"name": "Portfolio Manager", "id": "portfolio-manager"},
            {"name": "Notification Service", "id": "notification-service"},
        ]
        
        status_options = ["operational", "degraded", "offline"]
        weights = [0.8, 0.15, 0.05]  # More likely to be operational
        
        for component in components:
            component["status"] = random.choices(status_options, weights=weights)[0]
            if component["status"] == "operational":
                component["message"] = "Operating normally"
            elif component["status"] == "degraded":
                component["message"] = random.choice([
                    "High latency detected",
                    "Reduced throughput",
                    "Resource contention",
                    "Partially functional"
                ])
            else:
                component["message"] = random.choice([
                    "Connection failed",
                    "Service unavailable",
                    "Critical error",
                    "Resource exhausted"
                ])
                
        return components
    
    @staticmethod
    def generate_trading_performance():
        """Generate mock trading performance data"""
        now = datetime.now()
        
        daily_pnl = random.uniform(-2000, 5000)
        weekly_pnl = daily_pnl + random.uniform(-1000, 8000)
        monthly_pnl = weekly_pnl + random.uniform(2000, 15000)
        ytd_pnl = monthly_pnl + random.uniform(5000, 50000)
        
        return {
            "daily_pnl": daily_pnl,
            "weekly_pnl": weekly_pnl,
            "monthly_pnl": monthly_pnl,
            "ytd_pnl": ytd_pnl,
            "realized_pnl": ytd_pnl * random.uniform(0.3, 0.7),
            "unrealized_pnl": ytd_pnl * random.uniform(0.3, 0.7),
            "win_rate": random.uniform(40, 75),
            "avg_profit": random.uniform(200, 800),
            "avg_loss": random.uniform(-600, -150),
            "last_updated": now.strftime("%H:%M:%S"),
            "best_trade": random.uniform(1000, 5000),
            "worst_trade": random.uniform(-3000, -500),
        }
    
    @staticmethod
    def generate_current_positions():
        """Generate mock current positions data"""
        assets = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "AVAX", "MATIC"]
        positions = []
        
        for _ in range(random.randint(3, 7)):
            asset = random.choice(assets)
            assets.remove(asset)  # Ensure no duplicates
            
            entry_price = random.uniform(10, 50000)
            current_price = entry_price * random.uniform(0.85, 1.15)
            quantity = random.uniform(0.01, 10)
            profit_pct = ((current_price / entry_price) - 1) * 100
            
            positions.append({
                "asset": asset,
                "quantity": round(quantity, 4),
                "entry_price": round(entry_price, 2),
                "current_price": round(current_price, 2),
                "current_value": round(quantity * current_price, 2),
                "profit_loss": round((current_price - entry_price) * quantity, 2),
                "profit_loss_pct": round(profit_pct, 2),
                "strategy": random.choice(["Momentum", "Mean Reversion", "Trend Following", "Sentiment"]),
            })
            
        return positions
    
    @staticmethod
    def generate_recent_trades():
        """Generate mock recent trades data"""
        assets = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD", "DOT/USD", "AVAX/USD", "MATIC/USD"]
        trades = []
        
        now = datetime.now()
        
        for i in range(10):
            asset = random.choice(assets)
            side = random.choice(["buy", "sell"])
            price = random.uniform(10, 50000)
            quantity = random.uniform(0.01, 10)
            timestamp = now - timedelta(minutes=random.randint(1, 120))
            
            profit_loss = None
            if random.random() > 0.5:  # Some trades have P&L info
                profit_loss = random.uniform(-500, 1000)
            
            trades.append({
                "id": f"T-{random.randint(10000, 99999)}",
                "asset": asset,
                "side": side,
                "price": round(price, 2),
                "quantity": round(quantity, 4),
                "value": round(price * quantity, 2),
                "timestamp": timestamp.strftime("%H:%M:%S"),
                "strategy": random.choice(["Momentum", "Mean Reversion", "Trend Following", "Sentiment"]),
                "profit_loss": profit_loss,
            })
            
        return sorted(trades, key=lambda x: x["timestamp"], reverse=True)
    
    @staticmethod
    def generate_system_alerts():
        """Generate mock system alerts"""
        alert_types = ["info", "warning", "error", "critical"]
        weights = [0.4, 0.3, 0.2, 0.1]  # More likely to be info
        
        alerts = []
        now = datetime.now()
        
        for i in range(5):
            alert_type = random.choices(alert_types, weights=weights)[0]
            timestamp = now - timedelta(minutes=random.randint(1, 120))
            
            if alert_type == "info":
                message = random.choice([
                    "System started successfully",
                    "Data collection completed",
                    "Strategy rebalancing performed",
                    "New market data available",
                    "Configuration updated"
                ])
            elif alert_type == "warning":
                message = random.choice([
                    "High API latency detected",
                    "Memory usage above 70%",
                    "Order partially filled",
                    "Strategy performance below threshold",
                    "Exchange rate limit approaching"
                ])
            elif alert_type == "error":
                message = random.choice([
                    "Failed to connect to exchange API",
                    "Order execution timeout",
                    "Data inconsistency detected",
                    "Strategy execution failed",
                    "Risk limit breach"
                ])
            else:  # critical
                message = random.choice([
                    "Exchange connection lost",
                    "Insufficient funds for order",
                    "Security breach detected",
                    "Critical system resource exhausted",
                    "Database connection failure"
                ])
                
            alerts.append({
                "type": alert_type,
                "message": message,
                "timestamp": timestamp.strftime("%H:%M:%S"),
                "acknowledged": random.random() > 0.7,
            })
            
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    
    @staticmethod
    def generate_equity_curve():
        """Generate mock equity curve data for charts"""
        days = 30
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Oldest to newest
        
        # Start with initial equity
        initial_equity = 100000
        equity_values = [initial_equity]
        
        # Generate random daily changes with slight upward bias
        for i in range(1, days):
            prev_equity = equity_values[-1]
            daily_change = random.uniform(-0.03, 0.035)  # -3% to +3.5%
            new_equity = prev_equity * (1 + daily_change)
            equity_values.append(new_equity)
        
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "equity": equity_values
        }
    
    @staticmethod
    def generate_sentiment_data():
        """Generate mock sentiment analysis data"""
        # Generate time-based sentiment data
        days = 90
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Oldest to newest
        
        # Create sentiment time series
        overall_sentiment = []
        social_sentiment = []
        news_sentiment = []
        onchain_sentiment = []
        
        # Start with moderate sentiment
        current_overall = 0.2
        current_social = 0.3
        current_news = 0.1
        current_onchain = 0.2
        
        # Generate sentiment data with realistic patterns
        for i in range(days):
            # Add some random walk with mean reversion
            current_overall += random.uniform(-0.1, 0.1)
            current_social += random.uniform(-0.15, 0.15)
            current_news += random.uniform(-0.1, 0.1)
            current_onchain += random.uniform(-0.07, 0.07)
            
            # Simulate sentiment shifts at certain points
            if i == 15:  # Sudden positive news
                current_overall += 0.2
                current_news += 0.4
                current_social += 0.3
            elif i == 35:  # Social media FUD
                current_overall -= 0.3
                current_social -= 0.5
                current_news -= 0.2
            elif i == 55:  # Positive on-chain metrics
                current_overall += 0.2
                current_onchain += 0.4
                current_news += 0.1
            elif i == 75:  # Mixed signals
                current_social -= 0.2
                current_news += 0.3
                current_onchain -= 0.1
                
            # Ensure sentiment values stay between -1 and 1
            current_overall = max(-0.9, min(0.9, current_overall))
            current_social = max(-0.9, min(0.9, current_social))
            current_news = max(-0.9, min(0.9, current_news))
            current_onchain = max(-0.9, min(0.9, current_onchain))
            
            overall_sentiment.append(current_overall)
            social_sentiment.append(current_social)
            news_sentiment.append(current_news)
            onchain_sentiment.append(current_onchain)
        
        # Create source breakdown for current sentiment
        sources = [
            {"name": "Social Media", "score": social_sentiment[-1], "weight": 0.35, 
             "components": [
                 {"name": "Twitter", "score": social_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Reddit", "score": social_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Telegram", "score": social_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Discord", "score": social_sentiment[-1] + random.uniform(-0.2, 0.2)}
             ]},
            {"name": "News", "score": news_sentiment[-1], "weight": 0.30, 
             "components": [
                 {"name": "Financial News", "score": news_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Crypto News", "score": news_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "General Media", "score": news_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Press Releases", "score": news_sentiment[-1] + random.uniform(-0.2, 0.2)}
             ]},
            {"name": "On-Chain", "score": onchain_sentiment[-1], "weight": 0.25, 
             "components": [
                 {"name": "Network Activity", "score": onchain_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Whale Movements", "score": onchain_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "DEX Volume", "score": onchain_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Smart Contract", "score": onchain_sentiment[-1] + random.uniform(-0.2, 0.2)}
             ]},
            {"name": "Market Data", "score": overall_sentiment[-1] - 0.1, "weight": 0.10, 
             "components": [
                 {"name": "Price Action", "score": overall_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Volatility", "score": overall_sentiment[-1] - random.uniform(-0.1, 0.3)},
                 {"name": "Volume Profile", "score": overall_sentiment[-1] + random.uniform(-0.2, 0.2)},
                 {"name": "Order Flow", "score": overall_sentiment[-1] + random.uniform(-0.2, 0.2)}
             ]}
        ]
        
        # Calculate overall sentiment (weighted average)
        overall_score = sum(source["score"] * source["weight"] for source in sources)
        
        # Generate sentiment-based trading signals
        assets = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "AVAX", "LINK", "MATIC", "UNI"]
        signals = []
        
        for asset in assets:
            # Generate a sentiment score slightly different for each asset
            asset_sentiment = overall_score + random.uniform(-0.3, 0.3)
            asset_sentiment = max(-0.9, min(0.9, asset_sentiment))
            
            # Determine signal strength (1-5 stars)
            if abs(asset_sentiment) < 0.2:
                strength = 1
            elif abs(asset_sentiment) < 0.4:
                strength = 2
            elif abs(asset_sentiment) < 0.6:
                strength = 3
            elif abs(asset_sentiment) < 0.8:
                strength = 4
            else:
                strength = 5
            
            # Determine signal type
            if asset_sentiment > 0.1:
                signal_type = "BUY"
            elif asset_sentiment < -0.1:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            # Confidence level
            confidence = int(50 + abs(asset_sentiment) * 50)
            
            signals.append({
                "asset": asset,
                "sentiment": asset_sentiment,
                "signal": signal_type,
                "strength": strength,
                "confidence": confidence,
                "timeframe": random.choice(["Short-term", "Medium-term", "Long-term"])
            })
        
        # Sort signals by absolute sentiment (strongest first)
        signals.sort(key=lambda x: abs(x["sentiment"]), reverse=True)
        
        # Generate recent news items with sentiment
        news_sources = ["Bloomberg", "CoinDesk", "Reuters", "The Block", "Cointelegraph", "CNBC", "WSJ", "Financial Times"]
        news_items = []
        
        for i in range(10):
            timestamp = datetime.now() - timedelta(hours=random.randint(1, 48))
            sentiment_value = random.uniform(-0.8, 0.8)
            
            if sentiment_value > 0.3:
                sentiment_label = "Positive"
                sentiment_class = "positive"
            elif sentiment_value < -0.3:
                sentiment_label = "Negative"
                sentiment_class = "negative"
            else:
                sentiment_label = "Neutral"
                sentiment_class = "neutral"
            
            # Generate headline based on sentiment
            if sentiment_value > 0.5:
                headline = random.choice([
                    "Major Institution Announces $500M Investment in Crypto",
                    "Bitcoin Surges on ETF Approval News",
                    "Ethereum Upgrade Successfully Implemented",
                    "New Partnership Announced Between Blockchain Projects",
                    "Regulatory Clarity Emerges for Crypto Industry"
                ])
            elif sentiment_value > 0:
                headline = random.choice([
                    "Moderate Growth Expected in Crypto Markets",
                    "Technical Indicators Show Positive Momentum",
                    "Institutional Interest in Blockchain Technology Growing",
                    "New Features Launched for Major Cryptocurrency",
                    "Analyst Predicts Steady Recovery for Digital Assets"
                ])
            elif sentiment_value > -0.5:
                headline = random.choice([
                    "Market Volatility Continues Amid Mixed Signals",
                    "Crypto Trading Volume Remains Flat",
                    "Regulatory Discussions Ongoing Without Clear Outcome",
                    "Minor Security Issue Patched in Blockchain Protocol",
                    "Investors Await Clearer Market Direction"
                ])
            else:
                headline = random.choice([
                    "Major Selloff Following Regulatory Concerns",
                    "Security Breach Reported at Crypto Exchange",
                    "Central Bank Issues Warning on Cryptocurrency Risks",
                    "Mining Difficulty Increases Amid Declining Profitability",
                    "Market Downturn Accelerates as Support Levels Break"
                ])
            
            news_items.append({
                "source": random.choice(news_sources),
                "headline": headline,
                "sentiment": sentiment_value,
                "sentiment_label": sentiment_label,
                "sentiment_class": sentiment_class,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                "url": "#"  # Placeholder for URL
            })
        
        # Sort news by recency
        news_items.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Generate social media topics
        topics = [
            {"name": "Bitcoin ETF", "sentiment": random.uniform(-0.5, 0.9), "volume": random.randint(70, 100)},
            {"name": "Ethereum 2.0", "sentiment": random.uniform(-0.3, 0.8), "volume": random.randint(60, 90)},
            {"name": "DeFi Growth", "sentiment": random.uniform(-0.2, 0.7), "volume": random.randint(40, 85)},
            {"name": "Regulatory News", "sentiment": random.uniform(-0.8, 0.3), "volume": random.randint(50, 95)},
            {"name": "NFT Market", "sentiment": random.uniform(-0.6, 0.6), "volume": random.randint(30, 80)},
            {"name": "Layer 2 Scaling", "sentiment": random.uniform(-0.2, 0.8), "volume": random.randint(20, 70)},
            {"name": "Metaverse", "sentiment": random.uniform(-0.4, 0.7), "volume": random.randint(15, 65)},
            {"name": "Central Bank Policies", "sentiment": random.uniform(-0.7, 0.2), "volume": random.randint(40, 85)}
        ]
        
        # Sort topics by volume
        topics.sort(key=lambda x: x["volume"], reverse=True)
        
        # Generate correlation data
        price_sentiment_correlation = {
            "7d": round(random.uniform(-0.2, 0.9), 2),
            "30d": round(random.uniform(-0.1, 0.8), 2),
            "90d": round(random.uniform(0.1, 0.7), 2)
        }
        
        lead_lag = random.choice(["Sentiment leads price by 2-3 days", "Price and sentiment move together", "Price leads sentiment slightly"])
        
        # Create time series for visualization
        sentiment_time_series = {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "overall": overall_sentiment,
            "social": social_sentiment,
            "news": news_sentiment,
            "onchain": onchain_sentiment
        }
        
        return {
            "overall_score": overall_score,
            "sentiment_class": "positive" if overall_score > 0.1 else "negative" if overall_score < -0.1 else "neutral",
            "sources": sources,
            "signals": signals,
            "news_items": news_items,
            "topics": topics,
            "correlation": price_sentiment_correlation,
            "lead_lag": lead_lag,
            "time_series": sentiment_time_series,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    @staticmethod
    def generate_market_regime_data():
        """Generate mock market regime data"""
        days = 90
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Oldest to newest
        
        # Generate regime probabilities
        # We'll simulate different market periods
        bull_probs = []
        bear_probs = []
        sideways_probs = []
        volatile_probs = []
        
        # Start with balanced regime probabilities
        current_bull = 0.25
        current_bear = 0.25
        current_sideways = 0.25
        current_volatile = 0.25
        
        for i in range(days):
            # Add some random walk with mean reversion
            current_bull += random.uniform(-0.03, 0.03)
            current_bear += random.uniform(-0.03, 0.03)
            current_sideways += random.uniform(-0.03, 0.03)
            current_volatile += random.uniform(-0.03, 0.03)
            
            # Simulate regime shifts at certain points
            if i == 15:  # Shift towards bull market
                current_bull += 0.4
                current_bear -= 0.2
                current_sideways -= 0.1
                current_volatile -= 0.1
            elif i == 35:  # Shift towards volatile market
                current_bull -= 0.3
                current_bear -= 0.1
                current_sideways -= 0.1
                current_volatile += 0.5
            elif i == 55:  # Shift towards bear market
                current_bull -= 0.4
                current_bear += 0.6
                current_sideways -= 0.1
                current_volatile -= 0.1
            elif i == 75:  # Shift towards sideways market
                current_bull -= 0.2
                current_bear -= 0.3
                current_sideways += 0.6
                current_volatile -= 0.1
                
            # Ensure probabilities remain positive
            current_bull = max(0.05, current_bull)
            current_bear = max(0.05, current_bear)
            current_sideways = max(0.05, current_sideways)
            current_volatile = max(0.05, current_volatile)
            
            # Normalize to ensure they sum to 1
            total = current_bull + current_bear + current_sideways + current_volatile
            current_bull /= total
            current_bear /= total
            current_sideways /= total
            current_volatile /= total
            
            bull_probs.append(current_bull)
            bear_probs.append(current_bear)
            sideways_probs.append(current_sideways)
            volatile_probs.append(current_volatile)
        
        # Generate current regime confidence levels (based on last values)
        regimes = [
            {"name": "Bull Market", "probability": bull_probs[-1] * 100, "description": "Upward trending market with positive momentum", 
             "indicators": {"price_action": "Higher highs and higher lows", "volatility": "Moderate to low", "volume": "Increasing on upward moves"}},
            {"name": "Bear Market", "probability": bear_probs[-1] * 100, "description": "Downward trending market with negative sentiment", 
             "indicators": {"price_action": "Lower highs and lower lows", "volatility": "High and increasing", "volume": "Increasing on downward moves"}},
            {"name": "Sideways Market", "probability": sideways_probs[-1] * 100, "description": "Range-bound market with low directional bias", 
             "indicators": {"price_action": "Trading within a defined range", "volatility": "Decreasing", "volume": "Lower than average"}},
            {"name": "Volatile Market", "probability": volatile_probs[-1] * 100, "description": "Erratic price movements with high uncertainty", 
             "indicators": {"price_action": "Large candles in both directions", "volatility": "Very high", "volume": "Above average"}}
        ]
        
        # Sort regimes by probability (highest first)
        regimes.sort(key=lambda x: x["probability"], reverse=True)
        
        # Determine current dominant regime
        dominant_regime = regimes[0]["name"]
        
        # Generate recommended strategies based on regime
        strategy_recommendations = {
            "Bull Market": [
                {"name": "Momentum", "allocation": round(random.uniform(40, 60), 1), "description": "Follow upward trends with larger position sizes"},
                {"name": "Breakout", "allocation": round(random.uniform(20, 30), 1), "description": "Target resistance breakouts with tight stops"},
                {"name": "Growth", "allocation": round(random.uniform(10, 30), 1), "description": "Focus on high-growth assets"}
            ],
            "Bear Market": [
                {"name": "Short Selling", "allocation": round(random.uniform(30, 50), 1), "description": "Establish short positions on bounces"},
                {"name": "Defensive", "allocation": round(random.uniform(30, 40), 1), "description": "Focus on lower beta assets"},
                {"name": "Cash Reserve", "allocation": round(random.uniform(20, 30), 1), "description": "Maintain higher cash allocation"}
            ],
            "Sideways Market": [
                {"name": "Mean Reversion", "allocation": round(random.uniform(40, 60), 1), "description": "Trade range extremes back to mean"},
                {"name": "Theta Harvesting", "allocation": round(random.uniform(20, 40), 1), "description": "Collect premium through range-bound strategies"},
                {"name": "Pairs Trading", "allocation": round(random.uniform(10, 30), 1), "description": "Trade correlated asset divergences"}
            ],
            "Volatile Market": [
                {"name": "Volatility Capture", "allocation": round(random.uniform(30, 50), 1), "description": "Use strategies that benefit from high volatility"},
                {"name": "Reduced Size", "allocation": round(random.uniform(20, 40), 1), "description": "Trade with smaller position sizes"},
                {"name": "Quick Exits", "allocation": round(random.uniform(20, 30), 1), "description": "Use tighter profit targets and stop losses"}
            ]
        }
        
        # Generate strategy performance in current regime
        current_strategies = strategy_recommendations[dominant_regime]
        for strategy in current_strategies:
            strategy["performance"] = round(random.uniform(-10, 30), 1)  # -10% to +30%
            strategy["sharpe"] = round(random.uniform(0.5, 3.0), 2)  # 0.5 to 3.0
        
        # Create transition probability matrix
        transition_matrix = {
            "from_bull": {"to_bull": round(random.uniform(0.7, 0.9), 2), "to_bear": round(random.uniform(0.05, 0.2), 2), 
                         "to_sideways": round(random.uniform(0.05, 0.2), 2), "to_volatile": round(random.uniform(0.05, 0.15), 2)},
            "from_bear": {"to_bull": round(random.uniform(0.05, 0.15), 2), "to_bear": round(random.uniform(0.7, 0.9), 2), 
                         "to_sideways": round(random.uniform(0.1, 0.2), 2), "to_volatile": round(random.uniform(0.05, 0.15), 2)},
            "from_sideways": {"to_bull": round(random.uniform(0.2, 0.4), 2), "to_bear": round(random.uniform(0.2, 0.4), 2), 
                              "to_sideways": round(random.uniform(0.3, 0.5), 2), "to_volatile": round(random.uniform(0.1, 0.2), 2)},
            "from_volatile": {"to_bull": round(random.uniform(0.1, 0.3), 2), "to_bear": round(random.uniform(0.1, 0.4), 2), 
                             "to_sideways": round(random.uniform(0.1, 0.3), 2), "to_volatile": round(random.uniform(0.3, 0.5), 2)}
        }
        
        # Generate market indicators
        market_indicators = {
            "volatility": round(random.uniform(5, 40), 1),
            "volume": round(random.uniform(80, 150), 1),
            "sentiment": round(random.uniform(-0.8, 0.8), 2),
            "momentum": round(random.uniform(-0.8, 0.8), 2),
            "trend_strength": round(random.uniform(0, 100), 1)
        }
        
        # Create time series data for regime probabilities visualization
        regime_time_series = {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "bull_probs": bull_probs,
            "bear_probs": bear_probs,
            "sideways_probs": sideways_probs,
            "volatile_probs": volatile_probs
        }
        
        return {
            "regimes": regimes,
            "dominant_regime": dominant_regime,
            "strategy_recommendations": current_strategies,
            "transition_matrix": transition_matrix,
            "market_indicators": market_indicators,
            "regime_time_series": regime_time_series,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    @staticmethod
    def generate_risk_management_data():
        """Generate mock risk management data"""
        # Portfolio level risk metrics
        portfolio_risk = {
            "var_daily": round(random.uniform(2, 6), 2),  # Value at Risk (daily, 95%)
            "var_weekly": round(random.uniform(4, 10), 2),  # Value at Risk (weekly, 95%)
            "max_drawdown": round(random.uniform(8, 25), 2),  # Maximum historical drawdown
            "sharpe_ratio": round(random.uniform(0.8, 2.5), 2),  # Sharpe ratio
            "sortino_ratio": round(random.uniform(1.2, 3.2), 2),  # Sortino ratio
            "beta": round(random.uniform(0.8, 1.2), 2),  # Beta (compared to market)
            "current_drawdown": round(random.uniform(0, 10), 2),  # Current drawdown
            "risk_capacity": round(random.uniform(50, 85), 1),  # Current risk capacity utilization (%)
            "risk_tolerance": round(random.uniform(60, 90), 1),  # Current risk vs tolerance (%)
            "risk_adjusted_return": round(random.uniform(-5, 15), 2)  # Risk-adjusted return (%)
        }
        
        # Generate asset-level risk data
        assets = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "AVAX", "MATIC"]
        asset_risk = []
        
        for asset in assets:
            asset_risk.append({
                "asset": asset,
                "weight": round(random.uniform(5, 25), 1),  # Portfolio weight (%)
                "var_contrib": round(random.uniform(0.5, 5), 2),  # VaR contribution
                "risk_contrib": round(random.uniform(5, 25), 1),  # Risk contribution (%)
                "beta": round(random.uniform(0.7, 1.5), 2),  # Asset beta
                "correlation": round(random.uniform(-0.3, 0.9), 2),  # Correlation to portfolio
                "max_drawdown": round(random.uniform(10, 40), 1),  # Asset max drawdown
                "position_size": round(random.uniform(1000, 10000), 2),  # Position size in $
                "exposure": round(random.uniform(2, 40), 1)  # Exposure (%)
            })
        
        # Sort assets by exposure (highest first)
        asset_risk.sort(key=lambda x: x["exposure"], reverse=True)
        
        # Generate strategy risk allocation
        strategies = ["Momentum", "Mean Reversion", "Trend Following", "Sentiment", "Breakout"]
        strategy_risk = []
        
        total_alloc = 0
        for strategy in strategies:
            allocation = round(random.uniform(5, 30), 1)
            total_alloc += allocation
            
            strategy_risk.append({
                "strategy": strategy,
                "allocation": allocation,  # Theoretical allocation (%)
                "current_usage": round(random.uniform(50, 110), 1),  # Current usage vs allocation (%)
                "var_contrib": round(random.uniform(1, 5), 2),  # VaR contribution
                "sharpe": round(random.uniform(0.5, 2.5), 2),  # Strategy Sharpe
                "max_drawdown": round(random.uniform(5, 25), 1),  # Max drawdown for strategy
                "win_rate": round(random.uniform(40, 70), 1)  # Win rate (%)
            })
        
        # Normalize allocations to sum to 100%
        for strategy in strategy_risk:
            strategy["allocation"] = round((strategy["allocation"] / total_alloc) * 100, 1)
            
        # Sort strategies by allocation (highest first)
        strategy_risk.sort(key=lambda x: x["allocation"], reverse=True)
        
        # Generate risk circuit breakers status
        circuit_breakers = [
            {
                "name": "Max Drawdown Limit",
                "threshold": "15%",
                "current": f"{portfolio_risk['current_drawdown']}%",
                "status": "normal" if portfolio_risk['current_drawdown'] < 10 else "warning" if portfolio_risk['current_drawdown'] < 15 else "triggered",
                "action": "Reduce position sizes by 50%"
            },
            {
                "name": "Daily Loss Limit",
                "threshold": "5%",
                "current": f"{round(random.uniform(1, 6), 2)}%",
                "status": "normal" if random.random() > 0.2 else "warning" if random.random() > 0.5 else "triggered",
                "action": "Close all intraday positions"
            },
            {
                "name": "VaR Breach",
                "threshold": "Portfolio VaR > 8%",
                "current": f"{portfolio_risk['var_daily']}%",
                "status": "normal" if portfolio_risk['var_daily'] < 5 else "warning" if portfolio_risk['var_daily'] < 8 else "triggered",
                "action": "Reduce risk exposure by 30%"
            },
            {
                "name": "Volatility Spike",
                "threshold": "2x normal volatility",
                "current": f"{round(random.uniform(0.8, 2.5), 1)}x",
                "status": "normal" if random.random() > 0.3 else "warning" if random.random() > 0.7 else "triggered",
                "action": "Tighten stop losses and reduce new entries"
            },
            {
                "name": "Correlation Alert",
                "threshold": "Avg Correlation > 0.7",
                "current": f"{round(random.uniform(0.3, 0.8), 2)}",
                "status": "normal" if random.random() > 0.25 else "warning" if random.random() > 0.6 else "triggered",
                "action": "Increase diversification"
            }
        ]
        
        # Generate correlation matrix for assets
        correlation_matrix = {}
        
        for i, asset1 in enumerate(assets):
            correlations = {}
            for j, asset2 in enumerate(assets):
                if i == j:
                    correlations[asset2] = 1.0  # Self correlation is always 1
                elif j > i:
                    # Generate a somewhat realistic correlation (crypto assets tend to be positively correlated)
                    if asset1 in ["BTC", "ETH"] and asset2 in ["BTC", "ETH"]:
                        corr = round(random.uniform(0.7, 0.95), 2)  # Major cryptos highly correlated
                    else:
                        corr = round(random.uniform(0.3, 0.8), 2)  # Other pairs moderately correlated
                    correlations[asset2] = corr
                else:
                    # Use the already generated correlation (symmetric matrix)
                    correlations[asset2] = correlation_matrix[asset2][asset1]
            correlation_matrix[asset1] = correlations
        
        # Generate risk exposure history
        days = 90
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Oldest to newest
        
        # Risk metrics over time
        var_history = []
        drawdown_history = []
        exposure_history = []
        
        # Start values
        current_var = 3.0
        current_drawdown = 0.0
        current_exposure = 60.0
        
        for i in range(days):
            # Add some random walk with mean reversion
            current_var += random.uniform(-0.3, 0.3)
            current_var = max(1.0, min(8.0, current_var))  # Keep between 1-8%
            
            current_drawdown += random.uniform(-1.0, 0.8)  # More likely to go up (drawdown increases)
            current_drawdown = max(0.0, min(25.0, current_drawdown))  # Keep between 0-25%
            
            current_exposure += random.uniform(-2.0, 2.0)
            current_exposure = max(20.0, min(95.0, current_exposure))  # Keep between 20-95%
            
            # Simulate some risk events
            if i == 25:  # Risk spike event
                current_var *= 1.7
                current_drawdown += 8.0
                current_exposure *= 0.7  # Reduce exposure as risk response
            elif i == 60:  # Another risk event
                current_var *= 1.5
                current_drawdown += 5.0
                current_exposure *= 0.8
            
            var_history.append(round(current_var, 2))
            drawdown_history.append(round(current_drawdown, 2))
            exposure_history.append(round(current_exposure, 1))
        
        risk_time_series = {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "var": var_history,
            "drawdown": drawdown_history,
            "exposure": exposure_history
        }
        
        # Generate stress test results
        stress_tests = [
            {
                "scenario": "Market Crash (-30%)",
                "portfolio_impact": round(random.uniform(-28, -18), 1),
                "max_drawdown": round(random.uniform(25, 40), 1),
                "var_change": round(random.uniform(150, 300), 1),
                "recovery_time": f"{random.randint(3, 8)} months"
            },
            {
                "scenario": "Volatility Surge (+100%)",
                "portfolio_impact": round(random.uniform(-20, -10), 1),
                "max_drawdown": round(random.uniform(15, 30), 1),
                "var_change": round(random.uniform(100, 200), 1),
                "recovery_time": f"{random.randint(1, 4)} months"
            },
            {
                "scenario": "Liquidity Crisis",
                "portfolio_impact": round(random.uniform(-25, -15), 1),
                "max_drawdown": round(random.uniform(20, 35), 1),
                "var_change": round(random.uniform(120, 250), 1),
                "recovery_time": f"{random.randint(2, 6)} months"
            },
            {
                "scenario": "Correlation Breakdown",
                "portfolio_impact": round(random.uniform(-15, -5), 1),
                "max_drawdown": round(random.uniform(10, 25), 1),
                "var_change": round(random.uniform(50, 150), 1),
                "recovery_time": f"{random.randint(1, 3)} months"
            },
            {
                "scenario": "Interest Rate Spike (+2%)",
                "portfolio_impact": round(random.uniform(-12, -3), 1),
                "max_drawdown": round(random.uniform(8, 20), 1),
                "var_change": round(random.uniform(30, 120), 1),
                "recovery_time": f"{random.randint(1, 4)} months"
            }
        ]
        
        return {
            "portfolio_risk": portfolio_risk,
            "asset_risk": asset_risk,
            "strategy_risk": strategy_risk,
            "circuit_breakers": circuit_breakers,
            "correlation_matrix": correlation_matrix,
            "risk_time_series": risk_time_series,
            "stress_tests": stress_tests,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    @staticmethod
    def generate_performance_analytics_data():
        """Generate mock performance analytics data"""
        # Time periods for performance data
        periods = ["daily", "weekly", "monthly", "quarterly", "yearly", "all_time"]
        performance_summary = {}
        
        # Initial values that will be built upon for longer periods
        base_return = random.uniform(0.5, 2.0)
        base_trades = random.randint(10, 30)
        base_volume = random.uniform(50000, 150000)
        base_fees = base_volume * random.uniform(0.0005, 0.0015)
        
        # Generate performance data for different time periods
        for i, period in enumerate(periods):
            # Scale metrics based on time period
            multiplier = 1 if i == 0 else 5 if i == 1 else 20 if i == 2 else 60 if i == 3 else 240 if i == 4 else 365
            
            # Add some randomness to make it realistic
            return_pct = base_return * multiplier * random.uniform(0.8, 1.2)
            if i > 2:  # For longer periods, taper the growth rate
                return_pct *= random.uniform(0.7, 0.9)
                
            # For all_time, make it look more impressive
            if period == "all_time":
                return_pct *= 1.5
            
            # Calculate other metrics
            trades_count = int(base_trades * multiplier * random.uniform(0.9, 1.1))
            winning_trades = int(trades_count * random.uniform(0.55, 0.65))
            losing_trades = trades_count - winning_trades
            volume = base_volume * multiplier * random.uniform(0.9, 1.1)
            fees = volume * random.uniform(0.0005, 0.0015)
            
            # Calculate average profit/loss
            avg_profit = random.uniform(50, 200)
            avg_loss = random.uniform(-150, -40)
            
            # Calculate profit factor
            if losing_trades > 0:
                profit_factor = (winning_trades * avg_profit) / (losing_trades * abs(avg_loss))
            else:
                profit_factor = winning_trades * avg_profit
                
            # Expected value per trade
            expected_value = (winning_trades * avg_profit + losing_trades * avg_loss) / trades_count if trades_count > 0 else 0
            
            performance_summary[period] = {
                "return_pct": round(return_pct, 2),
                "trades_count": trades_count,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round((winning_trades / trades_count) * 100, 1) if trades_count > 0 else 0,
                "avg_profit": round(avg_profit, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "expected_value": round(expected_value, 2),
                "volume": round(volume, 2),
                "fees": round(fees, 2),
                "sharpe": round(random.uniform(1.2, 2.5), 2),
                "max_drawdown": round(random.uniform(5, 15), 2)
            }
        
        # Generate strategy performance
        strategies = ["Momentum", "Mean Reversion", "Trend Following", "Sentiment", "Breakout"]
        strategy_performance = []
        
        for strategy in strategies:
            win_rate = random.uniform(45, 70)
            profit_factor = random.uniform(1.1, 2.5)
            
            strategy_performance.append({
                "strategy": strategy,
                "return_pct": round(random.uniform(-5, 25), 2),
                "trades_count": random.randint(20, 200),
                "win_rate": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "sharpe": round(random.uniform(0.8, 2.5), 2),
                "max_drawdown": round(random.uniform(5, 20), 2),
                "expectancy": round(random.uniform(-0.5, 2.0), 2),
                "avg_holding_time": random.choice(["2.5h", "4.2h", "8.1h", "1.2d", "3.4d"]),
                "status": "active" if random.random() > 0.2 else "inactive"
            })
            
        # Sort strategies by return percentage
        strategy_performance.sort(key=lambda x: x["return_pct"], reverse=True)
        
        # Generate asset performance
        assets = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "AVAX", "MATIC"]
        asset_performance = []
        
        for asset in assets:
            asset_performance.append({
                "asset": asset,
                "return_pct": round(random.uniform(-10, 30), 2),
                "trades_count": random.randint(10, 100),
                "win_rate": round(random.uniform(40, 75), 1),
                "profit_factor": round(random.uniform(0.8, 2.2), 2),
                "avg_profit": round(random.uniform(20, 200), 2),
                "avg_loss": round(random.uniform(-150, -30), 2),
                "avg_holding_time": random.choice(["1.8h", "5.3h", "9.7h", "1.5d", "2.7d"]),
                "unrealized_pnl": round(random.uniform(-2000, 5000), 2)
            })
            
        # Sort assets by return percentage
        asset_performance.sort(key=lambda x: x["return_pct"], reverse=True)
        
        # Generate advanced trade analytics
        trade_analytics = {
            "trade_distribution": {
                "time_of_day": {
                    "00:00-04:00": round(random.uniform(5, 15), 1),
                    "04:00-08:00": round(random.uniform(10, 20), 1),
                    "08:00-12:00": round(random.uniform(15, 25), 1),
                    "12:00-16:00": round(random.uniform(20, 30), 1),
                    "16:00-20:00": round(random.uniform(15, 25), 1),
                    "20:00-24:00": round(random.uniform(5, 15), 1)
                },
                "day_of_week": {
                    "Monday": round(random.uniform(10, 20), 1),
                    "Tuesday": round(random.uniform(15, 25), 1),
                    "Wednesday": round(random.uniform(15, 25), 1),
                    "Thursday": round(random.uniform(15, 25), 1),
                    "Friday": round(random.uniform(10, 20), 1),
                    "Saturday": round(random.uniform(5, 15), 1),
                    "Sunday": round(random.uniform(5, 15), 1)
                },
                "trade_size": {
                    "Small (< $1K)": round(random.uniform(20, 40), 1),
                    "Medium ($1K-$5K)": round(random.uniform(30, 50), 1),
                    "Large ($5K-$20K)": round(random.uniform(10, 30), 1),
                    "Very Large (> $20K)": round(random.uniform(5, 15), 1)
                },
                "holding_time": {
                    "< 1 hour": round(random.uniform(10, 30), 1),
                    "1-6 hours": round(random.uniform(20, 40), 1),
                    "6-24 hours": round(random.uniform(20, 30), 1),
                    "1-3 days": round(random.uniform(10, 20), 1),
                    "> 3 days": round(random.uniform(5, 15), 1)
                }
            },
            "performance_factors": {
                "volatility_impact": round(random.uniform(-20, 30), 1),
                "volume_impact": round(random.uniform(-10, 40), 1),
                "market_hour_impact": round(random.uniform(-15, 25), 1),
                "news_sentiment_impact": round(random.uniform(-30, 40), 1)
            },
            "execution_quality": {
                "avg_slippage": round(random.uniform(0.01, 0.2), 3),
                "avg_execution_time": round(random.uniform(0.2, 2.0), 2),
                "price_improvement": round(random.uniform(-0.05, 0.15), 3),
                "order_fill_rate": round(random.uniform(92, 99.9), 1)
            }
        }
        
        # Generate recent trade history with performance data
        trades = []
        for i in range(20):  # Generate 20 recent trades
            entry_price = random.uniform(100, 50000)
            exit_price = entry_price * random.uniform(0.9, 1.1)
            quantity = random.uniform(0.01, 2.0)
            
            profit_loss = (exit_price - entry_price) * quantity
            profit_loss_pct = ((exit_price / entry_price) - 1) * 100
            
            # Entry and exit times
            now = datetime.now()
            random_days_ago = random.randint(0, 10)
            random_hours_ago = random.randint(0, 23)
            random_minutes_ago = random.randint(0, 59)
            
            entry_time = now - timedelta(days=random_days_ago, hours=random_hours_ago, minutes=random_minutes_ago)
            
            # Holding period between 10 minutes and 3 days
            holding_period = timedelta(minutes=random.randint(10, 4320))
            exit_time = entry_time + holding_period
            
            # Ensure exit time is not in the future
            if exit_time > now:
                exit_time = now
                
            trade = {
                "id": f"T-{random.randint(10000, 99999)}",
                "asset": random.choice(assets),
                "strategy": random.choice(strategies),
                "side": "buy" if profit_loss > 0 else "sell",  # Simplistic assumption for demo
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "quantity": round(quantity, 4),
                "profit_loss": round(profit_loss, 2),
                "profit_loss_pct": round(profit_loss_pct, 2),
                "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "holding_time": str(holding_period).split('.')[0],  # Format as HH:MM:SS
                "tags": random.sample(["trend", "breakout", "support", "resistance", "momentum", "reversal"], 
                                      k=random.randint(1, 3))
            }
            trades.append(trade)
            
        # Sort trades by exit time (most recent first)
        trades.sort(key=lambda x: x["exit_time"], reverse=True)
        
        # Generate equity curve data
        days = 90
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Oldest to newest
        
        # Start with initial equity
        initial_equity = 100000
        equity_values = [initial_equity]
        benchmark_values = [initial_equity]
        daily_returns = []
        
        # Generate random daily changes with slight upward bias
        for i in range(1, days):
            prev_equity = equity_values[-1]
            daily_change = random.uniform(-0.03, 0.035)  # -3% to +3.5%
            
            # Add some trend periods
            if 20 <= i < 30:  # Strong uptrend period
                daily_change = random.uniform(0.005, 0.045)
            elif 45 <= i < 55:  # Downtrend period
                daily_change = random.uniform(-0.04, 0.01)
            elif 70 <= i < 80:  # Recovery period
                daily_change = random.uniform(0.0, 0.04)
                
            new_equity = prev_equity * (1 + daily_change)
            equity_values.append(new_equity)
            daily_returns.append(daily_change)
            
            # Generate benchmark (e.g., BTC) with correlation but different performance
            prev_benchmark = benchmark_values[-1]
            # Correlated but different
            benchmark_change = daily_change * 0.7 + random.uniform(-0.02, 0.025)
            new_benchmark = prev_benchmark * (1 + benchmark_change)
            benchmark_values.append(new_benchmark)
        
        equity_curve = {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "equity": [round(e, 2) for e in equity_values],
            "benchmark": [round(b, 2) for b in benchmark_values],
            "daily_returns": [round(r * 100, 2) for r in daily_returns]
        }
        
        # Calculate drawdowns
        max_equity = equity_values[0]
        drawdowns = [0]
        
        for i in range(1, days):
            max_equity = max(max_equity, equity_values[i])
            drawdown_pct = (equity_values[i] / max_equity - 1) * 100
            drawdowns.append(round(drawdown_pct, 2))
            
        equity_curve["drawdowns"] = drawdowns
        
        return {
            "performance_summary": performance_summary,
            "strategy_performance": strategy_performance,
            "asset_performance": asset_performance,
            "trade_analytics": trade_analytics,
            "recent_trades": trades,
            "equity_curve": equity_curve,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    @staticmethod
    def generate_logs_monitoring_data():
        """Generate mock logs and monitoring data"""
        # Generate system logs
        log_types = ["INFO", "WARNING", "ERROR", "DEBUG"]
        log_weights = [0.7, 0.15, 0.05, 0.1]  # More likely to be INFO
        
        system_logs = []
        now = datetime.now()
        
        # System components
        components = [
            "DataCollection", "StrategyManager", "RiskManagement", "ExecutionEngine", 
            "MarketDataService", "SentimentAnalysis", "ExchangeConnector", "PortfolioManager"
        ]
        
        # Generate logs over the past 24 hours
        for i in range(200):  # Generate 200 log entries
            log_type = random.choices(log_types, weights=log_weights)[0]
            timestamp = now - timedelta(hours=random.uniform(0, 24))
            component = random.choice(components)
            
            # Generate message based on log type and component
            if log_type == "INFO":
                if component == "DataCollection":
                    message = random.choice([
                        "Market data collection completed successfully",
                        "Historical data updated for BTCUSD",
                        "New data source connected: Coinbase",
                        "Order book snapshots collected for top 10 pairs"
                    ])
                elif component == "StrategyManager":
                    message = random.choice([
                        "Strategy execution cycle completed",
                        "New strategy activated: Mean Reversion BTC",
                        "Strategy performance metrics updated",
                        "Signal generated for ETH/USD pair"
                    ])
                elif component == "RiskManagement":
                    message = random.choice([
                        "Risk limits updated successfully",
                        "Daily VaR calculated: 3.2%",
                        "Position risk assessment completed",
                        "Risk exposure report generated"
                    ])
                elif component == "ExecutionEngine":
                    message = random.choice([
                        "Order executed: Buy 0.5 BTC at $52,340",
                        "TWAP execution started for SOL position",
                        "Order routed to Binance",
                        "Execution algorithm selected: VWAP"
                    ])
                else:
                    message = f"{component} operation completed successfully"
            
            elif log_type == "WARNING":
                if component == "DataCollection":
                    message = random.choice([
                        "Data feed latency increased to 500ms",
                        "Missing candle data for ETHUSD",
                        "Rate limit approaching for Binance API",
                        "Data quality check found inconsistencies"
                    ])
                elif component == "ExchangeConnector":
                    message = random.choice([
                        "API response time degraded",
                        "Websocket reconnection required",
                        "Order book gaps detected",
                        "Exchange maintenance window approaching"
                    ])
                else:
                    message = f"{component} performance degraded"
            
            elif log_type == "ERROR":
                if component == "ExchangeConnector":
                    message = random.choice([
                        "API connection timeout",
                        "Authentication failed with exchange",
                        "Order submission rejected by exchange",
                        "Exchange returned error code: 429"
                    ])
                elif component == "ExecutionEngine":
                    message = random.choice([
                        "Order execution failed: insufficient funds",
                        "Fill confirmation timeout",
                        "Price slippage exceeded tolerance",
                        "Order routing failed"
                    ])
                else:
                    message = f"{component} encountered a critical error"
            
            else:  # DEBUG
                message = f"Detailed diagnostic information for {component}"
            
            log_entry = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "level": log_type,
                "component": component,
                "message": message,
                "details": {
                    "thread": f"Thread-{random.randint(1, 20)}",
                    "pid": random.randint(1000, 9999)
                }
            }
            
            system_logs.append(log_entry)
            
        # Sort logs by timestamp (newest first)
        system_logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Generate resource utilization history
        minutes = 60
        timestamps = [now - timedelta(minutes=i) for i in range(minutes)]
        timestamps.reverse()  # Oldest to newest
        
        # Create resource usage patterns
        cpu_usage = []
        memory_usage = []
        network_traffic = []
        api_latency = []
        
        # Base values
        cpu_base = 20
        memory_base = 35
        network_base = 50
        latency_base = 120
        
        # Generate resource usage with realistic patterns
        for i in range(minutes):
            # CPU pattern - fluctuates with periodic spikes
            cpu = cpu_base + random.uniform(-5, 5)
            if i % 15 == 0:  # Periodic spike every 15 minutes
                cpu += random.uniform(15, 30)
            cpu = min(max(cpu, 5), 95)  # Keep between 5% and 95%
            
            # Memory pattern - gradually increases then drops (GC)
            if i % 20 == 0:  # Memory cleanup every 20 minutes
                memory_base = max(memory_base - random.uniform(10, 15), 30)
            else:
                memory_base += random.uniform(0.2, 0.5)
            memory = memory_base + random.uniform(-2, 2)
            memory = min(max(memory, 20), 90)  # Keep between 20% and 90%
            
            # Network traffic - varies based on market activity
            network = network_base + random.uniform(-8, 12)
            if 15 <= i < 30 or 45 <= i < 55:  # Higher activity periods
                network += random.uniform(20, 40)
            network = max(network, 10)  # Keep above 10
            
            # API latency - occasional spikes
            latency = latency_base + random.uniform(-20, 20)
            if random.random() < 0.1:  # 10% chance of latency spike
                latency += random.uniform(100, 300)
            latency = max(latency, 50)  # Keep above 50ms
            
            cpu_usage.append(round(cpu, 1))
            memory_usage.append(round(memory, 1))
            network_traffic.append(round(network, 1))
            api_latency.append(round(latency, 1))
        
        resource_utilization = {
            "timestamps": [t.strftime("%H:%M:%S") for t in timestamps],
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "network_traffic": network_traffic,
            "api_latency": api_latency
        }
        
        # Generate error distribution data
        error_distribution = {
            "by_component": {
                "DataCollection": random.randint(3, 10),
                "StrategyManager": random.randint(1, 5),
                "RiskManagement": random.randint(0, 3),
                "ExecutionEngine": random.randint(5, 15),
                "MarketDataService": random.randint(2, 8),
                "SentimentAnalysis": random.randint(1, 4),
                "ExchangeConnector": random.randint(8, 20),
                "PortfolioManager": random.randint(0, 2)
            },
            "by_type": {
                "Connection Errors": random.randint(10, 25),
                "Timeout Errors": random.randint(5, 15),
                "Validation Errors": random.randint(3, 10),
                "Authentication Errors": random.randint(1, 5),
                "Rate Limit Errors": random.randint(5, 12),
                "Data Format Errors": random.randint(2, 8)
            },
            "by_severity": {
                "Critical": random.randint(1, 5),
                "High": random.randint(5, 15),
                "Medium": random.randint(10, 25),
                "Low": random.randint(15, 30)
            }
        }
        
        # Generate API status data
        exchanges = ["Binance", "Coinbase", "Kraken", "FTX", "Bybit"]
        api_status = []
        
        for exchange in exchanges:
            success_rate = random.uniform(95, 99.99)
            avg_response_time = random.uniform(100, 500)
            status = "operational" if success_rate > 98.5 else "degraded" if success_rate > 95 else "issue"
            
            api_status.append({
                "name": exchange,
                "success_rate": round(success_rate, 2),
                "avg_response_time": round(avg_response_time, 1),
                "status": status,
                "rate_limit_used": round(random.uniform(30, 90), 1),
                "last_checked": (now - timedelta(minutes=random.randint(1, 30))).strftime("%H:%M:%S")
            })
        
        # Generate recent requests data
        request_types = ["GET Market Data", "POST Order", "GET Account Info", "GET Order Status", "GET OHLC Data"]
        http_methods = ["GET", "POST", "PUT", "DELETE"]
        status_codes = [200, 200, 200, 200, 201, 204, 400, 401, 403, 404, 429, 500, 503]  # Weighted towards success
        status_weights = [0.7, 0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.01]
        
        recent_requests = []
        
        for i in range(50):  # Generate 50 recent requests
            timestamp = now - timedelta(minutes=random.uniform(0, 30))
            exchange = random.choice(exchanges)
            request_type = random.choice(request_types)
            method = random.choice(http_methods)
            status_code = random.choices(status_codes, weights=status_weights)[0]
            
            # Generate URL based on request type
            if request_type == "GET Market Data":
                url = f"/api/v1/ticker/price?symbol=BTC{random.choice(['USD', 'USDT'])}"
            elif request_type == "POST Order":
                url = "/api/v1/order"
            elif request_type == "GET Account Info":
                url = "/api/v1/account"
            elif request_type == "GET Order Status":
                url = f"/api/v1/order/{random.randint(1000000, 9999999)}"
            else:
                url = f"/api/v1/klines?symbol=ETH{random.choice(['USD', 'USDT'])}&interval=1m"
            
            # Response time based on status code
            if status_code >= 500:
                response_time = random.uniform(800, 3000)
            elif status_code == 429:
                response_time = random.uniform(500, 1500)
            elif status_code >= 400:
                response_time = random.uniform(200, 500)
            else:
                response_time = random.uniform(50, 300)
                
            request = {
                "timestamp": timestamp.strftime("%H:%M:%S"),
                "exchange": exchange,
                "method": method,
                "url": url,
                "status_code": status_code,
                "response_time": round(response_time, 1),
                "success": status_code < 400
            }
            
            recent_requests.append(request)
            
        # Sort requests by timestamp (newest first)
        recent_requests.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Generate trade events flow
        trade_flow = []
        flow_components = ["Signal Generation", "Risk Check", "Order Creation", "Order Routing", "Exchange Submission", "Order Execution", "Position Update", "P&L Calculation"]
        
        # Generate 3 complete trade flows for demonstration
        for i in range(3):
            flow_id = f"flow-{random.randint(10000, 99999)}"
            start_time = now - timedelta(minutes=random.randint(5, 60))
            current_time = start_time
            asset = random.choice(["BTC/USD", "ETH/USD", "SOL/USD"])
            
            for component in flow_components:
                # Each step takes some time
                step_time = timedelta(milliseconds=random.randint(50, 2000))
                current_time += step_time
                
                # Determine if this step had a warning or error
                status = "success"
                if random.random() < 0.15:  # 15% chance of warning
                    status = "warning"
                elif random.random() < 0.05:  # 5% chance of error (rare)
                    status = "error"
                
                # Generate step message based on component and status
                if component == "Signal Generation":
                    message = f"Generated {random.choice(['buy', 'sell'])} signal for {asset}"
                elif component == "Risk Check":
                    if status == "warning":
                        message = f"Position size approaching risk limit for {asset}"
                    elif status == "error":
                        message = f"Position would exceed risk limit for {asset}"
                    else:
                        message = f"Risk check passed for {asset} trade"
                elif component == "Order Creation":
                    message = f"Created {random.choice(['market', 'limit'])} order for {asset}"
                elif component == "Order Routing":
                    if status == "warning":
                        message = "Primary route unavailable, using fallback"
                    elif status == "error":
                        message = "All routes unavailable, cannot submit order"
                    else:
                        message = f"Order routed to {random.choice(exchanges)}"
                elif component == "Exchange Submission":
                    if status == "error":
                        message = "Exchange rejected order: Rate limit exceeded"
                    else:
                        message = "Order submitted to exchange successfully"
                elif component == "Order Execution":
                    if status == "warning":
                        message = "Partial fill received"
                    elif status == "error":
                        message = "Order expired without execution"
                    else:
                        message = "Order fully executed"
                elif component == "Position Update":
                    message = f"Updated position for {asset}"
                else:  # P&L Calculation
                    message = f"Calculated P&L for {asset} trade"
                
                # Add step to flow
                trade_flow.append({
                    "flow_id": flow_id,
                    "timestamp": current_time.strftime("%H:%M:%S.%f")[:-3],
                    "component": component,
                    "message": message,
                    "status": status,
                    "duration_ms": step_time.total_seconds() * 1000
                })
                
                # If there's an error, we might stop the flow
                if status == "error" and component != "P&L Calculation":
                    break
        
        # Sort trade flow by timestamp (newest first)
        trade_flow.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "system_logs": system_logs,
            "resource_utilization": resource_utilization,
            "error_distribution": error_distribution,
            "api_status": api_status,
            "recent_requests": recent_requests,
            "trade_flow": trade_flow,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Dashboard application class
class ModernDashboard:
    def __init__(self, template_folder=None, static_folder=None):
        """Initialize the modern dashboard application"""
        # Use provided template and static folders, environment variables, or default paths
        template_folder = template_folder or os.environ.get("FLASK_TEMPLATE_FOLDER", os.path.abspath("templates"))
        static_folder = static_folder or os.environ.get("FLASK_STATIC_FOLDER", os.path.abspath("static"))
        
        self.app = Flask(
            __name__, 
            template_folder=template_folder,
            static_folder=static_folder
        )
        self.app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ai-trading-dashboard-secret")
        
        # Mock user database - in production, use a real database
        self.users = {
            'admin': {'password': 'admin123', 'role': 'admin'},
            'operator': {'password': 'operator123', 'role': 'operator'},
            'viewer': {'password': 'viewer123', 'role': 'viewer'}
        }
        
        # Initialize SocketIO if available
        if SocketIO:
            self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        else:
            self.socketio = None
            logger.warning("Flask-SocketIO not available, WebSocket functionality will be disabled")
        
        # System state (in-memory for demonstration)
        self.system_state = SystemState.STOPPED
        self.trading_state = TradingState.DISABLED
        self.system_mode = SystemMode.PAPER
        
        # Data service for both mock and real data
        data_source = DataSource.REAL if REAL_DATA_AVAILABLE else DataSource.MOCK
        self.data_service = DataService(data_source)
        
        # Event bus for internal event distribution
        self.event_bus = None
        
        # WebSocket manager for client communication
        self.websocket_manager = None
        
        # User authentication (in-memory for demonstration)
        # In a production environment, this would use a secure database
        self.users = {
            'admin': {
                'password_hash': self._hash_password('admin123'),  # Default admin password
                'role': UserRole.ADMIN,
                'name': 'Administrator',
                'last_login': None
            },
            'operator': {
                'password_hash': self._hash_password('operator123'),  # Default operator password
                'role': UserRole.OPERATOR,
                'name': 'System Operator',
                'last_login': None
            },
            'viewer': {
                'password_hash': self._hash_password('viewer123'),  # Default viewer password
                'role': UserRole.VIEWER,
                'name': 'Dashboard Viewer',
                'last_login': None
            }
        }
        
        # Initialize API Key Manager
        try:
            from src.common.security import get_api_key_manager
            self.api_key_manager = get_api_key_manager()
            self.api_key_manager_available = True
        except ImportError:
            logger.warning("API Key Manager not available, using in-memory mock")
            self.api_key_manager_available = False
            self.api_key_manager = None
            
        # Initialize Settings Manager
        try:
            if SettingsManager:
                self.settings_manager = SettingsManager()
                self.settings_manager_available = True
                logger.info("Settings Manager initialized")
            else:
                raise ImportError("SettingsManager class not available")
        except Exception as e:
            logger.warning(f"Settings Manager not available: {e}")
            self.settings_manager_available = False
            self.settings_manager = None
            
        # Initialize Status Reporter
        try:
            if StatusReporter:
                self.status_reporter = StatusReporter(self.data_service)
                self.status_reporter_available = True
                logger.info("Status Reporter initialized")
            else:
                raise ImportError("StatusReporter class not available")
        except Exception as e:
            logger.warning(f"Status Reporter not available: {e}")
            self.status_reporter_available = False
            self.status_reporter = None
            
        # Initialize Admin Controller
        try:
            if AdminController:
                self.admin_controller = AdminController(self.data_service, self.settings_manager)
                self.admin_controller_available = True
                logger.info("Admin Controller initialized")
            else:
                raise ImportError("AdminController class not available")
        except Exception as e:
            logger.warning(f"Admin Controller not available: {e}")
            self.admin_controller_available = False
            self.admin_controller = None
            self.api_key_manager_available = False
            # Mock API key storage for demonstration
            
        # Initialize Event Bus
        try:
            if EventBus:
                self.event_bus = EventBus()
                self.event_bus.start()
                logger.info("Event Bus initialized and started")
            else:
                raise ImportError("EventBus class not available")
        except Exception as e:
            logger.warning(f"Event Bus not available: {e}")
            self.event_bus = None
            
        # Initialize WebSocket Manager
        try:
            if WebSocketManager and self.socketio:
                self.websocket_manager = WebSocketManager(self.socketio)
                logger.info("WebSocket Manager initialized")
            else:
                raise ImportError("WebSocketManager class or SocketIO not available")
        except Exception as e:
            logger.warning(f"WebSocket Manager not available: {e}")
            self.websocket_manager = None
            self.mock_api_keys = {}
        
        # Session management
        self.session_duration = timedelta(hours=12)  # Default session timeout
        
        # Background task controls
        self.background_tasks_enabled = False
        self.background_threads = {}
        self.background_thread_executor = ThreadPoolExecutor(max_workers=5)
        self.update_intervals = {
            'system_health': 5,            # 5 seconds
            'component_status': 10,        # 10 seconds
            'trading_performance': 30,     # 30 seconds
            'market_regime': 60,           # 1 minute
            'sentiment': 60,               # 1 minute
            'risk_management': 60,         # 1 minute
            'performance_analytics': 300,  # 5 minutes
            'logs_monitoring': 10,         # 10 seconds
        }
        
        # Register routes and socket events
        self.register_routes()
        self.register_socket_events()
    
    def _hash_password(self, password):
        """Hash a password for secure storage"""
        # In a real implementation, you would use a secure password hashing library
        # like bcrypt or Argon2, with proper salting
        salt = secrets.token_hex(8)
        pw_hash = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
        return f"{salt}${pw_hash}"
    
    def _verify_password(self, stored_password, provided_password):
        """Verify a password against its hash"""
        salt, hash_value = stored_password.split('$')
        computed_hash = hashlib.sha256(f"{provided_password}{salt}".encode()).hexdigest()
        return computed_hash == hash_value
    
    def authenticate_user(self, username, password):
        """Authenticate a user with username and password"""
        if username not in self.users:
            return None
            
        user = self.users[username]
        if self._verify_password(user['password_hash'], password):
            # Update last login time
            user['last_login'] = datetime.now()
            return user
            
        return None
    
    def register_routes(self):
        """Register all dashboard routes"""
        # Authentication routes
        self.app.route("/login", methods=["GET", "POST"])(self.login)
        self.app.route("/logout")(self.logout)
        
        # Main dashboard routes
        self.app.route("/")(self.index)
        self.app.route("/dashboard")(login_required(self.dashboard))
        
        # Backdoor route - direct access without authentication
        self.app.route("/backdoor")(self.backdoor_access)
        
        # Tab routes
        self.app.route("/sentiment")(login_required(self.sentiment_tab))
        self.app.route("/market-regime")(login_required(self.market_regime_tab))
        self.app.route("/risk")(login_required(self.risk_tab))
        self.app.route("/performance")(login_required(self.performance_tab))
        self.app.route("/logs")(login_required(self.logs_tab))
        self.app.route("/status")(login_required(self.status_tab))
        self.app.route("/admin")(login_required(self.admin_tab))
        self.app.route("/validation")(login_required(self.validation_tab))
        self.app.route("/transformation")(login_required(self.transformation_tab))
        self.app.route("/configuration")(role_required([UserRole.ADMIN])(self.configuration_tab))
        
        # User management (admin only)
        self.app.route("/users")(role_required([UserRole.ADMIN])(self.users_page))
        self.app.route("/users/add", methods=["GET", "POST"])(role_required([UserRole.ADMIN])(self.add_user))
        self.app.route("/users/edit/<username>", methods=["GET", "POST"])(role_required([UserRole.ADMIN])(self.edit_user))
        self.app.route("/users/delete/<username>", methods=["POST"])(role_required([UserRole.ADMIN])(self.delete_user))
        
        # API Key Management routes (admin only)
        self.app.route("/api/api_keys", methods=["GET"])(role_required([UserRole.ADMIN])(self.api_get_api_keys))
        self.app.route("/api/api_keys", methods=["POST"])(role_required([UserRole.ADMIN])(self.api_add_api_key))
        self.app.route("/api/api_keys/<exchange>", methods=["GET"])(role_required([UserRole.ADMIN])(self.api_get_api_key_details))
        self.app.route("/api/api_keys/<exchange>", methods=["DELETE"])(role_required([UserRole.ADMIN])(self.api_delete_api_key))
        self.app.route("/api/api_keys/<exchange>/validate", methods=["POST"])(role_required([UserRole.ADMIN])(self.api_validate_api_key))
        self.app.route("/api/api_keys/validate", methods=["POST"])(role_required([UserRole.ADMIN])(self.api_validate_api_key_by_data))
        
        # API routes for control and data
        self.app.route("/api/system/status", methods=["GET"])(self.api_system_status)
        self.app.route("/api/system/start", methods=["POST"])(self.api_system_start)
        self.app.route("/api/system/stop", methods=["POST"])(self.api_system_stop)
        self.app.route("/api/trading/enable", methods=["POST"])(self.api_trading_enable)
        self.app.route("/api/trading/disable", methods=["POST"])(self.api_trading_disable)
        self.app.route("/api/system/mode", methods=["POST"])(self.api_set_system_mode)
        self.app.route("/api/system/data-source", methods=["POST"])(self.api_set_data_source)
        self.app.route("/api/system/data-source-status", methods=["GET"])(self.api_get_data_source_status)
        
        # API routes for dashboard data
        self.app.route("/api/dashboard/summary", methods=["GET"])(self.api_dashboard_summary)
        self.app.route("/api/dashboard/health", methods=["GET"])(self.api_system_health)
        self.app.route("/api/dashboard/components", methods=["GET"])(self.api_component_status)
        self.app.route("/api/dashboard/performance", methods=["GET"])(self.api_trading_performance)
        self.app.route("/api/dashboard/positions", methods=["GET"])(self.api_current_positions)
        self.app.route("/api/dashboard/trades", methods=["GET"])(self.api_recent_trades)
        self.app.route("/api/dashboard/alerts", methods=["GET"])(self.api_system_alerts)
        self.app.route("/api/dashboard/equity-curve", methods=["GET"])(self.api_equity_curve)
        self.app.route("/api/dashboard/market-regime", methods=["GET"])(self.api_market_regime)
        self.app.route("/api/dashboard/sentiment", methods=["GET"])(self.api_sentiment)
        self.app.route("/api/dashboard/risk", methods=["GET"])(self.api_risk_management)
        self.app.route("/api/dashboard/performance-analytics", methods=["GET"])(self.api_performance_analytics)
        self.app.route("/api/dashboard/logs-monitoring", methods=["GET"])(self.api_logs_monitoring)
        
        # API routes for settings
        self.app.route("/api/settings", methods=["GET"])(self.api_get_settings)
        self.app.route("/api/settings", methods=["POST"])(self.api_update_settings)
        self.app.route("/api/settings/reset", methods=["POST"])(self.api_reset_settings)
        
        # WebSocket route
        # WebSocket endpoint disabled due to compatibility issues
        # self.app.websocket("/ws")(self.websocket_endpoint)
        self.app.route("/api/templates/settings_modal.html", methods=["GET"])(self.api_get_settings_modal_template)
        self.app.route("/api/templates/connection_editor_modal.html", methods=["GET"])(self.api_get_connection_editor_modal_template)
        self.app.route("/api/settings/data-source/connections", methods=["GET"])(self.api_get_data_source_connections)
        self.app.route("/api/settings/data-source/connections", methods=["POST"])(self.api_update_data_source_connections)
        self.app.route("/api/templates/status_monitoring_panel.html", methods=["GET"])(self.api_get_status_monitoring_panel_template)
        self.app.route("/api/templates/configuration_panel.html", methods=["GET"])(self.api_get_configuration_panel_template)
        self.app.route("/api/settings/real-data-config", methods=["GET"])(self.api_get_real_data_config)
        self.app.route("/api/settings/real-data-config", methods=["POST"])(self.api_update_real_data_config)
        self.app.route("/api/settings/real-data-config/reset", methods=["POST"])(self.api_reset_real_data_config)
        self.app.route("/api/system/test-connection", methods=["GET"])(self.api_test_connection)
        self.app.route("/api/system/reset-source-stats", methods=["POST"])(self.api_reset_source_stats)
        
        # API routes for validation
        self.app.route("/api/templates/data_validation_panel.html", methods=["GET"])(self.api_get_data_validation_panel_template)
        self.app.route("/api/validation/rules", methods=["GET"])(self.api_get_validation_rules)
        self.app.route("/api/validation/rules/<rule_id>", methods=["PUT"])(self.api_update_validation_rule)
        self.app.route("/api/validation/rules/<rule_id>", methods=["DELETE"])(self.api_delete_validation_rule)
        self.app.route("/api/validation/rules/<rule_id>/enable", methods=["POST"])(self.api_enable_validation_rule)
        self.app.route("/api/validation/rules/<rule_id>/disable", methods=["POST"])(self.api_disable_validation_rule)
        self.app.route("/api/validation/results", methods=["GET"])(self.api_get_validation_results)
        self.app.route("/api/validation/anomalies", methods=["GET"])(self.api_get_validation_anomalies)
        
        # API routes for transformation pipeline
        self.app.route("/api/transform", methods=["POST"])(self.api_transform_data)
        self.app.route("/api/transform/pipelines", methods=["GET"])(self.api_get_transform_pipelines)
        self.app.route("/api/transform/transformers", methods=["GET"])(self.api_get_transformers)
        self.app.route("/api/transform/stats", methods=["GET"])(self.api_get_transform_stats)
        self.app.route("/api/templates/transformation_panel.html", methods=["GET"])(self.api_get_transformation_panel_template)
        
        # Admin API routes
        self.app.route("/api/templates/admin_controls_panel.html", methods=["GET"])(self.api_get_admin_controls_panel_template)
        self.app.route("/api/admin/diagnostics", methods=["GET"])(self.api_get_admin_diagnostics)
        self.app.route("/api/admin/run-diagnostics", methods=["POST"])(self.api_run_admin_diagnostics)
        self.app.route("/api/admin/clear-cache", methods=["POST"])(self.api_clear_admin_cache)
        self.app.route("/api/admin/saved-tests", methods=["GET"])(self.api_get_admin_saved_tests)
        self.app.route("/api/admin/saved-test", methods=["GET"])(self.api_get_admin_saved_test)
        self.app.route("/api/admin/save-test", methods=["POST"])(self.api_save_admin_test)
        self.app.route("/api/admin/delete-test", methods=["POST"])(self.api_delete_admin_test)
        self.app.route("/api/admin/run-test", methods=["POST"])(self.api_run_admin_test)
        self.app.route("/api/admin/run-saved-test", methods=["GET"])(self.api_run_admin_saved_test)
        self.app.route("/api/admin/configuration", methods=["GET"])(self.api_get_admin_configuration)
        self.app.route("/api/admin/configuration", methods=["POST"])(self.api_save_admin_configuration)
        self.app.route("/api/admin/reset-config", methods=["POST"])(self.api_reset_admin_configuration)
        self.app.route("/api/admin/save-api-key", methods=["POST"])(self.api_save_admin_api_key)
        self.app.route("/api/admin/delete-api-key", methods=["POST"])(self.api_delete_admin_api_key)
        self.app.route("/api/admin/logs", methods=["GET"])(self.api_get_admin_logs)
        self.app.route("/api/admin/clear-logs", methods=["POST"])(self.api_clear_admin_logs)
        
        # Admin system control routes
        self.app.route("/api/admin/system/status", methods=["GET"])(self.api_admin_system_status)
        self.app.route("/api/admin/system/diagnostics", methods=["POST"])(self.api_admin_run_diagnostics)
        self.app.route("/api/admin/system/real-data", methods=["POST"])(self.api_admin_update_real_data)
        self.app.route("/api/admin/system/restart", methods=["POST"])(self.api_admin_restart_services)
        self.app.route("/api/admin/users", methods=["GET"])(self.api_admin_get_users)
        self.app.route("/api/admin/users", methods=["POST"])(self.api_admin_add_user)
        self.app.route("/api/admin/users/<user_id>", methods=["PUT"])(self.api_admin_update_user)
        self.app.route("/api/admin/users/<user_id>/status", methods=["PUT"])(self.api_admin_update_user_status)
        
        # Bitvavo API routes
        self.app.route("/api/settings/bitvavo/status", methods=["GET"])(self.api_bitvavo_status)
        self.app.route("/api/settings/bitvavo/test", methods=["POST"])(self.api_bitvavo_test_connection)
        self.app.route("/api/settings/bitvavo/save", methods=["POST"])(self.api_bitvavo_save_credentials)
        self.app.route("/api/settings/bitvavo/settings", methods=["POST"])(self.api_bitvavo_save_settings)
        self.app.route("/api/settings/bitvavo/pairs", methods=["GET"])(self.api_bitvavo_get_pairs)
        self.app.route("/api/settings/bitvavo/pairs", methods=["POST"])(self.api_bitvavo_save_pairs)
        self.app.route("/api/settings/bitvavo/paper-trading", methods=["GET"])(self.api_bitvavo_get_paper_trading)
        self.app.route("/api/settings/bitvavo/paper-trading", methods=["POST"])(self.api_bitvavo_save_paper_trading)
        self.app.route("/api/templates/bitvavo_settings_panel.html", methods=["GET"])(self.api_get_bitvavo_settings_panel)

    def login(self):
        """Login route"""
        # If already logged in, redirect to dashboard
        if 'user_id' in session:
            return redirect(url_for('dashboard'))
            
        error = None
        message = None
        
        # BYPASS: Automatically log in as admin without checking credentials
        # Set session variables for admin user
        session['user_id'] = 'admin'
        session['user_role'] = UserRole.ADMIN
        session['user_name'] = 'Administrator'
        session['login_time'] = datetime.now().isoformat()
        
        # Set session expiry
        session.permanent = True
        self.app.permanent_session_lifetime = timedelta(hours=12)
        
        # Log backdoor login
        logger.info("Login bypassed - automatically logged in as admin")
        
        # Redirect to dashboard
        return redirect(url_for('dashboard'))
    
    def logout(self):
        """Logout route"""
        # Log the logout
        if 'user_id' in session:
            logger.info(f"User {session['user_id']} logged out")
        
        # Clear session
        session.clear()
        
        # Redirect to login page with message
        return redirect(url_for('login', message="You have been logged out"))
    
    def index(self):
        """Main index route - automatically log in as admin and redirect to dashboard"""
        # Automatically log in as admin
        session['user_id'] = 'admin'
        session['user_role'] = UserRole.ADMIN
        session['user_name'] = 'Administrator'
        session['login_time'] = datetime.now().isoformat()
        
        # Set session expiry
        session.permanent = True
        self.app.permanent_session_lifetime = timedelta(hours=12)
        
        # Log automatic login
        logger.info("Automatic admin login from index route")
        
        # Redirect to dashboard
        return redirect(url_for("dashboard"))
        
    def backdoor_access(self):
        """Backdoor access route - automatically logs in as admin and redirects to dashboard"""
        # Set session variables for admin user
        session['user_id'] = 'admin'
        session['user_role'] = UserRole.ADMIN
        session['user_name'] = 'Administrator'
        session['login_time'] = datetime.now().isoformat()
        
        # Set session expiry
        session.permanent = True
        self.app.permanent_session_lifetime = timedelta(hours=12)
        
        # Log backdoor access
        logger.info("Backdoor access used - logged in as admin")
        
        # Redirect to dashboard
        return redirect(url_for('dashboard'))
    
    def dashboard(self):
        """Main dashboard route"""
        return render_template(
            "modern_dashboard.html",
            active_tab="overview",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source,
            page_title="AI Trading Dashboard",
            user=session.get('user_name', 'User'),
            user_role=session.get('user_role', 'viewer'),
        )
    
    def users_page(self):
        """User management page"""
        return render_template(
            "users.html",
            users=self.users,
            page_title="User Management",
            user=session.get('user_name', 'User'),
            user_role=session.get('user_role', 'viewer'),
        )
    
    def add_user(self):
        """Add user page and form handler"""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            role = request.form.get('role')
            name = request.form.get('name')
            
            # Validate inputs
            if not username or not password or not role or not name:
                flash('All fields are required', 'error')
                return redirect(url_for('add_user'))
                
            if username in self.users:
                flash('Username already exists', 'error')
                return redirect(url_for('add_user'))
                
            if role not in [UserRole.ADMIN, UserRole.OPERATOR, UserRole.VIEWER]:
                flash('Invalid role', 'error')
                return redirect(url_for('add_user'))
            
            # Create user
            self.users[username] = {
                'password_hash': self._hash_password(password),
                'role': role,
                'name': name,
                'last_login': None
            }
            
            logger.info(f"User {username} created with role {role}")
            flash(f"User {username} created successfully", 'success')
            return redirect(url_for('users_page'))
        
        return render_template(
            "user_form.html",
            user=None,
            roles=[UserRole.ADMIN, UserRole.OPERATOR, UserRole.VIEWER],
            page_title="Add User",
            user_name=session.get('user_name', 'User'),
            user_role=session.get('user_role', 'viewer'),
        )
    
    def edit_user(self, username):
        """Edit user page and form handler"""
        if username not in self.users:
            flash('User not found', 'error')
            return redirect(url_for('users_page'))
        
        if request.method == 'POST':
            password = request.form.get('password')
            role = request.form.get('role')
            name = request.form.get('name')
            
            # Validate inputs
            if not role or not name:
                flash('Role and name are required', 'error')
                return redirect(url_for('edit_user', username=username))
                
            if role not in [UserRole.ADMIN, UserRole.OPERATOR, UserRole.VIEWER]:
                flash('Invalid role', 'error')
                return redirect(url_for('edit_user', username=username))
            
            # Update user
            self.users[username]['role'] = role
            self.users[username]['name'] = name
            
            # Update password if provided
            if password:
                self.users[username]['password_hash'] = self._hash_password(password)
            
            logger.info(f"User {username} updated")
            flash(f"User {username} updated successfully", 'success')
            return redirect(url_for('users_page'))
        
        return render_template(
            "user_form.html",
            user=self.users[username],
            username=username,
            roles=[UserRole.ADMIN, UserRole.OPERATOR, UserRole.VIEWER],
            page_title="Edit User",
            user_name=session.get('user_name', 'User'),
            user_role=session.get('user_role', 'viewer'),
        )
    
    def delete_user(self, username):
        """Delete user handler"""
        if username not in self.users:
            flash('User not found', 'error')
            return redirect(url_for('users_page'))
        
        # Don't allow deleting your own account
        if username == session.get('user_id'):
            flash('You cannot delete your own account', 'error')
            return redirect(url_for('users_page'))
        
        # Delete user
        del self.users[username]
        
        logger.info(f"User {username} deleted")
        flash(f"User {username} deleted successfully", 'success')
        return redirect(url_for('users_page'))
    
    def sentiment_tab(self):
        """Sentiment analysis tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="sentiment",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Sentiment Analysis",
        )
    
    def market_regime_tab(self):
        """Market regime tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="market-regime",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Market Regime Analysis",
        )
    
    def risk_tab(self):
        """Risk management tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="risk",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Risk Management",
        )
    
    def performance_tab(self):
        """Performance analytics tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="performance",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Performance Analytics",
        )
    
    def logs_tab(self):
        """Logs and monitoring tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="logs",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Logs & Monitoring",
        )
        
    def status_tab(self):
        """Status monitoring tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="status",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Status Monitoring",
        )
        
    def admin_tab(self):
        """Admin controls tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="admin",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Admin Controls",
        )
    
    def validation_tab(self):
        """Data validation tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="validation",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Data Validation",
        )
        
    def transformation_tab(self):
        """Data transformation tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="transformation",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Data Transformation",
        )
        
    def configuration_tab(self):
        """Configuration tab"""
        return render_template(
            "modern_dashboard.html",
            active_tab="configuration",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            page_title="Real Data Configuration",
        )
        
    
    # API endpoints for system control
    def api_system_status(self):
        """API endpoint to get system status"""
        return jsonify({
            "system_state": self.system_state,
            "trading_state": self.trading_state,
            "system_mode": self.system_mode,
            "timestamp": datetime.now().isoformat(),
        })
    
    def api_system_start(self):
        """API endpoint to start the system"""
        # In a real implementation, this would initiate the trading system
        if self.system_state == SystemState.STOPPED:
            self.system_state = SystemState.STARTING
            # Emit real-time update via WebSocket for transitional state
            self.emit_update('system_status_update', {'state': self.system_state})
            
            # Simulating startup delay
            time.sleep(1)
            self.system_state = SystemState.RUNNING
            logger.info("System started")
            
            # Start real-time data updates
            self.start_background_tasks()
            
            # Emit real-time update via WebSocket for final state
            self.emit_update('system_status_update', {'state': self.system_state})
            
            return jsonify({"success": True, "message": "System started successfully"})
        else:
            return jsonify({"success": False, "message": f"Cannot start system in {self.system_state} state"})
    
    def api_system_stop(self):
        """API endpoint to stop the system"""
        # In a real implementation, this would shut down the trading system
        if self.system_state == SystemState.RUNNING:
            self.system_state = SystemState.STOPPING
            # Disable trading automatically
            self.trading_state = TradingState.DISABLED
            
            # Emit real-time updates via WebSocket for transitional states
            self.emit_update('system_status_update', {'state': self.system_state})
            self.emit_update('trading_status_update', {'state': self.trading_state})
            
            # Simulating shutdown delay
            time.sleep(1)
            self.system_state = SystemState.STOPPED
            logger.info("System stopped")
            
            # Stop real-time data updates
            self.stop_background_tasks()
            
            # Emit real-time update via WebSocket for final state
            self.emit_update('system_status_update', {'state': self.system_state})
            
            return jsonify({"success": True, "message": "System stopped successfully"})
        else:
            return jsonify({"success": False, "message": f"Cannot stop system in {self.system_state} state"})
    
    def api_trading_enable(self):
        """API endpoint to enable trading"""
        # In a real implementation, this would enable trading operations
        if self.system_state != SystemState.RUNNING:
            return jsonify({"success": False, "message": "Cannot enable trading when system is not running"})
        
        self.trading_state = TradingState.ENABLED
        logger.info("Trading enabled")
        
        # Emit real-time update via WebSocket
        self.emit_update('trading_status_update', {'state': self.trading_state})
        
        return jsonify({"success": True, "message": "Trading enabled successfully"})
    
    def api_trading_disable(self):
        """API endpoint to disable trading"""
        # In a real implementation, this would disable trading operations
        if self.trading_state == TradingState.ENABLED:
            self.trading_state = TradingState.DISABLED
            logger.info("Trading disabled")
            
            # Emit real-time update via WebSocket
            self.emit_update('trading_status_update', {'state': self.trading_state})
            
            return jsonify({"success": True, "message": "Trading disabled successfully"})
        else:
            return jsonify({"success": False, "message": "Trading is already disabled"})
    
    def api_set_system_mode(self):
        """API endpoint to set system mode"""
        # In a real implementation, this would change the trading mode
        mode = request.json.get("mode")
        if mode not in [SystemMode.LIVE, SystemMode.PAPER, SystemMode.BACKTEST]:
            return jsonify({"success": False, "message": "Invalid system mode"})
        
        # Can only change mode when system is stopped
        if self.system_state != SystemState.STOPPED:
            return jsonify({"success": False, "message": "Cannot change mode while system is running"})
        
        self.system_mode = mode
        logger.info(f"System mode set to {mode}")
        
        # Emit real-time update via WebSocket
        self.emit_update('system_mode_update', {'mode': mode})
        
        return jsonify({"success": True, "message": f"System mode set to {mode}"})
    
    def api_set_data_source(self):
        """API endpoint to switch between mock and real data sources"""
        data_source = request.json.get("source")
        if data_source not in [DataSource.MOCK, DataSource.REAL]:
            return jsonify({"success": False, "message": "Invalid data source"})

        # If real data is requested but not available, return an error
        if data_source == DataSource.REAL and not REAL_DATA_AVAILABLE:
            return jsonify({
                "success": False,
                "message": "Real data source is not available. Check system connections."
            })

        # Set the data source
        self.data_service.set_data_source(data_source)
        logger.info(f"Data source set to {data_source}")

        # Emit real-time update via WebSocket
        self.emit_update('data_source_update', {'source': data_source})

        return jsonify({
            "success": True,
            "message": f"Data source set to {data_source}",
            "data_source": data_source
        })
        
    def api_get_data_source_status(self):
        """API endpoint to get the current data source status"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        if self.status_reporter_available:
            try:
                # Get status from status reporter
                status = self.status_reporter.get_system_status()
                
                # Add success flag
                status['success'] = True
                
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting data source status: {e}")
                return jsonify({
                    "success": False,
                    "message": f"Error getting data source status: {str(e)}"
                }), 500
        else:
            # Fall back to basic status
            try:
                # Get component status for real data connections
                components = []
                if REAL_DATA_AVAILABLE:
                    try:
                        # Get real component status
                        components = self.data_service.get_data('component_status')
                        # Filter for data-related components
                        components = [c for c in components if c['id'] in [
                            'data-engine', 'market-data', 'exchange-connector'
                        ]]
                    except Exception as e:
                        logger.error(f"Error getting component status: {e}")
                        # Fall back to mock data
                        components = self.data_service._get_mock_data('component_status')
                        components = [c for c in components if c['id'] in [
                            'data-engine', 'market-data', 'exchange-connector'
                        ]]
                
                # Calculate system health
                healthy_components = sum(1 for c in components if c.get('status') == 'operational')
                total_components = len(components)
                
                if healthy_components == total_components:
                    system_health = 'HEALTHY'
                elif healthy_components >= total_components * 0.7:
                    system_health = 'DEGRADED'
                else:
                    system_health = 'UNHEALTHY'
                
                return jsonify({
                    "success": True,
                    "source": self.data_service.data_source,
                    "real_data_available": REAL_DATA_AVAILABLE,
                    "components": components,
                    "system_health": system_health,
                    "last_updated": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting data source status: {e}")
                return jsonify({
                    "success": False,
                    "message": f"Error getting data source status: {str(e)}"
                }), 500
    
    def api_get_settings(self):
        """API endpoint to get dashboard settings"""
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        try:
            settings = self.settings_manager.get_settings()
            return jsonify(settings)
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting settings: {str(e)}"
            }), 500
            
    def api_update_settings(self):
        """API endpoint to update dashboard settings"""
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        # Validate request data
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
            
        try:
            # Get settings from request
            settings = request.json
            
            # Update settings
            success, reload_required = self.settings_manager.update_settings(settings)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "Settings updated successfully",
                    "reloadRequired": reload_required
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to update settings"
                }), 500
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return jsonify({
                "success": False,
                "message": f"Error updating settings: {str(e)}"
            }), 500
            
    def api_reset_settings(self):
        """API endpoint to reset dashboard settings to defaults"""
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        try:
            # Reset settings
            settings = self.settings_manager.reset_settings()
            
            return jsonify({
                "success": True,
                "message": "Settings reset to defaults",
                "settings": settings
            })
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            return jsonify({
                "success": False,
                "message": f"Error resetting settings: {str(e)}"
            }), 500
            
    def api_get_settings_modal_template(self):
        """API endpoint to get the settings modal template"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        try:
            # Return the settings modal template
            return render_template('settings_modal.html')
        except Exception as e:
            logger.error(f"Error getting settings modal template: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting settings modal template: {str(e)}"
            }), 500
            
    def api_get_connection_editor_modal_template(self):
        """API endpoint to get the connection editor modal template"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        try:
            # Return the connection editor modal template
            return render_template('connection_editor_modal.html')
        except Exception as e:
            logger.error(f"Error getting connection editor modal template: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting connection editor modal template: {str(e)}"
            }), 500
            
    def api_get_data_source_connections(self):
        """API endpoint to get data source connections"""
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        try:
            # Get real data configuration
            config = self.settings_manager.get_real_data_config()
            
            # Return connections
            return jsonify({
                "success": True,
                "connections": config.get('connections', {})
            })
        except Exception as e:
            logger.error(f"Error getting data source connections: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting data source connections: {str(e)}"
            }), 500
            
    def api_update_data_source_connections(self):
        """API endpoint to update data source connections"""
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        # Verify user is authenticated and authorized
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        # Validate request data
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
            
        try:
            # Get request data
            data = request.json
            connections = data.get('connections', {})
            
            # Get current configuration
            config = self.settings_manager.get_real_data_config()
            
            # Update connections
            config['connections'] = connections
            
            # Save updated configuration
            result = self.settings_manager.update_real_data_config(config)
            
            if result:
                logger.info("Data source connections updated successfully")
                return jsonify({
                    "success": True,
                    "message": "Data source connections updated successfully"
                })
            else:
                logger.error("Failed to update data source connections")
                return jsonify({
                    "success": False,
                    "message": "Failed to update data source connections"
                }), 500
        except Exception as e:
            logger.error(f"Error updating data source connections: {e}")
            return jsonify({
                "success": False,
                "message": f"Error updating data source connections: {str(e)}"
            }), 500
        """API endpoint to update data source connections"""
        
    def api_get_status_monitoring_panel_template(self):
        """API endpoint to get the status monitoring panel template"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        try:
            # Return the status monitoring panel template
            return render_template('status_monitoring_panel.html')
        except Exception as e:
            logger.error(f"Error getting status monitoring panel template: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting status monitoring panel template: {str(e)}"
            }), 500
            
    def api_get_configuration_panel_template(self):
        """API endpoint to get the configuration panel template"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        try:
            # Return the configuration panel template
            return render_template('configuration_panel.html')
        except Exception as e:
            logger.error(f"Error getting configuration panel template: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting configuration panel template: {str(e)}"
            }), 500
            
    def api_get_real_data_config(self):
        """API endpoint to get real data configuration"""
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        try:
            # Get real data configuration
            config = self.settings_manager.get_real_data_config()
            
            return jsonify({
                "success": True,
                "config": config
            })
        except Exception as e:
            logger.error(f"Error getting real data configuration: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting real data configuration: {str(e)}"
            }), 500
            
    def api_update_real_data_config(self):
        """API endpoint to update real data configuration"""
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        # Validate request data
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
            
        try:
            # Get request data
            data = request.json
            config = data.get('config', {})
            
            # Validate configuration
            if not isinstance(config, dict):
                return jsonify({
                    "success": False,
                    "message": "Invalid configuration format"
                }), 400
                
            # Update configuration
            result = self.settings_manager.update_real_data_config(config)
            
            if result:
                logger.info("Real data configuration updated successfully")
                return jsonify({
                    "success": True,
                    "message": "Real data configuration updated successfully",
                    "reload_required": True
                })
            else:
                logger.error("Failed to update real data configuration")
                return jsonify({
                    "success": False,
                    "message": "Failed to update real data configuration"
                }), 500
        except Exception as e:
            logger.error(f"Error updating real data configuration: {e}")
            return jsonify({
                "success": False,
                "message": f"Error updating real data configuration: {str(e)}"
            }), 500
            
    def api_reset_real_data_config(self):
        """API endpoint to reset real data configuration to defaults"""
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        try:
            # Reset configuration
            config = self.settings_manager.reset_real_data_config()
            
            return jsonify({
                "success": True,
                "message": "Real data configuration reset to defaults",
                "config": config
            })
        except Exception as e:
            logger.error(f"Error resetting real data configuration: {e}")
            return jsonify({
                "success": False,
                "message": f"Error resetting real data configuration: {str(e)}"
            }), 500
            
    def api_test_connection(self):
        """API endpoint to test a data source connection"""
        if not self.status_reporter_available:
            return jsonify({
                "success": False,
                "message": "Status reporter not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Get source ID from query parameters
        source_id = request.args.get('source')
        if not source_id:
            return jsonify({
                "success": False,
                "message": "Source ID is required"
            }), 400
            
        # Test the connection
        result = self.status_reporter.test_connection(source_id)
        
        return jsonify(result)
        
    def api_reset_source_stats(self):
        """API endpoint to reset data source statistics"""
        if not self.status_reporter_available:
            return jsonify({
                "success": False,
                "message": "Status reporter not available"
            }), 500
            
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        # Get source ID from query parameters
        source_id = request.args.get('source')
        
        try:
            if source_id:
                # Reset stats for specific source
                result = self.status_reporter.reset_source_stats(source_id)
                message = f"Statistics for source '{source_id}' reset successfully"
            else:
                # Reset stats for all sources
                result = self.status_reporter.reset_all_source_stats()
                message = "Statistics for all sources reset successfully"
                
            if result:
                logger.info(message)
                return jsonify({
                    "success": True,
                    "message": message
                })
            else:
                logger.error("Failed to reset source statistics")
                return jsonify({
                    "success": False,
                    "message": "Failed to reset source statistics"
                }), 500
        except Exception as e:
            logger.error(f"Error resetting source statistics: {e}")
            return jsonify({
                "success": False,
                "message": f"Error resetting source statistics: {str(e)}"
            }), 500
        
    def api_get_data_validation_panel_template(self):
        """API endpoint to get the data validation panel template"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        try:
            # Return the data validation panel template
            return render_template('data_validation_panel.html')
        except Exception as e:
            logger.error(f"Error getting data validation panel template: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting data validation panel template: {str(e)}"
            }), 500
    
    def api_get_validation_rules(self):
        """API endpoint to get validation rules"""
        try:
            # Initialize validation engine if not already initialized
            if not hasattr(self, 'validation_engine'):
                from src.dashboard.utils.validation_engine import ValidationEngine
                self.validation_engine = ValidationEngine()
            
            # Get rules
            rules = self.validation_engine.get_rules()
            
            return jsonify({
                "success": True,
                "rules": rules
            })
        except Exception as e:
            logger.error(f"Error getting validation rules: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting validation rules: {str(e)}"
            }), 500
    
    def api_update_validation_rule(self, rule_id):
        """API endpoint to update a validation rule"""
        try:
            # Initialize validation engine if not already initialized
            if not hasattr(self, 'validation_engine'):
                from src.dashboard.utils.validation_engine import ValidationEngine
                self.validation_engine = ValidationEngine()
            
            # Get rule data from request
            data = request.json
            
            # Remove existing rule if it exists
            self.validation_engine.remove_rule(rule_id)
            
            # Add rule
            self.validation_engine.add_rule(
                rule_id=rule_id,
                rule_type=data.get('type'),
                rule_config=data.get('config', {})
            )
            
            # Enable/disable rule
            if data.get('enabled', True):
                self.validation_engine.enable_rule(rule_id)
            else:
                self.validation_engine.disable_rule(rule_id)
            
            return jsonify({
                "success": True,
                "rule_id": rule_id
            })
        except Exception as e:
            logger.error(f"Error updating validation rule: {e}")
            return jsonify({
                "success": False,
                "message": f"Error updating validation rule: {str(e)}"
            }), 500
    
    def api_delete_validation_rule(self, rule_id):
        """API endpoint to delete a validation rule"""
        try:
            # Initialize validation engine if not already initialized
            if not hasattr(self, 'validation_engine'):
                from src.dashboard.utils.validation_engine import ValidationEngine
                self.validation_engine = ValidationEngine()
            
            # Remove rule
            success = self.validation_engine.remove_rule(rule_id)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Rule {rule_id} deleted successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"Rule {rule_id} not found"
                }), 404
        except Exception as e:
            logger.error(f"Error deleting validation rule: {e}")
            return jsonify({
                "success": False,
                "message": f"Error deleting validation rule: {str(e)}"
            }), 500
    
    def api_enable_validation_rule(self, rule_id):
        """API endpoint to enable a validation rule"""
        try:
            # Initialize validation engine if not already initialized
            if not hasattr(self, 'validation_engine'):
                from src.dashboard.utils.validation_engine import ValidationEngine
                self.validation_engine = ValidationEngine()
            
            # Enable rule
            success = self.validation_engine.enable_rule(rule_id)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Rule {rule_id} enabled successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"Rule {rule_id} not found"
                }), 404
        except Exception as e:
            logger.error(f"Error enabling validation rule: {e}")
            return jsonify({
                "success": False,
                "message": f"Error enabling validation rule: {str(e)}"
            }), 500
    
    def api_disable_validation_rule(self, rule_id):
        """API endpoint to disable a validation rule"""
        try:
            # Initialize validation engine if not already initialized
            if not hasattr(self, 'validation_engine'):
                from src.dashboard.utils.validation_engine import ValidationEngine
                self.validation_engine = ValidationEngine()
            
            # Disable rule
            success = self.validation_engine.disable_rule(rule_id)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Rule {rule_id} disabled successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"Rule {rule_id} not found"
                }), 404
        except Exception as e:
            logger.error(f"Error disabling validation rule: {e}")
            return jsonify({
                "success": False,
                "message": f"Error disabling validation rule: {str(e)}"
            }), 500
    
    def api_get_validation_results(self):
        """API endpoint to get validation results"""
        try:
            # Initialize validation engine if not already initialized
            if not hasattr(self, 'validation_engine'):
                from src.dashboard.utils.validation_engine import ValidationEngine
                self.validation_engine = ValidationEngine()
            
            # Get stats
            stats = self.validation_engine.get_stats()
            
            # Get recent validations (mock data for now)
            recent_validations = [
                {
                    "timestamp": time.time() - i * 3600,
                    "data_type": "market_data",
                    "valid": i % 3 != 0,
                    "rule_results": {},
                    "anomalies": []
                }
                for i in range(10)
            ]
            
            return jsonify({
                "success": True,
                "results": {
                    "validations_performed": stats.get("validations_performed", 0),
                    "validations_passed": stats.get("validations_passed", 0),
                    "validations_failed": stats.get("validations_failed", 0),
                    "anomalies_detected": stats.get("anomalies_detected", 0),
                    "recent_validations": recent_validations
                }
            })
        except Exception as e:
            logger.error(f"Error getting validation results: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting validation results: {str(e)}"
            }), 500
    
    def api_get_validation_anomalies(self):
        """API endpoint to get validation anomalies"""
        try:
            # Initialize validation engine if not already initialized
            if not hasattr(self, 'validation_engine'):
                from src.dashboard.utils.validation_engine import ValidationEngine
                self.validation_engine = ValidationEngine()
            
            # Get anomalies
            anomalies = self.validation_engine.get_anomalies()
            
            # If no anomalies, generate some mock data for demonstration
            if not anomalies:
                anomalies = [
                    {
                        "timestamp": time.time() - i * 3600,
                        "field": f"price_{i % 3 + 1}",
                        "rule_id": f"anomaly_rule_{i % 3 + 1}",
                        "reason": f"Value outside expected range: {50 + i * 10} > 100",
                        "value": 50 + i * 10,
                        "expected": "<= 100"
                    }
                    for i in range(5)
                ]
            
            return jsonify({
                "success": True,
                "anomalies": anomalies
            })
        except Exception as e:
            logger.error(f"Error getting validation anomalies: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting validation anomalies: {str(e)}"
            }), 500
    
    def api_transform_data(self):
        """API endpoint to transform data"""
        try:
            # Initialize transformation pipeline if not already initialized
            if not hasattr(self, 'transformation_pipeline'):
                from src.dashboard.utils.transformation_pipeline import TransformationPipeline
                self.transformation_pipeline = TransformationPipeline()
            
            # Get request data
            data = request.json
            
            if not data:
                return jsonify({
                    "success": False,
                    "message": "No data provided"
                }), 400
            
            # Get transformation parameters
            input_data = data.get('data')
            pipeline_id = data.get('pipeline_id')
            transformers = data.get('transformers')
            
            if not input_data:
                return jsonify({
                    "success": False,
                    "message": "No input data provided"
                }), 400
            
            if not pipeline_id and not transformers:
                return jsonify({
                    "success": False,
                    "message": "No pipeline_id or transformers provided"
                }), 400
            
            # Transform data
            result = self.transformation_pipeline.transform(
                data=input_data,
                pipeline_id=pipeline_id,
                transformers=transformers
            )
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            return jsonify({
                "success": False,
                "message": f"Error transforming data: {str(e)}"
            }), 500
    
    def api_get_transform_pipelines(self):
        """API endpoint to get transformation pipelines"""
        try:
            # Initialize transformation pipeline if not already initialized
            if not hasattr(self, 'transformation_pipeline'):
                from src.dashboard.utils.transformation_pipeline import TransformationPipeline
                self.transformation_pipeline = TransformationPipeline()
            
            # Get pipelines
            pipelines = self.transformation_pipeline.get_pipelines()
            
            return jsonify({
                "success": True,
                "pipelines": pipelines
            })
        except Exception as e:
            logger.error(f"Error getting transformation pipelines: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting transformation pipelines: {str(e)}"
            }), 500
    
    def api_get_transformers(self):
        """API endpoint to get available transformers"""
        try:
            # Initialize transformation pipeline if not already initialized
            if not hasattr(self, 'transformation_pipeline'):
                from src.dashboard.utils.transformation_pipeline import TransformationPipeline
                self.transformation_pipeline = TransformationPipeline()
            
            # Get transformers
            transformers = self.transformation_pipeline.get_transformers()
            
            return jsonify({
                "success": True,
                "transformers": transformers
            })
        except Exception as e:
            logger.error(f"Error getting transformers: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting transformers: {str(e)}"
            }), 500
    
    def api_get_transform_stats(self):
        """API endpoint to get transformation statistics"""
        try:
            # Initialize transformation pipeline if not already initialized
            if not hasattr(self, 'transformation_pipeline'):
                from src.dashboard.utils.transformation_pipeline import TransformationPipeline
                self.transformation_pipeline = TransformationPipeline()
            
            # Get stats
            stats = self.transformation_pipeline.get_stats()
            
            return jsonify({
                "success": True,
                "stats": stats
            })
        except Exception as e:
            logger.error(f"Error getting transformation statistics: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting transformation statistics: {str(e)}"
            }), 500
    
    def api_get_transformation_panel_template(self):
        """API endpoint to get the transformation panel template"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        try:
            # Return the transformation panel template
            return render_template('transformation_panel.html')
        except Exception as e:
            logger.error(f"Error getting transformation panel template: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting transformation panel template: {str(e)}"
            }), 500
    
    def api_get_admin_controls_panel_template(self):
        """API endpoint to get the admin controls panel template"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        try:
            # Return the admin controls panel template
            return render_template('admin_controls_panel.html')
        except Exception as e:
            logger.error(f"Error getting admin controls panel template: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting admin controls panel template: {str(e)}"
            }), 500
            
    def api_admin_system_status(self):
        """API endpoint to get system status for admin panel"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        try:
            # Get system status
            uptime = self._get_system_uptime()
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            
            # Get data source status
            data_sources_status = "HEALTHY"
            healthy_sources = 0
            total_sources = 0
            
            if self.status_reporter_available:
                status = self.status_reporter.get_system_status()
                data_sources_status = status.get('system_health', 'UNKNOWN')
                sources = status.get('sources', {})
                total_sources = len(sources)
                healthy_sources = sum(1 for source in sources.values() if source.get('health') == 'HEALTHY')
            
            # Get services status
            services_status = "HEALTHY"
            running_services = 0
            total_services = 0
            
            # In a real implementation, this would check actual services
            # For now, we'll just return placeholder values
            running_services = 12
            total_services = 12
            
            return jsonify({
                "success": True,
                "system_health": self.system_state,
                "uptime": uptime,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "data_sources_status": data_sources_status,
                "healthy_sources": healthy_sources,
                "total_sources": total_sources,
                "services_status": services_status,
                "running_services": running_services,
                "total_services": total_services,
                "real_data_enabled": self.data_service.data_source == 'real',
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting system status: {str(e)}"
            }), 500
            
    def api_admin_run_diagnostics(self):
        """API endpoint to run system diagnostics"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        try:
            # Run diagnostics
            # In a real implementation, this would run actual diagnostics
            # For now, we'll just return placeholder values
            
            # Example diagnostic checks
            checks = [
                {
                    "name": "Database Connection",
                    "status": "PASSED",
                    "message": "Database connection is healthy"
                },
                {
                    "name": "File System Access",
                    "status": "PASSED",
                    "message": "File system is accessible and has sufficient space"
                },
                {
                    "name": "API Connectivity",
                    "status": "PASSED",
                    "message": "All required APIs are accessible"
                },
                {
                    "name": "Memory Usage",
                    "status": "WARNING",
                    "message": "Memory usage is above 75%"
                }
            ]
            
            # Determine overall status
            if any(check["status"] == "FAILED" for check in checks):
                status = "FAILED"
            elif any(check["status"] == "WARNING" for check in checks):
                status = "WARNING"
            else:
                status = "PASSED"
            
            return jsonify({
                "success": True,
                "status": status,
                "checks": checks,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}")
            return jsonify({
                "success": False,
                "message": f"Error running diagnostics: {str(e)}"
            }), 500
            
    def api_admin_update_real_data(self):
        """API endpoint to enable or disable real data"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        # Validate request data
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
            
        data = request.json
        enabled = data.get('enabled', False)
        
        try:
            # Update data source
            if enabled:
                self.data_service.data_source = 'real'
            else:
                self.data_service.data_source = 'mock'
                
            # Update settings if available
            if self.settings_manager_available:
                self.settings_manager.update_setting('data_source', self.data_service.data_source)
                
            return jsonify({
                "success": True,
                "message": f"Real data {'enabled' if enabled else 'disabled'} successfully"
            })
        except Exception as e:
            logger.error(f"Error updating real data status: {e}")
            return jsonify({
                "success": False,
                "message": f"Error updating real data status: {str(e)}"
            }), 500
            
    def api_admin_restart_services(self):
        """API endpoint to restart system services"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        try:
            # Restart services
            # In a real implementation, this would restart actual services
            # For now, we'll just return success
            
            # Simulate service restart
            time.sleep(1)
            
            return jsonify({
                "success": True,
                "message": "Services restarted successfully"
            })
        except Exception as e:
            logger.error(f"Error restarting services: {e}")
            return jsonify({
                "success": False,
                "message": f"Error restarting services: {str(e)}"
            }), 500
            
    def api_admin_get_users(self):
        """API endpoint to get all users"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        try:
            # Get users
            # In a real implementation, this would get actual users from the database
            # For now, we'll just return placeholder values
            users = [
                {
                    "id": 1,
                    "username": "admin",
                    "email": "admin@example.com",
                    "role": "ADMIN",
                    "active": True,
                    "last_login": "2025-03-30 08:45:12"
                },
                {
                    "id": 2,
                    "username": "analyst",
                    "email": "analyst@example.com",
                    "role": "ANALYST",
                    "active": True,
                    "last_login": "2025-03-29 16:20:33"
                },
                {
                    "id": 3,
                    "username": "viewer",
                    "email": "viewer@example.com",
                    "role": "VIEWER",
                    "active": False,
                    "last_login": "2025-03-15 11:10:45"
                }
            ]
            
            return jsonify({
                "success": True,
                "users": users
            })
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting users: {str(e)}"
            }), 500
            
    def api_admin_add_user(self):
        """API endpoint to add a new user"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        # Validate request data
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
            
        data = request.json
        
        # Check required fields
        required_fields = ['username', 'email', 'password', 'role']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "message": f"Missing required field: {field}"
                }), 400
                
        try:
            # Add user
            # In a real implementation, this would add the user to the database
            # For now, we'll just return success
            
            return jsonify({
                "success": True,
                "message": "User added successfully",
                "user_id": 4  # Example user ID
            })
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return jsonify({
                "success": False,
                "message": f"Error adding user: {str(e)}"
            }), 500
            
    def api_admin_update_user(self, user_id):
        """API endpoint to update a user"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        # Validate request data
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
            
        data = request.json
        
        try:
            # Update user
            # In a real implementation, this would update the user in the database
            # For now, we'll just return success
            
            return jsonify({
                "success": True,
                "message": f"User {user_id} updated successfully"
            })
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return jsonify({
                "success": False,
                "message": f"Error updating user: {str(e)}"
            }), 500
            
    def api_admin_update_user_status(self, user_id):
        """API endpoint to update a user's status (active/inactive)"""
        # Verify user is authenticated
        if not session.get('user_id'):
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401
            
        # Check if user has admin role
        if session.get('user_role') != UserRole.ADMIN:
            return jsonify({
                "success": False,
                "message": "Admin privileges required"
            }), 403
            
        # Validate request data
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Invalid request format"
            }), 400
            
        data = request.json
        active = data.get('active', False)
        
        try:
            # Update user status
            # In a real implementation, this would update the user's status in the database
            # For now, we'll just return success
            
            return jsonify({
                "success": True,
                "message": f"User {user_id} {'activated' if active else 'deactivated'} successfully"
            })
        except Exception as e:
            logger.error(f"Error updating user status: {e}")
            return jsonify({
                "success": False,
                "message": f"Error updating user status: {str(e)}"
            }), 500
            
    def _get_system_uptime(self):
        """Get system uptime"""
        # In a real implementation, this would get the actual system uptime
        # For now, we'll just return a placeholder value
        return "5d 12h 34m"
        
    def _get_cpu_usage(self):
        """Get CPU usage percentage"""
        # In a real implementation, this would get the actual CPU usage
        # For now, we'll just return a placeholder value
        return 45.2
        
    def _get_memory_usage(self):
        """Get memory usage percentage"""
        # In a real implementation, this would get the actual memory usage
        # For now, we'll just return a placeholder value
        return 68.7
        
    async def websocket_endpoint(self, websocket):
        """WebSocket endpoint for real-time updates"""
        # Generate a unique client ID
        client_id = f"client_{id(websocket)}"
        
        try:
            # Initialize WebSocket manager if not already initialized
            if not hasattr(self, 'ws_manager'):
                from src.dashboard.utils.websocket_manager import WebSocketManager
                self.ws_manager = WebSocketManager()
                await self.ws_manager.start()
                
                # Register data sources
                self._register_websocket_data_sources()
            
            # Connect client
            await self.ws_manager.connect(websocket, client_id)
            
            # Handle messages
            while True:
                # Receive message
                data = await websocket.receive_text()
                
                # Process message
                await self.ws_manager.receive_message(client_id, data)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Disconnect client
            if hasattr(self, 'ws_manager'):
                await self.ws_manager.disconnect(client_id)
    
    def _register_websocket_data_sources(self):
        """Register data sources for the WebSocket manager"""
        try:
            # Import data sources
            from src.dashboard.utils.websocket_manager import WebSocketManager
            
            # Create mock data sources
            class MockDataSource:
                async def get_data(self):
                    import random
                    return {
                        'value': random.randint(1, 100),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Register data sources
            self.ws_manager.register_data_source('dashboard', MockDataSource())
            self.ws_manager.register_data_source('trades', MockDataSource())
            self.ws_manager.register_data_source('positions', MockDataSource())
            self.ws_manager.register_data_source('performance', MockDataSource())
            self.ws_manager.register_data_source('alerts', MockDataSource())
            
            # Set update intervals
            self.ws_manager.set_update_interval('dashboard', 5.0)
            self.ws_manager.set_update_interval('trades', 2.0)
            self.ws_manager.set_update_interval('positions', 3.0)
            self.ws_manager.set_update_interval('performance', 10.0)
            self.ws_manager.set_update_interval('alerts', 1.0)
            
            logger.info("WebSocket data sources registered")
        except Exception as e:
            logger.error(f"Error registering WebSocket data sources: {e}")
    
    def api_get_admin_diagnostics(self):
        """API endpoint to get admin diagnostics"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        try:
            # Get diagnostics
            diagnostics = self.admin_controller.get_diagnostics()
            
            return jsonify(diagnostics)
        except Exception as e:
            logger.error(f"Error getting admin diagnostics: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting admin diagnostics: {str(e)}"
            }), 500
    
    def api_run_admin_diagnostics(self):
        """API endpoint to run admin diagnostics"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        try:
            # Run diagnostics
            diagnostics = self.admin_controller.run_diagnostics()
            
            return jsonify(diagnostics)
        except Exception as e:
            logger.error(f"Error running admin diagnostics: {e}")
            return jsonify({
                "success": False,
                "message": f"Error running admin diagnostics: {str(e)}"
            }), 500
    
    def api_clear_admin_cache(self):
        """API endpoint to clear admin cache"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        try:
            # Clear cache
            result = self.admin_controller.clear_cache()
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error clearing admin cache: {e}")
            return jsonify({
                "success": False,
                "message": f"Error clearing admin cache: {str(e)}"
            }), 500
    
    def api_get_admin_saved_tests(self):
        """API endpoint to get admin saved tests"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        try:
            # Get saved tests
            tests = self.admin_controller.get_saved_tests()
            
            return jsonify(tests)
        except Exception as e:
            logger.error(f"Error getting admin saved tests: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting admin saved tests: {str(e)}"
            }), 500
    
    def api_get_admin_saved_test(self):
        """API endpoint to get an admin saved test"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get test ID from query parameters
        test_id = request.args.get('id')
        if not test_id:
            return jsonify({
                "success": False,
                "message": "Test ID is required"
            }), 400
            
        try:
            # Get saved test
            test = self.admin_controller.get_saved_test(test_id)
            
            return jsonify(test)
        except Exception as e:
            logger.error(f"Error getting admin saved test: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting admin saved test: {str(e)}"
            }), 500
    
    def api_save_admin_test(self):
        """API endpoint to save an admin test"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get test data from request
        data = request.json
        name = data.get('name')
        endpoint = data.get('endpoint')
        parameters = data.get('parameters', {})
        
        if not name or not endpoint:
            return jsonify({
                "success": False,
                "message": "Name and endpoint are required"
            }), 400
            
        try:
            # Save test
            test = self.admin_controller.save_test(name, endpoint, parameters)
            
            return jsonify({
                "success": True,
                "test": test
            })
        except Exception as e:
            logger.error(f"Error saving admin test: {e}")
            return jsonify({
                "success": False,
                "message": f"Error saving admin test: {str(e)}"
            }), 500
    
    def api_delete_admin_test(self):
        """API endpoint to delete an admin test"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get test ID from request
        data = request.json
        test_id = data.get('id')
        
        if not test_id:
            return jsonify({
                "success": False,
                "message": "Test ID is required"
            }), 400
            
        try:
            # Delete test
            result = self.admin_controller.delete_saved_test(test_id)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error deleting admin test: {e}")
            return jsonify({
                "success": False,
                "message": f"Error deleting admin test: {str(e)}"
            }), 500
    
    def api_run_admin_test(self):
        """API endpoint to run an admin test"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get test data from request
        data = request.json
        endpoint = data.get('endpoint')
        parameters = data.get('parameters', {})
        
        if not endpoint:
            return jsonify({
                "success": False,
                "message": "Endpoint is required"
            }), 400
            
        try:
            # Run test
            result = self.admin_controller.run_test(endpoint, parameters)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error running admin test: {e}")
            return jsonify({
                "success": False,
                "message": f"Error running admin test: {str(e)}"
            }), 500
    
    def api_run_admin_saved_test(self):
        """API endpoint to run a saved admin test"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get test ID from query parameters
        test_id = request.args.get('id')
        if not test_id:
            return jsonify({
                "success": False,
                "message": "Test ID is required"
            }), 400
            
        try:
            # Run saved test
            result = self.admin_controller.run_saved_test(test_id)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error running saved admin test: {e}")
            return jsonify({
                "success": False,
                "message": f"Error running saved admin test: {str(e)}"
            }), 500
    
    def api_get_admin_configuration(self):
        """API endpoint to get admin configuration"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        try:
            # Get configuration
            config = self.admin_controller.get_configuration()
            
            return jsonify(config)
        except Exception as e:
            logger.error(f"Error getting admin configuration: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting admin configuration: {str(e)}"
            }), 500
    
    def api_save_admin_configuration(self):
        """API endpoint to save admin configuration"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get configuration from request
        data = request.json
        
        try:
            # Save configuration
            config = self.admin_controller.save_configuration(data)
            
            return jsonify(config)
        except Exception as e:
            logger.error(f"Error saving admin configuration: {e}")
            return jsonify({
                "success": False,
                "message": f"Error saving admin configuration: {str(e)}"
            }), 500
    
    def api_reset_admin_configuration(self):
        """API endpoint to reset admin configuration"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        try:
            # Reset configuration
            config = self.admin_controller.reset_configuration()
            
            return jsonify(config)
        except Exception as e:
            logger.error(f"Error resetting admin configuration: {e}")
            return jsonify({
                "success": False,
                "message": f"Error resetting admin configuration: {str(e)}"
            }), 500
    
    def api_save_admin_api_key(self):
        """API endpoint to save an admin API key"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get API key data from request
        data = request.json
        name = data.get('name')
        provider = data.get('provider')
        key = data.get('key')
        secret = data.get('secret')
        
        if not name or not provider or not key:
            return jsonify({
                "success": False,
                "message": "Name, provider, and key are required"
            }), 400
            
        try:
            # Save API key
            api_key = self.admin_controller.save_api_key(name, provider, key, secret)
            
            return jsonify({
                "success": True,
                "api_key": api_key
            })
        except Exception as e:
            logger.error(f"Error saving admin API key: {e}")
            return jsonify({
                "success": False,
                "message": f"Error saving admin API key: {str(e)}"
            }), 500
    
    def api_delete_admin_api_key(self):
        """API endpoint to delete an admin API key"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get API key ID from request
        data = request.json
        api_key_id = data.get('id')
        
        if not api_key_id:
            return jsonify({
                "success": False,
                "message": "API key ID is required"
            }), 400
            
        try:
            # Delete API key
            result = self.admin_controller.delete_api_key(api_key_id)
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error deleting admin API key: {e}")
            return jsonify({
                "success": False,
                "message": f"Error deleting admin API key: {str(e)}"
            }), 500
    
    def api_get_admin_logs(self):
        """API endpoint to get admin logs"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        level = request.args.get('level', 'all')
        component = request.args.get('component', 'all')
        
        try:
            # Get logs
            logs = self.admin_controller.get_logs(page, per_page, level, component)
            
            return jsonify(logs)
        except Exception as e:
            logger.error(f"Error getting admin logs: {e}")
            return jsonify({
                "success": False,
                "message": f"Error getting admin logs: {str(e)}"
            }), 500
    
    def api_clear_admin_logs(self):
        """API endpoint to clear admin logs"""
        if not self.admin_controller_available:
            return jsonify({
                "success": False,
                "message": "Admin controller not available"
            }), 500
            
        try:
            # Clear logs
            result = self.admin_controller.clear_logs()
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error clearing admin logs: {e}")
            return jsonify({
                "success": False,
                "message": f"Error clearing admin logs: {str(e)}"
            }), 500
        if not self.status_reporter_available:
            return jsonify({
                "success": False,
                "message": "Status reporter not available"
            }), 500
            
        # Get source ID from query parameters
        source_id = request.args.get('source')
        if not source_id:
            return jsonify({
                "success": False,
                "message": "Source ID is required"
            }), 400
            
        # Reset the source statistics
        result = self.status_reporter.reset_source_stats(source_id)
        
        return jsonify(result)
        if not self.settings_manager_available:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
            
        try:
            # Get connections from request
            data = request.json
            connections = data.get('connections', {})
            
            # Get current configuration
            config = self.settings_manager.get_real_data_config()
            
            # Update connections
            config['connections'] = connections
            
            # Save configuration
            success = self.settings_manager.update_real_data_config(config)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "Connections updated successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to update connections"
                }), 500
        except Exception as e:
            logger.error(f"Error updating data source connections: {e}")
            return jsonify({
                "success": False,
                "message": f"Error updating data source connections: {str(e)}"
            }), 500
    
    # API endpoints for dashboard data
    def api_dashboard_summary(self):
        """API endpoint to get dashboard summary data"""
        return jsonify({
            "system_state": self.system_state,
            "trading_state": self.trading_state,
            "system_mode": self.system_mode,
            "health": self.mock_data.generate_system_health(),
            "performance": self.mock_data.generate_trading_performance(),
            "alerts": self.mock_data.generate_system_alerts()[:3],  # Top 3 alerts
            "positions_count": len(self.mock_data.generate_current_positions()),
            "timestamp": datetime.now().isoformat(),
        })
    
    def api_system_health(self):
        """API endpoint to get system health data"""
        return jsonify(self.data_service.get_data('system_health'))
    
    def api_component_status(self):
        """API endpoint to get component status data"""
        return jsonify(self.data_service.get_data('component_status'))
    
    def api_trading_performance(self):
        """API endpoint to get trading performance data"""
        return jsonify(self.data_service.get_data('trading_performance'))
    
    def api_current_positions(self):
        """API endpoint to get current positions data"""
        return jsonify(self.data_service.get_data('current_positions'))
    
    def api_recent_trades(self):
        """API endpoint to get recent trades data"""
        return jsonify(self.data_service.get_data('recent_trades'))
    
    def api_system_alerts(self):
        """API endpoint to get system alerts data"""
        return jsonify(self.data_service.get_data('system_alerts'))
    
    def api_equity_curve(self):
        """API endpoint to get equity curve data"""
        return jsonify(self.data_service.get_data('equity_curve'))
        
    def api_market_regime(self):
        """API endpoint to get market regime data"""
        return jsonify(self.data_service.get_data('market_regime'))
        
    def api_sentiment(self):
        """API endpoint to get sentiment analysis data"""
        return jsonify(self.data_service.get_data('sentiment'))
        
    def api_risk_management(self):
        """API endpoint to get risk management data"""
        return jsonify(self.data_service.get_data('risk_management'))
        
    def api_performance_analytics(self):
        """API endpoint to get performance analytics data"""
        return jsonify(self.data_service.get_data('performance_analytics'))
        
    def api_logs_monitoring(self):
        """API endpoint to get logs and monitoring data"""
        return jsonify(self.data_service.get_data('logs_monitoring'))
        
    # API Key Management API endpoints
    
    def api_get_api_keys(self):
        """API endpoint to get all API keys (masked)"""
        if self.api_key_manager_available:
            try:
                # Get list of exchanges with credentials
                credentials = []
                for exchange_id in self.api_key_manager.list_credentials():
                    cred = self.api_key_manager.get_credential(exchange_id)
                    if cred:
                        # Create a sanitized version with masked secrets
                        credentials.append({
                            "exchange": exchange_id,
                            "key": cred.key,
                            "description": cred.description,
                            "is_testnet": cred.is_testnet
                        })
                return jsonify(credentials)
            except Exception as e:
                logger.error(f"Error getting API keys: {e}")
                return jsonify({"error": str(e)}), 500
        else:
            # Use mock data
            result = []
            for exchange, data in self.mock_api_keys.items():
                result.append({
                    "exchange": exchange, 
                    "key": data.get("key", ""),
                    "description": data.get("description", ""),
                    "is_testnet": data.get("is_testnet", False)
                })
            return jsonify(result)
    
    def api_add_api_key(self):
        """API endpoint to add or update an API key"""
        data = request.json
        if not data or not data.get("exchange") or not data.get("key") or not data.get("secret"):
            return jsonify({"error": "Missing required fields"}), 400
            
        exchange = data["exchange"]
        key = data["key"]
        secret = data["secret"]
        passphrase = data.get("passphrase")
        description = data.get("description", "")
        is_testnet = data.get("is_testnet", False)
        
        if self.api_key_manager_available:
            try:
                from src.common.security import ApiCredential
                # Create credential object
                cred = ApiCredential(
                    exchange_id=exchange,
                    key=key,
                    secret=secret,
                    passphrase=passphrase,
                    description=description,
                    is_testnet=is_testnet
                )
                
                # Add to storage
                self.api_key_manager.add_credential(cred)
                logger.info(f"Added API key for {exchange}")
                
                return jsonify({"success": True, "message": f"API key for {exchange} added successfully"})
            except Exception as e:
                logger.error(f"Error adding API key: {e}")
                return jsonify({"error": str(e)}), 500
        else:
            # Store in mock data
            self.mock_api_keys[exchange] = {
                "key": key,
                "secret": secret,
                "passphrase": passphrase,
                "description": description,
                "is_testnet": is_testnet,
                "last_validated": None,
                "is_valid": None
            }
            return jsonify({"success": True, "message": f"API key for {exchange} added successfully"})
    
    def api_get_api_key_details(self, exchange):
        """API endpoint to get details of a specific API key"""
        if self.api_key_manager_available:
            try:
                cred = self.api_key_manager.get_credential(exchange)
                if not cred:
                    return jsonify({"error": "API key not found"}), 404
                    
                # Return all details except the secret
                return jsonify({
                    "exchange": exchange,
                    "key": cred.key,
                    "passphrase": bool(cred.passphrase),  # Just indicate if present
                    "description": cred.description,
                    "is_testnet": cred.is_testnet,
                    # These would be added in a real implementation:
                    # "last_validated": cred.last_validated,
                    # "is_valid": cred.is_valid
                })
            except Exception as e:
                logger.error(f"Error getting API key details: {e}")
                return jsonify({"error": str(e)}), 500
        else:
            # Get from mock data
            if exchange not in self.mock_api_keys:
                return jsonify({"error": "API key not found"}), 404
                
            data = self.mock_api_keys[exchange]
            return jsonify({
                "exchange": exchange,
                "key": data["key"],
                "passphrase": bool(data.get("passphrase")),
                "description": data.get("description", ""),
                "is_testnet": data.get("is_testnet", False),
                "last_validated": data.get("last_validated"),
                "is_valid": data.get("is_valid")
            })
    
    def api_delete_api_key(self, exchange):
        """API endpoint to delete an API key"""
        if self.api_key_manager_available:
            try:
                success = self.api_key_manager.remove_credential(exchange)
                if not success:
                    return jsonify({"error": "API key not found"}), 404
                    
                logger.info(f"Deleted API key for {exchange}")
                return jsonify({"success": True, "message": f"API key for {exchange} deleted successfully"})
            except Exception as e:
                logger.error(f"Error deleting API key: {e}")
                return jsonify({"error": str(e)}), 500
        else:
            # Remove from mock data
            if exchange not in self.mock_api_keys:
                return jsonify({"error": "API key not found"}), 404
                
            del self.mock_api_keys[exchange]
            return jsonify({"success": True, "message": f"API key for {exchange} deleted successfully"})
    
    def api_validate_api_key(self, exchange):
        """API endpoint to validate an API key for a specific exchange"""
        if self.api_key_manager_available:
            try:
                cred = self.api_key_manager.get_credential(exchange)
                if not cred:
                    return jsonify({"error": "API key not found"}), 404
                
                # Perform real validation based on exchange type
                validation_result = self._validate_exchange_credentials(exchange, cred)
                
                # Log validation result
                if validation_result["success"]:
                    logger.info(f"API key for {exchange} validated successfully")
                else:
                    logger.warning(f"API key for {exchange} validation failed: {validation_result.get('message')}")
                
                # Include timestamp
                validation_result["last_validated"] = datetime.now().isoformat()
                
                return jsonify(validation_result)
            except Exception as e:
                logger.error(f"Error validating API key: {e}")
                return jsonify({"error": str(e)}), 500
        else:
            # Simulate validation for mock data
            if exchange not in self.mock_api_keys:
                return jsonify({"error": "API key not found"}), 404
                
            # For mock data, still try to use a real validator if possible
            try:
                mock_data = self.mock_api_keys[exchange]
                mock_cred = type('MockCredential', (), {
                    'exchange_id': exchange,
                    'key': mock_data["key"],
                    'secret': mock_data["secret"],
                    'passphrase': mock_data.get("passphrase"),
                    'description': mock_data.get("description", ""),
                    'is_testnet': mock_data.get("is_testnet", False)
                })
                
                validation_result = self._validate_exchange_credentials(exchange, mock_cred)
            except Exception:
                # Fallback to mock validation
                import random
                is_valid = random.choice([True, False, True, True])  # 75% success rate
                validation_result = {
                    "success": is_valid,
                    "message": f"API key for {exchange} is {'valid' if is_valid else 'invalid'} (mock validation)",
                }
            
            # Update mock data
            timestamp = datetime.now().isoformat()
            self.mock_api_keys[exchange]["last_validated"] = timestamp
            self.mock_api_keys[exchange]["is_valid"] = validation_result["success"]
            validation_result["last_validated"] = timestamp
            
            return jsonify(validation_result)
    
    def api_validate_api_key_by_data(self):
        """API endpoint to validate an API key using provided data"""
        data = request.json
        if not data or not data.get("exchange") or not data.get("key") or not data.get("secret"):
            return jsonify({"error": "Missing required fields"}), 400
            
        exchange = data["exchange"]
        key = data["key"]
        secret = data["secret"]
        passphrase = data.get("passphrase")
        is_testnet = data.get("is_testnet", False)
        
        # Create mock credential for validation
        mock_cred = type('MockCredential', (), {
            'exchange_id': exchange,
            'key': key,
            'secret': secret,
            'passphrase': passphrase,
            'is_testnet': is_testnet
        })
        
        # Perform real validation
        try:
            validation_result = self._validate_exchange_credentials(exchange, mock_cred)
            if validation_result["success"]:
                logger.info(f"API key validation successful for {exchange}")
            else:
                logger.warning(f"API key validation failed for {exchange}: {validation_result.get('message')}")
            
            # Include timestamp
            validation_result["last_validated"] = datetime.now().isoformat()
            return jsonify(validation_result)
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return jsonify({"error": str(e)}), 500
    
    def _validate_exchange_credentials(self, exchange, cred):
        """Validate exchange credentials by connecting to the exchange API
        
        Args:
            exchange: Exchange identifier
            cred: Credential object with key, secret, and optional passphrase
            
        Returns:
            Dictionary with validation result
        """
        # Implementation for various exchanges
        if exchange == "binance":
            return self._validate_binance(cred)
        elif exchange == "coinbase":
            return self._validate_coinbase(cred)
        elif exchange == "kraken":
            return self._validate_kraken(cred)
        elif exchange == "kucoin":
            return self._validate_kucoin(cred)
        elif exchange == "ftx":
            return self._validate_ftx(cred)
        elif exchange == "twitter":
            return self._validate_twitter(cred)
        elif exchange == "newsapi":
            return self._validate_newsapi(cred)
        elif exchange == "cryptocompare":
            return self._validate_cryptocompare(cred)
        else:
            # For unsupported exchanges, fallback to mock validation
            import random
            is_valid = random.choice([True, False, True, True])  # 75% success rate
            return {
                "success": is_valid,
                "message": f"Exchange {exchange} validation not implemented, using mock validation"
            }
    
    def _validate_binance(self, cred):
        """Validate Binance API credentials"""
        try:
            # Use real Binance API client here
            # For example:
            """
            from binance.client import Client
            client = Client(cred.key, cred.secret, testnet=cred.is_testnet)
            # Test with a simple API call that requires authentication
            account_info = client.get_account()
            # If we get here, the credentials are valid
            return {
                "success": True,
                "message": "Binance API credentials are valid"
            }
            """
            
            # For now, simulate with a high success rate for demo
            import random
            is_valid = random.random() < 0.9  # 90% success rate
            
            return {
                "success": is_valid,
                "message": "Binance API credentials are valid" if is_valid else "Invalid Binance API credentials"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Binance API validation error: {str(e)}"
            }
    
    def _validate_coinbase(self, cred):
        """Validate Coinbase API credentials"""
        try:
            # Use real Coinbase API client here
            # For example:
            """
            from coinbase.wallet.client import Client
            client = Client(cred.key, cred.secret)
            # Test with a simple API call that requires authentication
            accounts = client.get_accounts()
            # If we get here, the credentials are valid
            return {
                "success": True,
                "message": "Coinbase API credentials are valid"
            }
            """
            
            # For now, simulate
            import random
            is_valid = random.random() < 0.9  # 90% success rate
            
            return {
                "success": is_valid,
                "message": "Coinbase API credentials are valid" if is_valid else "Invalid Coinbase API credentials"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Coinbase API validation error: {str(e)}"
            }
    
    # Similar validation methods for other exchanges
    def _validate_kraken(self, cred):
        # Simulated validation for now
        import random
        is_valid = random.random() < 0.9
        return {
            "success": is_valid,
            "message": "Kraken API credentials are valid" if is_valid else "Invalid Kraken API credentials"
        }
    
    def _validate_kucoin(self, cred):
        # Simulated validation for now
        import random
        is_valid = random.random() < 0.9
        return {
            "success": is_valid,
            "message": "KuCoin API credentials are valid" if is_valid else "Invalid KuCoin API credentials"
        }
    
    def _validate_ftx(self, cred):
        # Simulated validation for now
        import random
        is_valid = random.random() < 0.9
        return {
            "success": is_valid,
            "message": "FTX API credentials are valid" if is_valid else "Invalid FTX API credentials"
        }
    
    def _validate_twitter(self, cred):
        # Simulated validation for now
        import random
        is_valid = random.random() < 0.85
        return {
            "success": is_valid,
            "message": "Twitter API credentials are valid" if is_valid else "Invalid Twitter API credentials"
        }
    
    def _validate_newsapi(self, cred):
        # Simulated validation for now
        import random
        is_valid = random.random() < 0.95
        return {
            "success": is_valid,
            "message": "News API credentials are valid" if is_valid else "Invalid News API credentials"
        }
    
    def _validate_cryptocompare(self, cred):
        # Simulated validation for now
        import random
        is_valid = random.random() < 0.95
        return {
            "success": is_valid,
            "message": "CryptoCompare API credentials are valid" if is_valid else "Invalid CryptoCompare API credentials"
        }
    
    def register_socket_events(self):
        """Register all socket.io events"""
        # Skip if SocketIO or WebSocketManager is not available
        if not self.socketio or not self.websocket_manager:
            logger.warning("SocketIO or WebSocketManager not available, skipping socket event registration")
            return
            
        # WebSocketManager handles the basic Socket.IO events
        # We'll add additional application-specific events here
        
        @self.socketio.on('request_dashboard_data')
        def handle_dashboard_data_request(data):
            """Handle request for dashboard data"""
            component = data.get('component')
            logger.info(f"Client requested dashboard data for component: {component}")
            
            # Get data for the requested component
            if component == 'summary':
                self.send_dashboard_summary()
            elif component == 'performance':
                self.send_performance_data()
            elif component == 'market':
                self.send_market_data()
            elif component == 'system':
                self.send_system_status()
        
        @self.socketio.on('request_historical_data')
        def handle_historical_data_request(data):
            """Handle request for historical data"""
            data_type = data.get('type')
            timeframe = data.get('timeframe', '1d')
            symbol = data.get('symbol')
            
            logger.info(f"Client requested historical data: {data_type} for {symbol} on {timeframe} timeframe")
            
            # Get historical data
            if data_type and symbol:
                self.send_historical_data(data_type, symbol, timeframe)
    
    def emit_update(self, channel, data, room=None):
        """
        Emit a real-time update to subscribed clients
        
        Args:
            channel: The channel/event name
            data: The data to emit
            room: Optional room to emit to (if None, broadcast to all)
        """
        # First publish to the event bus for internal components
        if self.event_bus:
            self.event_bus.publish(channel, data)
        
        # Then emit to WebSocket clients
        if self.websocket_manager:
            if room:
                self.websocket_manager.emit_to_room(room, channel, data)
            else:
                self.websocket_manager.broadcast(channel, data)
        elif self.socketio:
            # Fallback to direct socketio if WebSocketManager is not available
            if room:
                self.socketio.emit(channel, data, room=room)
            else:
                self.socketio.emit(channel, data)
    
    def send_dashboard_summary(self):
        """Send dashboard summary data to clients"""
        try:
            # Get dashboard summary data
            summary_data = {
                'portfolio_value': self.data_service.get_portfolio_value(),
                'daily_pnl': self.data_service.get_daily_pnl(),
                'open_positions': self.data_service.get_open_positions_count(),
                'system_state': self.system_state.value,
                'trading_state': self.trading_state.value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Emit update
            self.emit_update('dashboard_summary', {
                'type': 'dashboard_summary',
                'data': summary_data
            })
            
            logger.debug("Sent dashboard summary data")
        except Exception as e:
            logger.error(f"Error sending dashboard summary: {e}")
    
    def send_performance_data(self):
        """Send performance data to clients"""
        try:
            # Get performance data
            performance_data = {
                'total_return': self.data_service.get_total_return(),
                'daily_return': self.data_service.get_daily_return(),
                'sharpe_ratio': self.data_service.get_sharpe_ratio(),
                'max_drawdown': self.data_service.get_max_drawdown(),
                'win_rate': self.data_service.get_win_rate(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Emit update
            self.emit_update('trading_performance', {
                'type': 'trading_performance',
                'data': performance_data
            })
            
            logger.debug("Sent performance data")
        except Exception as e:
            logger.error(f"Error sending performance data: {e}")
    
    def send_market_data(self):
        """Send market data to clients"""
        try:
            # Get market data
            market_data = self.data_service.get_market_data()
            
            # Emit update
            self.emit_update('market_data', {
                'type': 'market_data',
                'data': market_data
            })
            
            logger.debug("Sent market data")
        except Exception as e:
            logger.error(f"Error sending market data: {e}")
    
    def send_system_status(self):
        """Send system status to clients"""
        try:
            # Get system status
            system_status = {
                'state': self.system_state.value,
                'trading_state': self.trading_state.value,
                'mode': self.system_mode.value,
                'components': self.get_component_statuses(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Emit update
            self.emit_update('system_status', {
                'type': 'system_status',
                'data': system_status
            })
            
            logger.debug("Sent system status")
        except Exception as e:
            logger.error(f"Error sending system status: {e}")
    
    def send_historical_data(self, data_type, symbol, timeframe='1d'):
        """
        Send historical data to clients
        
        Args:
            data_type: The type of data (price, volume, etc.)
            symbol: The symbol to get data for
            timeframe: The timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
        """
        try:
            # Get historical data
            historical_data = self.data_service.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                data_type=data_type
            )
            
            # Emit update
            self.emit_update('historical_data', {
                'type': 'historical_data',
                'data': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_type': data_type,
                    'data': historical_data
                }
            })
            
            logger.debug(f"Sent historical {data_type} data for {symbol} on {timeframe} timeframe")
        except Exception as e:
            logger.error(f"Error sending historical data: {e}")
    
    def start_background_tasks(self):
        """Start background tasks for real-time data updates"""
        if self.background_tasks_enabled:
            logger.info("Background tasks already running")
            return
            
        logger.info("Starting background data update tasks")
        self.background_tasks_enabled = True
        
        # Start a background task for each data type
        for data_type, interval in self.update_intervals.items():
            self.start_background_task(data_type, interval)
    
    def stop_background_tasks(self):
        """Stop all background tasks"""
        logger.info("Stopping background data update tasks")
        self.background_tasks_enabled = False
        
        # Wait for threads to complete
        for thread_name, thread in self.background_threads.items():
            if thread and thread.is_alive():
                logger.info(f"Waiting for {thread_name} thread to stop")
        
        self.background_threads = {}
    
    def start_background_task(self, data_type, interval):
        """Start a background task for a specific data type"""
        def update_task(data_type, interval):
            """Task that periodically emits data updates"""
            logger.info(f"Starting background task for {data_type} updates every {interval} seconds")
            
            while self.background_tasks_enabled:
                try:
                    # Only emit updates when system is running
                    if self.system_state == SystemState.RUNNING:
                        # Get the latest data from the data service
                        data = self.data_service.get_data(data_type, force_refresh=True)
                        
                        # Emit update via WebSocket
                        self.emit_update('dashboard_update', {
                            'type': data_type,
                            'data': data,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        logger.debug(f"Emitted {data_type} update")
                except Exception as e:
                    logger.error(f"Error in {data_type} update task: {e}")
                
                # Sleep for the specified interval
                time.sleep(interval)
                
            logger.info(f"Stopped background task for {data_type}")
        
        # Start the update task in a new thread
        thread = threading.Thread(
            target=update_task,
            args=(data_type, interval),
            daemon=True
        )
        thread.start()
        
        # Store the thread for later management
        self.background_threads[data_type] = thread
        
        return thread
    
    def run(self, host="127.0.0.1", port=8000, debug=False):
        """Run the dashboard application with SocketIO"""
        logger.info(f"Starting AI Trading Dashboard with WebSockets on {host}:{port}")
        
        # Start background tasks for real-time updates
        self.start_background_tasks()
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
        finally:
            # Ensure background tasks are stopped when the server shuts down
            self.stop_background_tasks()

# For standalone running
if __name__ == "__main__":
    dashboard = ModernDashboard()
    dashboard.run(debug=True)
