"""
Mock Data Generator Module

This module provides realistic mock data for the AI Trading Agent dashboard.
It generates market patterns with trends, seasonality, and noise that mimic
real market behavior across different market regimes.

Following modular design principles, this component is separated from the
main dashboard to maintain single responsibility and keep files under size limits.
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

class MockDataGenerator:
    """
    Generates realistic mock data for dashboard components.
    
    Features:
    - Realistic market patterns (trends, seasonality, noise)
    - Different market regimes (bull, bear, sideways, volatile, recovery, crash)
    - Correlated performance metrics based on market conditions
    """
    
    def __init__(self):
        """Initialize the mock data generator with common parameters"""
        # Market regime parameters
        self.current_regime = self._select_random_regime()
        self.regime_start_time = datetime.now() - timedelta(days=random.randint(1, 30))
        self.regime_duration = timedelta(days=random.randint(5, 45))
        
        # Performance tracking
        self.starting_equity = 100000.0
        self.current_equity = self.starting_equity
        self.daily_returns = []
        self.current_positions = []
        self.recent_trades = []
        
        # Cache for complex data
        self.cached_data = {}
        self.cache_expiry = {}
    
    def _select_random_regime(self) -> str:
        """Select a random market regime with realistic probabilities"""
        regimes = {
            "bull": 0.35,      # 35% chance of bull market
            "bear": 0.25,      # 25% chance of bear market
            "sideways": 0.25,  # 25% chance of sideways market
            "volatile": 0.1,   # 10% chance of volatile market
            "recovery": 0.03,  # 3% chance of recovery
            "crash": 0.02      # 2% chance of crash
        }
        
        rand_val = random.random()
        cumulative = 0
        for regime, probability in regimes.items():
            cumulative += probability
            if rand_val <= cumulative:
                return regime
        
        return "sideways"  # Default fallback
    
    def _get_regime_characteristics(self) -> Dict[str, Any]:
        """Get the characteristics of the current market regime"""
        characteristics = {
            "bull": {
                "trend": 0.05,           # 5% upward trend
                "volatility": 0.01,      # Low volatility
                "sharpe_ratio": 2.5,     # High Sharpe ratio
                "win_rate": 0.65,        # High win rate
                "drawdown": 0.03,        # Low drawdown
                "color": "#28a745"       # Green
            },
            "bear": {
                "trend": -0.04,          # 4% downward trend
                "volatility": 0.02,      # Medium volatility
                "sharpe_ratio": -1.5,    # Negative Sharpe ratio
                "win_rate": 0.35,        # Low win rate
                "drawdown": 0.15,        # High drawdown
                "color": "#dc3545"       # Red
            },
            "sideways": {
                "trend": 0.005,          # Very slight upward bias
                "volatility": 0.008,     # Low volatility
                "sharpe_ratio": 0.2,     # Near-zero Sharpe ratio
                "win_rate": 0.52,        # Slightly better than random
                "drawdown": 0.05,        # Medium drawdown
                "color": "#6c757d"       # Gray
            },
            "volatile": {
                "trend": 0.01,           # Slight upward bias
                "volatility": 0.035,     # High volatility
                "sharpe_ratio": 0.8,     # Modest Sharpe ratio
                "win_rate": 0.55,        # Modest win rate
                "drawdown": 0.12,        # High drawdown
                "color": "#ffc107"       # Yellow
            },
            "recovery": {
                "trend": 0.07,           # Strong upward trend
                "volatility": 0.025,     # Medium-high volatility
                "sharpe_ratio": 2.2,     # High Sharpe ratio but with more risk
                "win_rate": 0.7,         # Very high win rate
                "drawdown": 0.08,        # Medium drawdown
                "color": "#17a2b8"       # Blue
            },
            "crash": {
                "trend": -0.15,          # Severe downward trend
                "volatility": 0.06,      # Extreme volatility
                "sharpe_ratio": -2.8,    # Very negative Sharpe ratio
                "win_rate": 0.25,        # Very low win rate
                "drawdown": 0.3,         # Extreme drawdown
                "color": "#6f42c1"       # Purple
            }
        }
        
        return characteristics.get(self.current_regime, characteristics["sideways"])
    
    def generate_system_health(self) -> Dict[str, Any]:
        """Generate mock system health data"""
        # Base health indicators
        cpu_usage = random.uniform(10, 60)
        memory_usage = random.uniform(20, 70)
        disk_usage = random.uniform(30, 80)
        network_latency = random.uniform(5, 100)
        
        # Add some correlated randomness
        is_high_load = random.random() < 0.15
        if is_high_load:
            cpu_usage += random.uniform(20, 35)
            memory_usage += random.uniform(10, 25)
            network_latency += random.uniform(50, 150)
        
        # Ensure constraints
        cpu_usage = min(cpu_usage, 100)
        memory_usage = min(memory_usage, 100)
        disk_usage = min(disk_usage, 100)
        
        # Health status based on thresholds
        health_status = "good"
        if cpu_usage > 80 or memory_usage > 85 or disk_usage > 90 or network_latency > 200:
            health_status = "warning"
        if cpu_usage > 95 or memory_usage > 95 or disk_usage > 95 or network_latency > 500:
            health_status = "critical"
        
        return {
            "status": health_status,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "network_latency": network_latency,
            "uptime": random.randint(1, 30) * 86400 + random.randint(0, 86399),  # 1-30 days in seconds
            "last_update": datetime.now().isoformat(),
            "processes": random.randint(20, 60),
            "temperature": random.uniform(40, 70),  # CPU temperature in Celsius
            "io_wait": random.uniform(0.5, 5.0)
        }
    
    def generate_component_status(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate mock component status data"""
        components = [
            {
                "name": "Data Provider",
                "status": random.choices(["running", "warning", "error"], weights=[0.9, 0.08, 0.02])[0],
                "last_update": (datetime.now() - timedelta(seconds=random.randint(5, 60))).isoformat(),
                "version": "1.2.3",
                "load": random.uniform(10, 80),
                "message": ""
            },
            {
                "name": "Strategy Engine",
                "status": random.choices(["running", "warning", "error"], weights=[0.85, 0.1, 0.05])[0],
                "last_update": (datetime.now() - timedelta(seconds=random.randint(5, 60))).isoformat(),
                "version": "2.1.0",
                "load": random.uniform(20, 90),
                "message": ""
            },
            {
                "name": "Market Connector",
                "status": random.choices(["running", "warning", "error"], weights=[0.8, 0.15, 0.05])[0],
                "last_update": (datetime.now() - timedelta(seconds=random.randint(5, 60))).isoformat(),
                "version": "1.8.5",
                "load": random.uniform(15, 70),
                "message": ""
            },
            {
                "name": "Risk Manager",
                "status": random.choices(["running", "warning", "error"], weights=[0.95, 0.04, 0.01])[0],
                "last_update": (datetime.now() - timedelta(seconds=random.randint(5, 60))).isoformat(),
                "version": "1.1.7",
                "load": random.uniform(5, 50),
                "message": ""
            },
            {
                "name": "Order Executor",
                "status": random.choices(["running", "warning", "error"], weights=[0.9, 0.07, 0.03])[0],
                "last_update": (datetime.now() - timedelta(seconds=random.randint(5, 60))).isoformat(),
                "version": "1.3.2",
                "load": random.uniform(10, 60),
                "message": ""
            },
            {
                "name": "Machine Learning Pipeline",
                "status": random.choices(["running", "warning", "error"], weights=[0.75, 0.2, 0.05])[0],
                "last_update": (datetime.now() - timedelta(seconds=random.randint(5, 60))).isoformat(),
                "version": "2.0.1",
                "load": random.uniform(30, 95),
                "message": ""
            }
        ]
        
        # Add error messages for non-running components
        error_messages = {
            "warning": [
                "High latency detected",
                "Reduced performance",
                "Resource contention",
                "Intermittent timeouts",
                "Degraded service"
            ],
            "error": [
                "Failed to connect to data source",
                "Service crashed",
                "Out of memory",
                "Database connection failed",
                "Critical dependency failure"
            ]
        }
        
        for component in components:
            if component["status"] != "running":
                component["message"] = random.choice(error_messages[component["status"]])
        
        return {"components": components}
    
    def generate_trading_performance(self) -> Dict[str, Any]:
        """Generate mock trading performance data based on current regime"""
        regime = self._get_regime_characteristics()
        
        # Calculate performance based on regime characteristics
        daily_return = random.normalvariate(regime["trend"], regime["volatility"])
        self.current_equity *= (1 + daily_return)
        self.daily_returns.append(daily_return)
        
        # Calculate metrics
        returns = sum(self.daily_returns)
        volatility = max(0.001, (sum([r**2 for r in self.daily_returns]) / len(self.daily_returns)) ** 0.5)
        sharpe = returns / volatility if volatility > 0 else 0
        
        # Generate trades
        win_rate = regime["win_rate"]
        trades_today = random.randint(3, 15)
        winning_trades = int(trades_today * win_rate)
        losing_trades = trades_today - winning_trades
        
        # Calculate PnL
        daily_pnl = self.current_equity - self.starting_equity
        daily_pnl_pct = (daily_pnl / self.starting_equity) * 100
        
        return {
            "equity": round(self.current_equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "win_rate": round(win_rate * 100, 1),
            "trades_today": trades_today,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(regime["drawdown"] * 100, 1),
            "volatility": round(volatility * 100, 2),
            "regime": self.current_regime
        }
    
    def generate_market_regime_data(self) -> Dict[str, Any]:
        """Generate mock market regime data"""
        # Determine how long until regime change
        now = datetime.now()
        regime_end_time = self.regime_start_time + self.regime_duration
        time_in_regime = (now - self.regime_start_time).total_seconds()
        time_left_in_regime = max(0, (regime_end_time - now).total_seconds())
        
        # If regime is ending, select a new one
        if time_left_in_regime <= 0:
            self.current_regime = self._select_random_regime()
            self.regime_start_time = now
            self.regime_duration = timedelta(days=random.randint(5, 45))
            time_in_regime = 0
            time_left_in_regime = self.regime_duration.total_seconds()
        
        # Get current regime characteristics
        regime_chars = self._get_regime_characteristics()
        
        # Calculate regime probabilities - bias toward current regime but allow others
        regime_probs = {
            "bull": 0.1,
            "bear": 0.1,
            "sideways": 0.1,
            "volatile": 0.1,
            "recovery": 0.05,
            "crash": 0.05
        }
        
        # Current regime gets higher probability
        regime_probs[self.current_regime] = 0.5
        
        # Normalize probabilities
        total = sum(regime_probs.values())
        regime_probs = {k: v/total for k, v in regime_probs.items()}
        
        return {
            "current_regime": self.current_regime,
            "regime_start": self.regime_start_time.isoformat(),
            "time_in_regime": time_in_regime,
            "estimated_duration": self.regime_duration.total_seconds(),
            "regime_confidence": random.uniform(0.7, 0.95),
            "regime_characteristics": {
                "trend": regime_chars["trend"],
                "volatility": regime_chars["volatility"],
                "color": regime_chars["color"]
            },
            "regime_probabilities": regime_probs,
            "signals": {
                "price_action": random.uniform(-1, 1),
                "volume": random.uniform(-1, 1),
                "volatility": random.uniform(-1, 1),
                "momentum": random.uniform(-1, 1),
                "sentiment": random.uniform(-1, 1)
            }
        }
