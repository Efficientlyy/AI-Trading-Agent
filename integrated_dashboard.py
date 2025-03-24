"""
AI Trading Agent - Integrated Dashboard

This dashboard provides a unified interface with navigation between different dashboard views:
- Main Monitoring Dashboard
- Sentiment Analysis Dashboard
- Risk Management Dashboard
- Log Dashboard
- Market Regime Analysis Dashboard

Each module follows the Single Responsibility Principle by focusing on specific functionality.
"""

import random
import uuid
import logging
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AI Trading Agent Integrated Dashboard")

# Create necessary directories
templates_dir = Path("templates")
static_dir = Path("static")
css_dir = static_dir / "css"

templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)
css_dir.mkdir(exist_ok=True)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enums for consistent status and alert types
class ComponentStatus(str, Enum):
    OK = "OK"
    WARNING = "Warning"
    ERROR = "Error"

class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertCategory(str, Enum):
    SYSTEM = "system"
    EXCHANGE = "exchange"
    ORDER = "order"
    POSITION = "position"
    STRATEGY = "strategy"
    RISK = "risk"
    SECURITY = "security"

class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class OrderType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

class MarketRegimeType(str, Enum):
    BULL = "Bull Market"
    BEAR = "Bear Market"
    SIDEWAYS = "Sideways Market"
    VOLATILE = "Volatile Market"
    RECOVERY = "Recovery Market"
    CRASH = "Crash Market"

class RegimeTrend(str, Enum):
    RISING = "Rising"
    FALLING = "Falling"
    STABLE = "Stable"
    VOLATILE = "Volatile"

#
# Mock Data Generation Functions
# Each function follows single responsibility principle by focusing on one data type
#

def generate_time_series_data(days: int = 30, pattern: str = "random"):
    """
    Generate realistic time series data with specified pattern.
    Implements realistic market patterns based on user's memory.
    
    Args:
        days: Number of days to generate data for
        pattern: Market pattern type (bull, bear, sideways, volatile, recovery, crash)
        
    Returns:
        DataFrame with dates and values
    """
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()  # Oldest to newest
    
    # Base parameters
    trend = 0
    volatility = 1
    seasonality_amp = 0.2
    
    # Adjust parameters based on pattern
    if pattern == "bull":
        trend = 0.5
        volatility = 0.8
    elif pattern == "bear":
        trend = -0.4
        volatility = 1.2
    elif pattern == "sideways":
        trend = 0
        volatility = 0.5
    elif pattern == "volatile":
        trend = 0
        volatility = 2.5
    elif pattern == "recovery":
        trend = np.concatenate([np.linspace(-0.5, 0.8, days//2), np.ones(days - days//2) * 0.8])
        volatility = 1.5
    elif pattern == "crash":
        # Normal then sudden drop then recovery
        trend = np.ones(days) * 0.1
        crash_idx = days // 3
        recovery_idx = crash_idx + days // 6
        trend[crash_idx:recovery_idx] = -3
        trend[recovery_idx:] = 0.3
        volatility = 2
    
    # Generate base values
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, volatility, days)
    
    if isinstance(trend, (int, float)):
        trend_component = np.linspace(0, trend * days, days)
    else:
        trend_component = trend.cumsum()
    
    # Add seasonality (e.g., weekly patterns)
    seasonality = [seasonality_amp * np.sin(2 * np.pi * i / 7) for i in range(days)]
    
    # Combine components
    values = 100 + trend_component + noise + seasonality
    
    # Ensure no negative values
    values = np.maximum(values, 1)
    
    return pd.DataFrame({'date': dates, 'value': values})

def generate_mock_system_data():
    """Generate mock system monitoring data."""
    uptime = f"{random.randint(1, 24)}h {random.randint(1, 59)}m"
    errors = random.randint(0, 5)
    warnings = random.randint(0, 10)
    
    components = [
        {"name": "Data Collector", "status": ComponentStatus.OK},
        {"name": "Strategy Engine", "status": ComponentStatus.OK},
        {"name": "Order Manager", "status": ComponentStatus.OK},
        {"name": "Risk Manager", "status": ComponentStatus.OK},
        {"name": "Exchange Connector", "status": ComponentStatus.WARNING if random.random() < 0.3 else ComponentStatus.OK},
    ]
    
    execution_metrics = {
        "slippage": round(random.uniform(0.01, 0.1), 2),
        "fill_rate": round(random.uniform(90, 99.9), 1),
        "latency": random.randint(100, 500),
        "fees": round(random.uniform(10, 100), 2)
    }
    
    return {
        "uptime": uptime,
        "errors": errors,
        "warnings": warnings,
        "components": components,
        "execution_metrics": execution_metrics,
    }

def generate_mock_alerts(count: int = 10):
    """Generate mock system alerts."""
    alerts = []
    alert_counts = {"critical": 0, "warning": 0, "info": 0, "all": 0}
    
    alert_messages = {
        AlertLevel.CRITICAL: [
            "API authentication failed for exchange Binance",
            "Critical margin usage: 92.5%",
            "Order rejected: Insufficient funds",
            "Exchange connection lost",
            "Order execution timeout",
            "API rate limit exceeded on Kraken"
        ],
        AlertLevel.WARNING: [
            "High latency detected with exchange API",
            "Order partially filled",
            "Strategy performance below threshold",
            "Position size approaching limits",
            "Market data delay detected"
        ],
        AlertLevel.INFO: [
            "New strategy version deployed",
            "Daily risk limits reset",
            "Exchange maintenance scheduled",
            "New trading pair added",
            "System backup completed"
        ]
    }
    
    # Generate specified number of alerts
    now = datetime.now()
    
    for i in range(count):
        # More recent alerts are more likely to be critical
        level_weights = [0.1, 0.3, 0.6] if i < count/3 else [0.6, 0.3, 0.1]
        level = random.choices(
            [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL], 
            weights=level_weights
        )[0]
        
        category = random.choice(list(AlertCategory))
        message = random.choice(alert_messages[level])
        timestamp = (now - timedelta(minutes=random.randint(0, 60))).strftime("%H:%M:%S")
        
        alerts.append({
            "timestamp": timestamp,
            "level": level,
            "category": category,
            "message": message
        })
        
        # Update alert counts
        alert_counts[level] += 1
        alert_counts["all"] += 1
    
    # Sort by time (most recent first)
    alerts.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return alerts, alert_counts

def generate_mock_orders_and_trades():
    """Generate mock orders and trades data."""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BTC/EUR", "ETH/BTC"]
    
    active_orders = []
    for _ in range(random.randint(3, 8)):
        symbol = random.choice(symbols)
        order_type = random.choice(list(OrderType))
        status = random.choice(list(OrderStatus))
        price = round(random.uniform(100, 50000), 2) if "BTC" in symbol else round(random.uniform(10, 5000), 2)
        
        active_orders.append({
            "symbol": symbol,
            "type": order_type,
            "id": f"ID-ORD-{random.randint(100000, 999999)}",
            "price": f"${price}" if "USDT" in symbol or "EUR" in symbol else f"{price}",
            "status": status
        })
    
    recent_trades = []
    for _ in range(random.randint(5, 10)):
        symbol = random.choice(symbols)
        trade_type = random.choice(list(OrderType))
        price = round(random.uniform(100, 50000), 2) if "BTC" in symbol else round(random.uniform(10, 5000), 2)
        quantity = round(random.uniform(0.1, 5), 4)
        
        recent_trades.append({
            "symbol": symbol,
            "type": trade_type,
            "price": f"${price}" if "USDT" in symbol or "EUR" in symbol else f"{price}",
            "quantity": quantity
        })
    
    return {
        "active_orders": active_orders,
        "recent_trades": recent_trades
    }

def generate_mock_sentiment_data():
    """Generate mock sentiment analysis data with regime detection."""
    # Generate time series data with various patterns
    days = 90
    timestamps = [datetime.now() - timedelta(days=i) for i in range(days)]
    timestamps.reverse()  # Oldest to newest
    
    # Generate different market regimes
    bull_market = generate_time_series_data(days=30, pattern="bull")
    bear_market = generate_time_series_data(days=30, pattern="bear")
    volatile_market = generate_time_series_data(days=30, pattern="volatile")
    
    # Combine into one series (using proper DataFrame concatenation)
    price_data = pd.concat([bull_market, bear_market, volatile_market]).reset_index(drop=True)
    
    # Generate sentiment scores (-1 to 1)
    # Sentiment leads price changes slightly (predictive)
    sentiment_bull = [0.2 + random.uniform(0.3, 0.7) + random.uniform(-0.2, 0.2) for _ in range(30)]
    sentiment_bear = [-0.2 + random.uniform(-0.7, -0.3) + random.uniform(-0.2, 0.2) for _ in range(30)]
    sentiment_volatile = [random.uniform(-0.8, 0.8) for _ in range(30)]
    sentiment_data = sentiment_bull + sentiment_bear + sentiment_volatile
    
    # Generate regime probabilities
    bull_regime = [0.8 + random.uniform(-0.2, 0.1) for _ in range(30)] + \
                 [0.3 + random.uniform(-0.2, 0.1) for _ in range(30)] + \
                 [0.4 + random.uniform(-0.3, 0.3) for _ in range(30)]
                 
    bear_regime = [0.1 + random.uniform(-0.1, 0.1) for _ in range(30)] + \
                 [0.6 + random.uniform(-0.1, 0.2) for _ in range(30)] + \
                 [0.3 + random.uniform(-0.2, 0.2) for _ in range(30)]
                 
    neutral_regime = [0.1 + random.uniform(-0.1, 0.1) for _ in range(30)] + \
                    [0.1 + random.uniform(-0.1, 0.1) for _ in range(30)] + \
                    [0.3 + random.uniform(-0.1, 0.5) for _ in range(30)]
    
    # Create Plotly figures
    # Sentiment vs Price chart
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(
        go.Scatter(
            x=price_data['date'], 
            y=price_data['value'], 
            name="Asset Price",
            line=dict(color="#3366CC", width=2)
        ),
        secondary_y=False,
    )
    
    fig1.add_trace(
        go.Scatter(
            x=price_data['date'], 
            y=sentiment_data, 
            name="Sentiment Score",
            line=dict(color="#FF9900", width=2, dash="dash")
        ),
        secondary_y=True,
    )
    
    fig1.update_layout(
        title="Sentiment Analysis vs Price Movement",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified"
    )
    
    fig1.update_yaxes(title_text="Asset Price", secondary_y=False)
    fig1.update_yaxes(title_text="Sentiment Score (-1 to 1)", secondary_y=True)
    
    # Regime Detection chart
    fig2 = go.Figure()
    
    fig2.add_trace(
        go.Scatter(
            x=price_data['date'], 
            y=bull_regime, 
            name="Bullish Regime",
            line=dict(color="#00CC96", width=2),
            stackgroup="regime"
        )
    )
    
    fig2.add_trace(
        go.Scatter(
            x=price_data['date'], 
            y=bear_regime, 
            name="Bearish Regime",
            line=dict(color="#EF553B", width=2),
            stackgroup="regime"
        )
    )
    
    fig2.add_trace(
        go.Scatter(
            x=price_data['date'], 
            y=neutral_regime, 
            name="Neutral Regime",
            line=dict(color="#AB63FA", width=2),
            stackgroup="regime"
        )
    )
    
    fig2.update_layout(
        title="Market Regime Detection",
        xaxis_title="Date",
        yaxis_title="Regime Probability",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified"
    )
    
    # Correlation Analysis
    correlations = []
    for i in range(10):
        asset_name = f"Asset {i+1}"
        correlation = []
        for j in range(10):
            if i == j:
                correlation.append(1.0)
            else:
                # Assets in same sectors have similar correlations
                sector_i = i // 3
                sector_j = j // 3
                if sector_i == sector_j:
                    correlation.append(0.6 + random.uniform(-0.3, 0.3))
                else:
                    correlation.append(0.2 + random.uniform(-0.5, 0.3))
        correlations.append(correlation)
    
    asset_names = [f"Asset {i+1}" for i in range(10)]
    
    fig3 = go.Figure(data=go.Heatmap(
        z=correlations,
        x=asset_names,
        y=asset_names,
        colorscale='RdBu_r',
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in correlations],
        texttemplate="%{text}",
        textfont={"size":10}
    ))
    
    fig3.update_layout(
        title="Asset Correlation Matrix",
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    
    # Trading signals based on sentiment and regime
    signals = []
    for i in range(5):
        signal_type = random.choice(["BUY", "SELL", "HOLD"])
        asset = f"Asset {random.randint(1, 10)}"
        strength = random.randint(1, 5)
        signals.append({
            "asset": asset,
            "signal": signal_type,
            "strength": "★" * strength
        })
    
    # Convert to JSON
    sentiment_chart_json = json.dumps({
        "data": fig1.data,
        "layout": fig1.layout
    }, cls=plotly.utils.PlotlyJSONEncoder)
    
    regime_chart_json = json.dumps({
        "data": fig2.data,
        "layout": fig2.layout
    }, cls=plotly.utils.PlotlyJSONEncoder)
    
    correlation_chart_json = json.dumps({
        "data": fig3.data,
        "layout": fig3.layout
    }, cls=plotly.utils.PlotlyJSONEncoder)
    
    return {
        "sentiment_chart": sentiment_chart_json,
        "regime_chart": regime_chart_json,
        "correlation_chart": correlation_chart_json,
        "trading_signals": signals,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def generate_mock_risk_data():
    """Generate mock risk management data with visualizations."""
    # Overall risk
    overall_risk_percentage = round(random.uniform(30, 95), 1)
    
    # Strategy risks
    strategies = ["Momentum", "Mean Reversion", "Trend Following", "Statistical Arbitrage"]
    strategy_risks = []
    
    for strategy in strategies:
        allocation = round(random.uniform(10, 40), 1)
        utilization = round(random.uniform(20, 100), 1)
        
        strategy_risks.append({
            "name": strategy,
            "allocation": allocation,
            "utilization": utilization
        })
    
    # Asset risks
    assets = ["BTC", "ETH", "SOL", "ADA", "DOT", "XRP"]
    asset_risks = []
    
    for asset in assets:
        allocation = round(random.uniform(5, 35), 1)
        utilization = round(random.uniform(20, 100), 1)
        
        asset_risks.append({
            "name": asset,
            "allocation": allocation,
            "utilization": utilization
        })
    
    # Risk metrics
    risk_metrics = {
        "var": f"${round(random.uniform(5000, 20000), 2)}",
        "cvar": f"${round(random.uniform(8000, 30000), 2)}",
        "max_drawdown": round(random.uniform(5, 25), 1),
        "sharpe": round(random.uniform(0.5, 2.5), 2)
    }
    
    # Risk alerts
    risk_alerts = []
    risk_alert_messages = [
        "Strategy 'Mean Reversion' risk utilization at 80%",
        "Asset 'BTC' in 'Momentum' strategy has reached 100% risk utilization",
        "Portfolio VaR approaching daily limit",
        "Correlation matrix shows increased systemic risk",
        "Maximum drawdown warning for 'Trend Following' strategy",
        "Leverage ratio above target for 'Statistical Arbitrage'",
        "Risk concentration in crypto assets above threshold"
    ]
    
    for _ in range(random.randint(3, 7)):
        level = random.choice([AlertLevel.WARNING, AlertLevel.CRITICAL])
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        risk_alerts.append({
            "level": level,
            "timestamp": timestamp,
            "message": random.choice(risk_alert_messages)
        })
    
    # Create risk visualizations
    # 1. Strategy allocation pie chart
    strategy_labels = [s["name"] for s in strategy_risks]
    strategy_values = [s["allocation"] for s in strategy_risks]
    
    strategy_pie = go.Figure(data=[go.Pie(
        labels=strategy_labels, 
        values=strategy_values,
        hole=.3,
        marker=dict(colors=px.colors.qualitative.Plotly)
    )])
    strategy_pie.update_layout(
        title='Strategy Allocation',
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    # 2. Asset allocation pie chart
    asset_labels = [a["name"] for a in asset_risks]
    asset_values = [a["allocation"] for a in asset_risks]
    
    asset_pie = go.Figure(data=[go.Pie(
        labels=asset_labels, 
        values=asset_values,
        hole=.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    asset_pie.update_layout(
        title='Asset Allocation',
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    # 3. Risk utilization bar chart
    util_fig = go.Figure()
    util_fig.add_trace(go.Bar(
        x=strategy_labels,
        y=[s["utilization"] for s in strategy_risks],
        marker_color=['#2E86C1' if u < 60 else '#F39C12' if u < 80 else '#C0392B' for u in [s["utilization"] for s in strategy_risks]]
    ))
    util_fig.update_layout(
        title='Risk Utilization by Strategy',
        yaxis=dict(title='Utilization %', range=[0, 100]),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    # Convert charts to JSON
    strategy_pie_json = json.dumps({
        "data": strategy_pie.data,
        "layout": strategy_pie.layout
    }, cls=plotly.utils.PlotlyJSONEncoder)
    asset_pie_json = json.dumps({
        "data": asset_pie.data,
        "layout": asset_pie.layout
    }, cls=plotly.utils.PlotlyJSONEncoder)
    util_chart_json = json.dumps({
        "data": util_fig.data,
        "layout": util_fig.layout
    }, cls=plotly.utils.PlotlyJSONEncoder)
    
    return {
        "overall_risk_percentage": overall_risk_percentage,
        "strategy_risks": strategy_risks,
        "asset_risks": asset_risks,
        "risk_metrics": risk_metrics,
        "risk_alerts": risk_alerts,
        # Charts
        "strategy_pie_chart": strategy_pie_json,
        "asset_pie_chart": asset_pie_json,
        "utilization_chart": util_chart_json
    }

def generate_mock_market_regime_data():
    """Generate mock market regime detection data for the dashboard."""
    # Generate current regime confidence levels
    bull_confidence = random.randint(65, 90)
    bear_confidence = random.randint(55, 85)
    sideways_confidence = random.randint(70, 95)
    
    # Determine trends for each regime
    bull_trend = random.choice([RegimeTrend.RISING, RegimeTrend.STABLE])
    bear_trend = random.choice([RegimeTrend.FALLING, RegimeTrend.STABLE])
    sideways_trend = RegimeTrend.STABLE
    
    # Generate strategy performance data across different regimes
    strategies = ["Momentum", "Value", "Mean Reversion", "Trend Following", "Volatility Arbitrage"]
    
    strategy_performance = []
    for strategy in strategies:
        # Generate realistic returns based on strategy type and market regime
        if strategy == "Momentum":
            # Momentum strategies typically perform well in bull markets
            bull_return = round(random.uniform(15.0, 30.0), 1)
            bear_return = round(random.uniform(-20.0, -5.0), 1)
            sideways_return = round(random.uniform(-5.0, 5.0), 1)
        elif strategy == "Value":
            # Value strategies often underperform in strong bull markets
            bull_return = round(random.uniform(8.0, 15.0), 1)
            bear_return = round(random.uniform(-10.0, 0.0), 1)
            sideways_return = round(random.uniform(0.0, 8.0), 1)
        elif strategy == "Mean Reversion":
            # Mean reversion strategies often perform well in sideways markets
            bull_return = round(random.uniform(5.0, 15.0), 1)
            bear_return = round(random.uniform(-15.0, 0.0), 1)
            sideways_return = round(random.uniform(8.0, 15.0), 1)
        elif strategy == "Trend Following":
            # Trend following strategies tend to perform well in strong directional markets
            bull_return = round(random.uniform(18.0, 35.0), 1)
            bear_return = round(random.uniform(10.0, 25.0), 1)  # Can be positive in bear markets too
            sideways_return = round(random.uniform(-10.0, 0.0), 1)
        else:  # Volatility Arbitrage
            # Volatility strategies often perform well in volatile markets
            bull_return = round(random.uniform(5.0, 15.0), 1)
            bear_return = round(random.uniform(10.0, 20.0), 1)
            sideways_return = round(random.uniform(0.0, 8.0), 1)
            
        # Generate max drawdown values (generally higher in bear markets)
        bull_drawdown = round(random.uniform(3.0, 10.0), 1)
        bear_drawdown = round(random.uniform(15.0, 30.0), 1)
        sideways_drawdown = round(random.uniform(5.0, 15.0), 1)
        
        # Generate Sharpe ratios (generally higher in bull markets)
        bull_sharpe = round(random.uniform(1.5, 3.0), 2)
        bear_sharpe = round(random.uniform(-0.5, 1.0), 2)
        sideways_sharpe = round(random.uniform(0.5, 1.5), 2)
        
        strategy_performance.append({
            "name": strategy,
            "returns": {
                "bull": bull_return,
                "bear": bear_return,
                "sideways": sideways_return
            },
            "drawdowns": {
                "bull": bull_drawdown,
                "bear": bear_drawdown, 
                "sideways": sideways_drawdown
            },
            "sharpe": {
                "bull": bull_sharpe,
                "bear": bear_sharpe,
                "sideways": sideways_sharpe
            }
        })
    
    # Generate regime transition probabilities
    transition_probs = {
        "bull_to_bull": round(random.uniform(0.8, 0.95), 2),
        "bull_to_bear": round(random.uniform(0.05, 0.2), 2),
        "bear_to_bear": round(random.uniform(0.7, 0.9), 2),
        "bear_to_bull": round(random.uniform(0.1, 0.3), 2),
        "sideways_to_bull": round(random.uniform(0.3, 0.5), 2),
        "sideways_to_bear": round(random.uniform(0.2, 0.4), 2)
    }
    
    # Format current signal details
    current_signals = {
        "bull": {
            "confidence": bull_confidence,
            "trend": bull_trend.value,
            "signal": "Hold long positions",
            "allocation": round(random.uniform(30, 60), 1)
        },
        "bear": {
            "confidence": bear_confidence,
            "trend": bear_trend.value,
            "signal": "Reduce risk exposure",
            "allocation": round(random.uniform(10, 30), 1)
        },
        "sideways": {
            "confidence": sideways_confidence,
            "trend": sideways_trend.value,
            "signal": "Implement range-bound strategies",
            "allocation": round(random.uniform(20, 40), 1)
        }
    }
    
    # Generate time series data for regime probabilities over time
    days = 90
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    dates.reverse()
    
    # Create realistic regime probability series with transitions
    bull_probs = []
    bear_probs = []
    sideways_probs = []
    
    # Start with balanced regime probabilities
    current_bull = 0.33
    current_bear = 0.33
    current_sideways = 0.34
    
    for i in range(days):
        # Add some random walk with mean reversion
        current_bull += random.uniform(-0.05, 0.05)
        current_bear += random.uniform(-0.05, 0.05)
        current_sideways += random.uniform(-0.05, 0.05)
        
        # Simulate regime shifts at certain points
        if i == 20:  # Shift towards bull market
            current_bull += 0.3
            current_bear -= 0.2
            current_sideways -= 0.1
        elif i == 50:  # Shift towards bear market
            current_bull -= 0.3
            current_bear += 0.4
            current_sideways -= 0.1
        elif i == 70:  # Shift towards sideways market
            current_bull -= 0.2
            current_bear -= 0.2
            current_sideways += 0.4
            
        # Ensure probabilities remain positive
        current_bull = max(0.05, current_bull)
        current_bear = max(0.05, current_bear)
        current_sideways = max(0.05, current_sideways)
        
        # Normalize to ensure they sum to 1
        total = current_bull + current_bear + current_sideways
        current_bull /= total
        current_bear /= total
        current_sideways /= total
        
        bull_probs.append(current_bull)
        bear_probs.append(current_bear)
        sideways_probs.append(current_sideways)
    
    # Create Plotly figure for regime probabilities over time
    regime_history_fig = go.Figure()
    
    regime_history_fig.add_trace(
        go.Scatter(
            x=dates,
            y=bull_probs,
            name="Bull Market",
            stackgroup="regime",
            fillcolor="rgba(76, 175, 80, 0.5)",
            line=dict(color="rgb(46, 125, 50)")
        )
    )
    
    regime_history_fig.add_trace(
        go.Scatter(
            x=dates,
            y=bear_probs,
            name="Bear Market",
            stackgroup="regime",
            fillcolor="rgba(244, 67, 54, 0.5)",
            line=dict(color="rgb(183, 28, 28)")
        )
    )
    
    regime_history_fig.add_trace(
        go.Scatter(
            x=dates,
            y=sideways_probs,
            name="Sideways Market",
            stackgroup="regime",
            fillcolor="rgba(33, 150, 243, 0.5)",
            line=dict(color="rgb(13, 71, 161)")
        )
    )
    
    regime_history_fig.update_layout(
        title="Market Regime Probabilities Over Time",
        xaxis_title="Date",
        yaxis_title="Regime Probability",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified"
    )
    
    # Convert to JSON
    regime_history_json = json.dumps({
        "data": regime_history_fig.data,
        "layout": regime_history_fig.layout
    }, cls=plotly.utils.PlotlyJSONEncoder)
    
    return {
        "current_signals": current_signals,
        "strategy_performance": strategy_performance,
        "transition_probs": transition_probs,
        "regime_history_chart": regime_history_json,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def generate_mock_logs(level: str = "all", count: int = 20):
    """Generate mock system logs."""
    logs = []
    
    log_messages = {
        LogLevel.DEBUG: [
            "Initialized data collector with default settings",
            "Strategy engine metrics: execution_time=0.23s, memory_usage=128MB",
            "Processing market data batch: symbols=['BTC/USD', 'ETH/USD'], datapoints=1000",
            "Order ID 12345 created with params: symbol=BTC/USD, side=buy, price=45000, amount=0.1",
            "Cache hit ratio: 0.85"
        ],
        LogLevel.INFO: [
            "Successfully connected to exchange API",
            "Strategy 'Momentum' activated for trading pairs: ['BTC/USD', 'ETH/USD']",
            "Order executed: ID=12345, Symbol=BTC/USD, Price=44950.00, Amount=0.1",
            "Daily performance calculation completed: PnL=$2500, Sharpe=1.8",
            "Position rebalanced: BTC=30%, ETH=25%, CASH=45%"
        ],
        LogLevel.WARNING: [
            "High latency detected with exchange API: response_time=2.3s",
            "Order partially filled: ID=12345, filled=60%",
            "Rate limit approaching: 80% of maximum API calls used",
            "Strategy 'Mean Reversion' performance below threshold: Sharpe=0.8",
            "Market data refresh delayed by 30 seconds"
        ],
        LogLevel.ERROR: [
            "Failed to connect to exchange API: ConnectionTimeout",
            "Order execution failed: ID=12345, reason='Insufficient funds'",
            "Strategy execution error: KeyError in data preprocessing",
            "Risk limit breach: strategy='Momentum', asset='BTC', allocation=35%",
            "Database connection error: max_connections exceeded"
        ]
    }
    
    # Filter log levels based on user selection
    available_levels = list(LogLevel)
    if level != "all" and level in [l.value for l in LogLevel]:
        available_levels = [next(l for l in LogLevel if l.value == level)]
    
    now = datetime.now()
    
    for i in range(count):
        selected_level = random.choice(available_levels)
        message = random.choice(log_messages[selected_level])
        timestamp = (now - timedelta(seconds=random.randint(1, 3600))).strftime("%Y-%m-%d %H:%M:%S")
        
        logs.append({
            "timestamp": timestamp,
            "level": selected_level,
            "message": message
        })
    
    # Sort by timestamp (newest first)
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return logs


@app.get("/", response_class=HTMLResponse)
async def integrated_dashboard(request: Request, tab: str = Query("main", description="Dashboard tab to display"), level: str = Query("all", description="Log level filter for logs tab")):
    """
    Render the integrated dashboard with the selected tab.
    
    Args:
        request: The request object
        tab: The active tab (main, sentiment, risk, logs)
        level: The log level filter (all, info, warning, error, debug)
    
    Returns:
        HTMLResponse: Rendered dashboard template
    """
    import json
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Validate tab parameter
    valid_tabs = ["main", "sentiment", "risk", "logs", "market_regime"]
    if tab not in valid_tabs:
        tab = "main"
    
    # Basic template data
    template_data = {
        "request": request,
        "active_tab": tab,
        "timestamp": timestamp
    }
    
    # Get data specific to the selected tab
    if tab == "main":
        # Get system monitoring data
        system_data = generate_mock_system_data()
        orders_trades_data = generate_mock_orders_and_trades()
        alerts, alert_counts = generate_mock_alerts()
        
        template_data.update(system_data)
        template_data.update(orders_trades_data)
        template_data.update({
            "alerts": alerts,
            "alert_counts": alert_counts
        })
    
    elif tab == "sentiment":
        # Get sentiment analysis data
        sentiment_data = generate_mock_sentiment_data()
        
        template_data.update(sentiment_data)
    
    elif tab == "risk":
        risk_data = generate_mock_risk_data()
        
        template_data.update(risk_data)
    
    elif tab == "logs":
        log_data = generate_mock_logs(level)
        template_data.update({
            "log_level": level,
            "logs": log_data
        })
    
    elif tab == "market_regime":
        market_regime_data = generate_mock_market_regime_data()
        template_data.update(market_regime_data)
    
    else:  # Default to main dashboard
        main_data = generate_mock_system_data()
        template_data.update(main_data)
    
    # Render the integrated dashboard template
    return templates.TemplateResponse("integrated_dashboard.html", template_data)

def main():
    """Run the integrated dashboard server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the AI Trading Agent integrated dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development mode)")
    args = parser.parse_args()
    
    # Log startup information
    logger.info(f"Starting integrated dashboard on http://{args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop the server")
    
    # Run the server
    uvicorn.run(
        app, 
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload
    )

if __name__ == "__main__":
    main()
