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

# Import necessary modules
import os
import sys
import time
import random
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum  # Add missing Enum import

# Try to import pandas and numpy, which we need for data generation
try:
    import pandas as pd
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    # Log the error and exit if these critical dependencies are missing
    sys.exit("Error: Required dependencies not found. Please install pandas, numpy, and plotly.")

# Flask imports - these will be patched by our compatibility layer if needed
try:
    from flask import Flask, render_template, request, jsonify, redirect, url_for
except ImportError:
    # Log a warning but continue - our compatibility layer will handle this
    logging.warning("Flask not found. Using compatibility layer.")
    # Set these to None to avoid further import errors
    Flask = None
    render_template = None
    request = None
    jsonify = None
    redirect = None
    url_for = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_trading_dashboard")

# Apply dateutil compatibility patches
from src.common.dateutil_compat import apply_dateutil_patches
apply_dateutil_patches()

# Import visualization and data libs
try:
    import plotly.express as px
except ImportError as e:
    print(f"Error importing data libraries: {e}")
    print("Make sure pandas, numpy, and plotly are installed:")
    print("pip install pandas numpy plotly")
    sys.exit(1)

# Setup Flask instead of FastAPI
try:
    logger.info("Successfully imported Flask and dependencies")
except ImportError as e:
    logger.error(f"Error importing Flask: {e}")
    print(f"Error importing Flask: {e}")
    print("Make sure Flask is installed:")
    print("pip install flask")
    sys.exit(1)

# Enums for consistent status values
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


# Create Flask application function
def create_app():
    """Create and configure a Flask application for the dashboard."""
    # Create Flask application
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Configure application
    app.config['SECRET_KEY'] = 'ai-trading-agent-dashboard'
    
    # Ensure template directory exists
    template_dir = Path('templates')
    template_dir.mkdir(exist_ok=True)
    
    # Generate data immediately
    mock_price_data = generate_time_series_data(30, "bull")
    mock_system_data = generate_mock_system_data()
    # Use the correct DataFrame operations based on memory about time series issues
    price_df = pd.DataFrame({'date': mock_price_data['date'].dt.strftime('%Y-%m-%d'), 
                           'price': mock_price_data['value']})
    
    # Create a simple default template if it doesn't exist
    template_path = template_dir / 'dashboard.html'
    if not template_path.exists():
        with open(template_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>AI Trading Agent Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .card-header {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .status-ok { color: #198754; font-weight: bold; }
        .status-warning { color: #ffc107; font-weight: bold; }
        .status-error { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">AI Trading Agent Dashboard</h1>
        
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">System Status</div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>Status:</strong> <span class="status-ok">Active</span></p>
                                <p><strong>Last Updated:</strong> <span id="last-updated"></span></p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Server Time:</strong> <span id="server-time"></span></p>
                                <p><strong>Agent Version:</strong> 1.0.0</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">Price Chart</div>
                    <div class="card-body">
                        <div id="price-chart" style="height: 300px;"></div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Recent Alerts</div>
                    <div class="card-body">
                        <div class="alert alert-warning">
                            <strong>Warning:</strong> High volatility detected
                        </div>
                        <div class="alert alert-info">
                            <strong>Info:</strong> Pattern detected: Bull flag
                        </div>
                        <div class="alert alert-success">
                            <strong>Success:</strong> Trade executed
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">Performance Metrics</div>
                    <div class="card-body">
                        <p><strong>Sharpe Ratio:</strong> 1.75</p>
                        <p><strong>Win Rate:</strong> 68%</p>
                        <p><strong>Drawdown:</strong> 8.2%</p>
                        <p><strong>Total Return:</strong> 24.5%</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Set current time
        function updateTime() {
            const now = new Date();
            document.getElementById('server-time').textContent = now.toLocaleTimeString();
            document.getElementById('last-updated').textContent = now.toLocaleString();
        }
        updateTime();
        setInterval(updateTime, 1000);
        
        // Create sample price chart with embedded data
        document.addEventListener('DOMContentLoaded', function() {
            // Directly embedded data to avoid loading issues
            const chartData = """ + price_df.to_json(orient='records') + """;
            
            const dates = chartData.map(item => item.date);
            const prices = chartData.map(item => item.price);
            
            // Calculate simple moving average
            const movingAvg = [];
            for (let i = 0; i < prices.length; i++) {
                if (i < 7) {
                    movingAvg.push(null);
                } else {
                    let sum = 0;
                    for (let j = i - 7; j < i; j++) {
                        sum += prices[j];
                    }
                    movingAvg.push(sum / 7);
                }
            }
            
            const trace1 = {
                x: dates,
                y: prices,
                type: 'scatter',
                mode: 'lines',
                name: 'Price',
                line: { color: '#2962FF' }
            };
            
            const trace2 = {
                x: dates,
                y: movingAvg,
                type: 'scatter',
                mode: 'lines',
                name: 'MA(7)',
                line: { color: '#FF6D00', dash: 'dash' }
            };
            
            const layout = {
                autosize: true,
                margin: { l: 40, r: 40, t: 20, b: 40 },
                xaxis: { showgrid: false },
                yaxis: { showgrid: true },
                legend: { orientation: 'h', y: 1.1 }
            };
            
            Plotly.newPlot('price-chart', [trace1, trace2], layout);
        });
    </script>
</body>
</html>""")
    
    # Define routes
    @app.route('/')
    def index():
        """Main dashboard route that renders the integrated dashboard."""
        return render_template('dashboard.html')
        
    @app.route('/sentiment')
    def sentiment_dashboard():
        """Sentiment analysis dashboard tab."""
        return render_template('dashboard.html')
        
    @app.route('/risk')
    def risk_dashboard():
        """Risk management dashboard tab."""
        return render_template('dashboard.html')
        
    @app.route('/market-regime')
    def market_regime_dashboard():
        """Market regime dashboard tab."""
        return render_template('dashboard.html')
        
    @app.route('/logs')
    def logs_dashboard():
        """Logs and monitoring dashboard tab."""
        return render_template('dashboard.html')
    
    # API endpoints for data
    @app.route('/api/status')
    def api_status():
        """API endpoint to check status."""
        return jsonify({
            "status": "ok", 
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_usage": random.uniform(5, 25),
                "memory_usage": random.uniform(30, 70),
                "active_threads": random.randint(2, 8)
            }
        })
    
    @app.route('/api/price-data')
    def api_price_data():
        """API endpoint for price chart data."""
        days = int(request.args.get('days', 30))
        pattern = request.args.get('pattern', 'bull')
        
        # Generate time series data using our existing function
        data = generate_time_series_data(days, pattern)
        
        # Convert to dictionary records
        return jsonify(data.to_dict('records'))
    
    @app.route('/api/dashboard-data')
    def api_dashboard_data():
        """API endpoint for main dashboard data."""
        # Generate mock system status data
        system_data = generate_mock_system_data()
        
        # Generate some basic mock data for sentiment
        sentiment_days = 30
        price_data = generate_time_series_data(sentiment_days, 'bull')
        
        # Create sentiment data with dates
        sentiment_data = {
            'current_sentiment': random.uniform(0.3, 0.8),
            'sentiment_status': random.choice(['Bullish', 'Neutral', 'Bearish']),
            'historical_sentiment': {
                'dates': [d.strftime('%Y-%m-%d') for d in price_data['date'].tolist()],
                'values': [100 + random.uniform(-10, 10) for _ in range(sentiment_days)]
            }
        }
        
        # Create risk data
        risk_data = {
            'metrics': {
                'sharpe': random.uniform(0.8, 2.5),
                'var_95': random.uniform(3.5, 12.0)
            },
            'historical_volatility': {
                'dates': [d.strftime('%Y-%m-%d') for d in price_data['date'].tolist()],
                'values': [random.uniform(8, 25) for _ in range(sentiment_days)]
            }
        }
        
        # Create market regime data
        market_regime_data = {
            'current_regime': random.choice(['Bull Market', 'Bear Market', 'Sideways Market', 'Volatile Market']),
            'price_data': {
                'dates': [d.strftime('%Y-%m-%d') for d in price_data['date'].tolist()],
                'values': price_data['value'].tolist()
            }
        }
        
        # Create component status data
        components = [
            {
                'name': 'Data Collection',
                'status': 'OK',
                'message': 'Running normally'
            },
            {
                'name': 'Strategy Engine',
                'status': random.choice(['OK', 'Warning']),
                'message': 'Performance degraded' if random.random() < 0.3 else 'Running normally'
            },
            {
                'name': 'Order Manager',
                'status': 'OK',
                'message': 'Running normally'
            },
            {
                'name': 'Exchange Connection',
                'status': random.choice(['OK', 'Warning', 'Error']),
                'message': random.choice(['Running normally', 'High latency detected', 'Connection timed out'])
            },
            {
                'name': 'Risk Manager',
                'status': 'OK',
                'message': 'Running normally'
            }
        ]
        
        # Combine all data
        return jsonify({
            'system_status': {
                'uptime': f"{random.randint(1, 48)}h {random.randint(0, 59)}m",
                'errors': random.randint(0, 5),
                'warnings': random.randint(0, 10),
                'components': components
            },
            'sentiment': sentiment_data,
            'risk': risk_data,
            'market_regime': market_regime_data
        })
    
    return app

# For standalone testing
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=8001)
