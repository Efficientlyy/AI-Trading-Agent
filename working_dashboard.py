"""
AI Trading Agent - Working Dashboard (Python 3.13 Compatible)

This dashboard provides a unified interface with navigation between different dashboard views:
- Main Monitoring Dashboard
- Sentiment Analysis Dashboard
- Risk Management Dashboard
- Log Dashboard
- Market Regime Analysis Dashboard

Each module follows the Single Responsibility Principle by focusing on specific functionality.
"""

import os
import sys
import random
import uuid
import json
import logging
import numpy as np
import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# First check if Flask is installed and accessible
try:
    from flask import Flask, render_template, jsonify, request, redirect
    flask_installed = True
except ImportError as e:
    flask_installed = False
    print(f"Import Error: {e}")
    print(f"Python Path: {sys.path}")
    print(f"Python Executable: {sys.executable}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required directory setup
templates_dir = Path("templates")
static_dir = Path("static")
css_dir = static_dir / "css"

templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)
css_dir.mkdir(exist_ok=True)

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

class MarketRegimeType(str, Enum):
    BULL = "Bull Market"
    BEAR = "Bear Market"
    SIDEWAYS = "Sideways Market"
    VOLATILE = "Volatile Market"
    RECOVERY = "Recovery Market"
    CRASH = "Crash Market"

# Generate mock time series data (Python 3.13 compatible)
def generate_time_series(days=30, pattern="random"):
    """Generate time series data with realistic market patterns."""
    dates = [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d") 
             for i in range(days)]
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
    
    return {"dates": dates, "values": values.tolist()}

# Generate mock system status data
def generate_system_status():
    """Generate mock system monitoring data."""
    uptime = f"{random.randint(1, 24)}h {random.randint(1, 59)}m"
    errors = random.randint(0, 5)
    warnings = random.randint(0, 10)
    
    components = [
        {
            "name": "Data Ingestion",
            "status": random.choice([ComponentStatus.OK, ComponentStatus.OK, ComponentStatus.WARNING]),
            "message": "Processing streaming data"
        },
        {
            "name": "Signal Generation",
            "status": random.choice([ComponentStatus.OK, ComponentStatus.OK, ComponentStatus.OK, ComponentStatus.WARNING]),
            "message": "Analyzing market signals"
        },
        {
            "name": "Risk Management",
            "status": random.choice([ComponentStatus.OK, ComponentStatus.OK, ComponentStatus.OK, ComponentStatus.OK, ComponentStatus.ERROR]),
            "message": "Monitoring position risk"
        },
        {
            "name": "Order Execution",
            "status": random.choice([ComponentStatus.OK, ComponentStatus.OK, ComponentStatus.WARNING, ComponentStatus.ERROR]),
            "message": "Executing trades"
        },
        {
            "name": "Portfolio Management",
            "status": random.choice([ComponentStatus.OK, ComponentStatus.OK, ComponentStatus.OK]),
            "message": "Optimizing allocations"
        }
    ]
    
    return {
        "uptime": uptime,
        "errors": errors,
        "warnings": warnings,
        "components": components,
        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Generate mock sentiment analysis data
def generate_sentiment_data():
    """Generate mock sentiment analysis with visualizations."""
    sentiment_scores = generate_time_series(30, random.choice(["bull", "bear", "volatile"]))
    
    # Add correlated metrics based on market scenario
    news_influence = [random.uniform(0.4, 0.9) for _ in range(30)]
    social_media_influence = [random.uniform(0.3, 0.8) for _ in range(30)]
    
    # Calculate aggregated sentiment
    current_sentiment = sentiment_scores["values"][-1] - 100
    sentiment_status = "Bullish" if current_sentiment > 5 else "Bearish" if current_sentiment < -5 else "Neutral"
    
    # Sentiment by source
    sources = {
        "Financial News": random.uniform(-0.8, 0.8),
        "Social Media": random.uniform(-0.5, 0.5),
        "Company Reports": random.uniform(-0.3, 0.3),
        "Expert Opinions": random.uniform(-0.6, 0.6),
        "Market Data": random.uniform(-0.7, 0.7)
    }
    
    return {
        "current_sentiment": current_sentiment,
        "sentiment_status": sentiment_status,
        "historical_sentiment": sentiment_scores,
        "news_influence": news_influence,
        "social_media_influence": social_media_influence,
        "sources": sources,
        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Generate mock risk data
def generate_risk_data():
    """Generate mock risk management data."""
    # Portfolio allocation
    allocation = {
        "Equities": random.uniform(0.3, 0.6),
        "Bonds": random.uniform(0.1, 0.3),
        "Commodities": random.uniform(0.05, 0.15),
        "Crypto": random.uniform(0.05, 0.2),
        "Cash": random.uniform(0.05, 0.1)
    }
    
    # Normalize to 100%
    total = sum(allocation.values())
    allocation = {k: v/total for k, v in allocation.items()}
    
    # Risk metrics
    var_95 = random.uniform(1.5, 4.5)
    var_99 = var_95 * random.uniform(1.2, 1.5)
    
    sharpe = random.uniform(0.8, 2.2)
    sortino = sharpe * random.uniform(1.1, 1.5)
    max_drawdown = random.uniform(5, 15)
    
    # Position risk
    positions = [
        {"asset": "AAPL", "allocation": random.uniform(0.05, 0.15), "risk_contribution": random.uniform(0.05, 0.2)},
        {"asset": "MSFT", "allocation": random.uniform(0.05, 0.15), "risk_contribution": random.uniform(0.05, 0.2)},
        {"asset": "GOOG", "allocation": random.uniform(0.05, 0.1), "risk_contribution": random.uniform(0.05, 0.15)},
        {"asset": "AMZN", "allocation": random.uniform(0.05, 0.1), "risk_contribution": random.uniform(0.05, 0.15)},
        {"asset": "TSLA", "allocation": random.uniform(0.03, 0.08), "risk_contribution": random.uniform(0.05, 0.25)}
    ]
    
    # Historical volatility
    volatility = generate_time_series(60, "volatile")
    
    return {
        "allocation": allocation,
        "metrics": {
            "var_95": var_95,
            "var_99": var_99,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "current_volatility": volatility["values"][-1] / 100
        },
        "positions": positions,
        "historical_volatility": volatility,
        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Generate mock market regime data
def generate_market_regime_data():
    """Generate mock market regime analysis data."""
    # Current regime
    current_regime = random.choice(list(MarketRegimeType))
    
    # Regime probabilities
    regime_probs = {
        MarketRegimeType.BULL: random.uniform(0, 1),
        MarketRegimeType.BEAR: random.uniform(0, 1),
        MarketRegimeType.SIDEWAYS: random.uniform(0, 1),
        MarketRegimeType.VOLATILE: random.uniform(0, 1),
        MarketRegimeType.RECOVERY: random.uniform(0, 1),
        MarketRegimeType.CRASH: random.uniform(0, 1),
    }
    
    # Normalize to 100%
    total = sum(regime_probs.values())
    regime_probs = {k: v/total for k, v in regime_probs.items()}
    
    # Historical regimes (last 12 months)
    months = 12
    historical_regimes = []
    for i in range(months):
        month = (datetime.datetime.now() - datetime.timedelta(days=30*i)).strftime("%Y-%m")
        regime = random.choice(list(MarketRegimeType))
        historical_regimes.append({"month": month, "regime": regime})
    
    # Regime transitions (transition matrix)
    regimes = list(MarketRegimeType)
    transitions = {}
    for from_regime in regimes:
        transitions[from_regime] = {}
        probs = [random.uniform(0, 1) for _ in range(len(regimes))]
        total = sum(probs)
        probs = [p/total for p in probs]
        for i, to_regime in enumerate(regimes):
            transitions[from_regime][to_regime] = probs[i]
    
    return {
        "current_regime": current_regime,
        "regime_probabilities": regime_probs,
        "historical_regimes": historical_regimes,
        "transition_matrix": transitions,
        "price_data": generate_time_series(90, str(current_regime).split('.')[1].lower()),
        "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Consolidated data generation for the dashboard
def generate_dashboard_data():
    """Generate all dashboard data in one function call."""
    return {
        "system_status": generate_system_status(),
        "sentiment": generate_sentiment_data(),
        "risk": generate_risk_data(),
        "market_regime": generate_market_regime_data(),
        "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Routes for API endpoints
def create_flask_app():
    """Create a Flask app with all necessary routes."""
    if not flask_installed:
        raise ImportError("Flask is not installed or accessible")
        
    app = Flask(__name__, 
                template_folder=str(templates_dir.absolute()),
                static_folder=str(static_dir.absolute()))
    
    # Create template for the dashboard
    dashboard_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Trading Agent Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }
            .navbar { background-color: #343a40; }
            .card { margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .card-header { font-weight: bold; background-color: #f8f9fa; }
            .status-ok { color: #28a745; }
            .status-warning { color: #ffc107; }
            .status-error { color: #dc3545; }
            .dashboard-title { margin-bottom: 30px; color: #343a40; }
            .metric-value { font-size: 1.8em; font-weight: bold; }
            .metric-label { font-size: 0.9em; color: #6c757d; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="/">AI Trading Agent Dashboard</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link active" href="/">Main</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/sentiment">Sentiment</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/risk">Risk</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/market-regime">Market Regime</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/logs">Logs</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>

        <div class="container mt-4">
            <h2 class="dashboard-title">Trading System Dashboard</h2>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">System Status</div>
                        <div class="card-body">
                            <p>Status: <span class="status-ok" id="system-status">Operational</span></p>
                            <p>Uptime: <span id="uptime">Loading...</span></p>
                            <p>Errors: <span id="errors">Loading...</span></p>
                            <p>Warnings: <span id="warnings">Loading...</span></p>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Market Sentiment</div>
                        <div class="card-body">
                            <p>Current Sentiment: <span id="sentiment-value">Loading...</span></p>
                            <p>Status: <span id="sentiment-status">Loading...</span></p>
                            <canvas id="sentimentChart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Risk Metrics</div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-6 text-center">
                                    <div class="metric-value" id="sharpe-ratio">Loading...</div>
                                    <div class="metric-label">Sharpe Ratio</div>
                                </div>
                                <div class="col-6 text-center">
                                    <div class="metric-value" id="var-value">Loading...</div>
                                    <div class="metric-label">Value at Risk (95%)</div>
                                </div>
                            </div>
                            <canvas id="volatilityChart" width="400" height="200" class="mt-3"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">Market Regime</div>
                        <div class="card-body">
                            <p>Current Regime: <span id="current-regime">Loading...</span></p>
                            <canvas id="regimeChart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">Component Status</div>
                        <div class="card-body">
                            <table class="table">
                                <thead>
                                    <tr>
                                        <th>Component</th>
                                        <th>Status</th>
                                        <th>Message</th>
                                    </tr>
                                </thead>
                                <tbody id="component-status">
                                    <tr>
                                        <td colspan="3" class="text-center">Loading components...</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Function to fetch dashboard data
            async function fetchDashboardData() {
                try {
                    const response = await fetch('/api/dashboard-data');
                    const data = await response.json();
                    updateDashboard(data);
                } catch (error) {
                    console.error('Error fetching dashboard data:', error);
                }
            }
            
            // Function to update dashboard with data
            function updateDashboard(data) {
                // System status
                document.getElementById('uptime').textContent = data.system_status.uptime;
                document.getElementById('errors').textContent = data.system_status.errors;
                document.getElementById('warnings').textContent = data.system_status.warnings;
                
                // Component status
                const componentTable = document.getElementById('component-status');
                componentTable.innerHTML = '';
                data.system_status.components.forEach(component => {
                    const row = document.createElement('tr');
                    
                    const nameCell = document.createElement('td');
                    nameCell.textContent = component.name;
                    
                    const statusCell = document.createElement('td');
                    let statusClass = 'status-ok';
                    if (component.status === 'Warning') {
                        statusClass = 'status-warning';
                    } else if (component.status === 'Error') {
                        statusClass = 'status-error';
                    }
                    statusCell.innerHTML = `<span class="${statusClass}">${component.status}</span>`;
                    
                    const messageCell = document.createElement('td');
                    messageCell.textContent = component.message;
                    
                    row.appendChild(nameCell);
                    row.appendChild(statusCell);
                    row.appendChild(messageCell);
                    componentTable.appendChild(row);
                });
                
                // Sentiment data
                document.getElementById('sentiment-value').textContent = 
                    data.sentiment.current_sentiment.toFixed(2);
                document.getElementById('sentiment-status').textContent = 
                    data.sentiment.sentiment_status;
                
                // Risk metrics
                document.getElementById('sharpe-ratio').textContent = 
                    data.risk.metrics.sharpe.toFixed(2);
                document.getElementById('var-value').textContent = 
                    data.risk.metrics.var_95.toFixed(2) + '%';
                
                // Market regime
                document.getElementById('current-regime').textContent = 
                    data.market_regime.current_regime;
                
                // Charts
                createSentimentChart(data.sentiment.historical_sentiment);
                createVolatilityChart(data.risk.historical_volatility);
                createRegimeChart(data.market_regime.price_data);
            }
            
            // Create sentiment chart
            function createSentimentChart(data) {
                const ctx = document.getElementById('sentimentChart').getContext('2d');
                
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.dates.slice(-14), // Last 14 days
                        datasets: [{
                            label: 'Sentiment Score',
                            data: data.values.slice(-14).map(v => v - 100), // Normalize around 0
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false
                            }
                        }
                    }
                });
            }
            
            // Create volatility chart
            function createVolatilityChart(data) {
                const ctx = document.getElementById('volatilityChart').getContext('2d');
                
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.dates.slice(-30), // Last 30 days
                        datasets: [{
                            label: 'Volatility',
                            data: data.values.slice(-30).map(v => v / 100), // Scale down
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true
                    }
                });
            }
            
            // Create market regime chart
            function createRegimeChart(data) {
                const ctx = document.getElementById('regimeChart').getContext('2d');
                
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.dates.slice(-30), // Last 30 days
                        datasets: [{
                            label: 'Price',
                            data: data.values.slice(-30),
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true
                    }
                });
            }
            
            // Fetch data immediately and then every 30 seconds
            fetchDashboardData();
            setInterval(fetchDashboardData, 30000);
        </script>
    </body>
    </html>
    """
    
    # Write template to file
    with open(templates_dir / "dashboard.html", "w") as f:
        f.write(dashboard_template)
    
    # Routes
    @app.route('/')
    def index():
        return render_template('dashboard.html')
    
    @app.route('/sentiment')
    def sentiment():
        return render_template('dashboard.html')  # Same template, JS will update content
    
    @app.route('/risk')
    def risk():
        return render_template('dashboard.html')
    
    @app.route('/market-regime')
    def market_regime():
        return render_template('dashboard.html')
    
    @app.route('/logs')
    def logs():
        return render_template('dashboard.html')
    
    @app.route('/api/dashboard-data')
    def api_dashboard_data():
        return jsonify(generate_dashboard_data())
    
    return app

def main():
    """Run the dashboard application."""
    # Verify Flask is installed
    if not flask_installed:
        print("=" * 60)
        print("ERROR: Flask is not available in this Python environment")
        print("=" * 60)
        print(f"Current Python interpreter: {sys.executable}")
        print(f"Current Python path: {sys.path}")
        print("\nTo install Flask, run:")
        print("pip install flask")
        
        # Check if we're in a virtual environment
        in_venv = sys.prefix != sys.base_prefix
        if in_venv:
            print("\nYou appear to be in a virtual environment.")
            print(f"Make sure Flask is installed in the current virtual environment: {sys.prefix}")
        else:
            print("\nYou don't appear to be using a virtual environment.")
            print("Consider creating one to manage dependencies:")
            print("python -m venv .venv")
            print("source .venv/bin/activate  # On Unix/Mac")
            print(".venv\\Scripts\\activate  # On Windows")
            print("pip install flask")
            
        sys.exit(1)
    
    try:
        # Create the Flask app
        app = create_flask_app()
        
        host = "127.0.0.1"
        port = 8050
        
        print(f"Starting AI Trading Agent Dashboard at http://{host}:{port}/")
        print("Press Ctrl+C to stop")
        
        app.run(host=host, port=port, debug=True)
    except Exception as e:
        print(f"Error starting Flask application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
