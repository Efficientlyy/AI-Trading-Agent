"""
AI Trading Agent - Simple Dashboard

A simplified version of the integrated dashboard that works with Python 3.13
by reducing dependencies on problematic packages.
"""

import os
import sys
from pathlib import Path
import json
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

# Apply Python 3.13 compatibility patches
if sys.version_info >= (3, 13):
    print("Applying Python 3.13 compatibility patches...")
    try:
        # Add the project root to the path
        project_root = Path(__file__).parent.absolute()
        sys.path.insert(0, str(project_root))
        
        # Import compatibility patch
        from py313_compatibility_patch import apply_mock_modules
        # Apply mock modules
        apply_mock_modules()
        
        print("Compatibility patches applied successfully")
    except Exception as e:
        print(f"Warning: Failed to apply compatibility patches: {e}")

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Define enums for mock data
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

# Initialize FastAPI app
app = FastAPI(title="AI Trading Agent Dashboard")

# Set up static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create templates directory if it doesn't exist
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Create a basic index.html template
with open(templates_dir / "index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AI Trading Agent Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .dashboard { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #eee; border-radius: 4px; cursor: pointer; }
        .tab.active { background: #007bff; color: white; }
        .panel { display: none; }
        .panel.active { display: block; }
        .card { background: white; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card-header { font-weight: bold; margin-bottom: 10px; color: #333; }
        .status { display: flex; gap: 10px; margin-bottom: 10px; }
        .status-item { display: flex; flex-direction: column; align-items: center; }
        .status-circle { width: 20px; height: 20px; border-radius: 50%; margin-bottom: 5px; }
        .ok { background-color: #28a745; }
        .warning { background-color: #ffc107; }
        .error { background-color: #dc3545; }
        .alert { padding: 10px; border-radius: 4px; margin-bottom: 5px; }
        .alert.info { background-color: #cce5ff; }
        .alert.warning { background-color: #fff3cd; }
        .alert.error { background-color: #f8d7da; }
        .alert.critical { background-color: #dc3545; color: white; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
        .metric { text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .metric-title { font-size: 14px; color: #666; }
        .chart-placeholder { background: #eee; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>AI Trading Agent Dashboard</h1>
            <div>Last Updated: {{ timestamp }}</div>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="main">Main</div>
            <div class="tab" data-tab="sentiment">Sentiment Analysis</div>
            <div class="tab" data-tab="risk">Risk Management</div>
            <div class="tab" data-tab="regime">Market Regime</div>
            <div class="tab" data-tab="logs">Logs</div>
        </div>
        
        <div class="panel active" id="main">
            <div class="card">
                <div class="card-header">System Status</div>
                <div class="status">
                    <div class="status-item">
                        <div class="status-circle ok"></div>
                        <div>API</div>
                    </div>
                    <div class="status-item">
                        <div class="status-circle ok"></div>
                        <div>Database</div>
                    </div>
                    <div class="status-item">
                        <div class="status-circle warning"></div>
                        <div>Exchange</div>
                    </div>
                    <div class="status-item">
                        <div class="status-circle ok"></div>
                        <div>Model</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Recent Alerts</div>
                <div class="alert warning">Warning: High volatility detected in ETH/USD</div>
                <div class="alert info">Trading strategy switched to conservative mode</div>
                <div class="alert error">API rate limit reached for Binance</div>
            </div>
            
            <div class="card">
                <div class="card-header">Performance Metrics</div>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-title">Daily P/L</div>
                        <div class="metric-value">+2.4%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Weekly P/L</div>
                        <div class="metric-value">+5.7%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Sharpe Ratio</div>
                        <div class="metric-value">1.85</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Drawdown</div>
                        <div class="metric-value">-3.2%</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Portfolio Allocation</div>
                <div class="chart-placeholder">Chart Visualization Placeholder</div>
            </div>
        </div>
        
        <div class="panel" id="sentiment">
            <div class="card">
                <div class="card-header">Sentiment Overview</div>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-title">Social Media</div>
                        <div class="metric-value">62%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">News</div>
                        <div class="metric-value">56%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">On-Chain</div>
                        <div class="metric-value">73%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Overall</div>
                        <div class="metric-value">65%</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Sentiment Trend</div>
                <div class="chart-placeholder">Sentiment Chart Placeholder</div>
            </div>
            
            <div class="card">
                <div class="card-header">Continuous Improvement Results</div>
                <div>
                    <p>Last experiment: Bayesian sentiment classifier vs. Neural network</p>
                    <p>Winner: Bayesian classifier (Confidence: 87%)</p>
                    <p>Automatic stopping engaged after 250 samples</p>
                </div>
            </div>
        </div>
        
        <div class="panel" id="risk">
            <div class="card">
                <div class="card-header">Risk Metrics</div>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-title">VaR (95%)</div>
                        <div class="metric-value">$2,450</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">CVaR</div>
                        <div class="metric-value">$3,120</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Max Drawdown</div>
                        <div class="metric-value">-8.4%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Volatility</div>
                        <div class="metric-value">18.2%</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Position Sizing</div>
                <div class="chart-placeholder">Position Allocation Chart Placeholder</div>
            </div>
        </div>
        
        <div class="panel" id="regime">
            <div class="card">
                <div class="card-header">Current Market Regime</div>
                <div class="grid">
                    <div class="metric">
                        <div class="metric-title">Regime Type</div>
                        <div class="metric-value">Volatile</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Confidence</div>
                        <div class="metric-value">82%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Trend</div>
                        <div class="metric-value">Falling</div>
                    </div>
                    <div class="metric">
                        <div class="metric-title">Duration</div>
                        <div class="metric-value">12 days</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">Regime Indicators</div>
                <div class="chart-placeholder">Regime Indicator Chart Placeholder</div>
            </div>
        </div>
        
        <div class="panel" id="logs">
            <div class="card">
                <div class="card-header">System Logs</div>
                <div style="max-height: 400px; overflow-y: auto;">
                    <div class="alert info">[INFO] 2025-03-24 18:25:32 - Model prediction completed for BTC/USD</div>
                    <div class="alert info">[INFO] 2025-03-24 18:24:15 - New market data received from Binance</div>
                    <div class="alert warning">[WARNING] 2025-03-24 18:22:47 - High latency detected in API response</div>
                    <div class="alert info">[INFO] 2025-03-24 18:21:03 - Rebalancing portfolio based on new risk parameters</div>
                    <div class="alert error">[ERROR] 2025-03-24 18:18:56 - Failed to connect to news sentiment provider</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Simple tab navigation
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.getAttribute('data-tab');
                
                // Update active tab
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // Show active panel
                document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
            });
        });
    </script>
</body>
</html>
    """)


# Define route for dashboard
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the dashboard template."""
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )


# Define redirect for /dashboard to root
@app.get("/dashboard")
async def dashboard_redirect():
    """Redirect /dashboard to root."""
    return RedirectResponse(url="/")


def main():
    """Run the dashboard server."""
    # Add command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="AI Trading Agent Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the dashboard on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the dashboard on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"Starting dashboard at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the dashboard")
    
    # Run the uvicorn server
    uvicorn.run(
        "simple_dashboard:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
