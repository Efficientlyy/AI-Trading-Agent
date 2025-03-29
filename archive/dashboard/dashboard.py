"""
Standalone dashboard for the trading system monitoring.

This is a self-contained script that doesn't depend on any other modules.
"""

import os
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required packages
try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Please make sure you have installed all required packages:")
    print("pip install fastapi uvicorn jinja2 starlette pandas numpy")
    exit(1)

# Try to import our dashboard components
try:
    from src.dashboard import sentiment_router
except ImportError as e:
    logger.warning(f"Could not import sentiment dashboard: {e}")
    sentiment_router = None

# Create templates directory
templates_dir = Path("dashboard_templates")
templates_dir.mkdir(exist_ok=True)

# Create static files directory 
static_dir = Path("dashboard_static")
static_dir.mkdir(exist_ok=True)

# Create the HTML template
index_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading System Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 10px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            margin: 0;
            font-size: 24px;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-top: 0;
            margin-bottom: 15px;
            color: #2c3e50;
            font-size: 18px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-ok {
            background-color: #2ecc71;
        }
        .status-warning {
            background-color: #f39c12;
        }
        .status-error {
            background-color: #e74c3c;
        }
        .status-critical {
            background-color: #8e44ad;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        th {
            font-weight: bold;
            color: #7f8c8d;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .refresh-message {
            text-align: center;
            margin-top: 20px;
            color: #7f8c8d;
            font-size: 12px;
        }
        .alert {
            border-left: 4px solid #ccc;
            padding: 8px 12px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
        }
        .alert-info {
            border-left-color: #3498db;
        }
        .alert-warning {
            border-left-color: #f39c12;
        }
        .alert-error {
            border-left-color: #e74c3c;
        }
        .alert-critical {
            border-left-color: #8e44ad;
        }
        .alert-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .alert-title {
            font-weight: bold;
            color: #2c3e50;
        }
        .alert-time {
            color: #7f8c8d;
            font-size: 12px;
        }
        .alert-message {
            margin-bottom: 5px;
        }
        .alert-category {
            display: inline-block;
            padding: 2px 6px;
            font-size: 12px;
            border-radius: 4px;
            background-color: #ecf0f1;
            color: #7f8c8d;
            margin-right: 5px;
        }
        .alert-details {
            margin-top: 5px;
            font-size: 12px;
            color: #7f8c8d;
        }
        .alert-counters {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .alert-counter {
            padding: 8px 15px;
            border-radius: 4px;
            text-align: center;
            color: white;
            flex: 1;
        }
        .alert-counter.info {
            background-color: #3498db;
        }
        .alert-counter.warning {
            background-color: #f39c12;
        }
        .alert-counter.error {
            background-color: #e74c3c;
        }
        .alert-counter.critical {
            background-color: #8e44ad;
        }
        .alert-counter-value {
            font-size: 20px;
            font-weight: bold;
        }
        .alert-counter-label {
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Trading System Dashboard</h1>
    </header>
    <div class="container">
        <div class="dashboard">
            <!-- System Status Card -->
            <div class="card">
                <h2>System Status</h2>
                <div>
                    <p><strong>Uptime:</strong> {{uptime_seconds}} seconds</p>
                    <p><strong>Error Count:</strong> {{error_count}}</p>
                </div>
                <h3>Components</h3>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Status</th>
                        <th>Message</th>
                    </tr>
                    <tr>
                        <td>execution</td>
                        <td>
                            <span class="status-indicator status-ok"></span>
                            ok
                        </td>
                        <td>Running normally</td>
                    </tr>
                    <tr>
                        <td>data</td>
                        <td>
                            <span class="status-indicator status-warning"></span>
                            warning
                        </td>
                        <td>Experiencing delays</td>
                    </tr>
                    <tr>
                        <td>strategy</td>
                        <td>
                            <span class="status-indicator status-error"></span>
                            error
                        </td>
                        <td>Encountered an error</td>
                    </tr>
                </table>
            </div>

            <!-- Alerts Card -->
            <div class="card">
                <h2>System Alerts</h2>
                <div class="alert-counters">
                    <div class="alert-counter info">
                        <div class="alert-counter-value">{{alert_counts.info}}</div>
                        <div class="alert-counter-label">Info</div>
                    </div>
                    <div class="alert-counter warning">
                        <div class="alert-counter-value">{{alert_counts.warning}}</div>
                        <div class="alert-counter-label">Warning</div>
                    </div>
                    <div class="alert-counter error">
                        <div class="alert-counter-value">{{alert_counts.error}}</div>
                        <div class="alert-counter-label">Error</div>
                    </div>
                    <div class="alert-counter critical">
                        <div class="alert-counter-value">{{alert_counts.critical}}</div>
                        <div class="alert-counter-label">Critical</div>
                    </div>
                </div>

                {% for alert in alerts %}
                <div class="alert alert-{{alert.level}}">
                    <div class="alert-header">
                        <span class="alert-title">{{alert.source}}</span>
                        <span class="alert-time">{{alert.time}}</span>
                    </div>
                    <div class="alert-message">{{alert.message}}</div>
                    <div>
                        <span class="alert-category">{{alert.category}}</span>
                        <span class="alert-category">{{alert.level}}</span>
                    </div>
                    {% if alert.details %}
                    <div class="alert-details">{{alert.details}}</div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>

            <!-- Active Orders Card -->
            <div class="card">
                <h2>Active Orders</h2>
                <table>
                    <tr>
                        <th>Order ID</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>test-order-123</td>
                        <td>BTC/USDT</td>
                        <td>BUY</td>
                        <td>FILLED</td>
                    </tr>
                    <tr>
                        <td>test-order-124</td>
                        <td>ETH/USDT</td>
                        <td>SELL</td>
                        <td>OPEN</td>
                    </tr>
                </table>
            </div>

            <!-- Recent Trades Card -->
            <div class="card">
                <h2>Recent Trades</h2>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Price</th>
                    </tr>
                    <tr>
                        <td>{{trade_time_1}}</td>
                        <td>BTC/USDT</td>
                        <td>BUY</td>
                        <td>49950.00</td>
                    </tr>
                    <tr>
                        <td>{{trade_time_2}}</td>
                        <td>ETH/USDT</td>
                        <td>SELL</td>
                        <td>2950.75</td>
                    </tr>
                </table>
            </div>

            <!-- Performance Metrics Card -->
            <div class="card">
                <h2>Execution Metrics</h2>
                <div class="metric-grid">
                    <div>
                        <div class="metric-value">45</div>
                        <div class="metric-label">Orders Submitted</div>
                    </div>
                    <div>
                        <div class="metric-value">40</div>
                        <div class="metric-label">Orders Filled</div>
                    </div>
                    <div>
                        <div class="metric-value">3</div>
                        <div class="metric-label">Orders Cancelled</div>
                    </div>
                    <div>
                        <div class="metric-value">2</div>
                        <div class="metric-label">Orders Rejected</div>
                    </div>
                    <div>
                        <div class="metric-value">250 ms</div>
                        <div class="metric-label">Avg Fill Time</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="refresh-message">
            <p>Last updated: {{current_time}} | Refresh the page to update data</p>
        </div>
    </div>
</body>
</html>
"""

# Create the template file
with open(templates_dir / "index.html", "w") as f:
    f.write(index_template)

# Alert levels
ALERT_LEVELS = ["info", "warning", "error", "critical"]

# Alert categories
ALERT_CATEGORIES = ["system", "exchange", "order", "position", "strategy", "risk", "security"]

# Create FastAPI app
app = FastAPI(title="Trading System Dashboard")

# Set up templates
templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")
    
# Include sentiment dashboard router if available
if sentiment_router:
    app.include_router(sentiment_router)
    logger.info("Sentiment dashboard mounted at /sentiment")

# Mock alert data
def generate_mock_alerts(count=10):
    """Generate mock alerts for demonstration."""
    alerts = []
    
    now = datetime.now()
    
    # System startup alert
    alerts.append({
        "id": str(uuid.uuid4()),
        "level": "info",
        "category": "system",
        "source": "monitoring.service",
        "message": "System started successfully",
        "time": (now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
        "details": None
    })
    
    # Add random alerts
    for i in range(count - 1):
        # Determine time (more recent for more severe alerts)
        severity = random.randint(0, 3)  # 0=info, 1=warning, 2=error, 3=critical
        time_offset = random.randint(1, 30) - (severity * 5)
        if time_offset < 1:
            time_offset = 1
        
        alert_time = now - timedelta(minutes=time_offset)
        
        # Generate alert
        level = ALERT_LEVELS[severity]
        category = random.choice(ALERT_CATEGORIES)
        
        if category == "exchange":
            source = random.choice(["binance", "coinbase", "kraken"])
            if level == "info":
                message = f"Connected to {source} exchange"
                details = None
            elif level == "warning":
                message = f"High latency detected on {source} exchange"
                details = "Latency: 450ms (threshold: 300ms)"
            elif level == "error":
                message = f"API request failed on {source} exchange"
                details = "Error: Rate limit exceeded"
            else:  # critical
                message = f"Connection lost to {source} exchange"
                details = "Attempting reconnection (attempt 3/5)"
                
        elif category == "order":
            order_id = f"order-{random.randint(1000, 9999)}"
            source = "execution.service"
            if level == "info":
                message = f"Order {order_id} submitted successfully"
                details = None
            elif level == "warning":
                message = f"Partial fill for order {order_id}"
                details = "Filled: 0.5 BTC of 1.0 BTC"
            elif level == "error":
                message = f"Order {order_id} rejected by exchange"
                details = "Reason: Insufficient funds"
            else:  # critical
                message = f"Order execution failed: {order_id}"
                details = "Error: Exchange not responding to order status request"
                
        elif category == "strategy":
            strategy_id = f"strategy-{random.randint(100, 999)}"
            source = f"strategy.{strategy_id}"
            if level == "info":
                message = f"Strategy {strategy_id} started"
                details = None
            elif level == "warning":
                message = f"Strategy {strategy_id} performance degraded"
                details = "Win rate: 45% (threshold: 50%)"
            elif level == "error":
                message = f"Strategy {strategy_id} failed to calculate signal"
                details = "Error: Division by zero in indicator calculation"
            else:  # critical
                message = f"Strategy {strategy_id} stopped due to critical error"
                details = "Error: Maximum drawdown exceeded"
                
        else:
            source = "system.monitor"
            if level == "info":
                message = "System performing routine maintenance"
                details = None
            elif level == "warning":
                message = "High memory usage detected"
                details = "Memory usage: 85% (threshold: 80%)"
            elif level == "error":
                message = "Database connection error"
                details = "Error: Connection timeout after 30 seconds"
            else:  # critical
                message = "Critical system error detected"
                details = "Error: Insufficient disk space for operation"
        
        alerts.append({
            "id": str(uuid.uuid4()),
            "level": level,
            "category": category,
            "source": source,
            "message": message,
            "time": alert_time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": details
        })
    
    # Sort by time (newest first)
    alerts.sort(key=lambda a: a["time"], reverse=True)
    return alerts

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Render the dashboard."""
    uptime = (datetime.now() - datetime(2025, 3, 1, 12, 0, 0)).total_seconds()
    
    # Generate some mock alerts
    alerts = generate_mock_alerts(8)
    
    # Count alerts by level
    alert_counts = {level: 0 for level in ALERT_LEVELS}
    for alert in alerts:
        alert_counts[alert["level"]] += 1
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "uptime_seconds": int(uptime),
            "error_count": random.randint(0, 5),
            "trade_time_1": (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "trade_time_2": (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alerts": alerts,
            "alert_counts": alert_counts
        }
    )

@app.get("/risk", response_class=HTMLResponse)
async def risk_management(request: Request):
    """
    Display the risk management dashboard.
    
    This shows real-time risk utilization across strategies and assets,
    as well as risk alerts and optimization opportunities.
    """
    try:
        # In a real implementation, this would fetch data from the risk management system
        # For now, we use placeholder data (this matches the HTML template)
        risk_data = {
            "timestamp": datetime.now().isoformat(),
            "system_risk": {
                "total_budget": 5.0,
                "current_risk": 2.8,
                "utilization": 56.0,
                "available_risk": 2.2
            },
            "strategies": {
                "trend_following": {
                    "max_risk": 2.5,
                    "current_risk": 1.4,
                    "utilization": 56.0,
                    "status": "low",
                    "markets": {
                        "crypto": {
                            "max_risk": 1.5,
                            "current_risk": 0.9,
                            "utilization": 60.0,
                            "assets": {
                                "BTC": {"max_risk": 0.6, "current_risk": 0.4, "utilization": 67.0},
                                "ETH": {"max_risk": 0.5, "current_risk": 0.3, "utilization": 60.0},
                                "SOL": {"max_risk": 0.4, "current_risk": 0.2, "utilization": 50.0}
                            }
                        },
                        "forex": {
                            "max_risk": 1.0,
                            "current_risk": 0.5,
                            "utilization": 50.0,
                            "assets": {
                                "EUR/USD": {"max_risk": 0.6, "current_risk": 0.3, "utilization": 50.0},
                                "GBP/USD": {"max_risk": 0.4, "current_risk": 0.2, "utilization": 50.0}
                            }
                        }
                    }
                },
                "mean_reversion": {
                    "max_risk": 1.5,
                    "current_risk": 1.2,
                    "utilization": 80.0,
                    "status": "medium",
                    "markets": {
                        "crypto": {
                            "max_risk": 1.0,
                            "current_risk": 0.9,
                            "utilization": 90.0,
                            "assets": {
                                "BTC": {"max_risk": 0.4, "current_risk": 0.4, "utilization": 100.0},
                                "ETH": {"max_risk": 0.3, "current_risk": 0.3, "utilization": 100.0},
                                "LINK": {"max_risk": 0.3, "current_risk": 0.2, "utilization": 67.0}
                            }
                        },
                        "commodities": {
                            "max_risk": 0.5,
                            "current_risk": 0.3,
                            "utilization": 60.0,
                            "assets": {
                                "GOLD": {"max_risk": 0.3, "current_risk": 0.2, "utilization": 67.0},
                                "SILVER": {"max_risk": 0.2, "current_risk": 0.1, "utilization": 50.0}
                            }
                        }
                    }
                },
                "breakout": {
                    "max_risk": 1.0,
                    "current_risk": 0.2,
                    "utilization": 20.0,
                    "status": "low",
                    "markets": {
                        "crypto": {
                            "max_risk": 1.0,
                            "current_risk": 0.2,
                            "utilization": 20.0,
                            "assets": {
                                "BTC": {"max_risk": 0.5, "current_risk": 0.1, "utilization": 20.0},
                                "ETH": {"max_risk": 0.3, "current_risk": 0.1, "utilization": 33.0},
                                "AVAX": {"max_risk": 0.2, "current_risk": 0.0, "utilization": 0.0}
                            }
                        }
                    }
                }
            },
            "alerts": [
                {
                    "level": "warning",
                    "message": "Strategy 'Mean Reversion' risk utilization at 80.0%"
                },
                {
                    "level": "critical",
                    "message": "Asset 'BTC' in 'Mean Reversion' strategy has reached 100% risk utilization"
                },
                {
                    "level": "critical",
                    "message": "Asset 'ETH' in 'Mean Reversion' strategy has reached 100% risk utilization"
                }
            ]
        }
        
        # Serve the risk management HTML template
        return templates.TemplateResponse("risk_management.html", {"request": request, "risk_data": risk_data})
    except Exception as e:
        return HTMLResponse(f"<h1>Error</h1><p>Failed to load risk management dashboard: {str(e)}</p>")

def main():
    """Run the dashboard server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the trading system monitoring dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development mode)")
    args = parser.parse_args()
    
    # Log startup information
    print(f"Starting dashboard on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the server
    uvicorn.run(
        app, 
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 