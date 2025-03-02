"""
FastAPI implementation of the monitoring dashboard API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
import random

# Create FastAPI app
app = FastAPI(title="Trading System Monitoring API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# Static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Create a basic HTML template
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
                    <p><strong>Uptime:</strong> {{status.system_stats.uptime_seconds}} seconds</p>
                    <p><strong>Error Count:</strong> {{status.system_stats.error_count}}</p>
                </div>
                <h3>Components</h3>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Status</th>
                        <th>Message</th>
                    </tr>
                    {% for component_id, component in status.component_status.items() %}
                    <tr>
                        <td>{{component_id}}</td>
                        <td>
                            <span class="status-indicator status-{{component.status}}"></span>
                            {{component.status}}
                        </td>
                        <td>{{component.message}}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            <!-- Active Orders Card -->
            <div class="card">
                <h2>Active Orders</h2>
                {% if active_orders %}
                <table>
                    <tr>
                        <th>Order ID</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Status</th>
                    </tr>
                    {% for order in active_orders %}
                    <tr>
                        <td>{{order.order_id}}</td>
                        <td>{{order.symbol}}</td>
                        <td>{{order.side}}</td>
                        <td>{{order.status}}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% else %}
                <p>No active orders</p>
                {% endif %}
            </div>

            <!-- Recent Trades Card -->
            <div class="card">
                <h2>Recent Trades</h2>
                {% if recent_trades %}
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Price</th>
                    </tr>
                    {% for trade in recent_trades %}
                    <tr>
                        <td>{{trade.time}}</td>
                        <td>{{trade.symbol}}</td>
                        <td>{{trade.side}}</td>
                        <td>{{trade.price}}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% else %}
                <p>No recent trades</p>
                {% endif %}
            </div>

            <!-- Performance Metrics Card -->
            <div class="card">
                <h2>Execution Metrics</h2>
                <div class="metric-grid">
                    <div>
                        <div class="metric-value">{{metrics.performance_metrics.execution_metrics.orders_submitted}}</div>
                        <div class="metric-label">Orders Submitted</div>
                    </div>
                    <div>
                        <div class="metric-value">{{metrics.performance_metrics.execution_metrics.orders_filled}}</div>
                        <div class="metric-label">Orders Filled</div>
                    </div>
                    <div>
                        <div class="metric-value">{{metrics.performance_metrics.execution_metrics.orders_cancelled}}</div>
                        <div class="metric-label">Orders Cancelled</div>
                    </div>
                    <div>
                        <div class="metric-value">{{metrics.performance_metrics.execution_metrics.orders_rejected}}</div>
                        <div class="metric-label">Orders Rejected</div>
                    </div>
                    <div>
                        <div class="metric-value">{{metrics.performance_metrics.execution_metrics.avg_fill_time_ms}} ms</div>
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

# Save the template
with open(templates_dir / "index.html", "w") as f:
    f.write(index_template)

# Setup templates
templates = Jinja2Templates(directory=templates_dir)

# Mock data
def get_system_status():
    return {
        "system_stats": {
            "uptime_seconds": (datetime.now() - datetime(2025, 3, 1, 12, 0, 0)).total_seconds(),
            "error_count": random.randint(0, 5)
        },
        "component_status": {
            "execution": {"status": "ok", "message": "Running normally"},
            "data": {"status": "warning", "message": "Experiencing delays"},
            "strategy": {"status": "error", "message": "Encountered an error"}
        }
    }

def get_active_orders():
    return [
        {"order_id": "test-order-123", "symbol": "BTC/USDT", "side": "BUY", "status": "FILLED"},
        {"order_id": "test-order-124", "symbol": "ETH/USDT", "side": "SELL", "status": "OPEN"}
    ]

def get_recent_trades():
    return [
        {"time": (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), "symbol": "BTC/USDT", "side": "BUY", "price": "49950.00"},
        {"time": (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"), "symbol": "ETH/USDT", "side": "SELL", "price": "2950.75"}
    ]

def get_performance_metrics():
    return {
        "performance_metrics": {
            "execution_metrics": {
                "orders_submitted": 45,
                "orders_filled": 40,
                "orders_cancelled": 3,
                "orders_rejected": 2,
                "avg_fill_time_ms": 250
            }
        }
    }

@app.get("/")
async def root(request: Request):
    """Render the dashboard homepage"""
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "status": get_system_status(),
            "active_orders": get_active_orders(),
            "recent_trades": get_recent_trades(),
            "metrics": get_performance_metrics(),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

@app.get("/api/status")
async def api_status():
    """Get system status and component health"""
    return JSONResponse(get_system_status())

@app.get("/api/orders/active")
async def api_active_orders():
    """Get active orders"""
    return JSONResponse(get_active_orders())

@app.get("/api/trades/recent")
async def api_recent_trades():
    """Get recent trades"""
    return JSONResponse(get_recent_trades())

@app.get("/api/metrics")
async def api_metrics():
    """Get performance metrics"""
    return JSONResponse(get_performance_metrics())

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}") 