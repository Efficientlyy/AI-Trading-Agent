"""
AI Trading Agent - Standalone Monitoring Dashboard

This dashboard provides a simplified view of the trading system's status,
with real-time monitoring of system components, active orders, recent trades,
and alerts.
"""

import random
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path

# Create necessary directories
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("static/css").mkdir(exist_ok=True)

# Create FastAPI app
app = FastAPI(title="AI Trading Agent Monitoring")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a basic CSS file
css_content = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f7fa;
    color: #333;
}

.dashboard {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
}

.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    position: relative;
}

.card h2 {
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 1.2rem;
    color: #2c3e50;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
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

.component-status {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    padding: 8px;
    border-radius: 4px;
    background-color: #f8f9fa;
}

.timestamp {
    font-size: 0.8rem;
    color: #7f8c8d;
    position: absolute;
    right: 20px;
    top: 20px;
}

.metrics-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
}

.metric {
    flex: 1;
    text-align: center;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 4px;
    margin: 0 5px;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: bold;
    color: #3498db;
}

.metric-name {
    font-size: 0.8rem;
    color: #7f8c8d;
}

.trades-list, .orders-list {
    max-height: 300px;
    overflow-y: auto;
}

.trade-item, .order-item {
    padding: 10px;
    border-bottom: 1px solid #eee;
}

.trade-item:last-child, .order-item:last-child {
    border-bottom: none;
}

.badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: bold;
}

.badge-buy {
    background-color: rgba(46, 204, 113, 0.2);
    color: #27ae60;
}

.badge-sell {
    background-color: rgba(231, 76, 60, 0.2);
    color: #c0392b;
}

.badge-filled {
    background-color: rgba(46, 204, 113, 0.2);
    color: #27ae60;
}

.badge-partial {
    background-color: rgba(241, 196, 15, 0.2);
    color: #f39c12;
}

.badge-pending {
    background-color: rgba(52, 152, 219, 0.2);
    color: #2980b9;
}

.badge-cancelled {
    background-color: rgba(127, 140, 141, 0.2);
    color: #7f8c8d;
}

.badge-rejected {
    background-color: rgba(231, 76, 60, 0.2);
    color: #c0392b;
}

/* Alert system styles */
.alert {
    padding: 10px 15px;
    border-radius: 4px;
    margin-bottom: 10px;
    position: relative;
}

.alert-info {
    background-color: rgba(52, 152, 219, 0.2);
    border-left: 4px solid #3498db;
}

.alert-warning {
    background-color: rgba(241, 196, 15, 0.2);
    border-left: 4px solid #f39c12;
}

.alert-error {
    background-color: rgba(231, 76, 60, 0.2);
    border-left: 4px solid #e74c3c;
}

.alert-critical {
    background-color: rgba(231, 76, 60, 0.3);
    border-left: 4px solid #c0392b;
}

.alert-counters {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
}

.alert-count {
    padding: 8px 12px;
    border-radius: 4px;
    text-align: center;
    flex: 1;
    margin: 0 5px;
}

.alert-count-critical {
    background-color: rgba(192, 57, 43, 0.2);
}

.alert-count-error {
    background-color: rgba(231, 76, 60, 0.2);
}

.alert-count-warning {
    background-color: rgba(243, 156, 18, 0.2);
}

.alert-count-info {
    background-color: rgba(52, 152, 219, 0.2);
}

.alert-category {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
    text-transform: uppercase;
    margin-right: 5px;
    background-color: rgba(0, 0, 0, 0.1);
}

.alert-time {
    font-size: 0.7rem;
    color: #7f8c8d;
    float: right;
}

.alerts-list {
    max-height: 300px;
    overflow-y: auto;
}
"""

Path("static/css/dashboard.css").write_text(css_content)

# Create HTML template
html_template = """<!DOCTYPE html>
<html>
<head>
    <title>AI Trading Agent Monitoring</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', path='/css/dashboard.css') }}">
</head>
<body>
    <h1>AI Trading Agent Monitoring Dashboard</h1>
    <p>Last updated: {{ timestamp }}</p>
    
    <div class="dashboard">
        <!-- System Status Card -->
        <div class="card">
            <h2>System Status</h2>
            <span class="timestamp">{{ timestamp }}</span>
            
            <div class="metrics-row">
                <div class="metric">
                    <div class="metric-value">{{ uptime }}</div>
                    <div class="metric-name">Uptime</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ error_count }}</div>
                    <div class="metric-name">Errors</div>
                </div>
            </div>
            
            <h3>Components</h3>
            {% for component in components %}
            <div class="component-status">
                <div>
                    <span class="status-indicator status-{{ component.status.lower() }}"></span>
                    {{ component.name }}
                </div>
                <div>{{ component.status }}</div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Alert Card -->
        <div class="card">
            <h2>System Alerts</h2>
            <span class="timestamp">{{ timestamp }}</span>
            
            <div class="alert-counters">
                <div class="alert-count alert-count-critical">
                    <div class="metric-value">{{ alert_counts.critical }}</div>
                    <div class="metric-name">Critical</div>
                </div>
                <div class="alert-count alert-count-error">
                    <div class="metric-value">{{ alert_counts.error }}</div>
                    <div class="metric-name">Error</div>
                </div>
                <div class="alert-count alert-count-warning">
                    <div class="metric-value">{{ alert_counts.warning }}</div>
                    <div class="metric-name">Warning</div>
                </div>
                <div class="alert-count alert-count-info">
                    <div class="metric-value">{{ alert_counts.info }}</div>
                    <div class="metric-name">Info</div>
                </div>
            </div>
            
            <div class="alerts-list">
                {% for alert in alerts %}
                <div class="alert alert-{{ alert.level }}">
                    <span class="alert-category">{{ alert.category }}</span>
                    <span class="alert-time">{{ alert.time }}</span>
                    <div>{{ alert.message }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Active Orders Card -->
        <div class="card">
            <h2>Active Orders</h2>
            <span class="timestamp">{{ timestamp }}</span>
            
            <div class="orders-list">
                {% for order in orders %}
                <div class="order-item">
                    <div>
                        <span class="badge badge-{{ order.side.lower() }}">{{ order.side }}</span>
                        <span class="badge badge-{{ order.status.lower() }}">{{ order.status }}</span>
                        {{ order.symbol }}
                    </div>
                    <div>
                        <small>ID: {{ order.id }}</small> | 
                        <small>Price: ${{ order.price }}</small> | 
                        <small>Qty: {{ order.quantity }}</small>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Recent Trades Card -->
        <div class="card">
            <h2>Recent Trades</h2>
            <span class="timestamp">{{ timestamp }}</span>
            
            <div class="trades-list">
                {% for trade in trades %}
                <div class="trade-item">
                    <div>
                        <span class="badge badge-{{ trade.side.lower() }}">{{ trade.side }}</span>
                        {{ trade.symbol }}
                    </div>
                    <div>
                        <small>{{ trade.time }}</small> | 
                        <small>Price: ${{ trade.price }}</small> | 
                        <small>Qty: {{ trade.quantity }}</small>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <!-- Performance Metrics Card -->
        <div class="card">
            <h2>Execution Metrics</h2>
            <span class="timestamp">{{ timestamp }}</span>
            
            <div class="metrics-row">
                <div class="metric">
                    <div class="metric-value">{{ metrics.slippage }}%</div>
                    <div class="metric-name">Avg. Slippage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{ metrics.fill_rate }}%</div>
                    <div class="metric-name">Fill Rate</div>
                </div>
            </div>
            
            <div class="metrics-row">
                <div class="metric">
                    <div class="metric-value">{{ metrics.latency }}ms</div>
                    <div class="metric-name">Avg. Latency</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${{ metrics.fees }}</div>
                    <div class="metric-name">Daily Fees</div>
                </div>
            </div>
        </div>
    </div>
    
    <footer>
        <p>AI Trading Agent Monitoring Dashboard | Refresh page to update data</p>
    </footer>
</body>
</html>
"""

Path("templates/dashboard.html").write_text(html_template)

# Alert levels and categories for mock data
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

def generate_mock_alerts(count: int = 10) -> List[Dict[str, Any]]:
    """Generate mock alerts for the dashboard"""
    alerts = []
    now = datetime.now()
    
    # Create some predefined alert scenarios
    alert_scenarios = [
        {
            "level": AlertLevel.INFO,
            "category": AlertCategory.SYSTEM,
            "message": "System started successfully",
            "time_offset": timedelta(minutes=30)
        },
        {
            "level": AlertLevel.WARNING,
            "category": AlertCategory.EXCHANGE,
            "message": "High latency detected on Binance exchange",
            "time_offset": timedelta(minutes=15)
        },
        {
            "level": AlertLevel.ERROR,
            "category": AlertCategory.ORDER,
            "message": "Order rejected: Insufficient funds",
            "time_offset": timedelta(minutes=10)
        },
        {
            "level": AlertLevel.CRITICAL,
            "category": AlertCategory.RISK,
            "message": "Critical margin usage: 92.5%",
            "time_offset": timedelta(minutes=5)
        },
        {
            "level": AlertLevel.WARNING,
            "category": AlertCategory.POSITION,
            "message": "Large position concentration in BTC: 55%",
            "time_offset": timedelta(minutes=8)
        },
        {
            "level": AlertLevel.INFO,
            "category": AlertCategory.STRATEGY,
            "message": "Strategy performance below threshold: 2.1%",
            "time_offset": timedelta(minutes=20)
        },
        {
            "level": AlertLevel.ERROR,
            "category": AlertCategory.EXCHANGE,
            "message": "API rate limit exceeded on Kraken",
            "time_offset": timedelta(minutes=12)
        }
    ]
    
    # Add predefined scenarios
    for scenario in alert_scenarios[:min(count, len(alert_scenarios))]:
        alert_time = now - scenario["time_offset"]
        alerts.append({
            "id": str(uuid.uuid4()),
            "level": scenario["level"],
            "category": scenario["category"],
            "message": scenario["message"],
            "time": alert_time.strftime("%H:%M:%S"),
            "timestamp": alert_time
        })
    
    # Add random alerts if needed
    while len(alerts) < count:
        level = random.choice(list(AlertLevel))
        category = random.choice(list(AlertCategory))
        
        # Time between 1 minute and 2 hours ago
        minutes_ago = random.randint(1, 120)
        alert_time = now - timedelta(minutes=minutes_ago)
        
        # Generate appropriate message based on level and category
        if level == AlertLevel.INFO:
            if category == AlertCategory.SYSTEM:
                message = "Component health check passed"
            elif category == AlertCategory.EXCHANGE:
                message = f"Connected to {random.choice(['Binance', 'Coinbase', 'Kraken'])} successfully"
            else:
                message = f"{category.capitalize()} status normal"
        
        elif level == AlertLevel.WARNING:
            if category == AlertCategory.EXCHANGE:
                message = f"Elevated latency on {random.choice(['Binance', 'Coinbase', 'Kraken'])}: {random.randint(200, 500)}ms"
            elif category == AlertCategory.RISK:
                message = f"Margin usage approaching limit: {random.randint(70, 80)}%"
            else:
                message = f"{category.capitalize()} warning condition detected"
        
        elif level == AlertLevel.ERROR:
            if category == AlertCategory.ORDER:
                message = f"Order {random.randint(1000, 9999)} execution failed"
            elif category == AlertCategory.EXCHANGE:
                message = f"Connection error on {random.choice(['Binance', 'Coinbase', 'Kraken'])}"
            else:
                message = f"{category.capitalize()} error occurred"
        
        elif level == AlertLevel.CRITICAL:
            if category == AlertCategory.SYSTEM:
                message = "Critical system error: database connection lost"
            elif category == AlertCategory.SECURITY:
                message = "Possible unauthorized access attempt detected"
            else:
                message = f"Critical {category} failure"
        
        alerts.append({
            "id": str(uuid.uuid4()),
            "level": level,
            "category": category,
            "message": message,
            "time": alert_time.strftime("%H:%M:%S"),
            "timestamp": alert_time
        })
    
    # Sort by time (newest first)
    alerts.sort(key=lambda x: x["timestamp"], reverse=True)
    return alerts


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the monitoring dashboard"""
    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Simulate uptime (random hours)
    hours = random.randint(5, 120)
    uptime = f"{hours}h {random.randint(1, 59)}m"
    
    # Random error count
    error_count = random.randint(0, 5)
    
    # Component statuses
    components = [
        {"name": "Data Collector", "status": random.choice(["OK", "OK", "OK", "Warning"])},
        {"name": "Strategy Engine", "status": random.choice(["OK", "OK", "Warning"])},
        {"name": "Order Manager", "status": random.choice(["OK", "OK", "OK", "Error"])},
        {"name": "Risk Manager", "status": random.choice(["OK", "OK", "OK", "OK"])},
        {"name": "Exchange Connector", "status": random.choice(["OK", "OK", "Warning"])},
    ]
    
    # Generate alerts
    alerts = generate_mock_alerts(10)
    
    # Alert counts
    alert_counts = {
        "critical": len([a for a in alerts if a["level"] == AlertLevel.CRITICAL]),
        "error": len([a for a in alerts if a["level"] == AlertLevel.ERROR]),
        "warning": len([a for a in alerts if a["level"] == AlertLevel.WARNING]),
        "info": len([a for a in alerts if a["level"] == AlertLevel.INFO]),
    }
    
    # Active orders
    orders = []
    for i in range(5):
        orders.append({
            "id": f"ORD-{random.randint(100000, 999999)}",
            "symbol": random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]),
            "side": random.choice(["BUY", "SELL"]),
            "status": random.choice(["PENDING", "PARTIAL", "FILLED"]),
            "price": round(random.uniform(100, 60000), 2),
            "quantity": round(random.uniform(0.1, 10), 4)
        })
    
    # Recent trades
    trades = []
    for i in range(7):
        # Time between now and 3 hours ago
        trade_time = datetime.now() - timedelta(hours=random.random() * 3)
        
        trades.append({
            "symbol": random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]),
            "side": random.choice(["BUY", "SELL"]),
            "price": round(random.uniform(100, 60000), 2),
            "quantity": round(random.uniform(0.1, 10), 4),
            "time": trade_time.strftime("%H:%M:%S")
        })
    
    # Sort trades by time (most recent first)
    trades.sort(key=lambda x: x["time"], reverse=True)
    
    # Performance metrics
    metrics = {
        "slippage": round(random.uniform(0.01, 0.3), 2),
        "fill_rate": round(random.uniform(90, 99.9), 1),
        "latency": round(random.uniform(50, 300)),
        "fees": round(random.uniform(5, 50), 2)
    }
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "timestamp": timestamp,
            "uptime": uptime,
            "error_count": error_count,
            "components": components,
            "alerts": alerts,
            "alert_counts": alert_counts,
            "orders": orders,
            "trades": trades,
            "metrics": metrics
        }
    )

def main():
    """Run the dashboard server"""
    uvicorn.run(app, host="127.0.0.1", port=8080)

if __name__ == "__main__":
    main() 