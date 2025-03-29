"""
AI Trading Agent - Minimal Dashboard

A lightweight dashboard that demonstrates the key features of our trading system
while ensuring Python 3.13 compatibility.
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

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

# Check if Flask is installed
try:
    from flask import Flask, render_template_string, jsonify
except ImportError:
    print("Error: Flask is not installed. Installing it now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    from flask import Flask, render_template_string, jsonify

# Create Flask app
app = Flask(__name__)

# Constants for mock data
MARKET_REGIMES = ["Bull Market", "Bear Market", "Sideways Market", "Volatile Market", "Recovery Market", "Crash Market"]
SENTIMENT_SOURCES = ["Social Media", "News", "On-Chain", "Overall"]

# Mock data generation functions
def generate_mock_system_data():
    """Generate mock system status data."""
    components = ["API", "Database", "Exchange", "Model", "Data Feeds"]
    statuses = ["OK", "Warning", "Error"]
    weights = [0.8, 0.15, 0.05]  # Most components should be OK
    
    return {
        component: random.choices(statuses, weights=weights, k=1)[0]
        for component in components
    }

def generate_mock_sentiment_data():
    """Generate mock sentiment data following realistic market patterns."""
    # Create sentiment values with some correlation between sources
    base_sentiment = random.uniform(0.4, 0.7)  # Base sentiment between 40% and 70%
    
    return {
        source: round(
            max(0, min(1, base_sentiment + random.uniform(-0.15, 0.15))),
            2
        ) * 100
        for source in SENTIMENT_SOURCES
    }

def generate_mock_risk_data():
    """Generate mock risk management metrics."""
    # Generate correlated risk metrics
    volatility = random.uniform(0.1, 0.3)  # 10% to 30%
    
    return {
        "VaR (95%)": f"${round(random.uniform(1000, 5000), 0):.0f}",
        "CVaR": f"${round(random.uniform(1500, 6000), 0):.0f}",
        "Max Drawdown": f"-{round(random.uniform(0.05, 0.15), 2) * 100:.1f}%",
        "Volatility": f"{volatility * 100:.1f}%",
        "Sharpe Ratio": f"{random.uniform(0.8, 2.5):.2f}",
    }

def generate_mock_market_regime_data():
    """Generate mock market regime data."""
    current_regime = random.choice(MARKET_REGIMES)
    confidence = random.uniform(0.7, 0.95)
    trends = ["Rising", "Falling", "Stable", "Volatile"]
    
    return {
        "Current Regime": current_regime,
        "Confidence": f"{confidence * 100:.1f}%",
        "Trend": random.choice(trends),
        "Duration": f"{random.randint(1, 30)} days",
    }

def generate_mock_performance_data():
    """Generate mock performance metrics."""
    # Generate somewhat realistic and correlated performance metrics
    daily_pl = random.uniform(-0.03, 0.035)
    
    # Weekly P/L should somewhat align with daily
    weekly_pl_direction = 1 if daily_pl > 0 else -1
    weekly_pl = weekly_pl_direction * random.uniform(0.02, 0.08)
    
    # Sharpe and drawdown should be somewhat related to P/L
    sharpe = 1.0 + (0.5 * weekly_pl_direction * random.uniform(0.3, 1.0))
    drawdown = -1 * random.uniform(0.01, 0.06)
    
    return {
        "Daily P/L": f"{daily_pl * 100:+.1f}%",
        "Weekly P/L": f"{weekly_pl * 100:+.1f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{drawdown * 100:.1f}%",
    }

def generate_mock_continuous_improvement_data():
    """Generate mock continuous improvement experiment data."""
    # Experiment types
    experiment_types = [
        "Bayesian sentiment classifier vs. Neural network",
        "LSTM vs. Transformer for price prediction",
        "Reinforcement Learning vs. Rule-based for order execution",
        "Random Forest vs. XGBoost for regime detection",
    ]
    variants = ["A", "B"]
    
    # Generate experiment data with stopping criteria info
    return {
        "Last Experiment": random.choice(experiment_types),
        "Winner": f"Variant {random.choice(variants)}",
        "Confidence": f"{random.uniform(0.7, 0.98) * 100:.1f}%",
        "Samples": random.randint(100, 500),
        "Stopping Criteria": random.choice([
            "Bayesian probability threshold",
            "Expected loss threshold",
            "Confidence interval width",
            "Sample size reached",
            "Time limit reached"
        ])
    }

# Generate all dashboard data
def generate_dashboard_data():
    """Generate all mock data for the dashboard."""
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_status": generate_mock_system_data(),
        "sentiment": generate_mock_sentiment_data(),
        "risk": generate_mock_risk_data(),
        "market_regime": generate_mock_market_regime_data(),
        "performance": generate_mock_performance_data(),
        "continuous_improvement": generate_mock_continuous_improvement_data(),
    }

# HTML template
DASHBOARD_TEMPLATE = """
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
        .status { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
        .status-item { display: flex; align-items: center; padding: 8px 12px; border-radius: 4px; }
        .status-circle { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-text { font-weight: bold; }
        .OK { background-color: #e8f5e9; }
        .OK .status-circle { background-color: #28a745; }
        .Warning { background-color: #fff8e1; }
        .Warning .status-circle { background-color: #ffc107; }
        .Error { background-color: #ffebee; }
        .Error .status-circle { background-color: #dc3545; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
        .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 4px; }
        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .metric-title { font-size: 14px; color: #666; }
        .section { margin-bottom: 30px; }
        .experiment-box { background: #f0f8ff; padding: 15px; border-radius: 4px; }
        .reload-btn { padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .reload-btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>AI Trading Agent Dashboard</h1>
            <div>
                <button class="reload-btn" onclick="reloadData()">Refresh Data</button>
                <div>Last Updated: {{ data.timestamp }}</div>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="main">Main</div>
            <div class="tab" data-tab="sentiment">Sentiment Analysis</div>
            <div class="tab" data-tab="risk">Risk Management</div>
            <div class="tab" data-tab="regime">Market Regime</div>
        </div>
        
        <div class="panel active" id="main">
            <div class="section">
                <h2>System Status</h2>
                <div class="status">
                    {% for component, status in data.system_status.items() %}
                    <div class="status-item {{ status }}">
                        <div class="status-circle"></div>
                        <div class="status-text">{{ component }}: {{ status }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="grid">
                    {% for title, value in data.performance.items() %}
                    <div class="metric">
                        <div class="metric-title">{{ title }}</div>
                        <div class="metric-value">{{ value }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="panel" id="sentiment">
            <div class="section">
                <h2>Sentiment Overview</h2>
                <div class="grid">
                    {% for source, value in data.sentiment.items() %}
                    <div class="metric">
                        <div class="metric-title">{{ source }}</div>
                        <div class="metric-value">{{ value }}%</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="section">
                <h2>Continuous Improvement</h2>
                <div class="experiment-box">
                    <h3>Latest Experiment</h3>
                    <p><strong>Experiment:</strong> {{ data.continuous_improvement.Last_Experiment }}</p>
                    <p><strong>Winner:</strong> {{ data.continuous_improvement.Winner }}</p>
                    <p><strong>Confidence:</strong> {{ data.continuous_improvement.Confidence }}</p>
                    <p><strong>Samples:</strong> {{ data.continuous_improvement.Samples }}</p>
                    <p><strong>Stopping Criteria Used:</strong> {{ data.continuous_improvement.Stopping_Criteria }}</p>
                </div>
            </div>
        </div>
        
        <div class="panel" id="risk">
            <div class="section">
                <h2>Risk Metrics</h2>
                <div class="grid">
                    {% for title, value in data.risk.items() %}
                    <div class="metric">
                        <div class="metric-title">{{ title }}</div>
                        <div class="metric-value">{{ value }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="panel" id="regime">
            <div class="section">
                <h2>Current Market Regime</h2>
                <div class="grid">
                    {% for title, value in data.market_regime.items() %}
                    <div class="metric">
                        <div class="metric-title">{{ title }}</div>
                        <div class="metric-value">{{ value }}</div>
                    </div>
                    {% endfor %}
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
        
        function reloadData() {
            window.location.reload();
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the dashboard."""
    data = generate_dashboard_data()
    return render_template_string(DASHBOARD_TEMPLATE, data=data)

@app.route('/api/data')
def api_data():
    """API endpoint to get dashboard data."""
    return jsonify(generate_dashboard_data())

def main():
    """Run the dashboard server."""
    host = "127.0.0.1"
    port = 8000
    
    print(f"Starting AI Trading Agent Dashboard at http://{host}:{port}")
    print("Press Ctrl+C to stop the dashboard")
    
    app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
    main()
