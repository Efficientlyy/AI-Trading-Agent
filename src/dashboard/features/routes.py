"""
Dashboard Feature Modules

This module provides the dashboard feature modules for the AI Trading Agent.
It includes views for different dashboard tabs and their associated functionality.
"""

import logging
from typing import Dict, Any, List
from flask import Blueprint, render_template, request, jsonify, session

from src.dashboard.utils.auth import login_required, role_required
from src.dashboard.data.service import DataServiceFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dashboard_features")

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__)

# Initialize data service
data_service = None

@dashboard_bp.route('/')
@dashboard_bp.route('/overview')
@login_required
def index():
    """Render the main dashboard overview page."""
    return render_template('dashboard/overview.html', 
                          user=session.get('user'),
                          active_tab='overview')

@dashboard_bp.route('/sentiment')
@login_required
def sentiment():
    """Render the sentiment analysis dashboard page."""
    return render_template('dashboard/sentiment.html', 
                          user=session.get('user'),
                          active_tab='sentiment')

@dashboard_bp.route('/market-regime')
@login_required
def market_regime():
    """Render the market regime dashboard page."""
    return render_template('dashboard/market_regime.html', 
                          user=session.get('user'),
                          active_tab='market_regime')

@dashboard_bp.route('/risk-management')
@login_required
def risk_management():
    """Render the risk management dashboard page."""
    return render_template('dashboard/risk_management.html', 
                          user=session.get('user'),
                          active_tab='risk_management')

@dashboard_bp.route('/performance')
@login_required
def performance():
    """Render the performance analytics dashboard page."""
    return render_template('dashboard/performance.html', 
                          user=session.get('user'),
                          active_tab='performance')

@dashboard_bp.route('/logs')
@login_required
def logs():
    """Render the logs & monitoring dashboard page."""
    return render_template('dashboard/logs.html', 
                          user=session.get('user'),
                          active_tab='logs')

@dashboard_bp.route('/status')
@login_required
def status():
    """Render the system status dashboard page."""
    return render_template('dashboard/status.html', 
                          user=session.get('user'),
                          active_tab='status')

@dashboard_bp.route('/admin')
@login_required
@role_required(['admin'])
def admin():
    """Render the admin controls dashboard page."""
    return render_template('dashboard/admin.html', 
                          user=session.get('user'),
                          active_tab='admin')

@dashboard_bp.route('/data-validation')
@login_required
def data_validation():
    """Render the data validation dashboard page."""
    return render_template('dashboard/data_validation.html', 
                          user=session.get('user'),
                          active_tab='data_validation')

@dashboard_bp.route('/data-transformation')
@login_required
def data_transformation():
    """Render the data transformation dashboard page."""
    return render_template('dashboard/data_transformation.html', 
                          user=session.get('user'),
                          active_tab='data_transformation')

@dashboard_bp.route('/configuration')
@login_required
def configuration():
    """Render the configuration dashboard page."""
    return render_template('dashboard/configuration.html', 
                          user=session.get('user'),
                          active_tab='configuration')

# API endpoints for dashboard data

@dashboard_bp.route('/api/market-data', methods=['GET'])
@login_required
def get_market_data():
    """Get market data for a symbol."""
    symbol = request.args.get('symbol', 'BTC/USD')
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', 100))
    
    if not data_service:
        return jsonify({'error': 'Data service not initialized'})
    
    data = await data_service.get_market_data(symbol, timeframe, limit)
    return jsonify(data)

@dashboard_bp.route('/api/sentiment-data', methods=['GET'])
@login_required
def get_sentiment_data():
    """Get sentiment data for a symbol."""
    symbol = request.args.get('symbol', 'BTC/USD')
    
    if not data_service:
        return jsonify({'error': 'Data service not initialized'})
    
    data = await data_service.get_sentiment_data(symbol)
    return jsonify(data)

@dashboard_bp.route('/api/performance-data', methods=['GET'])
@login_required
def get_performance_data():
    """Get performance data."""
    if not data_service:
        return jsonify({'error': 'Data service not initialized'})
    
    data = await data_service.get_performance_data()
    return jsonify(data)

@dashboard_bp.route('/api/alerts', methods=['GET'])
@login_required
def get_alerts():
    """Get alerts."""
    if not data_service:
        return jsonify({'error': 'Data service not initialized'})
    
    data = await data_service.get_alerts()
    return jsonify(data)

@dashboard_bp.route('/api/positions', methods=['GET'])
@login_required
def get_positions():
    """Get positions."""
    if not data_service:
        return jsonify({'error': 'Data service not initialized'})
    
    data = await data_service.get_positions()
    return jsonify(data)

@dashboard_bp.route('/api/trades', methods=['GET'])
@login_required
def get_trades():
    """Get trades."""
    limit = int(request.args.get('limit', 20))
    
    if not data_service:
        return jsonify({'error': 'Data service not initialized'})
    
    data = await data_service.get_trades(limit)
    return jsonify(data)

def init_app(app, use_real_data=False, config=None):
    """Initialize dashboard feature modules with the Flask app."""
    global data_service
    
    # Register blueprint
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    
    # Initialize data service
    data_service = DataServiceFactory.create_data_service(use_real_data, config)
    
    # Add data service to app context
    app.data_service = data_service
    
    logger.info("Dashboard feature modules initialized")
