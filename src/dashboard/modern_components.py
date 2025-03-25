"""
AI Trading Agent - Modern Dashboard Components

This module provides the components and factory functions for creating the modern dashboard.
It implements improvements including:
- DataService with caching mechanism
- WebSocket for real-time updates
- User authentication with roles
- Dark/light theme toggle
- Notifications center
- Settings management
- Lazy loading and performance optimizations
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

def create_modern_app():
    """
    Create and configure the modern dashboard Flask application.
    
    Returns:
        Flask application instance with modern dashboard features
    """
    try:
        # Import Flask
        from flask import Flask, render_template, request, jsonify
        
        # Create the Flask application
        app = Flask(
            __name__, 
            template_folder=os.path.abspath("templates"),
            static_folder=os.path.abspath("static")
        )
        
        # Set secret key
        app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ai-trading-dashboard-secret")
        
        # Register routes
        @app.route('/')
        def index():
            return render_template(
                'modern_dashboard.html',
                page_title="Modern Dashboard",
                active_tab="overview",
                use_modern_features=True
            )
        
        @app.route('/sentiment')
        def sentiment_tab():
            return render_template(
                'modern_dashboard.html',
                page_title="Sentiment Analysis",
                active_tab="sentiment",
                use_modern_features=True
            )
        
        @app.route('/risk')
        def risk_tab():
            return render_template(
                'modern_dashboard.html',
                page_title="Risk Management",
                active_tab="risk",
                use_modern_features=True
            )
        
        @app.route('/logs')
        def logs_tab():
            return render_template(
                'modern_dashboard.html',
                page_title="Logs & Monitoring",
                active_tab="logs",
                use_modern_features=True
            )
        
        @app.route('/market-regime')
        def market_regime_tab():
            return render_template(
                'modern_dashboard.html',
                page_title="Market Regime Analysis",
                active_tab="market-regime",
                use_modern_features=True
            )
        
        # API endpoints
        @app.route('/api/system/status', methods=['GET'])
        def api_system_status():
            from integrated_dashboard import generate_mock_system_data
            return jsonify(generate_mock_system_data())
        
        @app.route('/api/sentiment', methods=['GET'])
        def api_sentiment():
            from integrated_dashboard import generate_mock_sentiment_data
            return jsonify(generate_mock_sentiment_data())
        
        @app.route('/api/risk', methods=['GET'])
        def api_risk():
            from integrated_dashboard import generate_mock_risk_data
            return jsonify(generate_mock_risk_data())
        
        @app.route('/api/logs', methods=['GET'])
        def api_logs():
            from integrated_dashboard import generate_mock_logs
            return jsonify(generate_mock_logs())
        
        @app.route('/api/market-regime', methods=['GET'])
        def api_market_regime():
            from integrated_dashboard import generate_mock_market_regime_data
            return jsonify(generate_mock_market_regime_data())
            
        return app
        
    except Exception as e:
        logger.error(f"Error creating modern app: {e}")
        raise
