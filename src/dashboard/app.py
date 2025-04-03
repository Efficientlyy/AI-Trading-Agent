"""
Core Application Framework

This module provides the core application framework for the dashboard.
It initializes the Flask application and sets up the necessary components.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, session, redirect, url_for, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

from src.dashboard.utils.auth import AuthManager
from src.dashboard.utils.settings_manager import SettingsManager
from src.dashboard.utils.event_bus import EventBus
from src.dashboard.utils.enums import SystemState, TradingState, SystemMode, DataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dashboard_app")

class DashboardApp:
    """
    Core application framework for the dashboard.
    """
    
    def __init__(self, config_dir: str = "config", template_dir: str = "templates", static_dir: str = "static"):
        """
        Initialize the dashboard application.
        
        Args:
            config_dir: Directory where configuration files are stored
            template_dir: Directory where templates are stored
            static_dir: Directory where static files are stored
        """
        # Initialize Flask application
        self.app = Flask(__name__, 
                         template_folder=template_dir,
                         static_folder=static_dir)
        
        # Fix for running behind proxy
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_for=1, x_proto=1, x_host=1)
        
        # Configure Flask application
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_development_only')
        self.app.config['SESSION_TYPE'] = 'filesystem'
        
        # Initialize components
        self.auth_manager = AuthManager()
        self.settings_manager = SettingsManager(config_dir=config_dir)
        self.event_bus = EventBus()
        
        # System state
        self.system_state = SystemState.STOPPED
        self.trading_state = TradingState.DISABLED
        self.system_mode = SystemMode.PAPER
        self.data_source = DataSource.MOCK
        
        # Register routes
        self._register_routes()
        
        logger.info("Dashboard application initialized")
    
    def _register_routes(self):
        """
        Register routes for the dashboard application.
        """
        # Authentication routes
        self._register_auth_routes()
        
        # Dashboard routes
        self._register_dashboard_routes()
        
        # API routes
        self._register_api_routes()
        
        # Error handlers
        self._register_error_handlers()
    
    def _register_auth_routes(self):
        """
        Register authentication routes.
        """
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            # Handle login form submission
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                
                # Authenticate user
                user = self.auth_manager.authenticate_user(username, password)
                
                if user:
                    # Store user in session
                    session['user'] = user
                    return redirect(url_for('dashboard'))
                else:
                    # Authentication failed
                    return render_template('login.html', error='Invalid username or password')
            
            # Display login form
            return render_template('login.html')
        
        @self.app.route('/logout')
        def logout():
            # Clear session
            session.clear()
            return redirect(url_for('login'))
    
    def _register_dashboard_routes(self):
        """
        Register dashboard routes.
        """
        @self.app.route('/')
        @self.app.route('/dashboard')
        @self.auth_manager.login_required
        def dashboard():
            # Get settings
            settings = self.settings_manager.get_settings()
            
            # Render dashboard template
            return render_template('dashboard.html', 
                                  user=session['user'],
                                  system_state=self.system_state,
                                  trading_state=self.trading_state,
                                  system_mode=self.system_mode,
                                  data_source=self.data_source,
                                  settings=settings)
    
    def _register_api_routes(self):
        """
        Register API routes.
        """
        @self.app.route('/api/system/state', methods=['GET'])
        def get_system_state():
            # Return system state
            return jsonify({
                'system_state': self.system_state,
                'trading_state': self.trading_state,
                'system_mode': self.system_mode,
                'data_source': self.data_source
            })
        
        @self.app.route('/api/system/start', methods=['POST'])
        @self.auth_manager.role_required(['admin', 'operator'])
        def start_system():
            # Start system
            self.system_state = SystemState.STARTING
            
            # Publish event
            self.event_bus.publish('system.state', {
                'state': self.system_state
            }, retain=True)
            
            # Return success
            return jsonify({'success': True})
        
        @self.app.route('/api/system/stop', methods=['POST'])
        @self.auth_manager.role_required(['admin', 'operator'])
        def stop_system():
            # Stop system
            self.system_state = SystemState.STOPPING
            
            # Publish event
            self.event_bus.publish('system.state', {
                'state': self.system_state
            }, retain=True)
            
            # Return success
            return jsonify({'success': True})
        
        @self.app.route('/api/trading/enable', methods=['POST'])
        @self.auth_manager.role_required(['admin', 'operator'])
        def enable_trading():
            # Enable trading
            self.trading_state = TradingState.ENABLED
            
            # Publish event
            self.event_bus.publish('trading.state', {
                'state': self.trading_state
            }, retain=True)
            
            # Return success
            return jsonify({'success': True})
        
        @self.app.route('/api/trading/disable', methods=['POST'])
        @self.auth_manager.role_required(['admin', 'operator'])
        def disable_trading():
            # Disable trading
            self.trading_state = TradingState.DISABLED
            
            # Publish event
            self.event_bus.publish('trading.state', {
                'state': self.trading_state
            }, retain=True)
            
            # Return success
            return jsonify({'success': True})
        
        @self.app.route('/api/settings', methods=['GET'])
        def get_settings():
            # Get settings
            settings = self.settings_manager.get_settings()
            
            # Return settings
            return jsonify(settings)
        
        @self.app.route('/api/settings', methods=['POST'])
        @self.auth_manager.role_required(['admin'])
        def update_settings():
            # Get settings from request
            new_settings = request.json
            
            # Update settings
            success, reload_required = self.settings_manager.update_settings(new_settings)
            
            # Return result
            return jsonify({
                'success': success,
                'reload_required': reload_required
            })
    
    def _register_error_handlers(self):
        """
        Register error handlers.
        """
        @self.app.errorhandler(404)
        def page_not_found(e):
            return render_template('errors/404.html'), 404
        
        @self.app.errorhandler(500)
        def internal_server_error(e):
            return render_template('errors/500.html'), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """
        Run the dashboard application.
        
        Args:
            host: Host to run the application on
            port: Port to run the application on
            debug: Whether to run in debug mode
        """
        self.app.run(host=host, port=port, debug=debug)

def create_app(config_dir: str = "config", template_dir: str = "templates", static_dir: str = "static") -> Flask:
    """
    Create and configure the dashboard application.
    
    Args:
        config_dir: Directory where configuration files are stored
        template_dir: Directory where templates are stored
        static_dir: Directory where static files are stored
        
    Returns:
        The configured Flask application
    """
    # Create dashboard application
    dashboard_app = DashboardApp(config_dir=config_dir, template_dir=template_dir, static_dir=static_dir)
    
    # Return Flask application
    return dashboard_app.app
