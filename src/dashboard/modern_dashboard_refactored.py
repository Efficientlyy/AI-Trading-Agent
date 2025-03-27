"""
AI Trading Agent - Modern Dashboard Implementation (Refactored)

This module provides a modular implementation of the modern dashboard interface
for the AI Trading Agent system, using a clean component-based architecture.

Following the Single Responsibility Principle and keeping files under 300 lines.
"""

import os
import logging
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO

# Import modular components
from src.dashboard.utils.enums import SystemState, TradingState, SystemMode, UserRole, DataSource
from src.dashboard.utils.data_service import DataService, REAL_DATA_AVAILABLE
from src.dashboard.utils.auth import AuthManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("modern_dashboard")

class ModernDashboard:
    """
    Modern dashboard application with modular components
    
    This implementation follows clean architecture principles with
    separation of concerns and dependency injection.
    """
    
    def __init__(self, template_folder=None, static_folder=None):
        """Initialize the modern dashboard application"""
        # Use provided template and static folders, environment variables, or default paths
        template_folder = template_folder or os.environ.get("FLASK_TEMPLATE_FOLDER", os.path.abspath("templates"))
        static_folder = static_folder or os.environ.get("FLASK_STATIC_FOLDER", os.path.abspath("static"))
        
        # Initialize Flask
        self.app = Flask(
            __name__, 
            template_folder=template_folder,
            static_folder=static_folder
        )
        self.app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ai-trading-dashboard-secret")
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Initialize authentication manager
        self.auth_manager = AuthManager()
        
        # System state (in-memory for demonstration)
        self.system_state = SystemState.STOPPED
        self.trading_state = TradingState.DISABLED
        self.system_mode = SystemMode.PAPER
        
        # Data service for both mock and real data
        data_source = DataSource.REAL if REAL_DATA_AVAILABLE else DataSource.MOCK
        self.data_service = DataService(data_source)
        
        # Initialize API Key Manager
        try:
            from src.common.security import get_api_key_manager
            self.api_key_manager = get_api_key_manager()
            self.api_key_manager_available = True
        except ImportError:
            logger.warning("API Key Manager not available, using in-memory mock")
            self.api_key_manager_available = False
            # Mock API key storage for demonstration
            self.mock_api_keys = {}
        
        # Register routes and socket events
        self.register_routes()
        self.register_socket_events()
    
    def register_routes(self):
        """Register all dashboard routes"""
        # Authentication routes
        self.app.route('/login', methods=['GET', 'POST'])(self.login)
        self.app.route('/logout')(self.logout)
        
        # Main routes
        self.app.route('/')(self.index)
        self.app.route('/dashboard')(self.dashboard)
        
        # Tab routes
        self.app.route('/dashboard/market-regime')(self.market_regime_tab)
        self.app.route('/dashboard/sentiment')(self.sentiment_tab)
        self.app.route('/dashboard/risk')(self.risk_tab)
        self.app.route('/dashboard/performance')(self.performance_tab)
        self.app.route('/dashboard/logs')(self.logs_tab)
        
        # API routes
        self.app.route('/api/system/status')(self.api_system_status)
        self.app.route('/api/system/start', methods=['POST'])(self.api_system_start)
        self.app.route('/api/system/stop', methods=['POST'])(self.api_system_stop)
        self.app.route('/api/trading/enable', methods=['POST'])(self.api_trading_enable)
        self.app.route('/api/trading/disable', methods=['POST'])(self.api_trading_disable)
        self.app.route('/api/system/mode', methods=['POST'])(self.api_set_system_mode)
        self.app.route('/api/data/source', methods=['POST'])(self.api_set_data_source)
        
        # Data API routes
        self.app.route('/api/data/dashboard/summary')(self.api_dashboard_summary)
        self.app.route('/api/data/system/health')(self.api_system_health)
        self.app.route('/api/data/component/status')(self.api_component_status)
        self.app.route('/api/data/trading/performance')(self.api_trading_performance)
        self.app.route('/api/data/market/regime')(self.api_market_regime)
        
        # Health check endpoint
        self.app.route('/health')(self.health_check)
    
    def login(self):
        """Login route"""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            user = self.auth_manager.authenticate_user(username, password)
            if user:
                session['user'] = user
                flash(f'Welcome back, {user["name"]}!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'danger')
        
        return render_template('login.html')
    
    def logout(self):
        """Logout route"""
        session.pop('user', None)
        flash('You have been logged out', 'info')
        return redirect(url_for('login'))
    
    def index(self):
        """Main index route - redirect to dashboard"""
        if 'user' in session:
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))
    
    def dashboard(self):
        """Main dashboard route"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        return render_template(
            'modern_dashboard.html',
            user=session['user'],
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode
        )
    
    def market_regime_tab(self):
        """Market regime tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        return render_template(
            'market_regime.html',
            user=session['user']
        )
    
    def sentiment_tab(self):
        """Sentiment analysis tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        return render_template(
            'sentiment.html',
            user=session['user']
        )
    
    def risk_tab(self):
        """Risk management tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        return render_template(
            'risk.html',
            user=session['user']
        )
    
    def performance_tab(self):
        """Performance analytics tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        return render_template(
            'performance.html',
            user=session['user']
        )
    
    def logs_tab(self):
        """Logs and monitoring tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        return render_template(
            'logs.html',
            user=session['user']
        )
    
    def api_system_status(self):
        """API endpoint to get system status"""
        return jsonify({
            'system_state': self.system_state,
            'trading_state': self.trading_state,
            'system_mode': self.system_mode
        })
    
    def api_system_start(self):
        """API endpoint to start the system"""
        if 'user' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        if session['user']['role'] not in [UserRole.ADMIN, UserRole.OPERATOR]:
            return jsonify({'error': 'Not authorized'}), 403
        
        self.system_state = SystemState.RUNNING
        return jsonify({'success': True, 'system_state': self.system_state})
    
    def api_system_stop(self):
        """API endpoint to stop the system"""
        if 'user' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        if session['user']['role'] not in [UserRole.ADMIN, UserRole.OPERATOR]:
            return jsonify({'error': 'Not authorized'}), 403
        
        self.system_state = SystemState.STOPPED
        self.trading_state = TradingState.DISABLED
        return jsonify({'success': True, 'system_state': self.system_state})
    
    def api_trading_enable(self):
        """API endpoint to enable trading"""
        if 'user' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        if session['user']['role'] != UserRole.ADMIN:
            return jsonify({'error': 'Not authorized'}), 403
        
        if self.system_state != SystemState.RUNNING:
            return jsonify({'error': 'System must be running to enable trading'}), 400
        
        self.trading_state = TradingState.ENABLED
        return jsonify({'success': True, 'trading_state': self.trading_state})
    
    def api_trading_disable(self):
        """API endpoint to disable trading"""
        if 'user' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        if session['user']['role'] not in [UserRole.ADMIN, UserRole.OPERATOR]:
            return jsonify({'error': 'Not authorized'}), 403
        
        self.trading_state = TradingState.DISABLED
        return jsonify({'success': True, 'trading_state': self.trading_state})
    
    def api_set_system_mode(self):
        """API endpoint to set system mode"""
        if 'user' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        if session['user']['role'] != UserRole.ADMIN:
            return jsonify({'error': 'Not authorized'}), 403
        
        mode = request.json.get('mode')
        if mode not in [SystemMode.LIVE, SystemMode.PAPER, SystemMode.BACKTEST]:
            return jsonify({'error': 'Invalid mode'}), 400
        
        self.system_mode = mode
        return jsonify({'success': True, 'system_mode': self.system_mode})
    
    def api_set_data_source(self):
        """API endpoint to switch between mock and real data sources"""
        if 'user' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        source = request.json.get('source')
        if source not in [DataSource.MOCK, DataSource.REAL]:
            return jsonify({'error': 'Invalid data source'}), 400
        
        try:
            self.data_service.set_data_source(source)
            return jsonify({'success': True, 'data_source': source})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    
    def api_dashboard_summary(self):
        """API endpoint to get dashboard summary data"""
        # Combine data from multiple sources
        system_health = self.data_service.get_data('system_health')
        trading_performance = self.data_service.get_data('trading_performance')
        market_regime = self.data_service.get_data('market_regime')
        
        return jsonify({
            'system_health': system_health,
            'trading_performance': trading_performance,
            'market_regime': market_regime
        })
    
    def api_system_health(self):
        """API endpoint to get system health data"""
        return jsonify(self.data_service.get_data('system_health'))
    
    def api_component_status(self):
        """API endpoint to get component status data"""
        return jsonify(self.data_service.get_data('component_status'))
    
    def api_trading_performance(self):
        """API endpoint to get trading performance data"""
        return jsonify(self.data_service.get_data('trading_performance'))
    
    def api_market_regime(self):
        """API endpoint to get market regime data"""
        return jsonify(self.data_service.get_data('market_regime'))
    
    def health_check(self):
        """Health check endpoint for Docker/Kubernetes"""
        return jsonify({'status': 'healthy'})
    
    def register_socket_events(self):
        """Register all socket.io events"""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            channel = data.get('channel')
            logger.info(f"Client {request.sid} subscribed to {channel}")
    
    def run(self, host="127.0.0.1", port=5000, debug=False):
        """Run the dashboard application with SocketIO"""
        self.socketio.run(self.app, host=host, port=port, debug=debug)


# For standalone running
if __name__ == "__main__":
    dashboard = ModernDashboard()
    dashboard.run(debug=True)
