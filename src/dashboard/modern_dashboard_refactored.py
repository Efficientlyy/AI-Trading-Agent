"""
AI Trading Agent - Modern Dashboard Implementation (Refactored)

This module provides a modular implementation of the modern dashboard interface
for the AI Trading Agent system, using a clean component-based architecture.

Following the Single Responsibility Principle and keeping files under 300 lines.
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from dateutil import tz

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

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
        self.app.route('/dashboard/status')(self.status_tab)
        self.app.route('/dashboard/admin')(self.admin_tab)
        self.app.route('/dashboard/validation')(self.validation_tab)
        self.app.route('/dashboard/transformation')(self.transformation_tab)
        self.app.route('/dashboard/configuration')(self.configuration_tab)
        
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
        
        # Missing API endpoints that are causing 404 errors
        self.app.route('/api/system/data-source-status')(self.api_data_source_status)
        self.app.route('/api/validation/results')(self.api_validation_results)
        self.app.route('/api/transform')(self.api_transform)
        
        # Settings API routes
        self.app.route('/api/settings')(self.api_get_settings)
        self.app.route('/api/settings', methods=['POST'])(self.api_save_settings)
        
        # Health check endpoint
        self.app.route('/health')(self.health_check)
    
    def login(self):
        """Login route"""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            user = self.auth_manager.authenticate_user(username, password)
            if user:
                # Store user data in session
                session['user'] = {
                    'username': user.get('username', 'admin'),
                    'name': user.get('name', 'Administrator'),
                    'role': user.get('role', 'admin'),
                    'last_login': datetime.now(tz=tz.tzutc())
                }
                
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
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'modern_dashboard.html',
            user=username,
            user_role=user_role,
            page_title="Dashboard",
            active_tab="overview",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def market_regime_tab(self):
        """Market regime tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'market_regime.html',
            user=username,
            user_role=user_role,
            page_title="Market Regime Analysis",
            active_tab="market-regime",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def sentiment_tab(self):
        """Sentiment analysis tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'sentiment_dashboard.html',
            user=username,
            user_role=user_role,
            page_title="Sentiment Analysis",
            active_tab="sentiment",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def _get_user_info_from_session(self):
        """Helper method to extract user info from session in a consistent way"""
        user_info = session.get('user', {})
        if isinstance(user_info, dict):
            username = user_info.get('username', 'Guest')
            user_role = user_info.get('role', 'user')
        else:
            username = str(user_info)
            user_role = 'user'
        
        return username, user_role
    
    def risk_tab(self):
        """Risk management tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'risk_management.html',
            user=username,
            user_role=user_role,
            page_title="Risk Management",
            active_tab="risk",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def performance_tab(self):
        """Performance metrics tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'performance.html',
            user=username,
            user_role=user_role,
            page_title="Performance Metrics",
            active_tab="performance",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def logs_tab(self):
        """Logs and monitoring tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'logs.html',
            user=username,
            user_role=user_role,
            page_title="Logs & Monitoring",
            active_tab="logs",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def status_tab(self):
        """System status tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'monitoring_dashboard.html',
            user=username,
            user_role=user_role,
            page_title="System Status",
            active_tab="status",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def admin_tab(self):
        """Admin controls tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Check if user has admin role
        _, user_role = self._get_user_info_from_session()
        if user_role != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'admin_controls_panel.html',
            user=username,
            user_role=user_role,
            page_title="Admin Controls",
            active_tab="admin",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def validation_tab(self):
        """Data validation tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'data_validation_panel.html',
            user=username,
            user_role=user_role,
            page_title="Data Validation",
            active_tab="validation",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def transformation_tab(self):
        """Data transformation pipeline tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'transformation_pipeline.html',
            user=username,
            user_role=user_role,
            page_title="Transformation Pipeline",
            active_tab="transformation",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
        )
    
    def configuration_tab(self):
        """Configuration tab"""
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Check if user has admin role
        _, user_role = self._get_user_info_from_session()
        if user_role != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        
        # Extract user info
        username, user_role = self._get_user_info_from_session()
        
        return render_template(
            'configuration_panel.html',
            user=username, 
            user_role=user_role,
            page_title="Configuration",
            active_tab="configuration",
            system_state=self.system_state,
            trading_state=self.trading_state,
            system_mode=self.system_mode,
            data_source=self.data_service.data_source.value.lower()
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
    
    def api_data_source_status(self):
        """API endpoint to get data source status information"""
        from datetime import datetime, timedelta
        
        # Generate a timestamp 5 minutes ago
        timestamp = (datetime.now() - timedelta(minutes=5)).isoformat()
        
        # Mock status data
        status_data = {
            "success": True,  # Add success flag
            "timestamp": timestamp,
            "sources": {
                "market_data": {
                    "id": "market_data",
                    "name": "Market Data API",
                    "description": "Real-time financial market data provider",
                    "type": "API",
                    "health": "HEALTHY",
                    "status": "ONLINE",
                    "lastCheck": datetime.now().isoformat(),
                    "uptime": "99.8%",
                    "responseTime": 245,
                    "errors": [],
                    "warnings": [],
                    "metrics": {
                        "requests": 1248,
                        "failures": 2,
                        "avgResponseTime": 218
                    }
                },
                "news_feed": {
                    "id": "news_feed",
                    "name": "Financial News Feed",
                    "description": "Real-time news and events affecting markets",
                    "type": "RSS",
                    "health": "HEALTHY",
                    "status": "ONLINE",
                    "lastCheck": datetime.now().isoformat(),
                    "uptime": "99.5%",
                    "responseTime": 312,
                    "errors": [],
                    "warnings": [
                        {
                            "message": "Increased latency detected",
                            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                            "severity": "LOW"
                        }
                    ],
                    "metrics": {
                        "requests": 845,
                        "failures": 5,
                        "avgResponseTime": 295
                    }
                },
                "sentiment_api": {
                    "id": "sentiment_api",
                    "name": "Market Sentiment API",
                    "description": "Sentiment analysis for market data",
                    "type": "API",
                    "health": "WARNING",
                    "status": "DEGRADED",
                    "lastCheck": datetime.now().isoformat(),
                    "uptime": "97.2%",
                    "responseTime": 485,
                    "errors": [],
                    "warnings": [
                        {
                            "message": "High response time",
                            "timestamp": (datetime.now() - timedelta(minutes=22)).isoformat(),
                            "severity": "MEDIUM"
                        }
                    ],
                    "metrics": {
                        "requests": 723,
                        "failures": 18,
                        "avgResponseTime": 412
                    }
                },
                "historical_db": {
                    "id": "historical_db",
                    "name": "Historical Database",
                    "description": "Database for historical market data",
                    "type": "DATABASE",
                    "health": "HEALTHY",
                    "status": "ONLINE",
                    "lastCheck": datetime.now().isoformat(),
                    "uptime": "99.9%",
                    "responseTime": 158,
                    "errors": [],
                    "warnings": [],
                    "metrics": {
                        "requests": 2145,
                        "failures": 1,
                        "avgResponseTime": 154
                    }
                }
            }
        }
        
        return jsonify(status_data)
    
    def api_validation_results(self):
        """API endpoint to get validation results"""
        return jsonify({'validation_results': self.data_service.get_data('validation_results')})
    
    def api_transform(self):
        """API endpoint to transform data"""
        return jsonify({'transformed_data': self.data_service.transform_data()})
    
    def api_get_settings(self):
        """API endpoint to get settings"""
        # Return mock settings for now
        settings = {
            "theme": "dark",
            "autoRefresh": True,
            "refreshInterval": 30,
            "useRealData": False,
            "fallbackStrategy": "mock",
            "cacheDuration": 60,
            "chartStyle": "candles",
            "defaultTimeRange": "1d",
            "decimalPlaces": 2,
            "desktopNotifications": True,
            "notificationLevel": "info",
            "soundAlerts": False
        }
        return jsonify(settings)
    
    def api_save_settings(self):
        """API endpoint to save settings"""
        # Log received settings for debugging
        settings = request.json
        logger.info(f"Saving settings: {settings}")
        # In a real implementation, we would save these settings to a database or file
        return jsonify({'success': True, 'message': 'Settings saved successfully'})
    
    def health_check(self):
        """Health check endpoint for Docker/Kubernetes"""
        return jsonify({'status': 'healthy'})
    
    def register_socket_events(self):
        """Register all socket.io events"""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            # Emit initial connection confirmation
            self.socketio.emit('connection_established', {'status': 'connected'}, room=request.sid)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            if not data or not isinstance(data, dict):
                logger.warning(f"Invalid subscription data: {data}")
                return
            
            # Handle both formats: {channel: '...'} and {channels: ['...', '...']}
            if 'channel' in data:
                channels = [data.get('channel')]
            elif 'channels' in data and isinstance(data.get('channels'), list):
                channels = data.get('channels')
            else:
                logger.warning(f"No valid channels in subscription data: {data}")
                return
            
            for channel in channels:
                if not channel:
                    continue
                logger.info(f"Client {request.sid} subscribed to {channel}")
                
                # Send initial data upon subscription
                if channel == 'dashboard':
                    self.send_dashboard_update(request.sid)
                elif channel == 'trades':
                    self.send_trades_update(request.sid)
                elif channel == 'positions':
                    self.send_positions_update(request.sid)
                elif channel == 'performance':
                    self.send_performance_update(request.sid)
    
    def send_dashboard_update(self, client_id=None):
        """Send dashboard update to clients"""
        try:
            # Generate dashboard data
            data = {
                'timestamp': datetime.now().isoformat(),
                'systemState': self.system_state.value,
                'tradingState': self.trading_state.value,
                'systemMode': self.system_mode.value,
                'activeStrategies': 3,
                'activeAssets': 5,
                'lastUpdate': datetime.now().isoformat()
            }
            
            # Emit to specific client or all clients
            if client_id:
                self.socketio.emit('dashboard_update', data, room=client_id)
            else:
                self.socketio.emit('dashboard_update', data)
        except Exception as e:
            logger.error(f"Error sending dashboard update: {str(e)}")
    
    def send_trades_update(self, client_id=None):
        """Send trades update to clients"""
        try:
            # Generate mock trade data
            data = {
                'trades': [
                    {'id': 1, 'symbol': 'BTC', 'type': 'buy', 'amount': 0.5, 'price': 34500, 'timestamp': datetime.now().isoformat()},
                    {'id': 2, 'symbol': 'ETH', 'type': 'sell', 'amount': 2.0, 'price': 1850, 'timestamp': datetime.now().isoformat()}
                ]
            }
            
            # Emit to specific client or all clients
            if client_id:
                self.socketio.emit('trade_update', data, room=client_id)
            else:
                self.socketio.emit('trade_update', data)
        except Exception as e:
            logger.error(f"Error sending trades update: {str(e)}")
    
    def send_positions_update(self, client_id=None):
        """Send positions update to clients"""
        try:
            # Generate mock position data
            data = {
                'positions': [
                    {'symbol': 'BTC', 'amount': 1.2, 'averagePrice': 33000, 'currentPrice': 34500, 'profitLoss': 1500},
                    {'symbol': 'ETH', 'amount': 10, 'averagePrice': 1700, 'currentPrice': 1850, 'profitLoss': 1500}
                ]
            }
            
            # Emit to specific client or all clients
            if client_id:
                self.socketio.emit('position_update', data, room=client_id)
            else:
                self.socketio.emit('position_update', data)
        except Exception as e:
            logger.error(f"Error sending positions update: {str(e)}")
    
    def send_performance_update(self, client_id=None):
        """Send performance update to clients"""
        try:
            # Generate mock performance data
            data = {
                'performance': {
                    'totalPnL': 12500,
                    'dailyPnL': 1200,
                    'winRate': 0.68,
                    'averageWin': 450,
                    'averageLoss': 280,
                    'sharpeRatio': 1.8
                }
            }
            
            # Emit to specific client or all clients
            if client_id:
                self.socketio.emit('performance_update', data, room=client_id)
            else:
                self.socketio.emit('performance_update', data)
        except Exception as e:
            logger.error(f"Error sending performance update: {str(e)}")
    
    def run(self, host="127.0.0.1", port=5000, debug=False):
        """Run the dashboard application with SocketIO"""
        self.socketio.run(self.app, host=host, port=port, debug=debug)


# For standalone running
if __name__ == "__main__":
    dashboard = ModernDashboard()
    dashboard.run(debug=True)
