"""
Bitvavo Exchange Routes Module

This module provides Flask routes for Bitvavo exchange integration.
"""

import logging
from typing import Dict, Any, List
from flask import Blueprint, render_template, request, jsonify, session
import asyncio

from src.dashboard.utils.auth import login_required, role_required
from src.dashboard.utils.settings_manager import SettingsManager
from src.dashboard.exchanges.bitvavo.client import BitvavoClient, BitvavoExchangeConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bitvavo_routes")

# Create blueprint
bitvavo_bp = Blueprint('bitvavo', __name__)

# Initialize settings manager
settings_manager = SettingsManager()

# Initialize exchange connector
exchange_connector = None

@bitvavo_bp.route('/settings')
@login_required
@role_required(['admin', 'operator'])
def settings():
    """Render the Bitvavo settings page."""
    return render_template('exchanges/bitvavo_settings.html', 
                          user=session.get('user'),
                          active_tab='configuration')

@bitvavo_bp.route('/api/settings', methods=['GET'])
@login_required
@role_required(['admin', 'operator'])
def get_settings():
    """Get Bitvavo settings."""
    settings = settings_manager.get_settings('bitvavo', {
        'api_key': '',
        'paper_trading': True,
        'trading_pairs': ['BTC-EUR', 'ETH-EUR', 'SOL-EUR'],
        'default_timeframe': '1h'
    })
    
    # Mask API key
    if settings.get('api_key'):
        settings['api_key'] = '********'
    
    return jsonify(settings)

@bitvavo_bp.route('/api/settings', methods=['POST'])
@login_required
@role_required(['admin'])
def update_settings():
    """Update Bitvavo settings."""
    data = request.json
    
    # Validate data
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'})
    
    # Get current settings
    current_settings = settings_manager.get_settings('bitvavo', {})
    
    # Update settings
    if 'api_key' in data and data['api_key'] != '********':
        current_settings['api_key'] = data['api_key']
    
    if 'api_secret' in data and data['api_secret']:
        current_settings['api_secret'] = data['api_secret']
    
    if 'paper_trading' in data:
        current_settings['paper_trading'] = data['paper_trading']
    
    if 'trading_pairs' in data:
        current_settings['trading_pairs'] = data['trading_pairs']
    
    if 'default_timeframe' in data:
        current_settings['default_timeframe'] = data['default_timeframe']
    
    # Save settings
    settings_manager.save_settings('bitvavo', current_settings)
    
    # Reinitialize exchange connector
    _initialize_exchange_connector()
    
    return jsonify({'success': True, 'message': 'Settings updated successfully'})

@bitvavo_bp.route('/api/test-connection', methods=['POST'])
@login_required
@role_required(['admin', 'operator'])
def test_connection():
    """Test Bitvavo API connection."""
    data = request.json
    
    # Get API credentials
    api_key = data.get('api_key', '')
    api_secret = data.get('api_secret', '')
    
    # If no credentials provided, use stored credentials
    if not api_key or not api_secret:
        settings = settings_manager.get_settings('bitvavo', {})
        api_key = settings.get('api_key', '')
        api_secret = settings.get('api_secret', '')
    
    # Create temporary client
    client = BitvavoClient(api_key, api_secret)
    
    # Test connection
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(client.test_connection())
    loop.close()
    
    return jsonify(result)

@bitvavo_bp.route('/api/markets', methods=['GET'])
@login_required
def get_markets():
    """Get available markets."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get markets
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    markets = loop.run_until_complete(exchange_connector.get_markets())
    loop.close()
    
    return jsonify(markets)

@bitvavo_bp.route('/api/assets', methods=['GET'])
@login_required
def get_assets():
    """Get available assets."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get assets
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    assets = loop.run_until_complete(exchange_connector.get_assets())
    loop.close()
    
    return jsonify(assets)

@bitvavo_bp.route('/api/ticker', methods=['GET'])
@login_required
def get_ticker():
    """Get ticker data."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get market from query parameters
    market = request.args.get('market')
    
    # Get ticker
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ticker = loop.run_until_complete(exchange_connector.get_ticker(market))
    loop.close()
    
    return jsonify(ticker)

@bitvavo_bp.route('/api/candles', methods=['GET'])
@login_required
def get_candles():
    """Get candle data."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get parameters from query
    market = request.args.get('market', 'BTC-EUR')
    interval = request.args.get('interval', '1h')
    limit = int(request.args.get('limit', 100))
    
    # Get candles
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    candles = loop.run_until_complete(exchange_connector.get_candles(market, interval, limit))
    loop.close()
    
    return jsonify(candles)

@bitvavo_bp.route('/api/balance', methods=['GET'])
@login_required
@role_required(['admin', 'operator'])
def get_balance():
    """Get account balance."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get balance
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    balance = loop.run_until_complete(exchange_connector.get_balance())
    loop.close()
    
    return jsonify(balance)

@bitvavo_bp.route('/api/orders', methods=['GET'])
@login_required
@role_required(['admin', 'operator'])
def get_orders():
    """Get open orders."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get market from query parameters
    market = request.args.get('market')
    
    # Get orders
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    orders = loop.run_until_complete(exchange_connector.get_orders(market))
    loop.close()
    
    return jsonify(orders)

@bitvavo_bp.route('/api/orders', methods=['POST'])
@login_required
@role_required(['admin', 'operator'])
def create_order():
    """Create a new order."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get order data
    data = request.json
    
    # Validate data
    if not data:
        return jsonify({'error': 'No data provided'})
    
    required_fields = ['market', 'side', 'orderType', 'amount']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'})
    
    # Create order
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    order = loop.run_until_complete(exchange_connector.create_order(
        data['market'],
        data['side'],
        data['orderType'],
        data['amount'],
        data.get('price')
    ))
    loop.close()
    
    return jsonify(order)

@bitvavo_bp.route('/api/orders/<order_id>', methods=['DELETE'])
@login_required
@role_required(['admin', 'operator'])
def cancel_order(order_id):
    """Cancel an order."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get market from query parameters
    market = request.args.get('market')
    
    if not market:
        return jsonify({'error': 'Market parameter is required'})
    
    # Cancel order
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(exchange_connector.cancel_order(order_id, market))
    loop.close()
    
    return jsonify(result)

@bitvavo_bp.route('/api/trades', methods=['GET'])
@login_required
@role_required(['admin', 'operator'])
def get_trades():
    """Get recent trades."""
    if not exchange_connector:
        _initialize_exchange_connector()
    
    if not exchange_connector:
        return jsonify({'error': 'Exchange connector not initialized'})
    
    # Get limit from query parameters
    limit = int(request.args.get('limit', 20))
    
    # Get trades
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    trades = loop.run_until_complete(exchange_connector.get_trades(limit))
    loop.close()
    
    return jsonify(trades)

def _initialize_exchange_connector():
    """Initialize the exchange connector with settings."""
    global exchange_connector
    
    # Get settings
    settings = settings_manager.get_settings('bitvavo', {})
    
    # Get API credentials
    api_key = settings.get('api_key', '')
    api_secret = settings.get('api_secret', '')
    paper_trading = settings.get('paper_trading', True)
    
    # Create exchange connector
    exchange_connector = BitvavoExchangeConnector(api_key, api_secret, paper_trading)
    
    # Initialize connector
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(exchange_connector.initialize())
    loop.close()
    
    logger.info(f"Bitvavo exchange connector initialized (paper_trading={paper_trading})")

def init_app(app):
    """Initialize Bitvavo routes with the Flask app."""
    # Register blueprint
    app.register_blueprint(bitvavo_bp, url_prefix='/bitvavo')
    
    # Initialize exchange connector
    _initialize_exchange_connector()
    
    logger.info("Bitvavo routes initialized")
