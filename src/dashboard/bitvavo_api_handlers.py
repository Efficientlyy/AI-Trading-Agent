"""
Bitvavo API Handler Methods

This module contains the API handler methods for Bitvavo integration.
These methods will be added to the ModernDashboard class.
"""

import logging
import json
from flask import jsonify, request, send_file
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Bitvavo API Endpoints

def api_get_bitvavo_settings_panel(self):
    """API endpoint to get the Bitvavo settings panel template"""
    try:
        return send_file('templates/bitvavo_settings_panel.html')
    except Exception as e:
        logger.error(f"Error loading Bitvavo settings panel template: {e}")
        return "Error loading template", 500

def api_bitvavo_status(self):
    """API endpoint to get Bitvavo connection status"""
    try:
        # Check if API key manager is available
        if not hasattr(self, 'api_key_manager') or self.api_key_manager is None:
            return jsonify({
                "configured": False,
                "connected": False,
                "message": "API key manager not available"
            })
        
        # Check if Bitvavo credentials exist
        credential = self.api_key_manager.get_credential("bitvavo")
        if not credential:
            return jsonify({
                "configured": False,
                "connected": False,
                "message": "Bitvavo API not configured"
            })
        
        # Test connection to Bitvavo
        try:
            from src.execution.exchange.bitvavo import BitvavoConnector
            connector = BitvavoConnector(api_key=credential.key, api_secret=credential.secret)
            
            # Initialize connector and get account info
            success = connector.initialize()
            
            if success:
                return jsonify({
                    "configured": True,
                    "connected": True,
                    "message": "Bitvavo API configured and connected"
                })
            else:
                return jsonify({
                    "configured": True,
                    "connected": False,
                    "message": "Bitvavo API configured but connection failed"
                })
        except Exception as e:
            logger.error(f"Error connecting to Bitvavo: {e}")
            return jsonify({
                "configured": True,
                "connected": False,
                "message": f"Bitvavo API configured but connection failed: {str(e)}"
            })
    except Exception as e:
        logger.error(f"Error checking Bitvavo status: {e}")
        return jsonify({
            "configured": False,
            "connected": False,
            "message": f"Error checking Bitvavo status: {str(e)}"
        }), 500

def api_bitvavo_test_connection(self):
    """API endpoint to test Bitvavo API connection"""
    try:
        # Get API credentials from request
        data = request.json
        api_key = data.get('apiKey')
        api_secret = data.get('apiSecret')
        
        if not api_key or not api_secret:
            return jsonify({
                "success": False,
                "message": "API key and secret are required"
            }), 400
        
        # Test connection
        try:
            from src.execution.exchange.bitvavo import BitvavoConnector
            connector = BitvavoConnector(api_key=api_key, api_secret=api_secret)
            
            # Initialize connector and get account info
            success = connector.initialize()
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "Successfully connected to Bitvavo API"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to connect to Bitvavo API"
                })
        except Exception as e:
            logger.error(f"Error testing Bitvavo connection: {e}")
            return jsonify({
                "success": False,
                "message": f"Connection test failed: {str(e)}"
            })
    except Exception as e:
        logger.error(f"Error in Bitvavo test connection endpoint: {e}")
        return jsonify({
            "success": False,
            "message": f"Error testing connection: {str(e)}"
        }), 500

def api_bitvavo_save_credentials(self):
    """API endpoint to save Bitvavo API credentials"""
    try:
        # Check if API key manager is available
        if not hasattr(self, 'api_key_manager') or self.api_key_manager is None:
            return jsonify({
                "success": False,
                "message": "API key manager not available"
            }), 500
        
        # Get API credentials from request
        data = request.json
        api_key = data.get('apiKey')
        api_secret = data.get('apiSecret')
        
        if not api_key or not api_secret:
            return jsonify({
                "success": False,
                "message": "API key and secret are required"
            }), 400
        
        # Save credentials
        success = self.api_key_manager.add_bitvavo_credentials(api_key, api_secret)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Bitvavo credentials saved successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to save Bitvavo credentials"
            }), 500
    except Exception as e:
        logger.error(f"Error saving Bitvavo credentials: {e}")
        return jsonify({
            "success": False,
            "message": f"Error saving credentials: {str(e)}"
        }), 500

def api_bitvavo_save_settings(self):
    """API endpoint to save Bitvavo connection settings"""
    try:
        # Get settings from request
        data = request.json
        retry_attempts = data.get('retryAttempts', 3)
        timeout_seconds = data.get('timeoutSeconds', 10)
        cache_duration_seconds = data.get('cacheDurationSeconds', 60)
        
        # Validate settings
        if retry_attempts < 1 or retry_attempts > 10:
            return jsonify({
                "success": False,
                "message": "Retry attempts must be between 1 and 10"
            }), 400
            
        if timeout_seconds < 1 or timeout_seconds > 60:
            return jsonify({
                "success": False,
                "message": "Timeout must be between 1 and 60 seconds"
            }), 400
            
        if cache_duration_seconds < 0 or cache_duration_seconds > 3600:
            return jsonify({
                "success": False,
                "message": "Cache duration must be between 0 and 3600 seconds"
            }), 400
        
        # Get real data config
        if hasattr(self, 'settings_manager') and self.settings_manager is not None:
            config = self.settings_manager.get_real_data_config()
            
            # Update exchange_api settings
            if 'connections' in config and 'exchange_api' in config['connections']:
                config['connections']['exchange_api']['retry_attempts'] = retry_attempts
                config['connections']['exchange_api']['timeout_seconds'] = timeout_seconds
                config['connections']['exchange_api']['cache_duration_seconds'] = cache_duration_seconds
                
                # Save updated config
                self.settings_manager.save_real_data_config(config)
                
                return jsonify({
                    "success": True,
                    "message": "Bitvavo settings saved successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Invalid real data configuration structure"
                }), 500
        else:
            return jsonify({
                "success": False,
                "message": "Settings manager not available"
            }), 500
    except Exception as e:
        logger.error(f"Error saving Bitvavo settings: {e}")
        return jsonify({
            "success": False,
            "message": f"Error saving settings: {str(e)}"
        }), 500

def api_bitvavo_get_pairs(self):
    """API endpoint to get configured Bitvavo trading pairs"""
    try:
        # For now, return mock data
        # In a real implementation, this would fetch from a configuration file or database
        pairs = [
            {
                "symbol": "BTC/EUR",
                "enabled": True,
                "minOrderSize": 0.001,
                "maxPosition": 1.0
            },
            {
                "symbol": "ETH/EUR",
                "enabled": True,
                "minOrderSize": 0.01,
                "maxPosition": 10.0
            },
            {
                "symbol": "XRP/EUR",
                "enabled": False,
                "minOrderSize": 100,
                "maxPosition": 5000
            }
        ]
        
        return jsonify({
            "success": True,
            "pairs": pairs
        })
    except Exception as e:
        logger.error(f"Error getting Bitvavo pairs: {e}")
        return jsonify({
            "success": False,
            "message": f"Error getting trading pairs: {str(e)}"
        }), 500

def api_bitvavo_save_pairs(self):
    """API endpoint to save Bitvavo trading pairs configuration"""
    try:
        # Get pairs data from request
        data = request.json
        pairs = data.get('pairs', [])
        
        # Validate pairs data
        for pair in pairs:
            if 'symbol' not in pair:
                return jsonify({
                    "success": False,
                    "message": "Each pair must have a symbol"
                }), 400
        
        # In a real implementation, this would save to a configuration file or database
        # For now, just return success
        return jsonify({
            "success": True,
            "message": "Trading pairs configuration saved successfully"
        })
    except Exception as e:
        logger.error(f"Error saving Bitvavo pairs: {e}")
        return jsonify({
            "success": False,
            "message": f"Error saving trading pairs: {str(e)}"
        }), 500

def api_bitvavo_get_paper_trading(self):
    """API endpoint to get Bitvavo paper trading configuration"""
    try:
        # For now, return mock data
        # In a real implementation, this would fetch from a configuration file or database
        config = {
            "enabled": True,
            "initialBalances": {
                "EUR": 10000,
                "BTC": 0.5
            },
            "simulatedSlippage": 0.1,
            "simulatedLatency": 200
        }
        
        return jsonify({
            "success": True,
            "config": config
        })
    except Exception as e:
        logger.error(f"Error getting Bitvavo paper trading config: {e}")
        return jsonify({
            "success": False,
            "message": f"Error getting paper trading configuration: {str(e)}"
        }), 500

def api_bitvavo_save_paper_trading(self):
    """API endpoint to save Bitvavo paper trading configuration"""
    try:
        # Get config data from request
        data = request.json
        enabled = data.get('enabled', True)
        initial_balances = data.get('initialBalances', {})
        simulated_slippage = data.get('simulatedSlippage', 0.1)
        simulated_latency = data.get('simulatedLatency', 200)
        
        # Validate config data
        if simulated_slippage < 0 or simulated_slippage > 5:
            return jsonify({
                "success": False,
                "message": "Simulated slippage must be between 0 and 5 percent"
            }), 400
            
        if simulated_latency < 0 or simulated_latency > 2000:
            return jsonify({
                "success": False,
                "message": "Simulated latency must be between 0 and 2000 milliseconds"
            }), 400
        
        # In a real implementation, this would save to a configuration file or database
        # For now, just return success
        return jsonify({
            "success": True,
            "message": "Paper trading configuration saved successfully"
        })
    except Exception as e:
        logger.error(f"Error saving Bitvavo paper trading config: {e}")
        return jsonify({
            "success": False,
            "message": f"Error saving paper trading configuration: {str(e)}"
        }), 500

# Add a validation method for Bitvavo credentials
def _validate_bitvavo(self, cred):
    """Validate Bitvavo API credentials"""
    try:
        from src.execution.exchange.bitvavo import BitvavoConnector
        connector = BitvavoConnector(api_key=cred.get('key'), api_secret=cred.get('secret'))
        
        # Initialize connector and get account info
        success = connector.initialize()
        
        return {
            "success": success,
            "message": "Bitvavo API credentials are valid" if success else "Invalid Bitvavo API credentials"
        }
    except Exception as e:
        logger.error(f"Error validating Bitvavo credentials: {e}")
        return {
            "success": False,
            "message": f"Bitvavo API validation error: {str(e)}"
        }