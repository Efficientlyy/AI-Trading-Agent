# Bitvavo Integration Plan for AI Trading Agent

This document outlines the comprehensive plan for integrating Bitvavo exchange with the existing AI Trading Agent system, focusing on leveraging the current codebase and dashboard API management system.

## Table of Contents

1. [Overview](#overview)
2. [Extending the API Key Management Service](#extending-the-api-key-management-service)
3. [Adding Bitvavo to Admin Dashboard Interface](#adding-bitvavo-to-admin-dashboard-interface)
4. [Implementing Bitvavo Exchange Connector](#implementing-bitvavo-exchange-connector)
5. [Pattern Recognition System Integration](#pattern-recognition-system-integration)
6. [Paper Trading Implementation](#paper-trading-implementation)
7. [Implementation Timeline](#implementation-timeline)

## Overview

The integration plan focuses on adding Bitvavo exchange support to the existing AI Trading Agent system, enabling automated trading with paper money using real market data. The implementation leverages the existing dashboard API management system and follows the established architecture patterns.

Bitvavo was selected as the optimal exchange for Netherlands users due to:
- Excellent documentation quality
- Straightforward API setup
- Netherlands-based with perfect compliance
- Competitive fee structure (0.15% maker / 0.25% taker)
- Comprehensive API features

## Extending the API Key Management Service

### 1. Add Bitvavo Provider to Exchange Enum

```python
# In src/common/models/enums.py

class ExchangeProvider(Enum):
    BINANCE = "binance"
    KRAKEN = "kraken"
    COINBASE = "coinbase"
    BITVAVO = "bitvavo"  # Add Bitvavo to existing providers
```

### 2. Extend API Key Manager for Bitvavo

```python
# In src/common/security/api_key_manager.py

class APIKeyManager:
    # Existing methods...
    
    def add_bitvavo_credentials(self, api_key: str, api_secret: str) -> bool:
        """Add Bitvavo API credentials to the secure storage"""
        try:
            # Use existing encryption mechanism
            encrypted_key = self.encrypt_data(api_key)
            encrypted_secret = self.encrypt_data(api_secret)
            
            # Store in database using existing patterns
            self.db_manager.insert_or_update(
                "api_keys",
                {"provider": ExchangeProvider.BITVAVO.value},
                {
                    "api_key": encrypted_key,
                    "api_secret": encrypted_secret,
                    "is_active": True,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to add Bitvavo credentials: {str(e)}")
            return False
            
    def test_bitvavo_connection(self, api_key: str, api_secret: str) -> Tuple[bool, str]:
        """Test Bitvavo API connection with provided credentials"""
        try:
            # Create temporary client to test connection
            from src.execution.exchange.bitvavo import BitvavoConnector
            test_client = BitvavoConnector(api_key, api_secret)
            
            # Test API connection by fetching account info
            account_info = test_client.get_account_info()
            
            if account_info:
                return True, "Successfully connected to Bitvavo API"
            return False, "Failed to retrieve account information"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
```

### 3. Implement Bitvavo-specific Validation

```python
# In src/common/security/validators.py

def validate_bitvavo_credentials(api_key: str, api_secret: str) -> Tuple[bool, str]:
    """Validate Bitvavo API credentials format"""
    if not api_key or len(api_key) < 10:
        return False, "API key is too short or empty"
        
    if not api_secret or len(api_secret) < 10:
        return False, "API secret is too short or empty"
        
    # Bitvavo-specific format validation
    if not api_key.isalnum():
        return False, "API key should contain only alphanumeric characters"
        
    return True, "Credentials format is valid"
```

## Adding Bitvavo to Admin Dashboard Interface

### 1. Create Bitvavo API Settings Component

```jsx
// In src/dashboard/components/admin/BitvavoApiSettings.jsx

import React, { useState, useEffect } from 'react';
import { Form, Button, Alert, Card } from 'react-bootstrap';
import { testBitvavoConnection, saveBitvavoCredentials } from '../../services/apiService';

const BitvavoApiSettings = () => {
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [status, setStatus] = useState(null);
  
  const handleTestConnection = async () => {
    setIsLoading(true);
    try {
      const response = await testBitvavoConnection(apiKey, apiSecret);
      setStatus(response.success ? 'success' : 'danger');
      setMessage(response.message);
    } catch (error) {
      setStatus('danger');
      setMessage('Connection test failed: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleSave = async () => {
    setIsLoading(true);
    try {
      const response = await saveBitvavoCredentials(apiKey, apiSecret);
      setStatus(response.success ? 'success' : 'danger');
      setMessage(response.message);
    } catch (error) {
      setStatus('danger');
      setMessage('Failed to save credentials: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <Card className="mb-4">
      <Card.Header>Bitvavo API Settings</Card.Header>
      <Card.Body>
        <Form>
          <Form.Group className="mb-3">
            <Form.Label>API Key</Form.Label>
            <Form.Control 
              type="text" 
              value={apiKey} 
              onChange={(e) => setApiKey(e.target.value)} 
              placeholder="Enter Bitvavo API Key" 
            />
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>API Secret</Form.Label>
            <Form.Control 
              type="password" 
              value={apiSecret} 
              onChange={(e) => setApiSecret(e.target.value)} 
              placeholder="Enter Bitvavo API Secret" 
            />
          </Form.Group>
          
          {message && <Alert variant={status}>{message}</Alert>}
          
          <div className="d-flex gap-2">
            <Button 
              variant="secondary" 
              onClick={handleTestConnection} 
              disabled={isLoading}
            >
              {isLoading ? 'Testing...' : 'Test Connection'}
            </Button>
            <Button 
              variant="primary" 
              onClick={handleSave} 
              disabled={isLoading}
            >
              {isLoading ? 'Saving...' : 'Save Credentials'}
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
};

export default BitvavoApiSettings;
```

### 2. Add API Service Methods

```jsx
// In src/dashboard/services/apiService.js

// Existing imports and methods...

export const testBitvavoConnection = async (apiKey, apiSecret) => {
  try {
    const response = await axios.post('/api/settings/bitvavo/test', {
      apiKey,
      apiSecret
    });
    return response.data;
  } catch (error) {
    console.error('Error testing Bitvavo connection:', error);
    throw error;
  }
};

export const saveBitvavoCredentials = async (apiKey, apiSecret) => {
  try {
    const response = await axios.post('/api/settings/bitvavo/save', {
      apiKey,
      apiSecret
    });
    return response.data;
  } catch (error) {
    console.error('Error saving Bitvavo credentials:', error);
    throw error;
  }
};

export const getBitvavoStatus = async () => {
  try {
    const response = await axios.get('/api/settings/bitvavo/status');
    return response.data;
  } catch (error) {
    console.error('Error getting Bitvavo status:', error);
    throw error;
  }
};
```

### 3. Update Admin Settings Page

```jsx
// In src/dashboard/pages/AdminSettings.jsx

import BitvavoApiSettings from '../components/admin/BitvavoApiSettings';

// Inside the render method, add:
<Tab eventKey="bitvavo" title="Bitvavo">
  <BitvavoApiSettings />
</Tab>
```

## Implementing Bitvavo Exchange Connector

### 1. Create Bitvavo Connector Class

```python
# In src/execution/exchange/bitvavo.py

from typing import Dict, List, Optional, Any, Tuple
import time
import hmac
import hashlib
import json
import requests
import websocket
import threading
from datetime import datetime

from src.common.models.enums import OrderType, OrderSide, TimeInForce
from src.common.security.api_key_manager import APIKeyManager
from src.common.logging import get_logger
from src.execution.exchange.base import BaseExchangeConnector

class BitvavoConnector(BaseExchangeConnector):
    """Bitvavo exchange connector implementation"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__("bitvavo")
        self.logger = get_logger(__name__)
        self.base_url = "https://api.bitvavo.com/v2"
        self.ws_url = "wss://ws.bitvavo.com/v2"
        
        # Get API credentials from manager if not provided
        if not api_key or not api_secret:
            key_manager = APIKeyManager()
            credentials = key_manager.get_credentials(self.exchange_id)
            if credentials:
                api_key = credentials.get("api_key")
                api_secret = credentials.get("api_secret")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        
        # WebSocket related attributes
        self.ws = None
        self.ws_connected = False
        self.ws_subscriptions = set()
        self.ws_callbacks = {}
        self.ws_thread = None
        
    def _generate_signature(self, timestamp: int, method: str, url_path: str, 
                           body: Optional[Dict] = None) -> str:
        """Generate HMAC-SHA256 signature for Bitvavo API authentication"""
        body_str = "" if body is None else json.dumps(body)
        message = str(timestamp) + method + url_path + body_str
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    def _get_auth_headers(self, method: str, url_path: str, 
                         body: Optional[Dict] = None) -> Dict[str, str]:
        """Get authentication headers for Bitvavo API requests"""
        timestamp = int(time.time() * 1000)
        signature = self._generate_signature(timestamp, method, url_path, body)
        
        headers = {
            'Bitvavo-Access-Key': self.api_key,
            'Bitvavo-Access-Signature': signature,
            'Bitvavo-Access-Timestamp': str(timestamp),
            'Content-Type': 'application/json'
        }
        return headers
        
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                data: Optional[Dict] = None, auth_required: bool = False) -> Any:
        """Make a request to Bitvavo API"""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if auth_required:
            headers.update(self._get_auth_headers(method, endpoint, data))
            
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Response: {e.response.text}")
            raise
            
    # Implement required methods from BaseExchangeConnector
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        return self._request("GET", "/account", auth_required=True)
        
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        return self._request("GET", "/markets")
        
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information for a symbol"""
        return self._request("GET", f"/ticker/price?market={symbol}")
        
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol"""
        return self._request("GET", f"/orderbook?market={symbol}&depth={limit}")
        
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol"""
        return self._request("GET", f"/trades?market={symbol}&limit={limit}")
        
    def get_klines(self, symbol: str, interval: str, 
                  start_time: Optional[int] = None, 
                  end_time: Optional[int] = None,
                  limit: int = 1000) -> List[List]:
        """Get candlestick data for a symbol"""
        params = {
            "market": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["start"] = start_time
            
        if end_time:
            params["end"] = end_time
            
        return self._request("GET", "/candles", params=params)
        
    def create_order(self, symbol: str, side: OrderSide, 
                    order_type: OrderType, quantity: float,
                    price: Optional[float] = None,
                    time_in_force: TimeInForce = TimeInForce.GTC) -> Dict:
        """Create a new order"""
        data = {
            "market": symbol,
            "side": side.value.lower(),
            "orderType": order_type.value.lower()
        }
        
        if order_type == OrderType.LIMIT:
            if not price:
                raise ValueError("Price is required for limit orders")
            data["price"] = str(price)
            data["timeInForce"] = time_in_force.value
            
        data["amount"] = str(quantity)
        
        return self._request("POST", "/order", data=data, auth_required=True)
        
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an existing order"""
        data = {
            "market": symbol,
            "orderId": order_id
        }
        return self._request("DELETE", "/order", data=data, auth_required=True)
        
    def get_order(self, symbol: str, order_id: str) -> Dict:
        """Get information about an order"""
        params = {
            "market": symbol,
            "orderId": order_id
        }
        return self._request("GET", "/order", params=params, auth_required=True)
        
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        params = {}
        if symbol:
            params["market"] = symbol
            
        return self._request("GET", "/orders", params=params, auth_required=True)
        
    def get_account_balance(self) -> List[Dict]:
        """Get account balance"""
        return self._request("GET", "/balance", auth_required=True)
```

### 2. Register Bitvavo Connector in Factory

```python
# In src/execution/factory.py

from src.execution.exchange.bitvavo import BitvavoConnector

class ExchangeConnectorFactory:
    # Existing code...
    
    def create_connector(self, exchange_id: str) -> BaseExchangeConnector:
        """Create an exchange connector instance"""
        if exchange_id == ExchangeProvider.BINANCE.value:
            return BinanceConnector()
        elif exchange_id == ExchangeProvider.KRAKEN.value:
            return KrakenConnector()
        elif exchange_id == ExchangeProvider.BITVAVO.value:
            return BitvavoConnector()
        else:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
```

### 3. Add API Routes for Bitvavo Settings

```python
# In src/api/routes/settings.py

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.common.security.api_key_manager import APIKeyManager
from src.execution.exchange.bitvavo import BitvavoConnector

router = APIRouter(prefix="/settings", tags=["settings"])

class BitvavoCredentials(BaseModel):
    apiKey: str
    apiSecret: str

@router.post("/bitvavo/test")
async def test_bitvavo_connection(credentials: BitvavoCredentials):
    """Test Bitvavo API connection"""
    key_manager = APIKeyManager()
    success, message = key_manager.test_bitvavo_connection(
        credentials.apiKey, 
        credentials.apiSecret
    )
    
    return {"success": success, "message": message}

@router.post("/bitvavo/save")
async def save_bitvavo_credentials(credentials: BitvavoCredentials):
    """Save Bitvavo API credentials"""
    key_manager = APIKeyManager()
    success = key_manager.add_bitvavo_credentials(
        credentials.apiKey, 
        credentials.apiSecret
    )
    
    if success:
        return {"success": True, "message": "Bitvavo credentials saved successfully"}
    else:
        return {"success": False, "message": "Failed to save Bitvavo credentials"}

@router.get("/bitvavo/status")
async def get_bitvavo_status():
    """Get Bitvavo connection status"""
    key_manager = APIKeyManager()
    credentials = key_manager.get_credentials("bitvavo")
    
    if not credentials:
        return {"configured": False, "message": "Bitvavo API not configured"}
        
    try:
        connector = BitvavoConnector()
        account_info = connector.get_account_info()
        return {
            "configured": True, 
            "connected": True,
            "message": "Bitvavo API configured and connected"
        }
    except Exception as e:
        return {
            "configured": True,
            "connected": False,
            "message": f"Bitvavo API configured but connection failed: {str(e)}"
        }
```

## Pattern Recognition System Integration

The pattern recognition system will be implemented next, focusing on:

1. Technical indicator implementation
2. Chart pattern recognition algorithms
3. Machine learning feature engineering
4. Integration with the trading decision engine

This will build upon the existing analysis modules in the codebase and leverage the Bitvavo market data.

## Paper Trading Implementation

After completing the pattern recognition system, paper trading will be implemented to:

1. Use real market data from Bitvavo
2. Simulate order execution without real funds
3. Track performance metrics
4. Test trading strategies in a realistic environment

## Implementation Timeline

| Phase | Component | Timeline |
|-------|-----------|----------|
| 1 | API Key Management Extension | Week 1, Days 1-2 |
| 2 | Admin Dashboard Integration | Week 1, Days 3-4 |
| 3 | Bitvavo Exchange Connector | Week 1, Day 5 - Week 2, Day 3 |
| 4 | Pattern Recognition System | Week 2, Day 4 - Week 4, Day 3 |
| 5 | Paper Trading Implementation | Week 4, Day 4 - Week 5, Day 5 |
| 6 | Testing and Optimization | Week 6 |

This implementation plan provides a comprehensive roadmap for integrating Bitvavo with the existing AI Trading Agent system, focusing on leveraging the current codebase and dashboard API management system.
