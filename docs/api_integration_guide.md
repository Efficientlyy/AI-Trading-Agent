# AI Trading Agent API Integration Guide

This guide provides comprehensive documentation for integrating with the AI Trading Agent External API Gateway. The API Gateway enables secure, reliable access to the AI Trading Agent's features for external partners.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Partner Tiers](#partner-tiers)
- [API Endpoints](#api-endpoints)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Sample Integrations](#sample-integrations)

## Overview

The AI Trading Agent External API Gateway provides a secure interface for partners to integrate with our advanced trading intelligence platform. The API is designed with a RESTful architecture and uses JSON for request and response payloads.

### Base URL

```
https://api.aitrading-agent.com/api
```

## Authentication

The API supports multiple authentication methods to accommodate different integration scenarios.

### API Key Authentication

The simplest method is API key authentication. Partners are assigned a unique API key that must be included in the request header.

```http
GET /v1/market-data/symbols
X-API-Key: your_api_key_here
```

### OAuth2 Authentication

For more secure integrations, we support OAuth2 authentication. This is recommended for applications acting on behalf of end users.

1. **Get Authorization Code**:
   ```
   GET https://api.aitrading-agent.com/api/auth/authorize?
       client_id=your_client_id&
       response_type=code&
       redirect_uri=your_redirect_uri&
       scope=trading:read,market-data:read
   ```

2. **Exchange Code for Access Token**:
   ```
   POST https://api.aitrading-agent.com/api/auth/token
   Content-Type: application/x-www-form-urlencoded

   grant_type=authorization_code&
   code=authorization_code_from_step_1&
   client_id=your_client_id&
   client_secret=your_client_secret&
   redirect_uri=your_redirect_uri
   ```

3. **Use Access Token**:
   ```http
   GET /v1/market-data/symbols
   Authorization: Bearer your_access_token
   ```

### JWT Authentication

For server-to-server integrations, we support JWT authentication.

```http
GET /v1/market-data/symbols
Authorization: Bearer your_jwt_token
```

## Rate Limiting

To ensure fair usage and system stability, the API implements rate limiting based on partner tiers.

Rate limits are applied per API key and are reset every second. When rate limits are exceeded, the API returns a `429 Too Many Requests` status code.

Example rate limits by tier:
- Public: 5 requests per second
- Basic: 10 requests per second
- Premium: 50 requests per second
- Enterprise: 200 requests per second

## Partner Tiers

The API offers different tiers of access to accommodate various partner needs:

### Public Tier
- Limited to public market data
- Rate limit: 5 requests per second
- Monthly quota: 1,000 requests
- No access to trading, signals, or analytics

### Basic Tier
- Historical market data access
- Rate limit: 10 requests per second
- Monthly quota: 100,000 requests
- Standard support

### Premium Tier
- All Basic features
- Real-time market data
- Trading signals access
- Basic analytics
- Rate limit: 50 requests per second
- Monthly quota: 1,000,000 requests
- Priority support

### Enterprise Tier
- All Premium features
- Unlimited monthly quota
- Advanced analytics and backtesting
- Extended hours trading
- Complex order types
- Dedicated support
- Rate limit: 200 requests per second

## API Endpoints

The API is organized into functional domains:

### Market Data Endpoints

#### List Available Symbols
```
GET /v1/market-data/symbols
```

#### Get Symbol Metadata
```
GET /v1/market-data/symbols/{symbol}
```

#### Get Historical Data
```
POST /v1/market-data/historical
Content-Type: application/json

{
  "symbol": "AAPL",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-01-31T23:59:59Z",
  "interval": "1d"
}
```

#### Get Real-Time Quotes
```
GET /v1/market-data/quotes?symbols=AAPL,MSFT,GOOG
```

#### Get Market Hours
```
GET /v1/market-data/market-hours?markets=NASDAQ,NYSE&date=2023-06-01
```

### Trading Signals Endpoints

#### Generate Signals
```
POST /v1/signals
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT"],
  "timeframe": "daily",
  "signal_types": ["entry", "exit", "stop_loss"],
  "include_analytics": false
}
```

#### Get Historical Signals
```
GET /v1/signals/historical/AAPL?timeframe=daily&start_date=2023-01-01T00:00:00Z
```

#### Get Signal Performance
```
GET /v1/signals/performance?timeframe=daily&start_date=2023-01-01T00:00:00Z
```

### Analytics Endpoints

#### Generate Analytics
```
POST /v1/analytics
Content-Type: application/json

{
  "symbols": ["AAPL"],
  "analysis_types": ["technical", "fundamental"],
  "timeframe": "daily",
  "start_date": "2023-01-01T00:00:00Z",
  "indicators": ["rsi", "macd"]
}
```

#### Get Sector Analysis
```
GET /v1/analytics/sectors?sectors=Technology,Healthcare&timeframe=weekly&start_date=2023-01-01T00:00:00Z
```

#### Analyze Portfolio
```
POST /v1/analytics/portfolio
Content-Type: application/json

{
  "holdings": [
    {"symbol": "AAPL", "weight": 0.4},
    {"symbol": "MSFT", "weight": 0.3},
    {"symbol": "GOOG", "weight": 0.3}
  ],
  "start_date": "2023-01-01T00:00:00Z",
  "benchmark": "SPY"
}
```

### Trading Endpoints

#### Place Order
```
POST /v1/trading/orders
Content-Type: application/json

{
  "symbol": "AAPL",
  "side": "buy",
  "type": "limit",
  "quantity": 10,
  "limit_price": 150.00,
  "time_in_force": "day"
}
```

#### List Orders
```
GET /v1/trading/orders?status=open&symbols=AAPL,MSFT
```

#### Get Order Details
```
GET /v1/trading/orders/{order_id}
```

#### Cancel Order
```
DELETE /v1/trading/orders/{order_id}
```

#### List Positions
```
GET /v1/trading/positions?symbols=AAPL,MSFT
```

#### Get Account Information
```
GET /v1/trading/account
```

### Health and Status Endpoints

#### Get API Health
```
GET /health
```

#### Get System Metrics
```
GET /health/metrics
```

#### Get Endpoint Performance
```
GET /health/endpoints
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure of requests. In case of an error, the response body will contain additional information about the error.

Example error response:
```json
{
  "detail": "Failed to fetch historical data: Symbol not found: XYZ"
}
```

Common error codes:
- `400 Bad Request`: The request was malformed or invalid
- `401 Unauthorized`: Authentication credentials are missing or invalid
- `403 Forbidden`: The authenticated user does not have access to the requested resource
- `404 Not Found`: The requested resource was not found
- `429 Too Many Requests`: Rate limit or quota exceeded
- `500 Internal Server Error`: An unexpected error occurred

## Best Practices

1. **Use Proper Authentication**: Always use the most secure authentication method available for your use case.

2. **Handle Rate Limits**: Implement retry logic with exponential backoff when encountering rate limit errors.

3. **Error Handling**: Properly handle and log errors to ensure a smooth integration.

4. **Request Optimization**: Batch requests where possible to minimize API calls.

5. **Webhook Integration**: For real-time updates, consider using webhooks instead of polling.

6. **Caching**: Implement caching for frequently accessed data that doesn't change often.

7. **Monitor Usage**: Keep track of your API usage to avoid hitting quotas unexpectedly.

## Sample Integrations

### Python Client

```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://api.aitrading-agent.com/api"

def get_historical_data(symbol, start_date, end_date, interval="1d"):
    url = f"{BASE_URL}/v1/market-data/historical"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Example usage
historical_data = get_historical_data("AAPL", "2023-01-01T00:00:00Z", "2023-01-31T23:59:59Z")
print(historical_data)
```

### JavaScript Client

```javascript
const API_KEY = 'your_api_key_here';
const BASE_URL = 'https://api.aitrading-agent.com/api';

async function getHistoricalData(symbol, startDate, endDate, interval = '1d') {
  const url = `${BASE_URL}/v1/market-data/historical`;
  const headers = {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
  };
  const data = {
    symbol,
    start_date: startDate,
    end_date: endDate,
    interval
  };
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(data)
    });
    
    if (response.ok) {
      return await response.json();
    } else {
      console.error(`Error: ${response.status}, ${await response.text()}`);
      return null;
    }
  } catch (error) {
    console.error('Failed to fetch historical data:', error);
    return null;
  }
}

// Example usage
getHistoricalData('AAPL', '2023-01-01T00:00:00Z', '2023-01-31T23:59:59Z')
  .then(data => console.log(data));
```

## Support

For any questions or issues regarding API integration, please contact our support team at api-support@aitrading-agent.com or visit our developer portal at https://developers.aitrading-agent.com.
