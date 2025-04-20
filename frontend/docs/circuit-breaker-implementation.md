# Circuit Breaker Pattern Implementation

## Overview

The circuit breaker pattern has been implemented across all trading APIs (Alpaca, Binance, Coinbase) to enhance error handling, improve system resilience, and create a consistent error management strategy. This document provides a comprehensive overview of the implementation.

## Core Components

### 1. Error Handling Utilities (`errorHandling.ts`)

The error handling utilities provide the foundation for the circuit breaker pattern:

- **Custom Error Classes**:
  - `ApiError`: For API-specific errors with status codes
  - `NetworkError`: For network-related errors

- **Retry Logic**:
  - `withRetry`: Implements exponential backoff retry mechanism
  - Configurable retry parameters (max retries, delay, etc.)
  - Intelligent retry decisions based on error type

- **API Call Execution**:
  - `executeApiCall`: Wrapper for API calls with retry, logging, and timing
  - Measures execution time and handles errors consistently

### 2. Monitoring Utilities (`monitoring.ts`)

The monitoring utilities track API call performance and manage circuit breaker state:

- **API Call Metrics**:
  - `recordApiCall`: Records metrics for API calls (success, failure, duration)
  - `getApiCallMetrics`: Retrieves metrics for specific API methods

- **Circuit Breaker State Management**:
  - `canMakeApiCall`: Determines if an API call should be allowed based on circuit state
  - `recordCircuitBreakerResult`: Updates circuit state based on API call results
  - `resetCircuitBreaker`: Manually resets circuit breaker state
  - `getCircuitBreakerState`: Retrieves current circuit breaker state

- **Circuit Breaker States**:
  - `closed`: Normal operation, all calls allowed
  - `open`: Circuit is tripped, calls are blocked
  - `half-open`: Testing if service has recovered, limited calls allowed

### 3. Trading API Integration

Each trading API has been refactored to use the circuit breaker pattern:

#### Alpaca Trading API (`alpacaTradingApi.ts`)

```typescript
// Example of executeAlpacaCall function
const executeAlpacaCall = async <T>(
  method: string,
  apiCall: () => Promise<T>,
  fallback?: () => Promise<T>
): Promise<T> => {
  // Check if circuit breaker allows the call
  if (!canMakeApiCall('Alpaca', method)) {
    if (fallback) {
      return fallback();
    }
    throw new Error('Circuit breaker is open');
  }

  try {
    // Record attempt
    recordApiCall('Alpaca', method, 'attempt');
    
    // Execute API call with retry
    const result = await executeApiCall(apiCall);
    
    // Record success
    recordApiCall('Alpaca', method, 'success');
    recordCircuitBreakerResult('Alpaca', method, true);
    
    return result;
  } catch (error) {
    // Record failure
    recordApiCall('Alpaca', method, 'failure');
    recordCircuitBreakerResult('Alpaca', method, false);
    
    // Use fallback if available
    if (fallback) {
      return fallback();
    }
    
    throw error;
  }
};
```

#### Binance Trading API (`binanceTradingApi.ts`)

```typescript
// Example of executeBinanceCall function
const executeBinanceCall = async <T>(
  method: string,
  apiCall: () => Promise<T>,
  fallback?: () => Promise<T>
): Promise<T> => {
  // Check if circuit breaker allows the call
  if (!canMakeApiCall('Binance', method)) {
    if (fallback) {
      return fallback();
    }
    throw new Error('Circuit breaker is open');
  }

  try {
    // Record attempt
    recordApiCall('Binance', method, 'attempt');
    
    // Execute API call with retry
    const result = await executeApiCall(apiCall);
    
    // Record success
    recordApiCall('Binance', method, 'success');
    recordCircuitBreakerResult('Binance', method, true);
    
    return result;
  } catch (error) {
    // Record failure
    recordApiCall('Binance', method, 'failure');
    recordCircuitBreakerResult('Binance', method, false);
    
    // Use fallback if available
    if (fallback) {
      return fallback();
    }
    
    throw error;
  }
};
```

#### Coinbase Trading API (`coinbaseTradingApi.ts`)

```typescript
// Example of executeCoinbaseCall function
const executeCoinbaseCall = async <T>(
  method: string,
  apiCall: () => Promise<T>,
  fallback?: () => Promise<T>
): Promise<T> => {
  // Check if circuit breaker allows the call
  if (!canMakeApiCall('Coinbase', method)) {
    if (fallback) {
      return fallback();
    }
    throw new Error('Circuit breaker is open');
  }

  try {
    // Record attempt
    recordApiCall('Coinbase', method, 'attempt');
    
    // Execute API call with retry
    const result = await executeApiCall(apiCall);
    
    // Record success
    recordApiCall('Coinbase', method, 'success');
    recordCircuitBreakerResult('Coinbase', method, true);
    
    return result;
  } catch (error) {
    // Record failure
    recordApiCall('Coinbase', method, 'failure');
    recordCircuitBreakerResult('Coinbase', method, false);
    
    // Use fallback if available
    if (fallback) {
      return fallback();
    }
    
    throw error;
  }
};
```

## Circuit Breaker Configuration

The circuit breaker pattern is configurable with the following parameters:

```typescript
// Default circuit breaker configuration
const defaultCircuitBreakerConfig: CircuitBreakerConfig = {
  failureThreshold: 3,       // Number of failures before opening circuit
  resetTimeoutMs: 30000,     // Time to wait before testing service again (30s)
};
```

## Fallback Strategies

Each trading API implements fallback strategies when the circuit breaker is open:

1. **Backend Fallback**: Use the backend API as a fallback source for data
2. **Cached Data**: Return cached data when available
3. **Default Values**: Return sensible defaults when no other fallback is available

## Monitoring and Metrics

The implementation includes comprehensive metrics tracking:

- **API Call Metrics**:
  - Total calls, success rate, failure rate
  - Response times (min, max, average)
  - Last error information

- **Circuit Breaker Metrics**:
  - Current state (closed, open, half-open)
  - Failure count
  - Last failure time
  - Next attempt time (for open circuits)

## Usage Examples

### Getting Portfolio Data with Circuit Breaker

```typescript
// In alpacaTradingApi.ts
export const getPortfolio = async (): Promise<Portfolio> => {
  return executeAlpacaCall(
    'getPortfolio',
    async () => {
      // Primary implementation using Alpaca API
      const accountInfo = await client.get('/v2/account');
      const positions = await client.get('/v2/positions');
      
      // Transform data to Portfolio format
      return {
        cash: parseFloat(accountInfo.data.cash),
        total_value: parseFloat(accountInfo.data.portfolio_value),
        // ... other portfolio data
      };
    },
    async () => {
      // Fallback implementation using backend API
      const response = await backendClient.get('/portfolio');
      return response.data.portfolio;
    }
  );
};
```

### Creating Orders with Circuit Breaker

```typescript
// In binanceTradingApi.ts
export const createOrder = async (orderRequest: OrderRequest): Promise<Order> => {
  return executeBinanceCall(
    'createOrder',
    async () => {
      // Primary implementation using Binance API
      const { symbol, side, order_type, quantity, price } = orderRequest;
      
      // Transform to Binance format
      const binanceOrder = {
        symbol: formatSymbol(symbol),
        side: side.toUpperCase(),
        type: order_type.toUpperCase(),
        quantity,
        price: price || undefined,
        timeInForce: price ? 'GTC' : undefined,
      };
      
      // Send order to Binance
      const response = await client.post('/api/v3/order', binanceOrder);
      
      // Transform response to Order format
      return convertBinanceOrder(response.data);
    },
    async () => {
      // Fallback implementation using backend API
      const response = await backendClient.post('/orders', orderRequest);
      return response.data.order;
    }
  );
};
```

## Benefits of the Implementation

1. **Improved Resilience**: The system can gracefully handle API failures without cascading failures
2. **Consistent Error Handling**: Standardized approach across all trading APIs
3. **Intelligent Retry Logic**: Exponential backoff with differentiation between retryable and non-retryable errors
4. **Comprehensive Monitoring**: Detailed metrics for API performance and circuit breaker state
5. **Fallback Strategies**: Multiple layers of fallbacks to ensure system availability

## Testing

The circuit breaker implementation has been tested with:

1. **Unit Tests**: Testing individual components (error handling, retry logic, circuit breaker state)
2. **Integration Tests**: Testing the interaction between components
3. **Scenario Tests**: Testing specific failure scenarios (API timeouts, rate limiting, authentication failures)

## Conclusion

The circuit breaker pattern implementation provides a robust error handling mechanism across all trading APIs, improving the overall reliability and fault tolerance of the AI Trading Agent platform. The standardized approach ensures consistent behavior and monitoring capabilities, while the fallback strategies maintain system availability even during external service disruptions.
