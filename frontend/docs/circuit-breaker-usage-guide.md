# Circuit Breaker Pattern Usage Guide

## Overview

The circuit breaker pattern is a design pattern used to detect failures and prevent cascading failures in distributed systems. This document provides a comprehensive guide to using the circuit breaker pattern implemented in the AI Trading Agent's cryptocurrency exchange APIs.

## Table of Contents

1. [Introduction](#introduction)
2. [Key Components](#key-components)
3. [Circuit Breaker States](#circuit-breaker-states)
4. [API Integration](#api-integration)
5. [Error Handling](#error-handling)
6. [Monitoring](#monitoring)
7. [Configuration](#configuration)
8. [Best Practices](#best-practices)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)
11. [Enhanced Logging System](#enhanced-logging-system)

## Introduction

The circuit breaker pattern is designed to:

- Prevent repeated calls to failing services
- Allow services time to recover
- Provide fallback mechanisms
- Maintain system stability during outages
- Collect metrics on API reliability

Our implementation provides a robust solution for handling API failures across multiple cryptocurrency exchanges (Alpaca, Binance, Coinbase) with consistent error handling and fallback strategies.

## Key Components

### 1. Circuit Breaker Core

The core functionality is implemented in the `monitoring.ts` file, which provides:

- Circuit state management
- Failure counting
- Timeout handling
- State transitions

### 2. API Call Wrappers

Each exchange API has a dedicated wrapper function:

- `executeAlpacaCall` - For Alpaca API calls
- `executeBinanceCall` - For Binance API calls
- `executeCoinbaseCall` - For Coinbase API calls

### 3. Error Handling Utilities

Custom error classes and utilities:

- `ApiError` - For API-specific errors
- `NetworkError` - For network-related errors
- Error classification (retryable vs. non-retryable)

### 4. Monitoring and Metrics

Comprehensive metrics tracking:

- Success/failure rates
- Response times
- Error types
- Circuit state changes

## Circuit Breaker States

Our circuit breaker implementation has three states:

### 1. Closed (Normal Operation)

- All API calls proceed normally
- Failures are counted
- When failure threshold is reached, circuit opens

### 2. Open (Blocking Calls)

- API calls are blocked
- Fallback mechanisms are used
- After timeout period, circuit transitions to half-open

### 3. Half-Open (Testing Recovery)

- Limited API calls are allowed
- Success returns circuit to closed state
- Failure returns circuit to open state

## API Integration

### Basic Usage Pattern

```typescript
// Example with Alpaca API
const executeAlpacaCall = async <T>(
  method: string,
  apiCall: () => Promise<T>,
  config: AlpacaConfig,
  fallback?: () => Promise<T>
): Promise<T> => {
  // Check if circuit is closed
  if (!canMakeApiCall('Alpaca', method)) {
    // Circuit is open, use fallback if available
    if (fallback) {
      return await fallback();
    }
    throw new Error(`Circuit open for Alpaca ${method}`);
  }

  // Record attempt
  recordApiCall('Alpaca', method, 'attempt');
  
  try {
    // Start timing
    const startTime = Date.now();
    
    // Execute API call
    const result = await apiCall();
    
    // Calculate duration
    const duration = Date.now() - startTime;
    
    // Record success
    recordApiCall('Alpaca', method, 'success', duration);
    recordCircuitBreakerResult('Alpaca', method, true);
    
    return result;
  } catch (error) {
    // Record failure
    recordApiCall('Alpaca', method, 'failure', 0, error);
    recordCircuitBreakerResult('Alpaca', method, false);
    
    // Use fallback if available
    if (fallback) {
      return await fallback();
    }
    
    // Rethrow error
    throw error;
  }
};
```

### Integration in API Methods

```typescript
// Example API method with circuit breaker
const getPortfolio = async (): Promise<Portfolio> => {
  return await executeAlpacaCall(
    'getPortfolio',
    async () => {
      // Primary API call implementation
      const account = await client.get('/v2/account');
      const positions = await client.get('/v2/positions');
      
      // Process and return data
      return formatPortfolio(account.data, positions.data);
    },
    config,
    async () => {
      // Fallback implementation
      const response = await backendClient.get('/portfolio');
      return response.data.portfolio;
    }
  );
};
```

## Error Handling

### Error Classification

Errors are classified into:

1. **Retryable Errors**
   - Network timeouts
   - Rate limiting
   - Temporary service unavailability

2. **Non-Retryable Errors**
   - Authentication failures
   - Invalid parameters
   - Insufficient funds
   - Permission issues

### Retry Strategy

Our implementation uses exponential backoff for retries:

```typescript
const retryWithBackoff = async <T>(
  fn: () => Promise<T>,
  retries: number = 3,
  delay: number = 300,
  backoff: number = 2
): Promise<T> => {
  try {
    return await fn();
  } catch (error) {
    // Check if error is retryable
    if (!isRetryableError(error) || retries <= 0) {
      throw error;
    }
    
    // Wait with exponential backoff
    await new Promise(resolve => setTimeout(resolve, delay));
    
    // Retry with increased delay
    return retryWithBackoff(fn, retries - 1, delay * backoff, backoff);
  }
};
```

## Monitoring

### Metrics Collection

The system collects the following metrics:

- **Total Calls**: Number of API calls attempted
- **Success Calls**: Number of successful API calls
- **Failed Calls**: Number of failed API calls
- **Success Rate**: Percentage of successful calls
- **Average Duration**: Average response time
- **Min/Max Duration**: Response time range
- **Last Error**: Most recent error encountered
- **Circuit State**: Current state of the circuit breaker

### Accessing Metrics

```typescript
// Get metrics for a specific API
const metrics = getApiCallMetrics('Binance', 'createOrder');

console.log(`Success rate: ${metrics.successCalls / metrics.totalCalls * 100}%`);
console.log(`Average duration: ${metrics.totalDuration / metrics.totalCalls}ms`);
```

## Configuration

### Default Configuration

```typescript
// Default circuit breaker configuration
const DEFAULT_FAILURE_THRESHOLD = 3;
const DEFAULT_RESET_TIMEOUT_MS = 30000; // 30 seconds
```

### Custom Configuration

You can customize the circuit breaker behavior per exchange:

```typescript
// Example custom configuration
const customConfig = {
  failureThreshold: 5,
  resetTimeoutMs: 60000, // 60 seconds
};

// Apply custom configuration
setCircuitBreakerConfig('Binance', 'getMarketData', customConfig);
```

## Best Practices

1. **Always Provide Fallbacks**
   - Implement fallback mechanisms for critical operations
   - Cache previous successful responses for use during outages

2. **Monitor Circuit State**
   - Regularly check circuit breaker status
   - Alert on frequent circuit openings

3. **Tune Thresholds**
   - Adjust failure thresholds based on API reliability
   - Set appropriate timeouts based on operation criticality

4. **Graceful Degradation**
   - Design UIs to handle partial data
   - Communicate service limitations to users

5. **Test Failure Scenarios**
   - Simulate API failures to verify circuit breaker behavior
   - Include circuit breaker tests in CI/CD pipeline

## Enhanced Logging System

The circuit breaker implementation includes a comprehensive logging system that provides detailed insights into API interactions, circuit breaker state changes, and fallback operations.

### Log Levels

The logging system supports multiple log levels to categorize events by severity:

- **DEBUG**: Detailed information for debugging purposes
- **INFO**: Normal operational information, successful API calls
- **WARNING**: Issues that don't prevent operation but require attention
- **ERROR**: Errors that prevent normal operation but don't cause system-wide failures
- **CRITICAL**: Severe errors that may lead to system-wide failures or data corruption

### Logging API Calls

API calls are automatically logged by the circuit breaker executor:

```typescript
import { executeCircuitBreaker } from './api/utils/circuitBreakerExecutor';

// This will automatically log the API call attempt, success/failure, and any fallbacks used
const result = await executeCircuitBreaker({
  exchange: 'binance',
  method: 'getAccountBalance',
  apiCall: () => binanceClient.getAccountBalance(),
});
```

### Manual Logging

You can also manually log events using the logging API:

```typescript
import { 
  logApiCallAttempt, 
  logApiCallSuccess, 
  logApiCallFailure,
  LogLevel 
} from './api/utils/enhancedLogging';

// Log an API call attempt
logApiCallAttempt('binance', 'getMarketData');

// Log a custom message
log({
  level: LogLevel.WARNING,
  exchange: 'binance',
  method: 'placeOrder',
  message: 'Order placed but confirmation delayed',
  metadata: { orderId: '12345', status: 'pending' }
});
```

### Accessing Logs

Logs can be accessed programmatically or through the API Logs Dashboard:

```typescript
import { getLogs } from './api/utils/enhancedLogging';

// Get all logs
const allLogs = getLogs();

// Get filtered logs
const binanceErrorLogs = getLogs({
  exchange: 'binance',
  level: LogLevel.ERROR,
  startTime: Date.now() - (24 * 60 * 60 * 1000) // Last 24 hours
});
```

### API Logs Dashboard

The API Logs Dashboard (`/api-logs`) provides a user-friendly interface for viewing, filtering, and analyzing logs:

- Filter logs by exchange, method, level, and time range
- View detailed information for each log entry
- Export logs for external analysis
- Clear logs when no longer needed

## Examples

### Basic Example: Market Price Retrieval

```typescript
// Get market price with circuit breaker
const getMarketPrice = async (symbol: string): Promise<number> => {
  return await executeBinanceCall(
    'getMarketPrice',
    async () => {
      const response = await client.get('/api/v3/ticker/price', {
        params: { symbol: formatSymbol(symbol) }
      });
      return parseFloat(response.data.price);
    },
    config,
    async () => {
      // Fallback to cached price or alternative source
      const cachedPrice = await getCachedPrice(symbol);
      if (cachedPrice) return cachedPrice;
      
      // Try alternative API
      const altResponse = await alternativeClient.get(`/prices/${symbol}`);
      return parseFloat(altResponse.data.price);
    }
  );
};
```

### Advanced Example: Order Creation with Retry

```typescript
// Create order with circuit breaker and retry
const createOrder = async (orderRequest: OrderRequest): Promise<Order> => {
  return await executeAlpacaCall(
    'createOrder',
    async () => {
      // Convert order to exchange format
      const alpacaOrder = convertToAlpacaOrder(orderRequest);
      
      // Use retry with backoff for order creation
      return await retryWithBackoff(
        async () => {
          const response = await client.post('/v2/orders', alpacaOrder);
          return convertFromAlpacaOrder(response.data);
        },
        3, // 3 retries
        500, // 500ms initial delay
        2 // Double delay each retry
      );
    },
    config,
    async () => {
      // Fallback to backend order creation
      const response = await backendClient.post('/orders', orderRequest);
      return response.data.order;
    }
  );
};
```

## Troubleshooting

### Common Issues

1. **Frequent Circuit Opening**
   - Check API health and stability
   - Review failure thresholds
   - Examine error patterns

2. **Slow Recovery**
   - Adjust reset timeout
   - Implement more aggressive backoff
   - Check for persistent API issues

3. **Fallback Failures**
   - Ensure fallback mechanisms are tested
   - Implement multiple fallback layers
   - Cache critical data

### Debugging

```typescript
// Enable debug logging
const enableDebugLogging = true;

// Debug circuit breaker state changes
const debugCircuitBreaker = (
  exchange: string,
  method: string,
  oldState: string,
  newState: string
) => {
  if (enableDebugLogging) {
    console.log(
      `[Circuit Breaker] ${exchange}.${method}: ${oldState} -> ${newState}`
    );
  }
};
```

## Conclusion

The circuit breaker pattern provides a robust solution for handling API failures in distributed systems. By implementing this pattern in your trading applications, you can ensure system stability, graceful degradation, and improved user experience even during service disruptions.

For more detailed information, refer to the implementation in:
- `frontend/src/api/utils/monitoring.ts`
- `frontend/src/api/trading/alpacaTradingApi.ts`
- `frontend/src/api/trading/binanceTradingApi.ts`
- `frontend/src/api/trading/coinbaseTradingApi.ts`
