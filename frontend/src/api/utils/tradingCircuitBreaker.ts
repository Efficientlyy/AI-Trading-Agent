/**
 * Trading-specific circuit breaker configuration and utilities.
 * This file provides common configurations and helper functions for trading APIs.
 */
import { CircuitBreakerOptions, executeWithCircuitBreaker } from './circuitBreakerExecutor';
import { CircuitBreakerConfig } from './monitoring';

// Standard circuit breaker configurations for different types of trading API calls
export const CircuitBreakerConfigs = {
  // Critical operations (e.g., order placement, cancellation)
  CRITICAL: {
    failureThreshold: 2, // Open circuit after just 2 failures
    resetTimeoutMs: 10000, // 10 seconds timeout before trying again
    halfOpenMaxCalls: 1, // Allow only 1 call in half-open state
    halfOpenSuccessThreshold: 2, // Require 2 successes to fully close circuit
  } as CircuitBreakerConfig,

  // Important operations (e.g., portfolio retrieval, market data)
  IMPORTANT: {
    failureThreshold: 3,
    resetTimeoutMs: 20000, // 20 seconds timeout
    halfOpenMaxCalls: 2,
    halfOpenSuccessThreshold: 1,
  } as CircuitBreakerConfig,

  // Standard operations (e.g., ticker data, general market info)
  STANDARD: {
    failureThreshold: 5,
    resetTimeoutMs: 30000, // 30 seconds timeout
    halfOpenMaxCalls: 3,
    halfOpenSuccessThreshold: 1,
  } as CircuitBreakerConfig,
};

// Operation types for different trading API methods
export const OperationType = {
  // Critical operations
  CREATE_ORDER: 'CRITICAL',
  CANCEL_ORDER: 'CRITICAL',
  MODIFY_ORDER: 'CRITICAL',
  CLOSE_POSITION: 'CRITICAL',

  // Important operations
  GET_PORTFOLIO: 'IMPORTANT',
  GET_POSITIONS: 'IMPORTANT',
  GET_ORDERS: 'IMPORTANT',
  GET_ACCOUNT_INFO: 'IMPORTANT',

  // Standard operations
  GET_MARKET_DATA: 'STANDARD',
  GET_PRICE: 'STANDARD',
  GET_TICKER: 'STANDARD',
  GET_HISTORY: 'STANDARD',
};

// Exchange-specific configurations (can override the standard configurations)
export const ExchangeConfigs = {
  Alpaca: {
    // Custom settings for specific Alpaca operations if needed
    CREATE_ORDER: {
      failureThreshold: 3, // More tolerant for Alpaca order creation
    },
  },
  Binance: {
    // Binance-specific settings
    GET_MARKET_DATA: {
      failureThreshold: 8, // More tolerant for rate limiting
      resetTimeoutMs: 45000, // Longer timeout for Binance API
    },
  },
  Coinbase: {
    // Coinbase-specific settings
  },
};

/**
 * Gets the appropriate circuit breaker config for a given exchange and operation
 */
export const getCircuitBreakerConfig = (
  exchange: string,
  operation: string
): CircuitBreakerConfig => {
  // Get the operation type (CRITICAL, IMPORTANT, STANDARD)
  const operationType = (OperationType as Record<string, string>)[operation] || 'STANDARD';

  // Get base config for this operation type
  const baseConfig = CircuitBreakerConfigs[operationType as keyof typeof CircuitBreakerConfigs] || CircuitBreakerConfigs.STANDARD;

  // Check for exchange-specific overrides
  const exchangeConfig = (ExchangeConfigs as Record<string, any>)[exchange];
  if (exchangeConfig && exchangeConfig[operation]) {
    return {
      ...baseConfig,
      ...exchangeConfig[operation],
    };
  }

  return baseConfig;
};

/**
 * Execute a trading API call with appropriate circuit breaker configuration
 * based on the exchange and operation type
 */
export const executeTrading = async <T>(
  apiCall: () => Promise<T>,
  exchange: string,
  operation: string,
  options: Partial<CircuitBreakerOptions<T>> = {}
): Promise<T> => {
  // Get appropriate circuit breaker configuration
  const circuitBreakerConfig = getCircuitBreakerConfig(exchange, operation);

  // Determine if this is a critical operation
  const isCritical = (OperationType as any)[operation] === 'CRITICAL';

  // Execute with circuit breaker
  return executeWithCircuitBreaker(apiCall, {
    exchange,
    method: operation,
    maxRetries: circuitBreakerConfig.failureThreshold,
    initialDelayMs: 500,
    maxDelayMs: 5000,
    isCritical,
    ...options,
  });
};

/**
 * Helper function to log circuit breaker events for trading APIs
 */
export const logTradingCircuitBreakerEvent = (
  exchange: string,
  operation: string,
  eventType: 'open' | 'close' | 'half-open' | 'success' | 'failure',
  details: any = {}
): void => {
  console.log(
    `[Circuit Breaker] ${exchange} ${operation} - ${eventType.toUpperCase()}`,
    details
  );
};