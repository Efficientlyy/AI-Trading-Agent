/**
 * Trading Agent Circuit Breaker integration
 * 
 * This file connects the circuit breaker system with the trading agent,
 * allowing for intelligent routing of API calls and handling of failures.
 */
import { TradingApi } from '../trading';
import { addCircuitBreakerEvent, getCircuitBreakerAlerts, getCircuitBreakerDashboard } from './circuitBreakerDashboard';
import { getCircuitBreakerState } from './monitoring';

// Interface for API health metrics
export interface ApiHealthMetrics {
  overallHealth: boolean;
  exchanges: Record<string, {
    name: string;
    health: boolean;
    successRate: number;
    apiStatus: 'available' | 'degraded' | 'unavailable';
  }>;
  alerts: string[];
}

// Interface for routing priority
export interface RoutingPriority {
  primary: string;
  secondary?: string;
  tertiary?: string;
}

/**
 * Get health metrics for all trading APIs
 */
export const getApiHealthMetrics = (): ApiHealthMetrics => {
  // Get dashboard data
  const dashboard = getCircuitBreakerDashboard();
  const alerts = getCircuitBreakerAlerts();

  // Build API health metrics
  const metrics: ApiHealthMetrics = {
    overallHealth: dashboard.overallHealth,
    exchanges: {},
    alerts,
  };

  // Process each exchange
  for (const exchangeName in dashboard.exchanges) {
    const exchangeData = dashboard.exchanges[exchangeName];
    const circuitIssues = Object.keys(exchangeData.circuitStates).length;

    // Determine API status
    let apiStatus: 'available' | 'degraded' | 'unavailable' = 'available';
    if (circuitIssues > 5 || !exchangeData.health) {
      apiStatus = 'unavailable';
    } else if (circuitIssues > 0) {
      apiStatus = 'degraded';
    }

    metrics.exchanges[exchangeName] = {
      name: exchangeName,
      health: exchangeData.health,
      successRate: exchangeData.successRate,
      apiStatus,
    };
  }

  return metrics;
};

/**
 * Determine the optimal routing priority for API calls based on health metrics
 * @param preferredExchange The preferred exchange to use (if available)
 */
export const determineRoutingPriority = (preferredExchange?: string): RoutingPriority => {
  const metrics = getApiHealthMetrics();
  const availableExchanges: string[] = [];
  const degradedExchanges: string[] = [];

  // Categorize exchanges by status
  for (const exchangeName in metrics.exchanges) {
    const exchange = metrics.exchanges[exchangeName];

    if (exchange.apiStatus === 'available') {
      availableExchanges.push(exchangeName);
    } else if (exchange.apiStatus === 'degraded') {
      degradedExchanges.push(exchangeName);
    }
  }

  // If preferred exchange is specified and available, use it as primary
  if (preferredExchange && metrics.exchanges[preferredExchange]?.apiStatus !== 'unavailable') {
    const priority: RoutingPriority = {
      primary: preferredExchange
    };

    // Add secondary and tertiary if available
    for (const exchange of availableExchanges) {
      if (exchange !== preferredExchange) {
        if (!priority.secondary) {
          priority.secondary = exchange;
        } else if (!priority.tertiary) {
          priority.tertiary = exchange;
          break;
        }
      }
    }

    // If we don't have enough, check degraded exchanges
    if (!priority.secondary || !priority.tertiary) {
      for (const exchange of degradedExchanges) {
        if (exchange !== preferredExchange && exchange !== priority.secondary) {
          if (!priority.secondary) {
            priority.secondary = exchange;
          } else if (!priority.tertiary) {
            priority.tertiary = exchange;
            break;
          }
        }
      }
    }

    return priority;
  }

  // No preferred exchange or it's unavailable, use best available
  if (availableExchanges.length > 0) {
    return {
      primary: availableExchanges[0],
      secondary: availableExchanges[1] || degradedExchanges[0],
      tertiary: availableExchanges[2] || degradedExchanges[0] || degradedExchanges[1],
    };
  }

  // Only degraded exchanges available
  if (degradedExchanges.length > 0) {
    return {
      primary: degradedExchanges[0],
      secondary: degradedExchanges[1],
      tertiary: degradedExchanges[2],
    };
  }

  // No exchanges available, return default
  return {
    primary: preferredExchange || 'Binance',
  };
};

/**
 * Execute a trading operation with automatic routing to the best available API
 * @param operation The operation name (e.g., GET_PORTFOLIO, CREATE_ORDER)
 * @param tradingApis Map of available trading APIs
 * @param executeFunc The function to execute on the selected API
 * @param preferredExchange The preferred exchange to use (if available)
 */
export const executeTradingOperation = async <T>(
  operation: string,
  tradingApis: Record<string, TradingApi>,
  executeFunc: (api: TradingApi) => Promise<T>,
  preferredExchange?: string
): Promise<T> => {
  // Determine routing priority
  const routing = determineRoutingPriority(preferredExchange);

  // Log the routing decision
  console.log(`[Routing] ${operation}: ${routing.primary} -> ${routing.secondary || 'none'} -> ${routing.tertiary || 'none'}`);

  // Execute on primary exchange
  try {
    const api = tradingApis[routing.primary];
    if (!api) {
      throw new Error(`API for ${routing.primary} not found`);
    }

    const result = await executeFunc(api);

    // Record success event
    addCircuitBreakerEvent({
      exchange: routing.primary,
      operation,
      event: 'close', // Assuming success keeps circuit closed
      timestamp: Date.now(),
    });

    return result;
  } catch (error) {
    console.error(`[Routing] ${operation} failed on ${routing.primary}:`, error);

    // Record failure event
    addCircuitBreakerEvent({
      exchange: routing.primary,
      operation,
      event: 'open',
      timestamp: Date.now(),
      errorType: error instanceof Error ? error.name : 'Unknown',
    });

    // Try secondary exchange if available
    if (routing.secondary && tradingApis[routing.secondary]) {
      try {
        console.log(`[Routing] ${operation}: Trying secondary exchange ${routing.secondary}`);
        const result = await executeFunc(tradingApis[routing.secondary]);

        // Record success event
        addCircuitBreakerEvent({
          exchange: routing.secondary,
          operation,
          event: 'close',
          timestamp: Date.now(),
        });

        return result;
      } catch (secondaryError) {
        console.error(`[Routing] ${operation} failed on ${routing.secondary}:`, secondaryError);

        // Record failure event
        addCircuitBreakerEvent({
          exchange: routing.secondary,
          operation,
          event: 'open',
          timestamp: Date.now(),
          errorType: secondaryError instanceof Error ? secondaryError.name : 'Unknown',
        });

        // Try tertiary exchange if available
        if (routing.tertiary && tradingApis[routing.tertiary]) {
          try {
            console.log(`[Routing] ${operation}: Trying tertiary exchange ${routing.tertiary}`);
            const result = await executeFunc(tradingApis[routing.tertiary]);

            // Record success event
            addCircuitBreakerEvent({
              exchange: routing.tertiary,
              operation,
              event: 'close',
              timestamp: Date.now(),
            });

            return result;
          } catch (tertiaryError) {
            console.error(`[Routing] ${operation} failed on ${routing.tertiary}:`, tertiaryError);

            // Record failure event
            addCircuitBreakerEvent({
              exchange: routing.tertiary,
              operation,
              event: 'open',
              timestamp: Date.now(),
              errorType: tertiaryError instanceof Error ? tertiaryError.name : 'Unknown',
            });

            // All exchanges failed, rethrow the original error
            throw error;
          }
        } else {
          // No tertiary exchange, rethrow the secondary error
          throw secondaryError;
        }
      }
    } else {
      // No secondary exchange, rethrow the original error
      throw error;
    }
  }
};

/**
 * Check if a specific operation is available on a specific exchange
 * @param exchange The exchange name
 * @param operation The operation to check
 */
export const isOperationAvailable = (exchange: string, operation: string): boolean => {
  const state = getCircuitBreakerState(exchange, operation);

  // If circuit breaker state doesn't exist or is closed, the operation is available
  if (!state || state.state === 'closed') {
    return true;
  }

  // If circuit breaker is half-open, check if we can make a call
  if (state.state === 'half-open' &&
    typeof state.halfOpenCallCount === 'number' &&
    typeof state.halfOpenSuccessCount === 'number') {
    // Default to 1 if halfOpenMaxCalls is not defined
    const maxCalls = 1;
    return state.halfOpenCallCount < maxCalls;
  }

  // Operation is unavailable
  return false;
};