/**
 * Circuit Breaker Dashboard
 * Provides utilities for monitoring and visualizing circuit breaker states
 */
import { getCircuitBreakerState, getAllMetrics, getApiHealthDashboard } from './monitoring';

// Dashboard data interface
export interface CircuitBreakerDashboardData {
  exchanges: Record<string, ExchangeDashboardData>;
  overallHealth: boolean;
  circuitBreakerEvents: CircuitBreakerEvent[];
  criticality: {
    critical: number;
    important: number;
    standard: number;
  };
}

export interface ExchangeDashboardData {
  name: string;
  health: boolean;
  circuitStates: Record<string, {
    operation: string;
    state: 'closed' | 'open' | 'half-open';
    sinceTime: number; // timestamp when state changed
    failureCount: number;
    totalOpenCount: number; // how many times circuit opened
    remainingTimeMs?: number;
  }>;
  successRate: number;
  callCount: number;
  failureCount: number;
}

export interface CircuitBreakerEvent {
  exchange: string;
  operation: string;
  event: 'open' | 'close' | 'half-open' | 'reset';
  timestamp: number;
  failureCount?: number;
  errorType?: string;
}

// In-memory store for events
const circuitBreakerEvents: CircuitBreakerEvent[] = [];

// Maximum number of events to keep in memory
const MAX_EVENTS = 100;

// Add a circuit breaker event
export const addCircuitBreakerEvent = (event: CircuitBreakerEvent): void => {
  circuitBreakerEvents.unshift(event);
  
  // Keep only the last MAX_EVENTS events
  if (circuitBreakerEvents.length > MAX_EVENTS) {
    circuitBreakerEvents.length = MAX_EVENTS;
  }
  
  // Optionally log to console
  console.log(`[CB Event] ${event.exchange} ${event.operation}: ${event.event}`);
};

// Get circuit breaker dashboard data
export const getCircuitBreakerDashboard = (): CircuitBreakerDashboardData => {
  // Get API health data
  const apiHealthData = getApiHealthDashboard();
  
  // Initialize dashboard data
  const dashboard: CircuitBreakerDashboardData = {
    exchanges: {},
    overallHealth: true,
    circuitBreakerEvents: [...circuitBreakerEvents],
    criticality: {
      critical: 0,
      important: 0,
      standard: 0,
    },
  };
  
  // Process each exchange
  for (const exchangeName in apiHealthData.exchanges) {
    const exchangeData = apiHealthData.exchanges[exchangeName];
    
    // Create exchange dashboard data
    dashboard.exchanges[exchangeName] = {
      name: exchangeName,
      health: exchangeData.health,
      circuitStates: {},
      successRate: exchangeData.successRate || 1,
      callCount: exchangeData.totalCalls || 0,
      failureCount: exchangeData.failedCalls || 0,
    };
    
    // Check health
    if (!exchangeData.health) {
      dashboard.overallHealth = false;
    }
    
    // Process circuit breaker states for each method
    for (const method in exchangeData.circuitBreakerStates) {
      const circuitState = exchangeData.circuitBreakerStates[method];
      
      if (!circuitState) continue;
      
      // Check if state is open or half-open
      if (circuitState.state === 'open' || circuitState.state === 'half-open') {
        const stateInfo = getCircuitBreakerState(exchangeName, method);
        
        if (stateInfo) {
          dashboard.exchanges[exchangeName].circuitStates[method] = {
            operation: method,
            state: circuitState.state as any,
            sinceTime: stateInfo.lastStateChangeTime || Date.now(),
            failureCount: stateInfo.failureCount || 0,
            totalOpenCount: stateInfo.openCount || 0,
            remainingTimeMs: stateInfo.remainingTimeMs || 0,
          };
          
          // Count by criticality
          if (method.includes('ORDER') || method === 'CREATE_ORDER' || method === 'CANCEL_ORDER') {
            dashboard.criticality.critical++;
          } else if (method.includes('PORTFOLIO') || method.includes('POSITION') || method === 'GET_PORTFOLIO') {
            dashboard.criticality.important++;
          } else {
            dashboard.criticality.standard++;
          }
        }
      }
    }
  }
  
  return dashboard;
};

// Get alerts for the dashboard
export const getCircuitBreakerAlerts = (): string[] => {
  const dashboard = getCircuitBreakerDashboard();
  const alerts: string[] = [];
  
  // Check for critical alerts
  if (dashboard.criticality.critical > 0) {
    alerts.push(`CRITICAL: ${dashboard.criticality.critical} critical operations are currently unavailable`);
  }
  
  // Check for important alerts
  if (dashboard.criticality.important > 0) {
    alerts.push(`WARNING: ${dashboard.criticality.important} important operations are degraded`);
  }
  
  // Check exchange health
  for (const exchangeName in dashboard.exchanges) {
    const exchangeData = dashboard.exchanges[exchangeName];
    if (!exchangeData.health) {
      alerts.push(`EXCHANGE: ${exchangeName} is experiencing issues (${Object.keys(exchangeData.circuitStates).length} operations affected)`);
    }
  }
  
  return alerts;
};