// Mock implementation for monitoring utilities
import { mockMonitoring } from '../tests/mocks/globalMocks';

// Define the ApiCallMetrics interface to match the real implementation
export interface ApiCallMetrics {
  totalCalls: number;
  successCalls: number;
  failedCalls: number;
  totalDuration: number;
  minDuration: number;
  maxDuration: number;
  lastCallTime: number;
  lastError?: Error;
}

// Get all mock functions from the global mock
const monitoringMocks = mockMonitoring();

// Export all the mock functions
export const canMakeApiCall = monitoringMocks.canMakeApiCall;
export const recordApiCall = monitoringMocks.recordApiCall;
export const recordCircuitBreakerResult = monitoringMocks.recordCircuitBreakerResult;
export const resetCircuitBreaker = monitoringMocks.resetCircuitBreaker;
export const getApiCallMetrics = monitoringMocks.getApiCallMetrics;
export const getCircuitBreakerState = monitoringMocks.getCircuitBreakerState;
export const getAllMetrics = monitoringMocks.getAllMetrics;
export const getSuccessRate = monitoringMocks.getSuccessRate;
export const getAverageDuration = monitoringMocks.getAverageDuration;
export const isApiHealthy = monitoringMocks.isApiHealthy;
export const getApiHealthDashboard = monitoringMocks.getApiHealthDashboard;

// For backward compatibility
export const getEnhancedApiMetrics = jest.fn().mockReturnValue({});
