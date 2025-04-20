/**
 * Simple tests for circuit breaker functionality
 */

// Import the functions we need to test
import { 
  canMakeApiCall, 
  recordCircuitBreakerResult, 
  resetCircuitBreaker,
  getCircuitBreakerState
} from './monitoring';

describe('Simple Circuit Breaker Tests', () => {
  // Store original Date.now
  const originalDateNow = global.Date.now;
  
  // Create a mockable time value
  let mockTime = 1000;
  
  beforeAll(() => {
    // Mock Date.now globally
    global.Date.now = jest.fn(() => mockTime);
  });
  
  afterAll(() => {
    // Restore original Date.now
    global.Date.now = originalDateNow;
  });
  
  beforeEach(() => {
    // Reset mock time before each test
    mockTime = 1000;
    jest.clearAllMocks();
  });
  
  test('basic circuit breaker functionality', () => {
    // Use unique exchange/method names
    const exchange = 'SimpleTest';
    const method = 'basicTest';
    
    // Reset circuit breaker
    resetCircuitBreaker(exchange, method);
    
    // Configure circuit breaker with low threshold
    const config = { failureThreshold: 2, resetTimeoutMs: 5000 };
    
    // Initially circuit should be closed
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record a failure
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should still be closed after one failure
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record another failure to exceed threshold
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should now be open
    expect(canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Advance time past reset timeout
    mockTime += 6000;
    
    // Circuit should transition to half-open
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record success in half-open state
    recordCircuitBreakerResult(exchange, method, true, config);
    
    // Circuit should be closed
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
  });
});
