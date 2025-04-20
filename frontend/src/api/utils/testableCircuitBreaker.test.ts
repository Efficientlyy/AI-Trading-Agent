/**
 * Tests for the testable circuit breaker implementation
 */
import { CircuitBreaker, CircuitBreakerRegistry } from './testableCircuitBreaker';

describe('CircuitBreaker', () => {
  // Mock time provider for predictable tests
  let mockTime = 1000;
  const mockTimeProvider = jest.fn(() => mockTime);
  
  beforeEach(() => {
    // Reset mock time before each test
    mockTime = 1000;
    mockTimeProvider.mockClear();
  });
  
  test('should start in closed state', () => {
    const breaker = new CircuitBreaker({}, mockTimeProvider);
    expect(breaker.canMakeCall()).toBe(true);
    expect(breaker.getState().state).toBe('closed');
  });
  
  test('should open after threshold failures', () => {
    const breaker = new CircuitBreaker({ failureThreshold: 3 }, mockTimeProvider);
    
    // Record two failures
    breaker.recordResult(false);
    breaker.recordResult(false);
    
    // Should still be closed
    expect(breaker.canMakeCall()).toBe(true);
    
    // Record one more failure to exceed threshold
    breaker.recordResult(false);
    
    // Should now be open
    expect(breaker.canMakeCall()).toBe(false);
    expect(breaker.getState().state).toBe('open');
  });
  
  test('should transition to half-open after timeout', () => {
    const breaker = new CircuitBreaker({ 
      failureThreshold: 3, 
      resetTimeoutMs: 5000 
    }, mockTimeProvider);
    
    // Open the circuit
    breaker.recordResult(false);
    breaker.recordResult(false);
    breaker.recordResult(false);
    
    // Should be open
    expect(breaker.canMakeCall()).toBe(false);
    expect(breaker.getState().state).toBe('open');
    
    // Advance time past timeout
    mockTime += 6000;
    
    // Should now be half-open (the call to canMakeCall triggers the state change)
    const canCall = breaker.canMakeCall();
    expect(canCall).toBe(true);
    expect(breaker.getState().state).toBe('half-open');
  });
  
  test('should close after success in half-open state', () => {
    const breaker = new CircuitBreaker({ 
      failureThreshold: 3, 
      resetTimeoutMs: 5000 
    }, mockTimeProvider);
    
    // Open the circuit
    breaker.recordResult(false);
    breaker.recordResult(false);
    breaker.recordResult(false);
    
    // Advance time past timeout
    mockTime += 6000;
    
    // Transition to half-open (the call to canMakeCall triggers the state change)
    breaker.canMakeCall();
    expect(breaker.getState().state).toBe('half-open');
    
    // Record success in half-open state
    breaker.recordResult(true);
    
    // Should now be closed
    expect(breaker.canMakeCall()).toBe(true);
    expect(breaker.getState().state).toBe('closed');
  });
  
  test('should reopen after failure in half-open state', () => {
    const breaker = new CircuitBreaker({ 
      failureThreshold: 3, 
      resetTimeoutMs: 5000 
    }, mockTimeProvider);
    
    // Open the circuit
    breaker.recordResult(false);
    breaker.recordResult(false);
    breaker.recordResult(false);
    
    // Advance time past timeout
    mockTime += 6000;
    
    // Transition to half-open (the call to canMakeCall triggers the state change)
    breaker.canMakeCall();
    expect(breaker.getState().state).toBe('half-open');
    
    // Record failure in half-open state
    breaker.recordResult(false);
    
    // Should be open again
    expect(breaker.canMakeCall()).toBe(false);
    expect(breaker.getState().state).toBe('open');
  });
  
  test('should reset to initial state', () => {
    const breaker = new CircuitBreaker({ 
      failureThreshold: 3, 
      resetTimeoutMs: 5000 
    }, mockTimeProvider);
    
    // Open the circuit
    breaker.recordResult(false);
    breaker.recordResult(false);
    breaker.recordResult(false);
    
    // Should be open
    expect(breaker.canMakeCall()).toBe(false);
    
    // Reset the circuit breaker
    breaker.reset();
    
    // Should be closed again
    expect(breaker.canMakeCall()).toBe(true);
    expect(breaker.getState().state).toBe('closed');
    expect(breaker.getState().failureCount).toBe(0);
  });
});

describe('CircuitBreakerRegistry', () => {
  // Mock time provider for predictable tests
  let mockTime = 1000;
  const mockTimeProvider = jest.fn(() => mockTime);
  
  beforeEach(() => {
    // Reset mock time before each test
    mockTime = 1000;
    mockTimeProvider.mockClear();
  });
  
  test('should manage multiple circuit breakers', () => {
    const registry = new CircuitBreakerRegistry({}, mockTimeProvider);
    
    // Create two circuit breakers
    const key1 = 'service1:method1';
    const key2 = 'service2:method2';
    
    // Both should start closed
    expect(registry.canMakeCall(key1)).toBe(true);
    expect(registry.canMakeCall(key2)).toBe(true);
    
    // Open the first circuit breaker
    registry.recordResult(key1, false, { failureThreshold: 1 });
    
    // First should be open, second still closed
    expect(registry.canMakeCall(key1)).toBe(false);
    expect(registry.canMakeCall(key2)).toBe(true);
    
    // Reset the first circuit breaker
    registry.reset(key1);
    
    // Both should be closed
    expect(registry.canMakeCall(key1)).toBe(true);
    expect(registry.canMakeCall(key2)).toBe(true);
  });
  
  test('should reset all circuit breakers', () => {
    const registry = new CircuitBreakerRegistry({}, mockTimeProvider);
    
    // Create two circuit breakers and open them
    const key1 = 'service1:method1';
    const key2 = 'service2:method2';
    
    registry.recordResult(key1, false, { failureThreshold: 1 });
    registry.recordResult(key2, false, { failureThreshold: 1 });
    
    // Both should be open
    expect(registry.canMakeCall(key1)).toBe(false);
    expect(registry.canMakeCall(key2)).toBe(false);
    
    // Reset all circuit breakers
    registry.resetAll();
    
    // Both should be closed
    expect(registry.canMakeCall(key1)).toBe(true);
    expect(registry.canMakeCall(key2)).toBe(true);
  });
});
