/**
 * Testable Circuit Breaker Utility
 * 
 * This module provides a testable implementation of the circuit breaker pattern
 * that can be easily used in tests without state interference.
 */

// Circuit breaker states
export type CircuitState = 'closed' | 'open' | 'half-open';

// Circuit breaker configuration
export interface CircuitBreakerConfig {
  failureThreshold: number;
  resetTimeoutMs: number;
}

// Default configuration
export const defaultCircuitBreakerConfig: CircuitBreakerConfig = {
  failureThreshold: 5,
  resetTimeoutMs: 60000, // 1 minute
};

// Circuit breaker class
export class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failureCount: number = 0;
  private lastFailureTime: number = 0;
  private nextAttemptTime: number = 0;
  private timeProvider: () => number;
  private config: CircuitBreakerConfig;

  /**
   * Create a new circuit breaker
   * @param config Circuit breaker configuration
   * @param timeProvider Function that returns the current time in milliseconds (defaults to Date.now)
   */
  constructor(
    config: Partial<CircuitBreakerConfig> = {}, 
    timeProvider: () => number = Date.now
  ) {
    this.config = { ...defaultCircuitBreakerConfig, ...config };
    this.timeProvider = timeProvider;
  }

  /**
   * Check if a call can be made through this circuit breaker
   * @returns True if the call is allowed, false otherwise
   */
  canMakeCall(): boolean {
    const now = this.timeProvider();

    if (this.state === 'open') {
      // Check if timeout has elapsed to transition to half-open
      if (now >= this.nextAttemptTime) {
        this.state = 'half-open';
        return true;
      }
      return false;
    }

    // Always allow calls in closed or half-open state
    return true;
  }

  /**
   * Record the result of a call
   * @param success Whether the call was successful
   */
  recordResult(success: boolean): void {
    const now = this.timeProvider();

    if (success) {
      // Success case
      if (this.state === 'half-open') {
        // Reset on success in half-open state
        this.state = 'closed';
        this.failureCount = 0;
      } else if (this.state === 'closed') {
        // Reset failure count on success
        this.failureCount = 0;
      }
    } else {
      // Failure case
      this.lastFailureTime = now;
      
      if (this.state === 'closed') {
        this.failureCount += 1;
        
        // Check if threshold exceeded
        if (this.failureCount >= this.config.failureThreshold) {
          this.state = 'open';
          this.nextAttemptTime = now + this.config.resetTimeoutMs;
        }
      } else if (this.state === 'half-open') {
        // Transition back to open on failure
        this.state = 'open';
        this.nextAttemptTime = now + this.config.resetTimeoutMs;
      }
    }
  }

  /**
   * Reset the circuit breaker to its initial state
   */
  reset(): void {
    this.state = 'closed';
    this.failureCount = 0;
    this.lastFailureTime = 0;
    this.nextAttemptTime = 0;
  }

  /**
   * Get the current state of the circuit breaker
   */
  getState(): { 
    state: CircuitState; 
    failureCount: number; 
    lastFailureTime: number; 
    nextAttemptTime: number;
  } {
    return {
      state: this.state,
      failureCount: this.failureCount,
      lastFailureTime: this.lastFailureTime,
      nextAttemptTime: this.nextAttemptTime,
    };
  }
}

/**
 * Create a circuit breaker registry that manages multiple circuit breakers
 */
export class CircuitBreakerRegistry {
  private breakers: Map<string, CircuitBreaker> = new Map();
  private timeProvider: () => number;
  private defaultConfig: Partial<CircuitBreakerConfig>;

  /**
   * Create a new circuit breaker registry
   * @param defaultConfig Default configuration for new circuit breakers
   * @param timeProvider Function that returns the current time in milliseconds
   */
  constructor(
    defaultConfig: Partial<CircuitBreakerConfig> = {}, 
    timeProvider: () => number = Date.now
  ) {
    this.defaultConfig = defaultConfig;
    this.timeProvider = timeProvider;
  }

  /**
   * Get or create a circuit breaker for the given key
   * @param key Unique identifier for the circuit breaker
   * @param config Optional configuration for this specific circuit breaker
   */
  getBreaker(key: string, config: Partial<CircuitBreakerConfig> = {}): CircuitBreaker {
    if (!this.breakers.has(key)) {
      const mergedConfig = { ...this.defaultConfig, ...config };
      this.breakers.set(key, new CircuitBreaker(mergedConfig, this.timeProvider));
    }
    return this.breakers.get(key)!;
  }

  /**
   * Reset all circuit breakers in the registry
   */
  resetAll(): void {
    this.breakers.forEach(breaker => breaker.reset());
  }

  /**
   * Reset a specific circuit breaker
   * @param key Unique identifier for the circuit breaker
   */
  reset(key: string): void {
    if (this.breakers.has(key)) {
      this.breakers.get(key)!.reset();
    }
  }

  /**
   * Check if a call can be made through the circuit breaker for the given key
   * @param key Unique identifier for the circuit breaker
   * @param config Optional configuration for this specific check
   */
  canMakeCall(key: string, config: Partial<CircuitBreakerConfig> = {}): boolean {
    return this.getBreaker(key, config).canMakeCall();
  }

  /**
   * Record the result of a call for the given key
   * @param key Unique identifier for the circuit breaker
   * @param success Whether the call was successful
   * @param config Optional configuration for this specific record
   */
  recordResult(key: string, success: boolean, config: Partial<CircuitBreakerConfig> = {}): void {
    this.getBreaker(key, config).recordResult(success);
  }

  /**
   * Get the state of a specific circuit breaker
   * @param key Unique identifier for the circuit breaker
   */
  getState(key: string): { 
    state: CircuitState; 
    failureCount: number; 
    lastFailureTime: number; 
    nextAttemptTime: number;
  } | undefined {
    if (this.breakers.has(key)) {
      return this.breakers.get(key)!.getState();
    }
    return undefined;
  }
}
