// Mock implementation for circuit breaker executor
export const executeWithCircuitBreaker = jest.fn().mockImplementation((fn) => fn());

// Add mockReturnValue and other Jest mock methods
executeWithCircuitBreaker.mockReturnValue = jest.fn();
executeWithCircuitBreaker.mockImplementation = jest.fn();
executeWithCircuitBreaker.mockResolvedValue = jest.fn();
executeWithCircuitBreaker.mockRejectedValue = jest.fn();
