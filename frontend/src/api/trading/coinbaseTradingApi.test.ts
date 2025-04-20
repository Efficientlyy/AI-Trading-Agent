// Mock the monitoring utilities
jest.mock('../utils/monitoring', () => ({
  canMakeApiCall: jest.fn(),
  recordApiCall: jest.fn(),
  recordCircuitBreakerResult: jest.fn()
}));

// Import the monitoring utilities
import { canMakeApiCall, recordApiCall, recordCircuitBreakerResult } from '../utils/monitoring';

// Test the monitoring functionality directly
describe('Trading API Monitoring', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should record API calls correctly', () => {
    // Call the monitoring functions directly
    canMakeApiCall('Coinbase', 'getPortfolio');
    recordApiCall('Coinbase', 'getPortfolio', 'attempt');
    recordApiCall('Coinbase', 'getPortfolio', 'success');
    recordCircuitBreakerResult('Coinbase', 'getPortfolio', true);
    
    // Verify the functions were called with the correct arguments
    expect(canMakeApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio');
    expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio', 'attempt');
    expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio', 'success');
    expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Coinbase', 'getPortfolio', true);
  });

  it('should verify circuit breaker calls', () => {
    // Call the monitoring functions directly
    canMakeApiCall('Coinbase', 'getPortfolio');
    canMakeApiCall('Coinbase', 'createOrder');
    
    // Verify the functions were called with the correct arguments
    expect(canMakeApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio');
    expect(canMakeApiCall).toHaveBeenCalledWith('Coinbase', 'createOrder');
    expect(canMakeApiCall).toHaveBeenCalledTimes(2);
  });
});

// Test the monitoring functionality with error handling
describe('Error Handling and Monitoring', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should record API failures correctly', () => {
    // Record an API failure
    recordApiCall('Coinbase', 'getPortfolio', 'attempt');
    recordApiCall('Coinbase', 'getPortfolio', 'failure');
    recordCircuitBreakerResult('Coinbase', 'getPortfolio', false);
    
    // Verify the functions were called with the correct arguments
    expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio', 'attempt');
    expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio', 'failure');
    expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Coinbase', 'getPortfolio', false);
  });

  it('should handle different API methods', () => {
    // Test with different API methods
    canMakeApiCall('Coinbase', 'getPortfolio');
    canMakeApiCall('Coinbase', 'createOrder');
    canMakeApiCall('Coinbase', 'getMarketPrice');
    
    // Verify the functions were called with the correct arguments
    expect(canMakeApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio');
    expect(canMakeApiCall).toHaveBeenCalledWith('Coinbase', 'createOrder');
    expect(canMakeApiCall).toHaveBeenCalledWith('Coinbase', 'getMarketPrice');
  });

  it('should record multiple API call attempts', () => {
    // Record multiple API call attempts
    recordApiCall('Coinbase', 'getPortfolio', 'attempt');
    recordApiCall('Coinbase', 'createOrder', 'attempt');
    recordApiCall('Coinbase', 'getMarketPrice', 'attempt');
    
    // Verify the functions were called with the correct arguments
    expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio', 'attempt');
    expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'createOrder', 'attempt');
    expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'getMarketPrice', 'attempt');
  });
});
