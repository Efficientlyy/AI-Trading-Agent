/**
 * @jest-environment jsdom
 */

// Properly type our mocks
type MockedTradingService = {
  getAvailableAssets: jest.Mock;
  getMarketData: jest.Mock;
  placeOrder: jest.Mock;
  getPortfolio: jest.Mock;
};

// Set up mocks before imports with proper typing
const mockTradingService: MockedTradingService = {
  getAvailableAssets: jest.fn(),
  getMarketData: jest.fn(),
  placeOrder: jest.fn(),
  getPortfolio: jest.fn()
};

jest.mock('../../services/tradingService', () => ({
  tradingService: mockTradingService
}));

// Continue with imports
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MemoryRouter } from 'react-router-dom';
import { ThemeProvider } from '../../context/ThemeContext';
import { AuthProvider } from '../../context/AuthContext';
import { DataSourceProvider } from '../../context/DataSourceContext';
import { SelectedAssetProvider } from '../../context/SelectedAssetContext';
import Trade from '../../pages/Trade';
import { tradingService } from '../../services/tradingService';

// Mock circuit breaker
jest.mock('../../api/utils/circuitBreakerExecutor', () => ({
  executeWithCircuitBreaker: jest.fn().mockImplementation((fn) => fn()),
  getCircuitBreakerState: jest.fn().mockReturnValue({ state: 'closed' }),
  resetCircuitBreaker: jest.fn()
}));

// Wrap component with all required providers
const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    React.createElement(
      MemoryRouter,
      null,
      React.createElement(
        ThemeProvider,
        { children: 
          React.createElement(
            AuthProvider,
            { children: 
              React.createElement(
                DataSourceProvider,
                { children: 
                  React.createElement(
                    SelectedAssetProvider,
                    { children: ui }
                  )
                }
              )
            }
          )
        }
      )
    )
  );
};

describe('Trade Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Default mock implementations
    mockTradingService.getAvailableAssets.mockResolvedValue([
      { symbol: 'BTC/USDT', name: 'Bitcoin', price: 50000 },
      { symbol: 'ETH/USDT', name: 'Ethereum', price: 3000 },
    ]);
    
    mockTradingService.getMarketData.mockResolvedValue({
      price: 50000,
      change: 2.5,
      high: 51000,
      low: 49000,
      volume: 10000,
    });
    
    mockTradingService.placeOrder.mockResolvedValue({
      id: '123',
      symbol: 'BTC/USDT',
      type: 'market',
      side: 'buy',
      quantity: 0.1,
      price: 50000,
      status: 'filled',
      timestamp: new Date().toISOString(),
    });
  });

  // Test that the trade page renders properly
  it('renders the trade page with all required components', async () => {
    renderWithProviders(React.createElement(Trade));

    // Check for main components
    expect(screen.getByText(/Market Order/i)).toBeInTheDocument();
    expect(screen.getByText(/Limit Order/i)).toBeInTheDocument();
    expect(screen.getByText(/Buy/i)).toBeInTheDocument();
    expect(screen.getByText(/Sell/i)).toBeInTheDocument();
    expect(screen.getByTestId('order-entry-form')).toBeInTheDocument();
    
    // Check that asset selector is present
    expect(screen.getByTestId('asset-selector')).toBeInTheDocument();
    
    // Check that market data is fetched and displayed
    await waitFor(() => {
      expect(mockTradingService.getMarketData).toHaveBeenCalled();
    });
  });

  // Test selecting an asset
  it('selects an asset and updates market data', async () => {
    renderWithProviders(React.createElement(Trade));

    // Wait for asset selector to be loaded
    await waitFor(() => {
      expect(screen.getByTestId('asset-selector')).toBeInTheDocument();
    });

    // Select a different asset
    fireEvent.click(screen.getByTestId('asset-selector'));
    fireEvent.click(screen.getByText('ETH/USDT'));
    
    // Verify the asset was selected and market data was fetched
    await waitFor(() => {
      expect(mockTradingService.getMarketData).toHaveBeenCalledWith('ETH/USDT');
    });
  });

  // Test form validation for market orders
  it('validates market order form inputs correctly', async () => {
    renderWithProviders(React.createElement(Trade));

    // Select market order tab if not already selected
    const marketOrderTab = screen.getByText(/Market Order/i);
    fireEvent.click(marketOrderTab);

    // Try to submit without quantity
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Check for validation message
    await waitFor(() => {
      expect(screen.getByText(/Quantity is required/i)).toBeInTheDocument();
    });
    
    // Fill in quantity but with an invalid value
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '-1' } });
    fireEvent.click(submitButton);
    
    // Check for validation message
    await waitFor(() => {
      expect(screen.getByText(/Quantity must be greater than 0/i)).toBeInTheDocument();
    });
    
    // Verify that the order was not placed
    expect(mockTradingService.placeOrder).not.toHaveBeenCalled();
  });

  // Test form validation for limit orders
  it('validates limit order form inputs correctly', async () => {
    renderWithProviders(React.createElement(Trade));

    // Select limit order tab
    const limitOrderTab = screen.getByText(/Limit Order/i);
    fireEvent.click(limitOrderTab);

    // Try to submit without price and quantity
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Check for validation messages
    await waitFor(() => {
      expect(screen.getByText(/Price is required/i)).toBeInTheDocument();
      expect(screen.getByText(/Quantity is required/i)).toBeInTheDocument();
    });
    
    // Fill in price but with an invalid value
    const priceInput = screen.getByLabelText(/Price/i);
    fireEvent.change(priceInput, { target: { value: '0' } });
    
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '0.1' } });
    
    fireEvent.click(submitButton);
    
    // Check for validation message
    await waitFor(() => {
      expect(screen.getByText(/Price must be greater than 0/i)).toBeInTheDocument();
    });
    
    // Verify that the order was not placed
    expect(mockTradingService.placeOrder).not.toHaveBeenCalled();
  });

  // Test submitting a market order
  it('submits market order successfully', async () => {
    renderWithProviders(React.createElement(Trade));

    // Select market order tab
    const marketOrderTab = screen.getByText(/Market Order/i);
    fireEvent.click(marketOrderTab);

    // Select buy side
    const buySideButton = screen.getByText(/Buy/i);
    fireEvent.click(buySideButton);
    
    // Fill in quantity
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '0.1' } });
    
    // Submit the order
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Verify that the order was placed with the correct parameters
    await waitFor(() => {
      expect(mockTradingService.placeOrder).toHaveBeenCalledWith({
        symbol: 'BTC/USDT',
        side: 'buy',
        type: 'market',
        quantity: 0.1,
      });
    });
    
    // Check for success message
    await waitFor(() => {
      expect(screen.getByText(/Order placed successfully/i)).toBeInTheDocument();
    });
  });

  // Test submitting a limit order
  it('submits limit order successfully', async () => {
    renderWithProviders(React.createElement(Trade));

    // Select limit order tab
    const limitOrderTab = screen.getByText(/Limit Order/i);
    fireEvent.click(limitOrderTab);

    // Select sell side
    const sellSideButton = screen.getByText(/Sell/i);
    fireEvent.click(sellSideButton);
    
    // Fill in price and quantity
    const priceInput = screen.getByLabelText(/Price/i);
    fireEvent.change(priceInput, { target: { value: '52000' } });
    
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '0.05' } });
    
    // Submit the order
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Verify that the order was placed with the correct parameters
    await waitFor(() => {
      expect(mockTradingService.placeOrder).toHaveBeenCalledWith({
        symbol: 'BTC/USDT',
        side: 'sell',
        type: 'limit',
        quantity: 0.05,
        price: 52000,
      });
    });
    
    // Check for success message
    await waitFor(() => {
      expect(screen.getByText(/Order placed successfully/i)).toBeInTheDocument();
    });
  });

  // Test error handling when placing an order
  it('handles API errors when placing an order', async () => {
    // Mock API error
    mockTradingService.placeOrder.mockRejectedValueOnce(new Error('API error'));

    renderWithProviders(React.createElement(Trade));

    // Wait for order form to be loaded
    await waitFor(() => {
      expect(screen.getByTestId('order-entry-form')).toBeInTheDocument();
    });

    // Select market order tab
    const marketOrderTab = screen.getByText(/Market Order/i);
    fireEvent.click(marketOrderTab);
    
    // Fill in quantity
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '0.1' } });
    
    // Submit the order
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Verify that the order was attempted
    await waitFor(() => {
      expect(mockTradingService.placeOrder).toHaveBeenCalled();
    });
    
    // Check for error message
    await waitFor(() => {
      expect(screen.getByText(/Failed to place order/i)).toBeInTheDocument();
    });
  });

  // Test circuit breaker usage
  it('uses circuit breaker for API calls', async () => {
    const circuitBreaker = require('../../api/utils/circuitBreakerExecutor');
    renderWithProviders(React.createElement(Trade));

    // Wait for components to load
    await waitFor(() => {
      expect(screen.getByTestId('order-entry-form')).toBeInTheDocument();
    });

    // Verify circuit breaker was used
    expect(circuitBreaker.executeWithCircuitBreaker).toHaveBeenCalled();
  });

  // Test performance optimizations
  it('uses performance optimizations', async () => {
    // Mock performance optimizations
    jest.mock('../../api/utils/performanceOptimizations', () => ({
      debounce: jest.fn((fn) => fn),
      memoize: jest.fn((fn) => fn),
      batchRequests: jest.fn((fn) => fn),
      initializeCache: jest.fn(),
    }));
    
    // Import after mocking
    const perfOptimizations = require('../../api/utils/performanceOptimizations');
    perfOptimizations.debounce.mockImplementation((fn: Function) => {
      return fn;
    });

    renderWithProviders(React.createElement(Trade));

    // Wait for components to load
    await waitFor(() => {
      expect(screen.getByTestId('order-entry-form')).toBeInTheDocument();
    });

    // Verify performance optimizations were used
    expect(perfOptimizations.debounce).toHaveBeenCalled();
  });
});
