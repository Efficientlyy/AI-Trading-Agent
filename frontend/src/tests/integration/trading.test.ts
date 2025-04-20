import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MemoryRouter } from 'react-router-dom';
import { AuthProvider } from '../../context/AuthContext';
import { DataSourceProvider } from '../../context/DataSourceContext';
import { SelectedAssetProvider } from '../../context/SelectedAssetContext';
import { ThemeProvider } from '../../context/ThemeContext';
import Trade from '../../pages/Trade';
// Mock Date for consistent test results
const mockDate = new Date('2023-01-01T00:00:00Z');
jest.spyOn(global, 'Date').mockImplementation(() => mockDate);

// Mock order response
const mockOrderResponse = {
  id: 'ord123456789',
  clientOrderId: 'client-ord-123',
  symbol: 'BTC-USD',
  side: 'buy',
  type: 'MARKET',
  status: 'FILLED',
  quantity: 0.1,
  price: 50000,
  filledQuantity: 0.1,
  filledPrice: 50000,
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  fees: 25,
  totalValue: 5000
};

// Mock market data
const mockMarketData: Record<string, any> = {
  'BTC-USD': {
    symbol: 'BTC-USD',
    price: 50000,
    change24h: 1200,
    changePercent24h: 2.4,
    high24h: 51200,
    low24h: 49200,
    volume24h: 1500000000,
    marketCap: 950000000000,
    lastUpdated: new Date().toISOString()
  },
  'ETH-USD': {
    symbol: 'ETH-USD',
    price: 3500,
    change24h: 120,
    changePercent24h: 3.5,
    high24h: 3600,
    low24h: 3400,
    volume24h: 800000000,
    marketCap: 420000000000,
    lastUpdated: new Date().toISOString()
  } };

// Mock available assets
const mockAssets = [
  {
    symbol: 'BTC-USD',
    name: 'Bitcoin',
    minOrderSize: 0.0001,
    maxOrderSize: 100,
    pricePrecision: 2,
    quantityPrecision: 8,
    status: 'ACTIVE'
  },
  {
    symbol: 'ETH-USD',
    name: 'Ethereum',
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    pricePrecision: 2,
    quantityPrecision: 6,
    status: 'ACTIVE'
  }
];

// Mock API services
jest.mock('../../api/services/tradingService', () => ({
  placeOrder: jest.fn().mockImplementation((order) => {
    return Promise.resolve(mockOrderResponse);
  }),
  getMarketData: jest.fn().mockImplementation((symbol) => {
    return Promise.resolve(mockMarketData[symbol] || mockMarketData['BTC-USD']);
  }),
  getAvailableAssets: jest.fn().mockImplementation(() => {
    return Promise.resolve(mockAssets);
  }),
  getOrderBook: jest.fn().mockImplementation((symbol) => {
    return Promise.resolve({
      bids: [
        { price: 49500, size: 1.2 },
        { price: 49450, size: 2.5 },
        { price: 49400, size: 3.1 }
      ],
      asks: [
        { price: 50000, size: 1.5 },
        { price: 50050, size: 2.0 },
        { price: 50100, size: 1.8 }
      ]
    });
  }),
  getRecentTrades: jest.fn().mockImplementation((symbol) => {
    return Promise.resolve([
      { id: '1', price: 49800, size: 0.1, side: 'buy', time: new Date().toISOString() },
      { id: '2', price: 49850, size: 0.2, side: 'sell', time: new Date().toISOString() }
    ]);
  })
}));

// Mock circuit breaker executor
jest.mock('../../api/utils/circuitBreakerExecutor', () => ({
  executeWithCircuitBreaker: jest.fn().mockImplementation((exchange, method, fn) => {
    // Properly handle the function call with the correct parameters
    if (typeof fn === 'function') {
      return Promise.resolve(fn();
    } else if (typeof method === 'function') {
      // Handle case where only two parameters are passed (method is actually the function)
      return Promise.resolve(method();
    }
    return Promise.resolve({});
  }),
  getCircuitBreakerStatus: jest.fn().mockReturnValue({
    state: 'CLOSED',
    failureCount: 0,
    lastFailureTime: null,
    resetTimeout: 30000
  }),
  getCircuitBreakerPerformanceMetrics: jest.fn().mockReturnValue({})
}));

// Wrap component with all required providers
const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <MemoryRouter>
      <ThemeProvider>
        <AuthProvider>
          <DataSourceProvider>
            <SelectedAssetProvider>
              {ui}
            </SelectedAssetProvider>
          </DataSourceProvider>
        </AuthProvider>
      </ThemeProvider>
    </MemoryRouter>
  );
};

describe('Trade Page Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the trade page with all required components', async () => {
    renderWithProviders(<Trade />);
    
    // Check for main components
    expect(screen.getByText(/Market Order/i)).toBeInTheDocument();
    expect(screen.getByText(/Limit Order/i)).toBeInTheDocument();
    expect(screen.getByText(/Buy/i)).toBeInTheDocument();
    expect(screen.getByText(/Sell/i)).toBeInTheDocument();
    
    // Check for asset selector
    expect(screen.getByText(/Select Asset/i)).toBeInTheDocument();
    
    // Verify market data is loaded
    await waitFor(() => {
      expect(screen.getByText(/Current Price/i)).toBeInTheDocument();
    });
  });

  it('selects an asset and updates market data', async () => {
    renderWithProviders(<Trade />);
    
    // Wait for asset selector to be loaded
    await waitFor(() => {
      expect(screen.getByTestId('asset-selector')).toBeInTheDocument();
    });
    
    // Select a different asset
    fireEvent.click(screen.getByTestId('asset-selector'));
    fireEvent.click(screen.getByText('ETH-USD'));
    
    // Verify market data is updated for the selected asset
    await waitFor(() => {
      expect(screen.getByText('ETH-USD')).toBeInTheDocument();
      expect(screen.getByText('$3,500.00')).toBeInTheDocument(); 
    });
  });

  it('validates market order form inputs correctly', async () => {
    renderWithProviders(<Trade />);
    
    // Select market order tab if not already selected
    const marketOrderTab = screen.getByText(/Market Order/i);
    fireEvent.click(marketOrderTab);
    
    // Try to submit without quantity
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Check for validation error
    await waitFor(() => {
      expect(screen.getByText(/Quantity is required/i)).toBeInTheDocument();
    });
    
    // Enter invalid quantity (too small)
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '0.00000001' } });
    fireEvent.click(submitButton);
    
    // Check for min quantity validation
    await waitFor(() => {
      expect(screen.getByText(/Minimum order size is/i)).toBeInTheDocument();
    });
  });

  it('validates limit order form inputs correctly', async () => {
    renderWithProviders(<Trade />);
    
    // Select limit order tab
    const limitOrderTab = screen.getByText(/Limit Order/i);
    fireEvent.click(limitOrderTab);
    
    // Try to submit without price and quantity
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Check for validation errors
    await waitFor(() => {
      expect(screen.getByText(/Price is required/i)).toBeInTheDocument();
      expect(screen.getByText(/Quantity is required/i)).toBeInTheDocument();
    });
    
    // Enter invalid price (negative)
    const priceInput = screen.getByLabelText(/Price/i);
    fireEvent.change(priceInput, { target: { value: '-100' } });
    
    // Enter invalid quantity (too large)
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '1000' } });
    fireEvent.click(submitButton);
    
    // Check for validation errors
    await waitFor(() => {
      expect(screen.getByText(/Price must be positive/i)).toBeInTheDocument();
      expect(screen.getByText(/Maximum order size is/i)).toBeInTheDocument();
    });
  });

  it('submits market order successfully', async () => {
    renderWithProviders(<Trade />);
    
    // Select market order tab
    const marketOrderTab = screen.getByText(/Market Order/i);
    fireEvent.click(marketOrderTab);
    
    // Select buy side
    const buySideButton = screen.getByText(/Buy/i);
    fireEvent.click(buySideButton);
    
    // Enter valid quantity
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '0.1' } });
    
    // Submit the order
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Check for success message
    await waitFor(() => {
      expect(screen.getByText(/Order placed successfully/i)).toBeInTheDocument();
      expect(screen.getByText(/Order ID: ord123456789/i)).toBeInTheDocument();
    });
    
    // Check order details are displayed
    expect(screen.getByText(/Status: FILLED/i)).toBeInTheDocument();
    expect(screen.getByText(/Total Value: \$5,000.00/i)).toBeInTheDocument();
  });

  it('submits limit order successfully', async () => {
    renderWithProviders(<Trade />);
    
    // Select limit order tab
    const limitOrderTab = screen.getByText(/Limit Order/i);
    fireEvent.click(limitOrderTab);
    
    // Select sell side
    const sellSideButton = screen.getByText(/Sell/i);
    fireEvent.click(sellSideButton);
    
    // Enter valid price and quantity
    const priceInput = screen.getByLabelText(/Price/i);
    fireEvent.change(priceInput, { target: { value: '51000' } });
    
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '0.05' } });
    
    // Submit the order
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Check for success message
    await waitFor(() => {
      expect(screen.getByText(/Order placed successfully/i)).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    // Mock API error
    const tradingService = require('../../api/services/tradingService');
    tradingService.placeOrder.mockRejectedValueOnce(new Error('API error'));
    
    renderWithProviders(<Trade />);
    
    // Wait for order form to be loaded
    await waitFor(() => {
      expect(screen.getByTestId('order-entry-form')).toBeInTheDocument();
    });
    
    // Select market order tab
    const marketOrderTab = screen.getByText(/Market Order/i);
    fireEvent.click(marketOrderTab);
    
    // Select buy side
    const buySideButton = screen.getByText(/Buy/i);
    fireEvent.click(buySideButton);
    
    // Enter valid quantity
    const quantityInput = screen.getByLabelText(/Quantity/i);
    fireEvent.change(quantityInput, { target: { value: '0.1' } });
    
    // Submit the order
    const submitButton = screen.getByText(/Place Order/i);
    fireEvent.click(submitButton);
    
    // Check for error message
    await waitFor(() => {
      expect(screen.getByText(/Failed to place order/i)).toBeInTheDocument();
      expect(screen.getByText(/API error/i)).toBeInTheDocument();
    });
  });

  it('uses circuit breaker for API calls', async () => {
    const circuitBreaker = require('../../api/utils/circuitBreakerExecutor');
    renderWithProviders(<Trade />);
    
    // Wait for components to load
    await waitFor(() => {
      expect(screen.getByTestId('order-entry-form')).toBeInTheDocument();
    });
    
    // Verify circuit breaker was used
    expect(circuitBreaker.executeWithCircuitBreaker).toHaveBeenCalled();
  });

  it('uses performance optimizations for API calls', async () => {
    // Mock the performance optimizations
    jest.mock('../../api/utils/performanceOptimizations', () => ({
      memoize: jest.fn().mockImplementation((fn) => fn),
      measureExecutionTime: jest.fn().mockImplementation((fn) => fn),
      createBatchProcessor: jest.fn().mockReturnValue({
        add: jest.fn(),
        flush: jest.fn(),
        clear: jest.fn()
      })
    }));
    
    renderWithProviders(<Trade />);
    
    // Wait for components to load
    await waitFor(() => {
      expect(screen.getByTestId('order-entry-form')).toBeInTheDocument();
    });
    
    // Verify performance optimizations were used
    const performanceOptimizations = require('../../api/utils/performanceOptimizations');
    expect(performanceOptimizations.memoize).toHaveBeenCalled();
    expect(performanceOptimizations.measureExecutionTime).toHaveBeenCalled();
  });
});
