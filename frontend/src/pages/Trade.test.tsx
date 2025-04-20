import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Trade from './Trade';
import { SelectedAssetProvider } from '../context/SelectedAssetContext';
import * as portfolioApi from '../api/portfolio';
import * as marketApi from '../api/market';
import * as tradesApi from '../api/trades';

// Mock the API modules
jest.mock('../api/portfolio', () => ({
  getPortfolio: jest.fn(),
  createOrder: jest.fn()
}));

jest.mock('../api/market', () => ({
  getAssets: jest.fn(),
  getHistoricalData: jest.fn()
}));

jest.mock('../api/trades', () => ({
  getRecentTrades: jest.fn()
}));

// Mock child components to simplify testing
jest.mock('../components/trading/AssetSelector', () => ({
  __esModule: true,
  default: jest.fn(({ assets }) => (
    <div data-testid="asset-selector">
      <select data-testid="asset-select">
        {assets.map((asset: string) => (
          <option key={asset} value={asset}>{asset}</option>
        ))}
      </select>
    </div>
  ))
}));

jest.mock('../components/trading/OrderBook', () => ({
  __esModule: true,
  default: jest.fn(() => <div data-testid="order-book">Order Book</div>)
}));

jest.mock('../components/trading/OrderEntryForm', () => ({
  __esModule: true,
  default: jest.fn(({ onSubmitOrder }) => (
    <div data-testid="order-entry-form">
      <button onClick={() => onSubmitOrder({ symbol: 'BTC/USD', side: 'buy', quantity: 1, price: 50000 })}>
        Submit Order
      </button>
    </div>
  ))
}));

jest.mock('../components/trading/OrderManagement', () => ({
  __esModule: true,
  default: jest.fn(() => <div data-testid="order-management">Order Management</div>)
}));

jest.mock('../components/trading/PositionDetails', () => ({
  __esModule: true,
  default: jest.fn(() => <div data-testid="position-details">Position Details</div>)
}));

jest.mock('../components/trading/TradeHistory', () => ({
  __esModule: true,
  default: jest.fn(() => <div data-testid="trade-history">Trade History</div>)
}));

jest.mock('../components/dashboard/TechnicalAnalysisChart', () => ({
  __esModule: true,
  default: jest.fn(() => <div data-testid="technical-analysis-chart">Chart</div>)
}));

describe('Trade Page', () => {
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup API mock responses
    (portfolioApi.getPortfolio as jest.Mock).mockImplementation(() => Promise.resolve({
      portfolio: {
        total_value: 100000,
        cash: 50000,
        positions: {
          'BTC/USD': {
            symbol: 'BTC/USD',
            quantity: 1,
            entry_price: 45000,
            current_price: 50000,
            market_value: 50000,
            unrealized_pnl: 5000
          } }
      } }));
    
    (marketApi.getAssets as jest.Mock).mockImplementation(() => Promise.resolve({
      assets: [
        { symbol: 'BTC/USD' },
        { symbol: 'ETH/USD' }
      ]
    }));
    
    (tradesApi.getRecentTrades as jest.Mock).mockImplementation(() => Promise.resolve({
      trades: [
        { id: '1', symbol: 'BTC/USD', side: 'buy', price: 45000, quantity: 1, timestamp: Date.now(), status: 'filled' }
      ]
    }));

    (marketApi.getHistoricalData as jest.Mock).mockImplementation(() => Promise.resolve({
      data: [
        { timestamp: '2023-01-01T00:00:00Z', open: 46000, high: 47000, low: 45000, close: 46500, volume: 100 }
      ]
    }));
  });

  it('renders Trade page with all components', async () => {
    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Trade />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Initially shows loading
    expect(screen.getByText(/Loading trading interface/i)).toBeInTheDocument();

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading trading interface/i)).not.toBeInTheDocument();
    });

    // Check that all main components are rendered
    expect(screen.getByTestId('asset-selector')).toBeInTheDocument();
    expect(screen.getByTestId('technical-analysis-chart')).toBeInTheDocument();
    expect(screen.getByTestId('order-entry-form')).toBeInTheDocument();
    expect(screen.getByTestId('order-book')).toBeInTheDocument();
    expect(screen.getByTestId('trade-history')).toBeInTheDocument();
    expect(screen.getByTestId('order-management')).toBeInTheDocument();
    expect(screen.getByTestId('position-details')).toBeInTheDocument();
  });

  it('handles order submission', async () => {
    (portfolioApi.createOrder as jest.Mock).mockImplementation(() => Promise.resolve({ success: true }));

    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Trade />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading trading interface/i)).not.toBeInTheDocument();
    });

    // Submit an order
    fireEvent.click(screen.getByText('Submit Order'));

    // Check that createOrder was called
    await waitFor(() => {
      expect(portfolioApi.createOrder).toHaveBeenCalledWith({
        symbol: 'BTC/USD',
        side: 'buy',
        quantity: 1,
        price: 50000
      });
    });

    // Check that success toast is shown
    expect(screen.getByText(/Order placed successfully/i)).toBeInTheDocument();
  });

  it('handles order submission error', async () => {
    (portfolioApi.createOrder as jest.Mock).mockImplementation(() => Promise.reject({ message: 'Insufficient funds' }));

    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Trade />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading trading interface/i)).not.toBeInTheDocument();
    });

    // Submit an order
    fireEvent.click(screen.getByText('Submit Order'));

    // Check that error toast is shown
    await waitFor(() => {
      expect(screen.getByText(/Insufficient funds/i)).toBeInTheDocument();
    });
  });
});
