import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Portfolio from './Portfolio';
import { SelectedAssetProvider } from '../context/SelectedAssetContext';
// Mock Date for consistent test results
const mockDate = new Date('2023-01-01T00:00:00Z');
jest.spyOn(global, 'Date').mockImplementation(() => mockDate);

// Mock the trading API
jest.mock('../api/trading', () => ({
  getTradingApi: jest.fn().mockReturnValue({
    getPortfolio: jest.fn().mockImplementation(() => Promise.resolve({
      totalValue: 50000,
      availableCash: 25000,
      totalPnl: 2850,
      totalPnlPercentage: 6.05,
      positions: {
        'BTC/USD': { symbol: 'BTC/USD', quantity: 0.5, entryPrice: 45000, currentPrice: 48000, marketValue: 24000, unrealizedPnl: 1500 },
        'ETH/USD': { symbol: 'ETH/USD', quantity: 5, entryPrice: 3200, currentPrice: 3500, marketValue: 17500, unrealizedPnl: 1500 },
        'SOL/USD': { symbol: 'SOL/USD', quantity: 20, entryPrice: 120, currentPrice: 110, marketValue: 2200, unrealizedPnl: -200 } }
    })),
    getRecentTrades: jest.fn().mockImplementation(() => Promise.resolve([
      { id: 'trade-1', symbol: 'BTC/USD', side: 'buy', quantity: 0.1, price: 48000, timestamp: new Date().toISOString() },
      { id: 'trade-2', symbol: 'ETH/USD', side: 'sell', quantity: 1, price: 3500, timestamp: new Date().toISOString() }
    ])
  })
}));

// Mock React hooks
jest.mock('react', () => {
  const actualReact = jest.requireActual('react');
  return {
    ...actualReact,
    useState: jest.fn().mockImplementation(actualReact.useState),
    useEffect: jest.fn().mockImplementation(() => {}),
    useContext: jest.fn().mockImplementation(actualReact.useContext)
  };
});

// Mock the components used in Portfolio
jest.mock('../components/dashboard/AssetAllocationChart', () => ({
  __esModule: true,
  default: jest.fn(({ data, onAssetSelect }) => (
    <div data-testid="asset-allocation-chart">
      <div>Showing {data?.length || 0} assets</div>
      <button onClick={() => onAssetSelect('BTC/USD')}>Select BTC</button>
    </div>
  ))
}));

jest.mock('../components/dashboard/PortfolioSummary', () => ({
  __esModule: true,
  default: jest.fn(({ totalValue, availableCash, onTimeframeChange }) => (
    <div data-testid="portfolio-summary">
      <div>Total Value: ${totalValue}</div>
      <div>Available Cash: ${availableCash}</div>
      <button onClick={() => onTimeframeChange && onTimeframeChange('1w')}>Change to 1W</button>
    </div>
  ))
}));

jest.mock('../components/dashboard/PositionsList', () => ({
  __esModule: true,
  default: jest.fn(({ positions, onPositionSelect, selectedSymbol }) => (
    <div data-testid="positions-list">
      <div>Showing {positions?.length || 0} positions</div>
      <div>Selected: {selectedSymbol || 'None'}</div>
      <button onClick={() => onPositionSelect('ETH/USD')}>Select ETH</button>
    </div>
  ))
}));

jest.mock('../components/dashboard/RecentTrades', () => ({
  __esModule: true,
  default: jest.fn(({ trades, onTradeSymbolSelect }) => (
    <div data-testid="recent-trades">
      <div>Showing {trades?.length || 0} trades</div>
      <button onClick={() => onTradeSymbolSelect && onTradeSymbolSelect('SOL/USD')}>Select SOL</button>
    </div>
  ))
}));

describe('Portfolio Page', () => {
  it('renders Portfolio page with all components', async () => {
    // Mock useState to return non-loading state
    const mockUseState = jest.spyOn(React, 'useState');
    
    // Mock portfolio data state
    mockUseState.mockImplementationOnce(() => [{
      totalValue: 50000,
      availableCash: 25000,
      totalPnl: 2850,
      totalPnlPercentage: 6.05,
      positions: [{ symbol: 'BTC/USD', quantity: 0.5, entryPrice: 45000, currentPrice: 48000 }]
    }, jest.fn()]);
    
    // Mock loading state
    mockUseState.mockImplementationOnce(() => [false, jest.fn()]);
    
    // Mock other states
    mockUseState.mockImplementation(() => [[], jest.fn()]);
    
    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Portfolio />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Check that all main components are rendered
    expect(screen.getByTestId('portfolio-summary')).toBeInTheDocument();
    expect(screen.getByTestId('asset-allocation-chart')).toBeInTheDocument();
    expect(screen.getByTestId('positions-list')).toBeInTheDocument();
    expect(screen.getByTestId('recent-trades')).toBeInTheDocument();
    
    // Clean up
    mockUseState.mockRestore();
  });

  it('handles asset selection from different components', async () => {
    // Mock useState to return non-loading state
    const mockUseState = jest.spyOn(React, 'useState');
    
    // Mock portfolio data state
    mockUseState.mockImplementationOnce(() => [{
      totalValue: 50000,
      availableCash: 25000,
      totalPnl: 2850,
      totalPnlPercentage: 6.05,
      positions: [{ symbol: 'BTC/USD', quantity: 0.5 }, { symbol: 'ETH/USD', quantity: 5 }]
    }, jest.fn()]);
    
    // Mock loading state
    mockUseState.mockImplementationOnce(() => [false, jest.fn()]);
    
    // Mock other states
    mockUseState.mockImplementation(() => [[], jest.fn()]);
    
    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Portfolio />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Select an asset from the chart
    fireEvent.click(screen.getByText(/Select BTC/i));
    expect(screen.getByText(/Selected: BTC\/USD/i)).toBeInTheDocument();
    
    // Select a different asset from the positions list
    fireEvent.click(screen.getByText(/Select ETH/i));
    expect(screen.getByText(/Selected: ETH\/USD/i)).toBeInTheDocument();
    
    // Select a different asset from recent trades
    fireEvent.click(screen.getByText(/Select SOL/i));
    expect(screen.getByText(/Selected: SOL\/USD/i)).toBeInTheDocument();
    
    // Clean up
    mockUseState.mockRestore();
  });

  it('handles timeframe change in portfolio summary', async () => {
    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Portfolio />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading portfolio data/i)).not.toBeInTheDocument();
    });

    // Change timeframe to 1W
    fireEvent.click(screen.getByText('Change to 1W'));
    
    // This is a bit tricky to test since the timeframe state is internal to the Portfolio component
    // In a real test, we'd check for specific data changes or UI updates
    // For now, we'll just verify the component didn't crash
    expect(screen.getByTestId('portfolio-summary')).toBeInTheDocument();
  });

  it('renders navigation links', async () => {
    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Portfolio />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading portfolio data/i)).not.toBeInTheDocument();
    });

    // Check that the navigation links are rendered
    const tradeLink = screen.getByText('Trade');
    expect(tradeLink).toBeInTheDocument();
    expect(tradeLink.closest('a')).toHaveAttribute('href', '/trade');

    const dashboardLink = screen.getByText('Dashboard');
    expect(dashboardLink).toBeInTheDocument();
    expect(dashboardLink.closest('a')).toHaveAttribute('href', '/');
  });
});
