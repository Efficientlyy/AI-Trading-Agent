import React from 'react';
import '@testing-library/jest-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Analytics from './Analytics';
import { SelectedAssetProvider } from '../context/SelectedAssetContext';

// Mock the components used in Analytics
jest.mock('../components/dashboard/EquityCurveChart', () => ({
  __esModule: true,
  default: jest.fn(({ data, isLoading }) => (
    <div data-testid="equity-curve-chart">
      {isLoading ? 'Loading...' : `Showing ${data?.length || 0} data points`}
    </div>
  ))
}));

jest.mock('../components/dashboard/TradeStatistics', () => ({
  __esModule: true,
  default: jest.fn(({ trades }) => (
    <div data-testid="trade-statistics">
      Showing statistics for {trades?.length || 0} trades
    </div>
  ))
}));

jest.mock('../components/dashboard/PerformanceAnalysis', () => ({
  __esModule: true,
  default: jest.fn(() => (
    <div data-testid="performance-analysis">
      Performance Analysis
    </div>
  ))
}));

jest.mock('../components/dashboard/TechnicalAnalysisChart', () => ({
  __esModule: true,
  default: jest.fn(({ symbol }) => (
    <div data-testid="technical-analysis-chart">
      Technical Analysis for {symbol}
    </div>
  ))
}));

describe('Analytics Page', () => {
  it('renders Analytics page with initial tab', async () => {
    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Analytics />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Initially shows loading
    expect(screen.getByText(/Loading analytics data/i)).toBeInTheDocument();

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading analytics data/i)).not.toBeInTheDocument();
    });

    // Check that the equity curve chart is rendered by default
    expect(screen.getByTestId('equity-curve-chart')).toBeInTheDocument();
    
    // Check that the other components are not rendered
    expect(screen.queryByTestId('trade-statistics')).not.toBeInTheDocument();
    expect(screen.queryByTestId('performance-analysis')).not.toBeInTheDocument();
    expect(screen.queryByTestId('technical-analysis-chart')).not.toBeInTheDocument();
  });

  it('switches between tabs correctly', async () => {
    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Analytics />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading analytics data/i)).not.toBeInTheDocument();
    });

    // Switch to Trade Statistics tab
    fireEvent.click(screen.getByText('Trade Statistics'));
    expect(screen.getByTestId('trade-statistics')).toBeInTheDocument();
    expect(screen.queryByTestId('equity-curve-chart')).not.toBeInTheDocument();

    // Switch to Performance Analysis tab
    fireEvent.click(screen.getByText('Performance Analysis'));
    expect(screen.getByTestId('performance-analysis')).toBeInTheDocument();
    expect(screen.queryByTestId('trade-statistics')).not.toBeInTheDocument();

    // Switch to Technical Analysis tab
    fireEvent.click(screen.getByText('Technical Analysis'));
    expect(screen.getByTestId('technical-analysis-chart')).toBeInTheDocument();
    expect(screen.queryByTestId('performance-analysis')).not.toBeInTheDocument();

    // Switch back to Equity Curve Chart tab
    fireEvent.click(screen.getByText('Equity Curve Chart'));
    expect(screen.getByTestId('equity-curve-chart')).toBeInTheDocument();
    expect(screen.queryByTestId('technical-analysis-chart')).not.toBeInTheDocument();
  });

  it('renders back to dashboard link', async () => {
    render(
      <BrowserRouter>
        <SelectedAssetProvider>
          <Analytics />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading analytics data/i)).not.toBeInTheDocument();
    });

    // Check that the back to dashboard link is rendered
    const backLink = screen.getByText('Back to Dashboard');
    expect(backLink).toBeInTheDocument();
    expect(backLink.closest('a')).toHaveAttribute('href', '/');
  });

  it('uses selected asset from context', async () => {
    render(
      <BrowserRouter>
        <SelectedAssetProvider initialSymbol="ETH/USD">
          <Analytics />
        </SelectedAssetProvider>
      </BrowserRouter>
    );

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText(/Loading analytics data/i)).not.toBeInTheDocument();
    });

    // Switch to Technical Analysis tab
    fireEvent.click(screen.getByText('Technical Analysis'));
    
    // Check that the selected asset is used
    expect(screen.getByText('Technical Analysis for ETH/USD')).toBeInTheDocument();
  });
});
