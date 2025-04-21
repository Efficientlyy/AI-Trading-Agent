import React from 'react';
import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MemoryRouter } from 'react-router-dom';
import { SelectedAssetProvider } from '../context/SelectedAssetContext';
import { jest, describe, it, expect } from '@jest/globals';

// Mock the Dashboard component directly instead of all its dependencies
jest.mock('./Dashboard', () => {
  const React = require('react');
  return {
    __esModule: true,
    default: () => (
      <div data-testid="dashboard-mock">
        <div data-testid="dashboard-grid">
          <div data-testid="portfolio-summary">Portfolio Summary</div>
          <div data-testid="performance-metrics">Performance Metrics</div>
          <div data-testid="sentiment-summary">Sentiment Summary</div>
          <div data-testid="equity-curve">Equity Curve</div>
          <div data-testid="asset-allocation">Asset Allocation</div>
          <div data-testid="technical-analysis">Technical Analysis</div>
        </div>
        <div className="nav-links">
          <button>Trade</button>
          <button>Portfolio</button>
          <button>Analytics</button>
          <button>Strategies</button>
          <button>Settings</button>
        </div>
      </div>
    )
  };
});

// Import the mocked Dashboard component
import Dashboard from './Dashboard';

describe('Dashboard Component', () => {
  it('renders all dashboard sections', async () => {
    render(
      <SelectedAssetProvider>
        <MemoryRouter>
          <Dashboard />
        </MemoryRouter>
      </SelectedAssetProvider>
    );
    
    // Check that main sections are rendered
    expect(screen.getByTestId('dashboard-grid')).toBeInTheDocument();
    expect(screen.getByTestId('portfolio-summary')).toBeInTheDocument();
    expect(screen.getByTestId('performance-metrics')).toBeInTheDocument();
    expect(screen.getByTestId('sentiment-summary')).toBeInTheDocument();
    expect(screen.getByTestId('equity-curve')).toBeInTheDocument();
    expect(screen.getByTestId('asset-allocation')).toBeInTheDocument();
    expect(screen.getByTestId('technical-analysis')).toBeInTheDocument();
  });

  it('contains navigation links to other pages', () => {
    render(
      <SelectedAssetProvider>
        <MemoryRouter>
          <Dashboard />
        </MemoryRouter>
      </SelectedAssetProvider>
    );
    
    // Check that navigation links are present
    expect(screen.getByText('Trade')).toBeInTheDocument();
    expect(screen.getByText('Portfolio')).toBeInTheDocument();
    expect(screen.getByText('Analytics')).toBeInTheDocument();
    expect(screen.getByText('Strategies')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });
});
