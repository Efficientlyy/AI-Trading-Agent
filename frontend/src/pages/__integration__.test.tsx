import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { SelectedAssetProvider } from '../context/SelectedAssetContext';
import { jest, describe, it, expect } from '@jest/globals';
import userEvent from '@testing-library/user-event';

// Create a simplified test that doesn't rely on complex components
describe('Integration: Asset selection across components', () => {
  it('persists selected asset across components', async () => {
    // Create a simple test component that uses the SelectedAssetContext
    const TestComponent = () => {
      const { useSelectedAsset } = require('../context/SelectedAssetContext');
      const { symbol, setSymbol } = useSelectedAsset();
      
      return (
        <div>
          <div data-testid="current-symbol">{symbol}</div>
          <select 
            data-testid="asset-selector"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          >
            <option value="BTC/USD">BTC/USD</option>
            <option value="ETH/USD">ETH/USD</option>
            <option value="AAPL">AAPL</option>
          </select>
        </div>
      );
    };
    
    // Render two instances of the test component wrapped in SelectedAssetProvider
    render(
      <SelectedAssetProvider>
        <MemoryRouter>
          <TestComponent />
          <TestComponent />
        </MemoryRouter>
      </SelectedAssetProvider>
    );
    
    // Check initial state - should be BTC/USD (default from provider)
    const symbolDisplays = screen.getAllByTestId('current-symbol');
    expect(symbolDisplays[0].textContent).toBe('BTC/USD');
    expect(symbolDisplays[1].textContent).toBe('BTC/USD');
    
    // Change the asset in the first selector
    const selectors = screen.getAllByTestId('asset-selector');
    await userEvent.selectOptions(selectors[0], 'ETH/USD');
    
    // Verify both components show the updated asset
    expect(symbolDisplays[0].textContent).toBe('ETH/USD');
    expect(symbolDisplays[1].textContent).toBe('ETH/USD');
  });
});
