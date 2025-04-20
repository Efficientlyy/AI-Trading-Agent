import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import userEvent from '@testing-library/user-event';
import { SelectedAssetProvider, useSelectedAsset } from './SelectedAssetContext';

// Basic test component that uses the SelectedAssetContext
const TestComponent = () => {
  const { symbol, setSymbol } = useSelectedAsset();
  return (
    <div>
      <span data-testid="symbol">{symbol}</span>
      <button onClick={() => setSymbol('ETH/USD')}>Set ETH/USD</button>
      <button onClick={() => setSymbol('XRP/USD')}>Set XRP/USD</button>
      <select 
        data-testid="symbol-selector"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
      >
        <option value="BTC/USD">BTC/USD</option>
        <option value="ETH/USD">ETH/USD</option>
        <option value="XRP/USD">XRP/USD</option>
      </select>
    </div>
  );
};

// Second component to test context sharing
const SecondComponent = () => {
  const { symbol } = useSelectedAsset();
  return <div data-testid="second-component-symbol">{symbol}</div>;
};

describe('SelectedAssetContext', () => {
  it('provides default symbol and updates symbol with button click', async () => {
    render(
      <SelectedAssetProvider>
        <TestComponent />
      </SelectedAssetProvider>
    );
    expect(screen.getByTestId('symbol').textContent).toBe('BTC/USD');
    await userEvent.click(screen.getByText('Set ETH/USD'));
    expect(screen.getByTestId('symbol').textContent).toBe('ETH/USD');
  });

  it('updates symbol using dropdown selector', () => {
    render(
      <SelectedAssetProvider>
        <TestComponent />
      </SelectedAssetProvider>
    );
    
    // Change the selected symbol using the dropdown
    const selector = screen.getByTestId('symbol-selector');
    fireEvent.change(selector, { target: { value: 'XRP/USD' } });
    
    // Check that the symbol was updated
    expect(screen.getByTestId('symbol').textContent).toBe('XRP/USD');
  });
  
  it('shares the selected asset across multiple components', async () => {
    render(
      <SelectedAssetProvider>
        <TestComponent />
        <SecondComponent />
      </SelectedAssetProvider>
    );
    
    // Check that both components show the default symbol
    expect(screen.getByTestId('symbol').textContent).toBe('BTC/USD');
    expect(screen.getByTestId('second-component-symbol').textContent).toBe('BTC/USD');
    
    // Change the symbol
    await userEvent.click(screen.getByText('Set XRP/USD'));
    
    // Check that both components show the updated symbol
    expect(screen.getByTestId('symbol').textContent).toBe('XRP/USD');
    expect(screen.getByTestId('second-component-symbol').textContent).toBe('XRP/USD');
  });
});
