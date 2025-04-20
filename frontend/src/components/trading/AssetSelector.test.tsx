import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { SelectedAssetProvider, useSelectedAsset } from '../../context/SelectedAssetContext';
import AssetSelector from './AssetSelector';

const assets = ['BTC/USD', 'ETH/USD', 'AAPL'];

const Consumer = () => {
  const { symbol } = useSelectedAsset();
  return <div data-testid="selected-symbol">{symbol}</div>;
};

describe('AssetSelector', () => {
  it('renders options and updates context on change', () => {
    render(
      <SelectedAssetProvider>
        <AssetSelector assets={assets} />
        <Consumer />
      </SelectedAssetProvider>
    );
    const select = screen.getByRole('combobox');
    expect(select.value).toBe('BTC/USD');
    fireEvent.change(select, { target: { value: 'AAPL' } });
    expect(screen.getByTestId('selected-symbol').textContent).toBe('AAPL');
  });
});
