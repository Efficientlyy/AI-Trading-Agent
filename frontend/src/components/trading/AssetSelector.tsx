import React from 'react';
import { useSelectedAsset } from '../../context/SelectedAssetContext';

interface AssetSelectorProps {
  assets: string[];
}

const AssetSelector: React.FC<AssetSelectorProps> = ({ assets }) => {
  const { symbol, setSymbol } = useSelectedAsset();

  return (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Select Asset</label>
      <select
        className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-200"
        value={symbol}
        onChange={e => setSymbol(e.target.value)}
      >
        {assets.map(asset => (
          <option key={asset} value={asset}>{asset}</option>
        ))}
      </select>
    </div>
  );
};

export default AssetSelector;
