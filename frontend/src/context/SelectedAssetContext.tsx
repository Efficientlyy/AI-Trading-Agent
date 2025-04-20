import React, { createContext, useContext, useState, ReactNode } from 'react';

export interface SelectedAssetContextType {
  symbol: string;
  setSymbol: (symbol: string) => void;
}

const SelectedAssetContext = createContext<SelectedAssetContextType | undefined>(undefined);

export const useSelectedAsset = () => {
  const context = useContext(SelectedAssetContext);
  if (!context) {
    throw new Error('useSelectedAsset must be used within a SelectedAssetProvider');
  }
  return context;
};

export const SelectedAssetProvider = ({ children }: { children: ReactNode }) => {
  const [symbol, setSymbol] = useState<string>('BTC/USD');
  return (
    <SelectedAssetContext.Provider value={{ symbol, setSymbol }}>
      {children}
    </SelectedAssetContext.Provider>
  );
};
