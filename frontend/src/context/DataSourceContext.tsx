import React, { createContext, useContext, useState, ReactNode } from 'react';

export type DataSourceType = 'mock' | 'real';

interface DataSourceContextProps {
  dataSource: DataSourceType;
  setDataSource: (ds: DataSourceType) => void;
}

const DataSourceContext = createContext<DataSourceContextProps | undefined>(undefined);

export const useDataSource = () => {
  const context = useContext(DataSourceContext);
  if (!context) {
    throw new Error('useDataSource must be used within a DataSourceProvider');
  }
  return context;
};

export const DataSourceProvider = ({ children }: { children: ReactNode }) => {
  const [dataSource, setDataSource] = useState<DataSourceType>('mock');

  return (
    <DataSourceContext.Provider value={{ dataSource, setDataSource }}>
      {children}
    </DataSourceContext.Provider>
  );
};
