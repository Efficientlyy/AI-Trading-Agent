import React, { createContext, ReactNode, useContext, useState, useEffect } from 'react';

interface MockDataContextType {
  useMockData: boolean;
  toggleMockData: () => void;
  setUseMockData: (value: boolean) => void;
}

const MockDataContext = createContext<MockDataContextType>({
  useMockData: process.env.NODE_ENV === 'development',
  toggleMockData: () => {},
  setUseMockData: () => {},
});

export const MockDataProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Default to using mock data in development, but not in production
  const [useMockData, setUseMockData] = useState<boolean>(() => {
    // Try to get the saved preference from localStorage
    const savedPreference = localStorage.getItem('useMockData');
    if (savedPreference !== null) {
      return savedPreference === 'true';
    }
    // Default to using mock data in development
    return process.env.NODE_ENV === 'development';
  });

  // Save the preference to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('useMockData', useMockData.toString());
  }, [useMockData]);

  const toggleMockData = () => {
    setUseMockData(prev => !prev);
  };

  const value = {
    useMockData,
    toggleMockData,
    setUseMockData,
  };

  return (
    <MockDataContext.Provider value={value}>
      {children}
    </MockDataContext.Provider>
  );
};

export const useMockData = () => useContext(MockDataContext);

export default MockDataContext;
