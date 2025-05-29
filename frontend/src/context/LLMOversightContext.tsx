import React, { createContext, ReactNode, useContext, useEffect, useState } from 'react';
import { getOpenRouterCredentials } from '../api/openRouterClient';
import { oversightClient } from '../api/oversightClient';

// Define types for our context
export interface LLMOversightStatus {
  isConnected: boolean;
  provider: string;
  model: string;
  lastChecked: string | null;
  health: 'healthy' | 'unhealthy' | 'unknown';
}

interface LLMOversightContextType {
  status: LLMOversightStatus;
  checkConnection: () => Promise<boolean>;
  runHealthCheck: () => Promise<void>;
}

// Default status
const defaultStatus: LLMOversightStatus = {
  isConnected: false,
  provider: 'None',
  model: 'None',
  lastChecked: null,
  health: 'unknown'
};

// Create the context with default values
const LLMOversightContext = createContext<LLMOversightContextType>({
  status: defaultStatus,
  checkConnection: async () => false,
  runHealthCheck: async () => {},
});

// Create a provider component
export const LLMOversightProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [status, setStatus] = useState<LLMOversightStatus>(defaultStatus);

  // Check if OpenRouter is connected by retrieving credentials from storage
  const checkConnection = async (): Promise<boolean> => {
    try {
      const credentials = getOpenRouterCredentials();
      
      if (credentials) {
        // Update the status with connection information
        setStatus({
          isConnected: true,
          provider: 'OpenRouter',
          model: credentials.model || 'Unknown Model',
          lastChecked: new Date().toISOString(),
          health: 'healthy' // Assume healthy if credentials exist
        });
        return true;
      } else {
        // No credentials found
        setStatus({
          ...defaultStatus,
          lastChecked: new Date().toISOString()
        });
        return false;
      }
    } catch (error) {
      console.error('Error checking LLM Oversight connection:', error);
      setStatus({
        ...defaultStatus,
        health: 'unhealthy',
        lastChecked: new Date().toISOString()
      });
      return false;
    }
  };

  // Run a health check against the LLM service
  const runHealthCheck = async (): Promise<void> => {
    try {
      // Try to get a simple model list as a health check
      const credentials = getOpenRouterCredentials();
      
      if (!credentials) {
        setStatus(prev => ({
          ...prev,
          health: 'unknown',
          lastChecked: new Date().toISOString()
        }));
        return;
      }
      
      // Perform a simple test to verify the LLM is operational
      const simpleDecision = {
        symbol: 'TEST',
        action: 'BUY',
        price: 100,
        quantity: 1,
        strategy: 'Test Strategy',
        reasoning: 'Health check test'
      };
      
      await oversightClient.analyzeTradingDecision(simpleDecision);
      
      // If we got here, the check passed
      setStatus(prev => ({
        ...prev,
        health: 'healthy',
        lastChecked: new Date().toISOString()
      }));
    } catch (error) {
      console.error('LLM Oversight health check failed:', error);
      setStatus(prev => ({
        ...prev,
        health: 'unhealthy',
        lastChecked: new Date().toISOString()
      }));
    }
  };

  // Check connection on initial load
  useEffect(() => {
    checkConnection();
    
    // Check connection status every 5 minutes
    const intervalId = setInterval(checkConnection, 5 * 60 * 1000);
    
    return () => clearInterval(intervalId);
  }, []);

  return (
    <LLMOversightContext.Provider value={{ status, checkConnection, runHealthCheck }}>
      {children}
    </LLMOversightContext.Provider>
  );
};

// Custom hook for using this context
export const useLLMOversight = () => useContext(LLMOversightContext);

export default LLMOversightContext;
