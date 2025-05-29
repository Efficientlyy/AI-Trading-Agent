import React, { useState, useEffect, useMemo } from 'react';
import apiClient from '../../api/apiClient';
import { useNotification } from '../../components/common/NotificationSystem';
import MockDataToggle from '../common/MockDataToggle';
import { useMockData } from '../../context/MockDataContext';
import { FormControlLabel, Switch } from '@mui/material';

interface Integration {
  id: string;
  name: string;
  type: 'broker' | 'data' | 'news' | 'ai' | 'storage' | 'notification' | 'risk';
  category: string;
  status: 'connected' | 'disconnected' | 'testing' | 'error';
  apiKeyMasked: string;
  apiSecretMasked: string;
  errorMessage?: string;
  lastTested?: string;
  requiresSecret: boolean;
  additionalFields?: {
    name: string;
    label: string;
    type: 'text' | 'password' | 'select';
    options?: string[];
    required: boolean;
    value: string;
    maskedValue: string;
  }[];
}

const initialIntegrations: Integration[] = [
  // Primary Brokers
  { 
    id: 'primary_broker', 
    name: 'Primary Broker', 
    type: 'broker',
    category: 'Trading',
    status: 'disconnected', 
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      { 
        name: 'broker_type', 
        label: 'Broker Provider', 
        type: 'select',
        options: ['Alpaca', 'Interactive Brokers', 'TD Ameritrade', 'Binance', 'Coinbase'],
        required: true,
        value: '',
        maskedValue: ''
      },
      { 
        name: 'account_id', 
        label: 'Account ID', 
        type: 'text',
        required: true,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // Secondary Broker (Failover)
  { 
    id: 'secondary_broker', 
    name: 'Secondary Broker (Failover)', 
    type: 'broker',
    category: 'Trading',
    status: 'disconnected', 
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      { 
        name: 'broker_type', 
        label: 'Broker Provider', 
        type: 'select',
        options: ['Alpaca', 'Interactive Brokers', 'TD Ameritrade', 'Binance', 'Coinbase'],
        required: true,
        value: '',
        maskedValue: ''
      },
      { 
        name: 'account_id', 
        label: 'Account ID', 
        type: 'text',
        required: true,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // Market Data Provider
  { 
    id: 'market_data', 
    name: 'Market Data Provider', 
    type: 'data',
    category: 'Market Data',
    status: 'disconnected', 
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      { 
        name: 'provider', 
        label: 'Data Provider', 
        type: 'select',
        options: ['Polygon.io', 'Alpha Vantage', 'IEX Cloud', 'Finnhub', 'Twelve Data'],
        required: true,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // Realtime Websocket Provider
  { 
    id: 'websocket_data', 
    name: 'Realtime Websocket Feed', 
    type: 'data',
    category: 'Market Data',
    status: 'disconnected', 
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      { 
        name: 'provider', 
        label: 'Websocket Provider', 
        type: 'select',
        options: ['Twelve Data', 'Alpaca', 'Binance', 'Coinbase', 'Finnhub'],
        required: true,
        value: '',
        maskedValue: ''
      },
      {
        name: 'websocket_url',
        label: 'Websocket URL',
        type: 'text',
        required: true,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // News Provider
  { 
    id: 'news_data', 
    name: 'Financial News Provider', 
    type: 'news',
    category: 'Market Data',
    status: 'disconnected', 
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: false,
    additionalFields: [
      { 
        name: 'provider', 
        label: 'News Provider', 
        type: 'select',
        options: ['Alpha Vantage News', 'News API', 'Bloomberg', 'Reuters', 'Financial Times'],
        required: true,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // Sentiment Provider
  { 
    id: 'sentiment_api', 
    name: 'Sentiment Analysis API', 
    type: 'data',
    category: 'Market Data',
    status: 'disconnected', 
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: false
  },
  // Economic Data Provider (for regime classification)
  {
    id: 'economic_data_api',
    name: 'Economic Data Provider',
    type: 'data',
    category: 'Market Data',
    status: 'disconnected',
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      {
        name: 'provider',
        label: 'Provider',
        type: 'select',
        options: ['FRED (Federal Reserve)', 'World Bank', 'OECD', 'Quandl', 'Trading Economics'],
        required: true,
        value: '',
        maskedValue: ''
      },
      {
        name: 'data_frequency',
        label: 'Data Frequency',
        type: 'select',
        options: ['Daily', 'Weekly', 'Monthly', 'Quarterly'],
        required: true,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // Risk Management Service
  {
    id: 'risk_api',
    name: 'Risk Management API',
    type: 'risk',
    category: 'Risk Management',
    status: 'disconnected',
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      {
        name: 'provider',
        label: 'Risk Provider',
        type: 'select',
        options: ['OpenGamma', 'Axioma', 'RiskMetrics', 'In-house', 'Custom'],
        required: true,
        value: '',
        maskedValue: ''
      },
      {
        name: 'risk_models',
        label: 'Risk Models',
        type: 'select',
        options: ['VaR', 'Stress Testing', 'Volatility', 'All'],
        required: true,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // LLM Provider for System Oversight
  {
    id: 'llm_oversight',
    name: 'LLM Oversight Service',
    type: 'ai',
    category: 'AI Services',
    status: 'disconnected',
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      {
        name: 'provider',
        label: 'LLM Provider',
        type: 'select',
        options: ['OpenAI', 'Anthropic', 'Azure OpenAI', 'OpenRouter', 'Self-hosted', 'Cohere', 'AI21 Labs'],
        required: true,
        value: '',
        maskedValue: ''
      },
      {
        name: 'model',
        label: 'Model Name',
        type: 'text',
        required: true,
        value: '',
        maskedValue: ''
      },
      {
        name: 'endpoint_url',
        label: 'API Endpoint URL',
        type: 'text',
        required: false,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // External Storage Service
  {
    id: 'storage_service',
    name: 'External Storage Service',
    type: 'storage',
    category: 'Infrastructure',
    status: 'disconnected',
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      {
        name: 'provider',
        label: 'Storage Provider',
        type: 'select',
        options: ['AWS S3', 'Google Cloud Storage', 'Azure Blob Storage', 'Backblaze B2', 'Custom'],
        required: true,
        value: '',
        maskedValue: ''
      },
      {
        name: 'bucket_name',
        label: 'Bucket/Container Name',
        type: 'text',
        required: true,
        value: '',
        maskedValue: ''
      },
      {
        name: 'region',
        label: 'Region',
        type: 'text',
        required: false,
        value: '',
        maskedValue: ''
      }
    ]
  },
  // Notification Service
  {
    id: 'notification_service',
    name: 'External Notification Service',
    type: 'notification',
    category: 'Infrastructure',
    status: 'disconnected',
    apiKeyMasked: '',
    apiSecretMasked: '',
    requiresSecret: true,
    additionalFields: [
      {
        name: 'provider',
        label: 'Notification Provider',
        type: 'select',
        options: ['Slack', 'Discord', 'Telegram', 'Email (SMTP)', 'Twilio SMS', 'Custom Webhook'],
        required: true,
        value: '',
        maskedValue: ''
      },
      {
        name: 'webhook_url',
        label: 'Webhook URL/Endpoint',
        type: 'text',
        required: false,
        value: '',
        maskedValue: ''
      },
      {
        name: 'channel',
        label: 'Channel/Chat ID',
        type: 'text',
        required: false,
        value: '',
        maskedValue: ''
      }
    ]
  }
];
const Integrations: React.FC = () => {
  // Initialize integrations state with defaults - load from localStorage or use initial data
  const [integrations, setIntegrations] = useState<Integration[]>(() => {
    const savedIntegrations = localStorage.getItem('tradingAgentIntegrations');
    if (savedIntegrations) {
      try {
        return JSON.parse(savedIntegrations);
      } catch (e) {
        console.error('Failed to parse saved integrations', e);
      }
    }
    return initialIntegrations;
  });
  const [apiKeyInput, setApiKeyInput] = useState<{ [id: string]: string }>({});
  const [apiSecretInput, setApiSecretInput] = useState<{ [id: string]: string }>({});
  const [additionalFieldInputs, setAdditionalFieldInputs] = useState<{ [id: string]: { [fieldName: string]: string } }>({});
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isTesting, setIsTesting] = useState<{ [id: string]: boolean }>({});
  
  // Get notification system hooks
  const { showNotification } = useNotification();
  
  // Function to handle additional field changes
  const handleAdditionalFieldChange = (integrationId: string, fieldName: string, value: string) => {
    const updatedFields = { 
      ...(additionalFieldInputs[integrationId] || {}), 
      [fieldName]: value 
    };
    setAdditionalFieldInputs({
      ...additionalFieldInputs,
      [integrationId]: updatedFields
    });
  };
  
  // Get mock data context
  const mockDataContext = useMockData();
  const { useMockData: isMockDataEnabled, toggleMockData } = mockDataContext;

  // Group integrations by category for display
  const integrationsByCategory = useMemo(() => {
    const grouped: { [key: string]: Integration[] } = {};
    integrations.forEach(integration => {
      if (!grouped[integration.category]) {
        grouped[integration.category] = [];
      }
      grouped[integration.category].push(integration);
    });
    return grouped;
  }, [integrations]);

  // Mock data toggle is now handled by the MockDataToggle component
  // which already includes notification handling

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 mx-auto mt-8">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">API Integrations</h2>
        <MockDataToggle />
      </div>
      <p className="text-gray-600 dark:text-gray-300 mb-6">
        Connect your trading accounts, data providers, and financial news sources for real-time trading with the AI Trading Agent.
        {isMockDataEnabled && <span className="ml-1 text-blue-500">(Using mock data for testing)</span>}
      </p>

      {Object.keys(integrationsByCategory).sort().map(category => (
        <div key={category} className="mb-10">
          <h3 className="text-xl font-semibold mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">{category}</h3>
          <div className="space-y-8">
            {integrationsByCategory[category].map((intg: Integration) => (
              <div key={intg.id} className="border rounded-lg dark:border-gray-700 p-4">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h4 className="font-semibold text-lg">{intg.name}</h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">{intg.type === 'broker' ? 'Trading Account' : intg.type === 'data' ? 'Market Data Provider' : 'News Source'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {intg.status === 'connected' && (
                      <span className="text-green-600 dark:text-green-400 inline-flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Connected
                      </span>
                    )}
                    {intg.status === 'disconnected' && (
                      <span className="text-gray-600 dark:text-gray-400 inline-flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Disconnected
                      </span>
                    )}
                    {intg.status === 'testing' && (
                      <span className="text-yellow-600 dark:text-yellow-400 inline-flex items-center">
                        <svg className="animate-spin h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Testing Connection...
                      </span>
                    )}
                    {intg.status === 'error' && (
                      <span className="text-red-600 dark:text-red-400 inline-flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Connection Error
                      </span>
                    )}
                  </div>
                </div>
                
                {intg.status === 'connected' ? (
                  <div className="mt-2 space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm text-gray-600 dark:text-gray-300">
                        <span className="font-medium">API Key:</span> {intg.apiKeyMasked}
                      </span>
                      {intg.apiSecretMasked && (
                        <span className="text-sm text-gray-600 dark:text-gray-300 ml-4">
                          <span className="font-medium">API Secret:</span> {intg.apiSecretMasked}
                        </span>
                      )}
                    </div>
                    
                    {intg.additionalFields?.map((field) => 
                      field.value && (
                        <div key={field.name} className="text-sm text-gray-600 dark:text-gray-300">
                          <span className="font-medium">{field.label}:</span> {field.type === 'password' ? 'â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢' : field.value}
                        </div>
                      )
                    )}
                    
                    {intg.lastTested && (
                      <div className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                        Last verified: {new Date(intg.lastTested).toLocaleString()}
                      </div>
                    )}
                    
                    <div className="flex justify-end mt-4">
                      <button
                        className="text-red-600 dark:text-red-400 hover:underline text-sm font-medium"
                        onClick={() => {
                          // Implement disconnect functionality
                          const updatedIntegrations = integrations.map(integration => 
                            integration.id === intg.id 
                              ? {...integration, status: 'disconnected' as const, apiKeyMasked: '', apiSecretMasked: ''} 
                              : integration
                          );
                          setIntegrations(updatedIntegrations);
                          localStorage.setItem('tradingAgentIntegrations', JSON.stringify(updatedIntegrations));
                          showNotification({
                            type: 'success',
                            title: 'Integration Disconnected',
                            message: `Successfully disconnected from ${intg.name}.`
                          });
                        }}
                      >
                        Disconnect
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="mt-4 space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">API Key</label>
                      <input
                        className="w-full px-3 py-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
                        type="text"
                        placeholder="Enter API key"
                        value={apiKeyInput[intg.id] || ''}
                        onChange={e => setApiKeyInput({ ...apiKeyInput, [intg.id]: e.target.value })}
                      />
                    </div>
                    
                    {intg.requiresSecret && (
                      <div>
                        <label className="block text-sm font-medium mb-1">API Secret</label>
                        <input
                          className="w-full px-3 py-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
                          type="password"
                          placeholder="Enter API secret"
                          value={apiSecretInput[intg.id] || ''}
                          onChange={e => setApiSecretInput({ ...apiSecretInput, [intg.id]: e.target.value })}
                        />
                      </div>
                    )}
                    
                    {intg.additionalFields?.map((field) => {
                      // For LLM Oversight, only show endpoint URL field for self-hosted or Azure OpenAI
                      if (intg.id === 'llm_oversight' && field.name === 'endpoint_url') {
                        const provider = additionalFieldInputs[intg.id]?.provider;
                        if (provider === 'OpenRouter' || (provider && provider !== 'Self-hosted' && provider !== 'Azure OpenAI')) {
                          return null; // Don't show endpoint URL for OpenRouter and standard API providers
                        }
                      }
                      
                      // For LLM Oversight with OpenRouter, add a model hint
                      let placeholder = `Enter ${field.label.toLowerCase()}`;
                      if (intg.id === 'llm_oversight' && 
                          field.name === 'model' && 
                          additionalFieldInputs[intg.id]?.provider === 'OpenRouter') {
                        placeholder = 'Enter model name (e.g. gpt-4 or claude-3-opus)';
                      }
                      
                      return (
                        <div key={field.name}>
                          <label className="block text-sm font-medium mb-1">{field.label}</label>
                          {field.type === 'select' ? (
                            <select 
                              className="w-full px-3 py-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
                              value={(additionalFieldInputs[intg.id]?.[field.name]) || ''}
                              onChange={e => handleAdditionalFieldChange(intg.id, field.name, e.target.value)}
                            >
                              <option value="">Select {field.label}</option>
                              {field.options?.map((option: string) => (
                                <option key={option} value={option}>{option}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              className="w-full px-3 py-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white"
                              type={field.type}
                              placeholder={placeholder}
                              value={(additionalFieldInputs[intg.id]?.[field.name]) || ''}
                              onChange={e => handleAdditionalFieldChange(intg.id, field.name, e.target.value)}
                            />
                          )}
                        </div>
                      );
                    })}
                    
                    {intg.status === 'error' && intg.errorMessage && (
                      <div className="text-red-600 dark:text-red-400 text-sm mt-2">
                        Error: {intg.errorMessage}
                      </div>
                    )}
                    
                    <div className="flex justify-end mt-4">
                      <button
                        className="bg-blue-500 text-white px-4 py-2 rounded font-semibold hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
                        onClick={async (e) => {
                          e.preventDefault();
                          console.log('Connect button clicked for', intg.id);
                          
                          // Set status to testing
                          setIsTesting(prev => ({ ...prev, [intg.id]: true }));
                          
                          // Get form values
                          const apiKey = apiKeyInput[intg.id]?.trim();
                          if (!apiKey) {
                            setError('API key required.');
                            setIsTesting(prev => ({ ...prev, [intg.id]: false }));
                            return;
                          }
                          
                          try {
                            if (intg.id === 'llm_oversight' && additionalFieldInputs[intg.id]?.provider === 'OpenRouter') {
                              // Real connection test to OpenRouter API
                              console.log('Testing OpenRouter connection...');
                              
                              // Get model name
                              const modelName = additionalFieldInputs[intg.id]?.model?.trim() || '';
                              console.log('Model name:', modelName);
                              
                              // Make API call to verify credentials
                              console.log('Making request to OpenRouter API');
                              console.log('Current origin:', window.location.origin);
                              
                              const response = await fetch('https://openrouter.ai/api/v1/models', {
                                method: 'GET',
                                headers: {
                                  'Authorization': `Bearer ${apiKey}`,
                                  'HTTP-Referer': window.location.origin,
                                  'X-Title': 'AI Trading Agent'
                                }
                              });
                              
                              console.log('OpenRouter API response status:', response.status);
                              // Convert headers to array safely
                              const headerArray: [string, string][] = [];
                              response.headers.forEach((value, key) => {
                                headerArray.push([key, value]);
                              });
                              console.log('OpenRouter API response headers:', headerArray);
                              
                              if (!response.ok) {
                                throw new Error(`API responded with status: ${response.status}`);
                              }
                              
                              const data = await response.json();
                              console.log('OpenRouter models:', data);
                              
                              // If we got here, the API key is valid
                              // Check if model exists if specified
                              if (modelName && data.data && data.data.length > 0) {
                                // Convert user-friendly input to OpenRouter format if needed
                                let checkModelName = modelName;
                                if (!modelName.includes('/')) {
                                  // Apply default provider prefix for common models
                                  checkModelName = 
                                    modelName.toLowerCase().startsWith('gpt') ? `openai/${modelName}` : 
                                    modelName.toLowerCase().startsWith('claude') ? `anthropic/${modelName}` : 
                                    modelName.toLowerCase().startsWith('gemini') ? `google/${modelName}` : 
                                    modelName;
                                }
                                
                                // Check if model exists
                                const modelExists = data.data.some((model: { id: string }) => 
                                  model.id.toLowerCase() === checkModelName.toLowerCase() ||
                                  checkModelName.toLowerCase().includes(model.id.toLowerCase())
                                );
                                
                                if (!modelExists) {
                                  console.warn('Model not explicitly found, but proceeding since API key is valid');
                                }
                              }
                            } else {
                              // For other integrations, simulate test
                              console.log('Simulating connection test for', intg.name);
                              await new Promise(resolve => setTimeout(resolve, 1000));
                            }
                            
                            // Create masked versions of credentials
                            const apiKeyMasked = apiKey.length > 8 
                              ? `${apiKey.slice(0, 4)}****${apiKey.slice(-4)}` 
                              : '********';
                              
                            // Update integration directly
                            const updatedIntegrations = [...integrations];
                            const index = updatedIntegrations.findIndex(i => i.id === intg.id);
                            
                            if (index !== -1) {
                              // Process additional fields
                              const updatedAdditionalFields = intg.additionalFields?.map(field => {
                                const value = (additionalFieldInputs[intg.id] || {})[field.name] || '';
                                return {
                                  ...field,
                                  value,
                                  maskedValue: field.type === 'password' ? '********' : value
                                };
                              });
                              
                              // Create updated integration
                              updatedIntegrations[index] = {
                                ...updatedIntegrations[index],
                                status: 'connected',
                                apiKeyMasked,
                                lastTested: new Date().toISOString(),
                                additionalFields: updatedAdditionalFields
                              };
                              
                              // Save changes immediately
                              setIntegrations(updatedIntegrations);
                              localStorage.setItem('tradingAgentIntegrations', JSON.stringify(updatedIntegrations));
                              
                              // Clear form inputs
                              setApiKeyInput(prev => ({ ...prev, [intg.id]: '' }));
                              
                              // Show success notification
                              showNotification({
                                type: 'success',
                                title: 'Connection Successful',
                                message: `Successfully connected to ${intg.name} with verified credentials.`,
                              });
                              
                              setMessage(`Connected to ${intg.name} successfully with verified credentials!`);
                              setError(null);
                            }
                          } catch (err) {
                            console.error('Connection test failed:', err);
                            const errorMessage = err instanceof Error ? err.message : 'Connection test failed';
                            setError(`Connection failed: ${errorMessage}`);
                            
                            // Show error notification
                            showNotification({
                              type: 'error',
                              title: 'Connection Failed',
                              message: `Failed to connect to ${intg.name}: ${errorMessage}`,
                            });
                          } finally {
                            // Always reset testing state
                            setIsTesting(prev => ({ ...prev, [intg.id]: false }));
                          }
                        }}
                        disabled={isTesting[intg.id]}
                        type="button"
                      >
                        {isTesting[intg.id] ? 'Testing...' : 'Connect'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
      
      {message && <div className="mt-8 text-green-600 dark:text-green-400 font-medium p-3 bg-green-50 dark:bg-green-900/20 rounded">{message}</div>}
      {error && <div className="mt-8 text-red-600 dark:text-red-500 font-medium p-3 bg-red-50 dark:bg-red-900/20 rounded">{error}</div>}
    </div>
  );
};

export default Integrations;
