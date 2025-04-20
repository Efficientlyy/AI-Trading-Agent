import React, { useState } from 'react';

interface Integration {
  id: string;
  name: string;
  status: 'connected' | 'disconnected';
  apiKeyMasked: string;
}

const initialIntegrations: Integration[] = [
  { id: 'binance', name: 'Binance', status: 'disconnected', apiKeyMasked: '' },
  { id: 'coinbase', name: 'Coinbase', status: 'disconnected', apiKeyMasked: '' },
  { id: 'alpaca', name: 'Alpaca', status: 'disconnected', apiKeyMasked: '' }
];

const Integrations: React.FC = () => {
  const [integrations, setIntegrations] = useState<Integration[]>(initialIntegrations);
  const [apiKeyInput, setApiKeyInput] = useState<{ [id: string]: string }>({});
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleConnect = (id: string) => {
    const apiKey = apiKeyInput[id]?.trim();
    if (!apiKey) {
      setError('API key required.');
      return;
    }
    setIntegrations(prev => prev.map(intg =>
      intg.id === id
        ? { ...intg, status: 'connected', apiKeyMasked: apiKey.slice(0, 4) + '****' + apiKey.slice(-4) }
        : intg
    ));
    setApiKeyInput(prev => ({ ...prev, [id]: '' }));
    setMessage('Integration connected.');
    setError(null);
  };

  const handleDisconnect = (id: string) => {
    setIntegrations(prev => prev.map(intg =>
      intg.id === id ? { ...intg, status: 'disconnected', apiKeyMasked: '' } : intg
    ));
    setMessage('Integration disconnected.');
    setError(null);
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-lg mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Integrations</h2>
      <div className="space-y-6">
        {integrations.map(intg => (
          <div key={intg.id} className="border-b border-gray-200 dark:border-gray-700 pb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold">{intg.name}</span>
              {intg.status === 'connected' ? (
                <span className="text-green-600 font-medium">Connected</span>
              ) : (
                <span className="text-red-500 font-medium">Disconnected</span>
              )}
            </div>
            {intg.status === 'connected' ? (
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-300">API Key: {intg.apiKeyMasked}</span>
                <button
                  className="ml-4 text-sm text-red-600 hover:underline"
                  onClick={() => handleDisconnect(intg.id)}
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2 mb-2">
                <input
                  className="w-64 px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
                  type="text"
                  placeholder="Enter API key"
                  value={apiKeyInput[intg.id] || ''}
                  onChange={e => setApiKeyInput({ ...apiKeyInput, [intg.id]: e.target.value })}
                />
                <button
                  className="bg-primary text-white px-4 py-2 rounded font-semibold hover:bg-primary-dark"
                  onClick={() => handleConnect(intg.id)}
                >
                  Connect
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
      {message && <div className="mt-4 text-green-600 font-medium">{message}</div>}
      {error && <div className="mt-4 text-red-600 font-medium">{error}</div>}
    </div>
  );
};

export default Integrations;
