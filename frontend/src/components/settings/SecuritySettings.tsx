import React, { useState } from 'react';

interface Session {
  id: string;
  device: string;
  location: string;
  lastActive: string;
}

const mockSessions: Session[] = [
  { id: '1', device: 'Windows PC', location: 'Berlin, DE', lastActive: '2025-04-17 19:58' },
  { id: '2', device: 'iPhone 14', location: 'Berlin, DE', lastActive: '2025-04-17 18:10' }
];

const SecuritySettings: React.FC = () => {
  const [twoFAEnabled, setTwoFAEnabled] = useState(false);
  const [sessions, setSessions] = useState<Session[]>(mockSessions);
  const [message, setMessage] = useState<string | null>(null);

  const handleToggle2FA = () => {
    setTwoFAEnabled(v => !v);
    setMessage(twoFAEnabled ? 'Two-factor authentication disabled.' : 'Two-factor authentication enabled.');
  };

  const handleRevokeSession = (id: string) => {
    setSessions(prev => prev.filter(session => session.id !== id));
    setMessage('Session revoked.');
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-lg mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Security Settings</h2>
      <div className="mb-6">
        <label className="block text-sm font-medium mb-1">Two-Factor Authentication (2FA)</label>
        <button
          className={`px-6 py-2 rounded font-semibold ${twoFAEnabled ? 'bg-red-600 text-white' : 'bg-primary text-white'} hover:bg-primary-dark`}
          onClick={handleToggle2FA}
        >
          {twoFAEnabled ? 'Disable 2FA' : 'Enable 2FA'}
        </button>
        <div className="text-xs text-gray-500 mt-2">
          {twoFAEnabled
            ? '2FA is enabled. You will be required to enter a code when logging in.'
            : '2FA is disabled. Enable for extra security.'}
        </div>
      </div>
      <hr className="my-6 border-gray-300 dark:border-gray-700" />
      <h3 className="text-lg font-semibold mb-2">Active Sessions</h3>
      <ul className="divide-y divide-gray-200 dark:divide-gray-700">
        {sessions.map(session => (
          <li key={session.id} className="py-2 flex items-center justify-between">
            <div>
              <div className="font-medium">{session.device}</div>
              <div className="text-xs text-gray-500">{session.location} &middot; Last active: {session.lastActive}</div>
            </div>
            <button className="text-sm text-red-600 hover:underline" onClick={() => handleRevokeSession(session.id)}>
              Revoke
            </button>
          </li>
        ))}
        {sessions.length === 0 && <li className="py-2 text-gray-500">No active sessions.</li>}
      </ul>
      {message && <div className="mt-4 text-green-600 font-medium">{message}</div>}
    </div>
  );
};

export default SecuritySettings;
