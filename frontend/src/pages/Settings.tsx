import React, { useState } from 'react';
import UserProfile from '../components/settings/UserProfile';
import Preferences from '../components/settings/Preferences';
import Integrations from '../components/settings/Integrations';
import SecuritySettings from '../components/settings/SecuritySettings';

const Settings: React.FC = () => {
  const [tab, setTab] = useState<'profile' | 'preferences' | 'integrations' | 'security'>('profile');

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 py-10">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">Settings</h1>
        <div className="flex justify-center gap-4 mb-8">
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'profile' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('profile')}
          >
            User Profile
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'preferences' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('preferences')}
          >
            Preferences
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'integrations' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('integrations')}
          >
            Integrations
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'security' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('security')}
          >
            Security Settings
          </button>
        </div>
        {tab === 'profile' && <UserProfile />}
        {tab === 'preferences' && <Preferences />}
        {tab === 'integrations' && <Integrations />}
        {tab === 'security' && <SecuritySettings />}
      </div>
    </div>
  );
};

export default Settings;
