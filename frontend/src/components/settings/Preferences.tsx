import React, { useState } from 'react';
import { useTheme, ThemeMode } from '../../context/ThemeContext';
import { useNotification } from '../../components/common/NotificationSystem';

const Preferences: React.FC = () => {
  const { mode, setMode, accent, setAccent } = useTheme();
  const { showNotification } = useNotification();
  const [notificationsEnabled, setNotificationsEnabled] = useState<boolean>(true);
  const [layout, setLayout] = useState<'default' | 'compact'>('default');
  const [message, setMessage] = useState<string | null>(null);

  const handleSave = () => {
    // Save preferences to backend or local storage if needed
    
    // Call the notification system to show success message
    showNotification({
      type: 'success',
      title: 'Preferences Updated',
      message: 'Your preferences have been saved successfully.'
    });
    
    // Also set the local message for inline feedback
    setMessage('Preferences saved successfully.');
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-lg mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Preferences</h2>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Theme</label>
        <select
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={mode}
          onChange={e => setMode(e.target.value as ThemeMode)}
        >
          <option value="light">Light</option>
          <option value="dark">Dark</option>
          <option value="system">System</option>
        </select>
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Notifications</label>
        <input
          type="checkbox"
          checked={notificationsEnabled}
          onChange={e => setNotificationsEnabled(e.target.checked)}
          className="mr-2"
        />
        <span>{notificationsEnabled ? 'Enabled' : 'Disabled'}</span>
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Layout</label>
        <select
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={layout}
          onChange={e => setLayout(e.target.value as 'default' | 'compact')}
        >
          <option value="default">Default</option>
          <option value="compact">Compact</option>
        </select>
      </div>
      <button className="bg-primary text-white px-6 py-2 rounded font-semibold hover:bg-primary-dark" onClick={handleSave}>Save Preferences</button>
      {message && <div className="mt-4 text-green-600 font-medium">{message}</div>}
    </div>
  );
};

export default Preferences;
