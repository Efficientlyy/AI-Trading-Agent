import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';

const UserProfile: React.FC = () => {
  const auth = useAuth();
  const [username, setUsername] = useState(auth?.authState?.user?.username || '');
  const [email, setEmail] = useState(auth?.authState?.user?.email || '');
  const [editing, setEditing] = useState(false);
  const [passwords, setPasswords] = useState({ current: '', new: '', confirm: '' });
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleProfileSave = async () => {
    setError(null);
    setMessage(null);
    try {
      // Mock profile update for now
      // In a real app, you would call an API or authState.updateProfile
      await new Promise(resolve => setTimeout(resolve, 500));
      setMessage('Profile updated successfully.');
      setEditing(false);
    } catch (e: any) {
      setError(e.message || 'Failed to update profile');
    }
  };

  const handlePasswordChange = async () => {
    setError(null);
    setMessage(null);
    if (passwords.new !== passwords.confirm) {
      setError('New passwords do not match.');
      return;
    }
    try {
      // Mock password change for now
      // In a real app, you would call an API or authState.changePassword
      await new Promise(resolve => setTimeout(resolve, 500));
      setMessage('Password changed successfully.');
      setPasswords({ current: '', new: '', confirm: '' });
    } catch (e: any) {
      setError(e.message || 'Failed to change password');
    }
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-lg mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">User Profile</h2>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Username</label>
        <input
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={username}
          onChange={e => setUsername(e.target.value)}
          disabled={!editing}
        />
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Email</label>
        <input
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={email}
          onChange={e => setEmail(e.target.value)}
          disabled={!editing}
        />
      </div>
      {editing ? (
        <button className="bg-primary text-white px-6 py-2 rounded font-semibold hover:bg-primary-dark mr-2" onClick={handleProfileSave}>Save</button>
      ) : (
        <button className="bg-primary text-white px-6 py-2 rounded font-semibold hover:bg-primary-dark mr-2" onClick={() => setEditing(true)}>Edit Profile</button>
      )}
      <hr className="my-6 border-gray-300 dark:border-gray-700" />
      <h3 className="text-lg font-semibold mb-2">Change Password</h3>
      <div className="mb-2">
        <label className="block text-sm font-medium mb-1">Current Password</label>
        <input
          type="password"
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={passwords.current}
          onChange={e => setPasswords({ ...passwords, current: e.target.value })}
        />
      </div>
      <div className="mb-2">
        <label className="block text-sm font-medium mb-1">New Password</label>
        <input
          type="password"
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={passwords.new}
          onChange={e => setPasswords({ ...passwords, new: e.target.value })}
        />
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Confirm New Password</label>
        <input
          type="password"
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={passwords.confirm}
          onChange={e => setPasswords({ ...passwords, confirm: e.target.value })}
        />
      </div>
      <button className="bg-primary text-white px-6 py-2 rounded font-semibold hover:bg-primary-dark" onClick={handlePasswordChange}>Change Password</button>
      {message && <div className="mt-4 text-green-600 font-medium">{message}</div>}
      {error && <div className="mt-4 text-red-600 font-medium">{error}</div>}
    </div>
  );
};

export default UserProfile;
