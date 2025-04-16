import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { AuthContextType } from '../../types';
import { 
  IconBell, 
  IconMoon, 
  IconSun, 
  IconUser, 
  IconLogout, 
  IconSettings
} from '@tabler/icons-react';
import { useDataSource } from '../../context/DataSourceContext';

interface HeaderProps {
  toggleDarkMode: () => void;
  isDarkMode: boolean;
}

const Header: React.FC<HeaderProps> = ({ toggleDarkMode, isDarkMode }) => {
  const auth = useAuth() as AuthContextType;
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const { dataSource, setDataSource } = useDataSource();

  const toggleUserMenu = () => {
    setIsUserMenuOpen(!isUserMenuOpen);
    if (isNotificationsOpen) setIsNotificationsOpen(false);
  };
  
  const toggleNotifications = () => {
    setIsNotificationsOpen(!isNotificationsOpen);
    if (isUserMenuOpen) setIsUserMenuOpen(false);
  };
  
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm">
      <div className="px-4 py-3 flex justify-between items-center">
        <div className="flex items-center">
          <h1 className="text-xl font-bold text-primary">AI Trading Agent</h1>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Data Source Toggle */}
          <div className="flex items-center space-x-1 bg-gray-100 dark:bg-gray-700 rounded px-2 py-1">
            <span className="text-xs text-gray-600 dark:text-gray-300 font-medium">Mock</span>
            <button
              className={`w-10 h-5 flex items-center bg-gray-300 dark:bg-gray-600 rounded-full p-1 transition-colors duration-200 focus:outline-none ${dataSource === 'real' ? 'bg-green-400 dark:bg-green-600' : ''}`}
              onClick={() => setDataSource(dataSource === 'mock' ? 'real' : 'mock')}
              aria-label="Toggle data source"
            >
              <span
                className={`w-4 h-4 bg-white rounded-full shadow-md transform transition-transform duration-200 ${dataSource === 'real' ? 'translate-x-5' : ''}`}
              />
            </button>
            <span className="text-xs text-gray-600 dark:text-gray-300 font-medium">Real</span>
          </div>
          {/* Theme toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
            aria-label="Toggle dark mode"
          >
            {isDarkMode ? <IconSun size={20} /> : <IconMoon size={20} />}
          </button>
          
          {/* Notifications */}
          <div className="relative">
            <button
              onClick={toggleNotifications}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 relative"
              aria-label="Notifications"
            >
              <IconBell size={20} />
              <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-500"></span>
            </button>
            
            {/* Notifications dropdown */}
            {isNotificationsOpen && (
              <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-md shadow-lg py-1 z-10 border border-gray-200 dark:border-gray-700">
                <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700">
                  <h3 className="text-sm font-semibold">Notifications</h3>
                </div>
                <div className="max-h-96 overflow-y-auto">
                  <div className="px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <p className="text-sm font-medium">New trade alert</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">BTC/USD buy signal detected</p>
                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">10 minutes ago</p>
                  </div>
                  <div className="px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700">
                    <p className="text-sm font-medium">Portfolio rebalanced</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">Automatic rebalancing completed</p>
                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">1 hour ago</p>
                  </div>
                </div>
                <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700">
                  <Link to="/notifications" className="text-xs text-primary hover:text-primary-dark">View all notifications</Link>
                </div>
              </div>
            )}
          </div>
          
          {/* User menu */}
          <div className="relative">
            <button
              onClick={toggleUserMenu}
              className="flex items-center space-x-2 p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
              aria-label="User menu"
            >
              <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-white">
                <IconUser size={16} />
              </div>
              <span className="hidden md:block text-sm font-medium">
                {auth.authState.user?.username || 'User'}
              </span>
            </button>
            
            {/* User dropdown */}
            {isUserMenuOpen && (
              <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg py-1 z-10 border border-gray-200 dark:border-gray-700">
                <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700">
                  <p className="text-sm font-semibold">{auth.authState.user?.username || 'User'}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{auth.authState.user?.role || 'User'}</p>
                </div>
                <Link to="/settings" className="block px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                  <div className="flex items-center">
                    <IconSettings size={16} className="mr-2" />
                    Settings
                  </div>
                </Link>
                <button
                  onClick={auth.logout}
                  className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <div className="flex items-center">
                    <IconLogout size={16} className="mr-2" />
                    Sign out
                  </div>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
