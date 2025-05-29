import React from 'react';
import { Link, useLocation } from 'react-router-dom';

/**
 * Main navigation component for the application
 */
const Navigation: React.FC = () => {
  const location = useLocation();
  
  // Check if the current path matches
  const isActive = (path: string) => {
    return location.pathname.startsWith(path);
  };
  
  return (
    <nav className="bg-gray-800 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <span className="text-xl font-bold">AI Trading Agent</span>
            </div>
            <div className="ml-10 flex items-baseline space-x-4">
              <Link
                to="/sentiment/enhanced"
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  isActive('/sentiment') 
                    ? 'bg-gray-900 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                Sentiment Dashboard
              </Link>
              <Link
                to="/technical"
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  isActive('/technical') 
                    ? 'bg-gray-900 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                Technical Analysis
              </Link>
              <Link
                to="/integrated"
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  isActive('/integrated') 
                    ? 'bg-gray-900 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                Integrated Signals
              </Link>
              <Link
                to="/settings"
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  isActive('/settings') 
                    ? 'bg-gray-900 text-white' 
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                Settings
              </Link>
            </div>
          </div>
          <div className="ml-4 flex items-center md:ml-6">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Active
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
