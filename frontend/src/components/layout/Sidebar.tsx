import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { AuthContextType } from '../../types';
import { 
  IconDashboard, 
  IconChartBar, 
  IconWallet, 
  IconRobot, 
  IconSettings, 
  IconChartLine,
  IconReportAnalytics,
  IconBrandOpenai,
  IconHeartbeat,
  IconCircuitSwitchClosed,
  IconFileAnalytics,
  IconTerminal2,
  IconAlertTriangle,
  IconSpeedboat,
  IconChartCandle,
  IconTestPipe
} from '@tabler/icons-react';

interface SidebarProps {
  isOpen: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen }) => {
  const auth = useAuth() as AuthContextType;
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: <IconDashboard size={20} /> },
    { path: '/portfolio', label: 'Portfolio', icon: <IconWallet size={20} /> },
    { path: '/trade', label: 'Trading', icon: <IconChartLine size={20} /> },
    { path: '/paper-trading', label: 'Paper Trading', icon: <IconTestPipe size={20} /> },
    { path: '/strategies', label: 'Strategies', icon: <IconRobot size={20} /> },
    { path: '/backtests', label: 'Backtest History', icon: <IconChartBar size={20} /> },
    { path: '/sentiment', label: 'Sentiment', icon: <IconReportAnalytics size={20} /> },
    { path: '/sentiment-analysis', label: 'Sentiment Analysis', icon: <IconBrandOpenai size={20} /> },
    { path: '/trading-signals', label: 'Trading Signals', icon: <IconChartCandle size={20} /> },
    { path: '/advanced-signals', label: 'Advanced Signals', icon: <IconChartLine size={20} /> },
    { 
      path: '/api-health', 
      label: 'API Health', 
      icon: <IconCircuitSwitchClosed size={20} /> 
    },
    { 
      path: '/api-logs', 
      label: 'API Logs', 
      icon: <IconTerminal2 size={20} /> 
    },
    { 
      path: '/alerts', 
      label: 'Alerts', 
      icon: <IconAlertTriangle size={20} /> 
    },
    { 
      path: '/performance', 
      label: 'Performance', 
      icon: <IconSpeedboat size={20} /> 
    },
    { 
      path: '/performance-test', 
      label: 'Performance Tests', 
      icon: <IconFileAnalytics size={20} /> 
    },
    { path: '/settings', label: 'Settings', icon: <IconSettings size={20} /> },
  ];
  
  return (
    <aside 
      className={`fixed inset-y-0 left-0 z-10 w-64 bg-white dark:bg-gray-800 shadow-md transform transition-transform duration-300 ease-in-out ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      } md:translate-x-0`}
    >
      <div className="h-full flex flex-col">
        {/* User info */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-full bg-primary flex items-center justify-center text-white">
              {auth.authState.user?.username?.charAt(0).toUpperCase() || 'U'}
            </div>
            <div>
              <p className="font-medium text-gray-900 dark:text-white">{auth.authState.user?.username || 'User'}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">{auth.authState.user?.role || 'User'}</p>
            </div>
          </div>
        </div>
        
        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4">
          <ul className="space-y-1 px-2">
            {navItems.map((item, index) => (
              <li key={index}>
                <NavLink
                  to={item.path}
                  className={({ isActive }) => 
                    `flex items-center space-x-3 px-3 py-2 rounded-md transition-colors ${
                      isActive 
                        ? 'bg-primary text-white' 
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`
                  }
                >
                  {item.icon}
                  <span>{item.label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        
        {/* App version */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400">Version 0.1.0</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
