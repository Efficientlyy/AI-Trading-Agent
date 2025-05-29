import React, { ReactNode } from 'react';

interface TabProps {
  id: string;
  label: string;
  children: ReactNode;
  className?: string;
}

const Tab: React.FC<TabProps> = ({ children, className = '' }) => {
  return (
    <div className={`tab-content ${className}`}>
      {children}
    </div>
  );
};

interface TabsProps {
  children: ReactNode;
  activeTab: string;
  onChange: (tabId: string) => void;
  className?: string;
}

const Tabs: React.FC<TabsProps> = ({
  children,
  activeTab,
  onChange,
  className = ''
}) => {
  // Extract tab information from children
  const tabs = React.Children.toArray(children)
    .filter((child): child is React.ReactElement => React.isValidElement(child))
    .map(child => ({
      id: child.props.id,
      label: child.props.label
    }));

  return (
    <div className={`tabs-container ${className}`}>
      <div className="border-b border-gray-700">
        <nav className="flex space-x-8">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`py-4 px-1 text-center border-b-2 font-medium text-sm whitespace-nowrap ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
              }`}
              onClick={() => onChange(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
      <div className="py-4">
        {React.Children.toArray(children)
          .filter((child): child is React.ReactElement => React.isValidElement(child))
          .find(child => child.props.id === activeTab)}
      </div>
    </div>
  );
};

// Export components
export { Tab, Tabs };
