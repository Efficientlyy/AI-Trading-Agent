import React from 'react';
import { useMockData } from '../../context/MockDataContext';
import { FormControlLabel, Switch } from '@mui/material';
import { useNotification } from './NotificationSystem';

interface MockDataToggleProps {
  className?: string;
  compact?: boolean;
}

const MockDataToggle: React.FC<MockDataToggleProps> = ({ className = '', compact = false }) => {
  const mockDataContext = useMockData();
  const isMockDataEnabled = mockDataContext.useMockData;
  const { toggleMockData } = mockDataContext;
  const { showNotification } = useNotification();

  const handleToggle = () => {
    toggleMockData();
    showNotification({
      type: 'info',
      title: isMockDataEnabled ? 'Switched to Real Data' : 'Switched to Mock Data',
      message: isMockDataEnabled 
        ? 'Using real data sources. API credentials will be required.' 
        : 'Using mock data for testing purposes.',
    });
  };

  if (compact) {
    return (
      <div className={`flex items-center ${className}`}>
        <span className="text-xs mr-2">{isMockDataEnabled ? 'Mock' : 'Real'}</span>
        <Switch
          checked={isMockDataEnabled}
          onChange={handleToggle}
          color="primary"
          size="small"
        />
      </div>
    );
  }

  return (
    <div className={`flex items-center ${className}`}>
      <FormControlLabel
        control={
          <Switch
            checked={isMockDataEnabled}
            onChange={handleToggle}
            color="primary"
            size="small"
          />
        }
        label={isMockDataEnabled ? "Using Mock Data" : "Using Real Data"}
      />
    </div>
  );
};

export default MockDataToggle;
