import React from 'react';
import { useTheme, ThemeMode, ThemeAccent } from '../../context/ThemeContext';

interface ThemeOptionProps {
  label: string;
  value: ThemeMode | ThemeAccent;
  currentValue: string;
  onClick: (value: any) => void;
  type: 'mode' | 'accent';
}

const ThemeOption: React.FC<ThemeOptionProps> = ({ 
  label, 
  value, 
  currentValue, 
  onClick, 
  type 
}) => {
  const isActive = value === currentValue;
  
  return (
    <button
      onClick={() => onClick(value)}
      className={`
        flex items-center justify-center p-2 rounded-md transition-colors
        ${isActive ? 'bg-accent-primary text-white' : 'bg-bg-secondary text-text-primary hover:bg-bg-tertiary'}
        ${type === 'accent' ? 'w-10 h-10' : 'px-4 py-2'}
      `}
      aria-label={`Set theme ${type} to ${label}`}
    >
      {type === 'accent' ? (
        <span 
          className="w-6 h-6 rounded-full" 
          style={{ 
            backgroundColor: 
              value === 'blue' ? 'var(--accent-blue-light)' : 
              value === 'green' ? 'var(--accent-green-light)' : 
              value === 'purple' ? 'var(--accent-purple-light)' : 
              value === 'orange' ? 'var(--accent-orange-light)' : 
              'var(--accent-red-light)' 
          }}
        />
      ) : (
        <span>{label}</span>
      )}
    </button>
  );
};

const ThemeSettings: React.FC = () => {
  const { mode, accent, setMode, setAccent } = useTheme();
  
  const themeOptions: { label: string; value: ThemeMode }[] = [
    { label: 'Light', value: 'light' },
    { label: 'Dark', value: 'dark' },
    { label: 'System', value: 'system' }
  ];
  
  const accentOptions: { label: string; value: ThemeAccent }[] = [
    { label: 'Blue', value: 'blue' },
    { label: 'Green', value: 'green' },
    { label: 'Purple', value: 'purple' },
    { label: 'Orange', value: 'orange' },
    { label: 'Red', value: 'red' }
  ];
  
  return (
    <div className="p-4 bg-bg-primary border border-border-color rounded-lg shadow-md transition-colors animate-fadeIn">
      <h3 className="text-lg font-semibold mb-4 text-text-primary">Theme Settings</h3>
      
      <div className="mb-6">
        <h4 className="text-sm font-medium mb-2 text-text-secondary">Theme Mode</h4>
        <div className="flex flex-wrap gap-2">
          {themeOptions.map(option => (
            <ThemeOption
              key={option.value}
              label={option.label}
              value={option.value}
              currentValue={mode}
              onClick={setMode}
              type="mode"
            />
          ))}
        </div>
      </div>
      
      <div>
        <h4 className="text-sm font-medium mb-2 text-text-secondary">Accent Color</h4>
        <div className="flex flex-wrap gap-2">
          {accentOptions.map(option => (
            <ThemeOption
              key={option.value}
              label={option.label}
              value={option.value}
              currentValue={accent}
              onClick={setAccent}
              type="accent"
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default ThemeSettings;
