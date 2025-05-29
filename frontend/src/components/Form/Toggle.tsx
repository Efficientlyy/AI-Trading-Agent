import React from 'react';

interface ToggleProps {
  id?: string;      // Add id prop
  label: string;
  checked: boolean;
  onChange: () => void;
  className?: string;
  disabled?: boolean;
}

/**
 * Toggle component for boolean inputs
 */
const Toggle: React.FC<ToggleProps> = ({
  id,
  label,
  checked,
  onChange,
  className = '',
  disabled = false
}) => {
  return (
    <div className={`flex items-center ${className}`}>
      <label className="inline-flex items-center cursor-pointer">
        <span className="mr-3 text-sm font-medium text-gray-700">{label}</span>
        <div className="relative">
          <input
            id={id}
            type="checkbox"
            className="sr-only"
            checked={checked}
            onChange={onChange}
            disabled={disabled}
          />
          <div
            className={`block ${
              disabled ? 'bg-gray-300' : checked ? 'bg-blue-600' : 'bg-gray-200'
            } w-10 h-6 rounded-full transition-colors duration-200`}
          ></div>
          <div
            className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 transform ${
              checked ? 'translate-x-4' : 'translate-x-0'
            } ${disabled ? 'opacity-70' : ''}`}
          ></div>
        </div>
      </label>
    </div>
  );
};

export default Toggle;
