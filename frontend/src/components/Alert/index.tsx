import React, { ReactNode } from 'react';

interface AlertProps {
  children: ReactNode;
  type?: 'info' | 'success' | 'warning' | 'error';
  className?: string;
  onClose?: () => void;
}

export const Alert: React.FC<AlertProps> = ({
  children,
  type = 'info',
  className = '',
  onClose
}) => {
  // Define styles based on alert type
  const typeStyles = {
    info: 'bg-blue-100 text-blue-800 border-blue-200',
    success: 'bg-green-100 text-green-800 border-green-200',
    warning: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    error: 'bg-red-100 text-red-800 border-red-200'
  };

  return (
    <div 
      className={`p-4 mb-4 border rounded-lg flex items-center justify-between ${typeStyles[type]} ${className}`}
      role="alert"
    >
      <div>{children}</div>
      {onClose && (
        <button
          type="button"
          onClick={onClose}
          className="ml-auto -mx-1.5 -my-1.5 bg-transparent inline-flex h-6 w-6 items-center justify-center rounded-lg p-1 hover:bg-gray-200 focus:ring-2 focus:ring-gray-300"
          aria-label="Close"
        >
          <span className="sr-only">Close</span>
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"></path>
          </svg>
        </button>
      )}
    </div>
  );
};

export default Alert;
