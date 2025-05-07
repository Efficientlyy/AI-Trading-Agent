import React, { useState, useEffect } from 'react';
import { Notification, NotificationType } from './NotificationSystem';
import { formatDateTime } from '../../utils/formatters';

interface NotificationToastProps {
  notification: Notification;
  onClose: () => void;
}

/**
 * Notification Toast Component
 * 
 * Displays a single notification toast with appropriate styling based on type
 */
const NotificationToast: React.FC<NotificationToastProps> = ({ notification, onClose }) => {
  const [isVisible, setIsVisible] = useState<boolean>(true);
  const [progress, setProgress] = useState<number>(100);
  
  // Handle auto-close with progress bar
  useEffect(() => {
    if (notification.duration) {
      const startTime = Date.now();
      const endTime = startTime + notification.duration;
      
      const updateProgress = () => {
        const now = Date.now();
        const remaining = Math.max(0, endTime - now);
        const percentage = (remaining / (notification.duration || 1)) * 100;
        
        setProgress(percentage);
        
        if (percentage <= 0) {
          setIsVisible(false);
          setTimeout(onClose, 300); // Allow time for exit animation
        } else {
          requestAnimationFrame(updateProgress);
        }
      };
      
      const animationFrame = requestAnimationFrame(updateProgress);
      
      return () => {
        cancelAnimationFrame(animationFrame);
      };
    }
  }, [notification.duration, onClose]);
  
  // Handle close animation
  const handleClose = () => {
    setIsVisible(false);
    setTimeout(onClose, 300); // Allow time for exit animation
  };
  
  // Get icon based on notification type
  const getIcon = () => {
    switch (notification.type) {
      case NotificationType.SUCCESS:
        return (
          <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
          </svg>
        );
      case NotificationType.WARNING:
        return (
          <svg className="w-6 h-6 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
          </svg>
        );
      case NotificationType.ERROR:
        return (
          <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        );
      case NotificationType.SIGNAL:
        return (
          <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
          </svg>
        );
      default:
        return (
          <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
        );
    }
  };
  
  // Get background color based on notification type
  const getBackgroundColor = () => {
    switch (notification.type) {
      case NotificationType.SUCCESS:
        return 'bg-green-50 border-green-200';
      case NotificationType.WARNING:
        return 'bg-yellow-50 border-yellow-200';
      case NotificationType.ERROR:
        return 'bg-red-50 border-red-200';
      case NotificationType.SIGNAL:
        return 'bg-blue-50 border-blue-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };
  
  // Get progress bar color based on notification type
  const getProgressColor = () => {
    switch (notification.type) {
      case NotificationType.SUCCESS:
        return 'bg-green-500';
      case NotificationType.WARNING:
        return 'bg-yellow-500';
      case NotificationType.ERROR:
        return 'bg-red-500';
      case NotificationType.SIGNAL:
        return 'bg-blue-500';
      default:
        return 'bg-gray-500';
    }
  };
  
  // Render signal-specific content if it's a signal notification
  const renderSignalContent = () => {
    if (notification.type === NotificationType.SIGNAL && notification.data) {
      const { symbol, direction, strength, confidence } = notification.data;
      
      // Determine color based on direction
      const directionColor = direction.includes('BUY') ? 'text-green-600' : 'text-red-600';
      
      return (
        <div className="mt-2 p-2 bg-white rounded-md border border-gray-100">
          <div className="flex justify-between items-center">
            <span className="font-bold">{symbol}</span>
            <span className={`font-medium ${directionColor}`}>{direction}</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm mt-1">
            <div>
              <span className="text-gray-500">Strength:</span>
              <span className="ml-1">{(strength * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-500">Confidence:</span>
              <span className="ml-1">{(confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      );
    }
    
    return null;
  };
  
  return (
    <div
      className={`transform transition-all duration-300 ease-in-out ${
        isVisible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'
      }`}
    >
      <div className={`rounded-lg shadow-lg border p-4 ${getBackgroundColor()}`}>
        <div className="flex justify-between items-start">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              {getIcon()}
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-gray-900">{notification.title}</h3>
              <div className="mt-1 text-sm text-gray-600">{notification.message}</div>
              
              {/* Signal-specific content */}
              {renderSignalContent()}
              
              {/* Notification actions */}
              {notification.actions && notification.actions.length > 0 && (
                <div className="mt-2 flex space-x-2">
                  {notification.actions.map((action, index) => (
                    <button
                      key={index}
                      onClick={action.onClick}
                      className="text-sm font-medium text-blue-600 hover:text-blue-800"
                    >
                      {action.label}
                    </button>
                  ))}
                </div>
              )}
              
              {/* Timestamp */}
              <div className="mt-1 text-xs text-gray-500">
                {formatDateTime(notification.timestamp.toISOString(), true)}
              </div>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="flex-shrink-0 ml-4 text-gray-400 hover:text-gray-600 focus:outline-none"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              ></path>
            </svg>
          </button>
        </div>
        
        {/* Progress bar for auto-close */}
        {notification.duration && (
          <div className="mt-2 w-full bg-gray-200 rounded-full h-1">
            <div
              className={`h-1 rounded-full ${getProgressColor()}`}
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NotificationToast;
