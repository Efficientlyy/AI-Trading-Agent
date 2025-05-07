import React, { useState, useEffect, createContext, useContext } from 'react';
import { createPortal } from 'react-dom';
import NotificationToast from './NotificationToast';

// Notification types
export enum NotificationType {
  INFO = 'info',
  SUCCESS = 'success',
  WARNING = 'warning',
  ERROR = 'error',
  SIGNAL = 'signal'
}

// Notification interface
export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  duration?: number; // Duration in milliseconds
  timestamp: Date;
  data?: any; // Additional data for the notification
  read?: boolean;
  actions?: Array<{
    label: string;
    onClick: () => void;
  }>;
}

// Notification context interface
interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  removeNotification: (id: string) => void;
  markAsRead: (id: string) => void;
  clearAll: () => void;
}

// Create notification context
const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// Hook to use notifications
export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

// Notification provider props
interface NotificationProviderProps {
  children: React.ReactNode;
  maxNotifications?: number;
}

// Notification provider component
export const NotificationProvider: React.FC<NotificationProviderProps> = ({
  children,
  maxNotifications = 5
}) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  
  // Add a new notification
  const addNotification = (notification: Omit<Notification, 'id' | 'timestamp'>): string => {
    const id = Math.random().toString(36).substring(2, 9);
    const newNotification: Notification = {
      ...notification,
      id,
      timestamp: new Date(),
      read: false
    };
    
    setNotifications(prev => {
      // Remove oldest notifications if exceeding max
      const updated = [newNotification, ...prev];
      return updated.slice(0, maxNotifications);
    });
    
    // Auto-remove notification after duration
    if (notification.duration) {
      setTimeout(() => {
        removeNotification(id);
      }, notification.duration);
    }
    
    return id;
  };
  
  // Remove a notification
  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  };
  
  // Mark a notification as read
  const markAsRead = (id: string) => {
    setNotifications(prev =>
      prev.map(notification =>
        notification.id === id ? { ...notification, read: true } : notification
      )
    );
  };
  
  // Clear all notifications
  const clearAll = () => {
    setNotifications([]);
  };
  
  // Context value
  const contextValue: NotificationContextType = {
    notifications,
    addNotification,
    removeNotification,
    markAsRead,
    clearAll
  };
  
  return (
    <NotificationContext.Provider value={contextValue}>
      {children}
      <NotificationContainer />
    </NotificationContext.Provider>
  );
};

// Notification container component
const NotificationContainer: React.FC = () => {
  const { notifications, removeNotification } = useNotifications();
  
  // Create portal for notifications
  return createPortal(
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-md">
      {notifications.map(notification => (
        <NotificationToast
          key={notification.id}
          notification={notification}
          onClose={() => removeNotification(notification.id)}
        />
      ))}
    </div>,
    document.body
  );
};

export default NotificationProvider;
