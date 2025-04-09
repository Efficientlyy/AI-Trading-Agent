import React, { useState, useEffect, createContext, useContext, useCallback } from 'react';

// Notification types
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

// Notification interface
export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  title?: string;
  autoClose?: boolean;
  duration?: number;
}

// Context interface
interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => string;
  removeNotification: (id: string) => void;
  clearAllNotifications: () => void;
}

// Create context
const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// Generate unique ID
const generateId = (): string => {
  return Math.random().toString(36).substring(2, 9);
};

// Provider component
export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  // Add notification
  const addNotification = useCallback((notification: Omit<Notification, 'id'>): string => {
    const id = generateId();
    const newNotification: Notification = {
      id,
      autoClose: true,
      duration: 5000,
      ...notification,
    };

    setNotifications((prevNotifications) => [...prevNotifications, newNotification]);
    return id;
  }, []);

  // Remove notification
  const removeNotification = useCallback((id: string): void => {
    setNotifications((prevNotifications) => 
      prevNotifications.filter((notification) => notification.id !== id)
    );
  }, []);

  // Clear all notifications
  const clearAllNotifications = useCallback((): void => {
    setNotifications([]);
  }, []);

  // Auto-close notifications
  useEffect(() => {
    const timers: NodeJS.Timeout[] = [];

    notifications.forEach((notification) => {
      if (notification.autoClose) {
        const timer = setTimeout(() => {
          removeNotification(notification.id);
        }, notification.duration);
        timers.push(timer);
      }
    });

    return () => {
      timers.forEach((timer) => clearTimeout(timer));
    };
  }, [notifications, removeNotification]);

  return (
    <NotificationContext.Provider
      value={{
        notifications,
        addNotification,
        removeNotification,
        clearAllNotifications,
      }}
    >
      {children}
      <NotificationContainer />
    </NotificationContext.Provider>
  );
};

// Hook to use notifications
export const useNotification = (): NotificationContextType => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

// Helper hooks for specific notification types
export const useSuccessNotification = () => {
  const { addNotification } = useNotification();
  return useCallback(
    (message: string, title?: string, options?: Partial<Omit<Notification, 'id' | 'type' | 'message' | 'title'>>) => {
      return addNotification({ type: 'success', message, title, ...options });
    },
    [addNotification]
  );
};

export const useErrorNotification = () => {
  const { addNotification } = useNotification();
  return useCallback(
    (message: string, title?: string, options?: Partial<Omit<Notification, 'id' | 'type' | 'message' | 'title'>>) => {
      return addNotification({ type: 'error', message, title, ...options });
    },
    [addNotification]
  );
};

export const useWarningNotification = () => {
  const { addNotification } = useNotification();
  return useCallback(
    (message: string, title?: string, options?: Partial<Omit<Notification, 'id' | 'type' | 'message' | 'title'>>) => {
      return addNotification({ type: 'warning', message, title, ...options });
    },
    [addNotification]
  );
};

export const useInfoNotification = () => {
  const { addNotification } = useNotification();
  return useCallback(
    (message: string, title?: string, options?: Partial<Omit<Notification, 'id' | 'type' | 'message' | 'title'>>) => {
      return addNotification({ type: 'info', message, title, ...options });
    },
    [addNotification]
  );
};

// Notification Item component
const NotificationItem: React.FC<{
  notification: Notification;
  onClose: (id: string) => void;
}> = ({ notification, onClose }) => {
  const { id, type, message, title } = notification;

  // Get icon and color based on notification type
  const getTypeStyles = (type: NotificationType) => {
    switch (type) {
      case 'success':
        return {
          icon: (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path>
            </svg>
          ),
          bgColor: 'bg-green-50 dark:bg-green-900/20',
          borderColor: 'border-green-500',
          textColor: 'text-green-800 dark:text-green-200',
          iconColor: 'text-green-500',
        };
      case 'error':
        return {
          icon: (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd"></path>
            </svg>
          ),
          bgColor: 'bg-red-50 dark:bg-red-900/20',
          borderColor: 'border-red-500',
          textColor: 'text-red-800 dark:text-red-200',
          iconColor: 'text-red-500',
        };
      case 'warning':
        return {
          icon: (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd"></path>
            </svg>
          ),
          bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
          borderColor: 'border-yellow-500',
          textColor: 'text-yellow-800 dark:text-yellow-200',
          iconColor: 'text-yellow-500',
        };
      case 'info':
      default:
        return {
          icon: (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd"></path>
            </svg>
          ),
          bgColor: 'bg-blue-50 dark:bg-blue-900/20',
          borderColor: 'border-blue-500',
          textColor: 'text-blue-800 dark:text-blue-200',
          iconColor: 'text-blue-500',
        };
    }
  };

  const styles = getTypeStyles(type);

  return (
    <div
      className={`flex items-center p-4 mb-3 rounded-lg border-l-4 shadow-md ${styles.bgColor} ${styles.borderColor}`}
      role="alert"
    >
      <div className={`inline-flex items-center justify-center flex-shrink-0 w-8 h-8 ${styles.iconColor}`}>
        {styles.icon}
      </div>
      <div className="ml-3 text-sm font-normal">
        {title && <div className="font-medium">{title}</div>}
        <div className={styles.textColor}>{message}</div>
      </div>
      <button
        type="button"
        className={`ml-auto -mx-1.5 -my-1.5 ${styles.textColor} rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 inline-flex h-8 w-8 focus:outline-none`}
        aria-label="Close"
        onClick={() => onClose(id)}
      >
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"></path>
        </svg>
      </button>
    </div>
  );
};

// Notification Container component
const NotificationContainer: React.FC = () => {
  const { notifications, removeNotification } = useNotification();

  return (
    <div className="fixed top-4 right-4 z-50 w-full max-w-xs">
      {notifications.map((notification) => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onClose={removeNotification}
        />
      ))}
    </div>
  );
};

export default NotificationProvider;
