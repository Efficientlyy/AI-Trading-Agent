import React, { createContext, useContext, useReducer, ReactNode, useCallback } from 'react';

// Define notification types
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

// Define notification interface
export interface Notification {
  id: string;
  message: string;
  type: NotificationType;
  duration?: number;
  timestamp: number;
}

// Define notification state
interface NotificationState {
  notifications: Notification[];
}

// Define notification actions
type NotificationAction = 
  | { type: 'ADD_NOTIFICATION'; payload: Omit<Notification, 'id' | 'timestamp'> }
  | { type: 'REMOVE_NOTIFICATION'; payload: { id: string } }
  | { type: 'CLEAR_ALL_NOTIFICATIONS' };

// Define notification context
interface NotificationContextType {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearAllNotifications: () => void;
}

// Create notification context
const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// Create notification reducer
const notificationReducer = (state: NotificationState, action: NotificationAction): NotificationState => {
  switch (action.type) {
    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [
          ...state.notifications,
          {
            ...action.payload,
            id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            timestamp: Date.now(),
          },
        ],
      };
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(
          (notification) => notification.id !== action.payload.id
        ),
      };
    case 'CLEAR_ALL_NOTIFICATIONS':
      return {
        ...state,
        notifications: [],
      };
    default:
      return state;
  }
};

// Create notification provider
export const NotificationProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(notificationReducer, { notifications: [] });

  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp'>) => {
    dispatch({ type: 'ADD_NOTIFICATION', payload: notification });
    
    // Auto-remove notification after duration (default: 5000ms)
    if (notification.duration !== 0) {
      const duration = notification.duration || 5000;
      setTimeout(() => {
        dispatch({
          type: 'REMOVE_NOTIFICATION',
          payload: { id: `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}` },
        });
      }, duration);
    }
  }, []);

  const removeNotification = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_NOTIFICATION', payload: { id } });
  }, []);

  const clearAllNotifications = useCallback(() => {
    dispatch({ type: 'CLEAR_ALL_NOTIFICATIONS' });
  }, []);

  return (
    <NotificationContext.Provider
      value={{
        notifications: state.notifications,
        addNotification,
        removeNotification,
        clearAllNotifications,
      }}
    >
      {children}
    </NotificationContext.Provider>
  );
};

// Create hook for using notification context
export const useNotification = (): NotificationContextType => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

// Helper functions for adding specific notification types
export const useNotificationHelpers = () => {
  const { addNotification } = useNotification();

  const success = useCallback((message: string, duration?: number) => {
    addNotification({ message, type: 'success', duration });
  }, [addNotification]);

  const error = useCallback((message: string, duration?: number) => {
    addNotification({ message, type: 'error', duration });
  }, [addNotification]);

  const warning = useCallback((message: string, duration?: number) => {
    addNotification({ message, type: 'warning', duration });
  }, [addNotification]);

  const info = useCallback((message: string, duration?: number) => {
    addNotification({ message, type: 'info', duration });
  }, [addNotification]);

  return { success, error, warning, info };
};
