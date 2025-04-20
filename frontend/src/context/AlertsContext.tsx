import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import {
  Alert,
  AlertFilterOptions,
  AlertSeverity,
  AlertStatus,
  AlertType,
  getAlerts,
  updateAlertStatus,
  createAlert,
  startAlertChecks,
  requestNotificationPermission
} from '../api/utils/alerts';

interface AlertsContextType {
  alerts: Alert[];
  activeAlerts: Alert[];
  getFilteredAlerts: (options: AlertFilterOptions) => Alert[];
  acknowledgeAlert: (alertId: string) => void;
  resolveAlert: (alertId: string) => void;
  ignoreAlert: (alertId: string) => void;
  createCustomAlert: (
    severity: AlertSeverity,
    type: AlertType,
    title: string,
    message: string,
    metadata?: any,
    exchange?: string,
    method?: string
  ) => Alert;
  hasUnacknowledgedAlerts: boolean;
  hasUnacknowledgedCriticalAlerts: boolean;
  notificationsEnabled: boolean;
  requestNotifications: () => Promise<boolean>;
  alertCheckInterval: number;
  setAlertCheckInterval: (interval: number) => void;
}

const AlertsContext = createContext<AlertsContextType | undefined>(undefined);

export const AlertsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<Alert[]>([]);
  const [notificationsEnabled, setNotificationsEnabled] = useState<boolean>(false);
  const [alertCheckInterval, setAlertCheckInterval] = useState<number>(60000); // 1 minute default
  const [stopAlertChecks, setStopAlertChecks] = useState<(() => void) | null>(null);

  // Load alerts on mount
  useEffect(() => {
    refreshAlerts();
    
    // Check if notifications are already enabled
    if (typeof window !== 'undefined' && 'Notification' in window) {
      setNotificationsEnabled(Notification.permission === 'granted');
    }
    
    // Start periodic alert checks
    const stopFn = startAlertChecks(alertCheckInterval);
    setStopAlertChecks(() => stopFn);
    
    return () => {
      if (stopAlertChecks) {
        stopAlertChecks();
      }
    };
  }, []);

  // Update alert checks when interval changes
  useEffect(() => {
    if (stopAlertChecks) {
      stopAlertChecks();
    }
    
    const stopFn = startAlertChecks(alertCheckInterval);
    setStopAlertChecks(() => stopFn);
    
    return () => {
      if (stopAlertChecks) {
        stopAlertChecks();
      }
    };
  }, [alertCheckInterval]);

  // Refresh alerts from the alerts utility
  const refreshAlerts = useCallback(() => {
    const allAlerts = getAlerts();
    setAlerts(allAlerts);
    
    // Filter for active alerts
    const active = allAlerts.filter(alert => alert.status === AlertStatus.ACTIVE);
    setActiveAlerts(active);
  }, []);

  // Get filtered alerts
  const getFilteredAlerts = useCallback((options: AlertFilterOptions): Alert[] => {
    return getAlerts(options);
  }, []);

  // Update alert status
  const updateStatus = useCallback((alertId: string, status: AlertStatus) => {
    const userId = 'current-user'; // This would come from auth context in a real app
    const updatedAlert = updateAlertStatus(alertId, status, userId);
    
    if (updatedAlert) {
      refreshAlerts();
    }
    
    return updatedAlert;
  }, [refreshAlerts]);

  // Acknowledge an alert
  const acknowledgeAlert = useCallback((alertId: string) => {
    return updateStatus(alertId, AlertStatus.ACKNOWLEDGED);
  }, [updateStatus]);

  // Resolve an alert
  const resolveAlert = useCallback((alertId: string) => {
    return updateStatus(alertId, AlertStatus.RESOLVED);
  }, [updateStatus]);

  // Ignore an alert
  const ignoreAlert = useCallback((alertId: string) => {
    return updateStatus(alertId, AlertStatus.IGNORED);
  }, [updateStatus]);

  // Create a custom alert
  const createCustomAlert = useCallback(
    (
      severity: AlertSeverity,
      type: AlertType,
      title: string,
      message: string,
      metadata?: any,
      exchange?: string,
      method?: string
    ): Alert => {
      const alert = createAlert(severity, type, title, message, metadata, exchange, method);
      refreshAlerts();
      return alert;
    },
    [refreshAlerts]
  );

  // Request notification permissions
  const requestNotifications = useCallback(async (): Promise<boolean> => {
    const granted = await requestNotificationPermission();
    setNotificationsEnabled(granted);
    return granted;
  }, []);

  // Calculate if there are unacknowledged alerts
  const hasUnacknowledgedAlerts = activeAlerts.length > 0;
  
  // Calculate if there are unacknowledged critical alerts
  const hasUnacknowledgedCriticalAlerts = activeAlerts.some(
    alert => alert.severity === AlertSeverity.CRITICAL || alert.severity === AlertSeverity.ERROR
  );

  // Set up periodic refresh of alerts
  useEffect(() => {
    const intervalId = setInterval(() => {
      refreshAlerts();
    }, 10000); // Refresh every 10 seconds
    
    return () => clearInterval(intervalId);
  }, [refreshAlerts]);

  const value = {
    alerts,
    activeAlerts,
    getFilteredAlerts,
    acknowledgeAlert,
    resolveAlert,
    ignoreAlert,
    createCustomAlert,
    hasUnacknowledgedAlerts,
    hasUnacknowledgedCriticalAlerts,
    notificationsEnabled,
    requestNotifications,
    alertCheckInterval,
    setAlertCheckInterval
  };

  return <AlertsContext.Provider value={value}>{children}</AlertsContext.Provider>;
};

export const useAlerts = (): AlertsContextType => {
  const context = useContext(AlertsContext);
  if (context === undefined) {
    throw new Error('useAlerts must be used within an AlertsProvider');
  }
  return context;
};

export default AlertsContext;
