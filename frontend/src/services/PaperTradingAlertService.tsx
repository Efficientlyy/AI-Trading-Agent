import { useState, useCallback } from 'react';
import { PaperTradingAlert } from '../types/paperTrading';
import { paperTradingApi } from '../api/paperTrading';
import { useNotification } from '../components/common/NotificationSystem';
import { generateId } from '../lib/utils';

// Define the alert service interface
interface PaperTradingAlertService {
  sessionStarted: (sessionId: string, config: any) => Omit<PaperTradingAlert, 'id'>;
  sessionCompleted: (sessionId: string, results: any) => Omit<PaperTradingAlert, 'id'>;
  sessionStopped: (sessionId: string) => Omit<PaperTradingAlert, 'id'>;
  tradePlaced: (sessionId: string, trade: any) => Omit<PaperTradingAlert, 'id'>;
  tradeClosed: (sessionId: string, trade: any) => Omit<PaperTradingAlert, 'id'>;
  checkProfitTarget: (sessionId: string, trade: any, threshold?: number) => Omit<PaperTradingAlert, 'id'> | null;
  checkStopLoss: (sessionId: string, trade: any, threshold?: number) => Omit<PaperTradingAlert, 'id'> | null;
  checkConsecutiveLosses: (sessionId: string, trade: any, threshold?: number) => Omit<PaperTradingAlert, 'id'> | null;
  checkLargeTrade: (sessionId: string, trade: any, threshold?: number) => Omit<PaperTradingAlert, 'id'> | null;
  addAlert: (sessionId: string, alert: Omit<PaperTradingAlert, 'id'>) => Omit<PaperTradingAlert, 'id'>;
}

export const usePaperTradingAlerts = (): PaperTradingAlertService => {
  const { showNotification } = useNotification();
  
  // Send alert to backend and show notification
  const sendAlert = useCallback(async (sessionId: string, alert: Omit<PaperTradingAlert, 'id'>) => {
    try {
      // Send alert to backend
      await paperTradingApi.addAlert(sessionId, alert);
      
      // Show notification
      showNotification({
        type: alert.severity,
        title: alert.title,
        message: alert.message
      });
      
      return alert;
    } catch (error) {
      console.error('Failed to send alert:', error);
      return alert;
    }
  }, [showNotification]);
  
  // Alert for session started
  const sessionStarted = useCallback((sessionId: string, config: any): Omit<PaperTradingAlert, 'id'> => {
    const alert = {
      type: 'info',
      title: 'Paper Trading Session Started',
      message: `Session ${sessionId} started with ${config.duration_minutes || config.duration} minutes duration`,
      severity: 'info',
      timestamp: new Date().toISOString()
    };
    
    sendAlert(sessionId, alert);
    return alert;
  }, [sendAlert]);
  
  // Alert for session completed
  const sessionCompleted = useCallback((sessionId: string, results: any): Omit<PaperTradingAlert, 'id'> => {
    const alert = {
      type: 'success',
      title: 'Paper Trading Session Completed',
      message: `Session ${sessionId} completed with ${results?.profit_loss_percentage || 0}% P&L`,
      severity: 'success',
      timestamp: new Date().toISOString()
    };
    
    sendAlert(sessionId, alert);
    return alert;
  }, [sendAlert]);
  
  // Alert for session stopped
  const sessionStopped = useCallback((sessionId: string): Omit<PaperTradingAlert, 'id'> => {
    const alert = {
      type: 'warning',
      title: 'Paper Trading Session Stopped',
      message: `Session ${sessionId} was manually stopped`,
      severity: 'warning',
      timestamp: new Date().toISOString()
    };
    
    sendAlert(sessionId, alert);
    return alert;
  }, [sendAlert]);
  
  // Alert for trade placed
  const tradePlaced = useCallback((sessionId: string, trade: any): Omit<PaperTradingAlert, 'id'> => {
    const alert = {
      type: 'info',
      title: 'Trade Placed',
      message: `${trade.side} ${trade.quantity} ${trade.symbol} at ${trade.price}`,
      severity: 'info',
      timestamp: new Date().toISOString()
    };
    
    sendAlert(sessionId, alert);
    return alert;
  }, [sendAlert]);
  
  // Alert for trade closed
  const tradeClosed = useCallback((sessionId: string, trade: any): Omit<PaperTradingAlert, 'id'> => {
    const alert = {
      type: 'info',
      title: 'Trade Closed',
      message: `Closed ${trade.side} ${trade.quantity} ${trade.symbol} with ${trade.profit_loss_percentage}% P&L`,
      severity: trade.profit_loss_percentage >= 0 ? 'success' : 'warning',
      timestamp: new Date().toISOString()
    };
    
    sendAlert(sessionId, alert);
    return alert;
  }, [sendAlert]);
  
  // Check if profit target reached
  const checkProfitTarget = useCallback((sessionId: string, trade: any, threshold: number = 5): Omit<PaperTradingAlert, 'id'> | null => {
    if (trade.profit_loss_percentage >= threshold) {
      const alert = {
        type: 'success',
        title: 'Profit Target Reached',
        message: `${trade.symbol} reached ${trade.profit_loss_percentage}% profit target`,
        severity: 'success',
        timestamp: new Date().toISOString()
      };
      
      sendAlert(sessionId, alert);
      return alert;
    }
    return null;
  }, [sendAlert]);
  
  // Check if stop loss triggered
  const checkStopLoss = useCallback((sessionId: string, trade: any, threshold: number = -3): Omit<PaperTradingAlert, 'id'> | null => {
    if (trade.profit_loss_percentage <= threshold) {
      const alert = {
        type: 'warning',
        title: 'Stop Loss Triggered',
        message: `${trade.symbol} hit ${trade.profit_loss_percentage}% stop loss level`,
        severity: 'warning',
        timestamp: new Date().toISOString()
      };
      
      sendAlert(sessionId, alert);
      return alert;
    }
    return null;
  }, [sendAlert]);
  
  // Check for consecutive losses
  const checkConsecutiveLosses = useCallback((sessionId: string, trade: any, threshold: number = 3): Omit<PaperTradingAlert, 'id'> | null => {
    if (trade.consecutive_losses >= threshold) {
      const alert = {
        type: 'error',
        title: 'Consecutive Losses Alert',
        message: `${trade.consecutive_losses} consecutive losing trades detected`,
        severity: 'error',
        timestamp: new Date().toISOString()
      };
      
      sendAlert(sessionId, alert);
      return alert;
    }
    return null;
  }, [sendAlert]);
  
  // Check for large trades
  const checkLargeTrade = useCallback((sessionId: string, trade: any, threshold: number = 1000): Omit<PaperTradingAlert, 'id'> | null => {
    if (trade.value >= threshold) {
      const alert = {
        type: 'warning',
        title: 'Large Trade Alert',
        message: `Large trade detected: ${trade.side} ${trade.quantity} ${trade.symbol} worth $${trade.value}`,
        severity: 'warning',
        timestamp: new Date().toISOString()
      };
      
      sendAlert(sessionId, alert);
      return alert;
    }
    return null;
  }, [sendAlert]);
  
  // Add custom alert
  const addAlert = useCallback((sessionId: string, alert: Omit<PaperTradingAlert, 'id'>): Omit<PaperTradingAlert, 'id'> => {
    const fullAlert = {
      ...alert,
      timestamp: alert.timestamp || new Date().toISOString()
    };
    
    sendAlert(sessionId, fullAlert);
    return fullAlert;
  }, [sendAlert]);
  
  return {
    sessionStarted,
    sessionCompleted,
    sessionStopped,
    tradePlaced,
    tradeClosed,
    checkProfitTarget,
    checkStopLoss,
    checkConsecutiveLosses,
    checkLargeTrade,
    addAlert
  };
};
