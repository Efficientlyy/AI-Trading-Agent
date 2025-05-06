import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '../../../components/ui/Card';
import { Badge } from '../../../components/ui/Badge';
import { Button } from '../../../components/ui/Button';
import { Switch } from '../../../components/ui/Switch';
import { Spinner } from '../../../components/common/Spinner';
import { 
  PaperTradingAlertType, 
  DEFAULT_ALERT_THRESHOLDS, 
  PaperTradingAlertThresholds 
} from '../../../services/PaperTradingAlertService';
import { paperTradingApi } from '../../../api/paperTrading';

const PaperTradingAlertPanel: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [alertsEnabled, setAlertsEnabled] = useState<boolean>(true);
  const [thresholds, setThresholds] = useState<PaperTradingAlertThresholds>(DEFAULT_ALERT_THRESHOLDS);
  const [alerts, setAlerts] = useState<any[]>([]);
  
  // Fetch alerts if sessionId is provided
  const { 
    data: alertsData,
    isLoading: alertsLoading,
    error: alertsError,
    refetch: refetchAlerts
  } = useQuery({
    queryKey: ['paperTradingAlerts', sessionId],
    queryFn: () => sessionId ? paperTradingApi.getSessionAlerts(sessionId) : null,
    enabled: !!sessionId && alertsEnabled,
    refetchInterval: alertsEnabled ? 10000 : false // Refetch every 10 seconds if enabled
  });
  
  // Update alerts when data changes
  useEffect(() => {
    if (alertsData?.alerts) {
      setAlerts(alertsData.alerts);
    }
  }, [alertsData]);
  
  // Get alert badge color based on type
  const getAlertBadgeColor = (type: PaperTradingAlertType) => {
    switch (type) {
      case PaperTradingAlertType.SESSION_STARTED:
      case PaperTradingAlertType.SESSION_COMPLETED:
      case PaperTradingAlertType.SIGNIFICANT_GAIN:
        return 'bg-green-500 text-white';
      case PaperTradingAlertType.SESSION_ERROR:
      case PaperTradingAlertType.SYSTEM_ERROR:
        return 'bg-red-500 text-white';
      case PaperTradingAlertType.LARGE_DRAWDOWN:
      case PaperTradingAlertType.CONSECUTIVE_LOSSES:
      case PaperTradingAlertType.POOR_PERFORMANCE:
        return 'bg-yellow-500 text-white';
      case PaperTradingAlertType.LARGE_TRADE:
      case PaperTradingAlertType.DATA_DELAY:
      case PaperTradingAlertType.STRATEGY_CHANGE:
        return 'bg-blue-500 text-white';
      default:
        return 'bg-gray-500 text-white';
    }
  };
  
  // Format alert timestamp
  const formatAlertTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };
  
  // Handle threshold change
  const handleThresholdChange = (key: keyof PaperTradingAlertThresholds, value: number) => {
    setThresholds(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  // Clear alerts
  const handleClearAlerts = () => {
    setAlerts([]);
  };
  
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          <span>Paper Trading Alerts</span>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {alertsEnabled ? 'Alerts Enabled' : 'Alerts Disabled'}
            </span>
            <Switch 
              checked={alertsEnabled} 
              onCheckedChange={setAlertsEnabled} 
              aria-label="Toggle alerts"
            />
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Alert Settings */}
        <div className="mb-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-md">
          <h3 className="text-sm font-medium mb-2">Alert Thresholds</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>Portfolio Drawdown:</div>
            <div className="flex items-center">
              <input
                type="number"
                className="w-16 h-6 px-1 text-xs border border-gray-300 dark:border-gray-600 rounded"
                value={thresholds.drawdownThreshold * 100}
                onChange={(e) => handleThresholdChange('drawdownThreshold', Number(e.target.value) / 100)}
                min="1"
                max="50"
                step="1"
              />
              <span className="ml-1">%</span>
            </div>
            
            <div>Large Trade Size:</div>
            <div className="flex items-center">
              <span className="mr-1">$</span>
              <input
                type="number"
                className="w-16 h-6 px-1 text-xs border border-gray-300 dark:border-gray-600 rounded"
                value={thresholds.largeTradeThreshold}
                onChange={(e) => handleThresholdChange('largeTradeThreshold', Number(e.target.value))}
                min="100"
                max="10000"
                step="100"
              />
            </div>
            
            <div>Consecutive Losses:</div>
            <div>
              <input
                type="number"
                className="w-16 h-6 px-1 text-xs border border-gray-300 dark:border-gray-600 rounded"
                value={thresholds.consecutiveLossesThreshold}
                onChange={(e) => handleThresholdChange('consecutiveLossesThreshold', Number(e.target.value))}
                min="1"
                max="10"
                step="1"
              />
            </div>
          </div>
        </div>
        
        {/* Alerts List */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <h3 className="font-medium">Recent Alerts</h3>
            <Button 
              outline={true} 
              variant="secondary" 
              size="sm" 
              onClick={handleClearAlerts}
              disabled={alerts.length === 0}
            >
              Clear
            </Button>
          </div>
          
          {alertsLoading ? (
            <div className="flex justify-center p-4">
              <Spinner />
            </div>
          ) : alertsError ? (
            <p className="text-red-500 text-center p-4">Error loading alerts</p>
          ) : alerts.length === 0 ? (
            <p className="text-gray-500 text-center p-4">No alerts</p>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {alerts.map((alert, index) => (
                <div 
                  key={index} 
                  className="p-2 border border-gray-200 dark:border-gray-700 rounded-md"
                >
                  <div className="flex justify-between items-start">
                    <Badge className={getAlertBadgeColor(alert.type)}>
                      {alert.type.replace(/_/g, ' ')}
                    </Badge>
                    <span className="text-xs text-gray-500">
                      {formatAlertTime(alert.timestamp)}
                    </span>
                  </div>
                  <p className="mt-1 text-sm">{alert.message}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default PaperTradingAlertPanel;