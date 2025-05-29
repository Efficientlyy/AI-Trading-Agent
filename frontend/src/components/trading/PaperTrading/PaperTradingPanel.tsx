import React, { useState, useEffect, useCallback } from 'react';
import { useNotification } from '../../../components/common/NotificationSystem';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { Spinner } from '../../../components/common/Spinner';
import { Card, CardHeader, CardTitle, CardContent } from '../../../components/ui/Card';
import { Button } from '../../../components/ui/Button';
import { Select } from '../../../components/ui/Select';
import { Input } from '../../../components/ui/Input';
import { Badge } from '../../ui/Badge';
import { Divider } from '../../ui/Divider';
import { Tooltip } from '../../ui/Tooltip';

// Import components
import AutonomousTradingButton from './AutonomousTradingButton';

// Import API client
import { paperTradingApi } from '../../../api/paperTrading';

interface PaperTradingConfig {
  configPath: string;
  duration: number;
  interval: number;
}

interface PaperTradingSession {
  session_id: string;
  status: 'starting' | 'running' | 'stopping' | 'completed' | 'error';
  start_time?: string;
  uptime_seconds?: number;
  symbols?: string[];
  current_portfolio?: any;
}

interface PaperTradingPanelProps {
  selectedSessionId?: string;
}

const PaperTradingPanel: React.FC<PaperTradingPanelProps> = ({ selectedSessionId }) => {
  const { showNotification } = useNotification();
  const queryClient = useQueryClient();
  const { state, startPaperTrading, stopPaperTrading, selectSession } = usePaperTrading();
  
  // State for form inputs
  const [configPath, setConfigPath] = useState<string>('config/trading_config.yaml');
  const [duration, setDuration] = useState<number>(60);
  const [interval, setInterval] = useState<number>(1);
  const [availableConfigs, setAvailableConfigs] = useState<string[]>([
    'config/trading_config.yaml',
    'config/crypto_trading_config.yaml',
    'config/sentiment_trading_config.yaml'
  ]);

  // Use data from context
  const statusData = state.currentStatus;
  const statusLoading = state.isLoading;
  const sessionsData = state.activeSessions;
  
  // Create mutations for starting and stopping paper trading
  const startMutation = useMutation({
    mutationFn: (config: PaperTradingConfig) => startPaperTrading(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paperTradingSessions'] });
      queryClient.invalidateQueries({ queryKey: ['paperTradingStatus'] });
    }
  });
  
  const stopMutation = useMutation({
    mutationFn: (sessionId: string) => stopPaperTrading(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['paperTradingSessions'] });
      queryClient.invalidateQueries({ queryKey: ['paperTradingStatus'] });
    }
  });

  // Fetch available config files
  useEffect(() => {
    // In a real implementation, we would fetch this from the API
    // For now, we'll use the hardcoded list
    setAvailableConfigs([
      'config/trading_config.yaml',
      'config/crypto_trading_config.yaml',
      'config/sentiment_trading_config.yaml'
    ]);
  }, []);

  // Handle starting paper trading
  const handleStartPaperTrading = useCallback(() => {
    startMutation.mutate({
      configPath,
      duration,
      interval
    });
  }, [configPath, duration, interval, startMutation]);

  // Handle stopping paper trading
  const handleStopPaperTrading = useCallback((sessionId: string) => {
    stopMutation.mutate(sessionId);
  }, [stopMutation]);

  // Get color for status badge
  const getStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'starting':
        return 'bg-blue-500';
      case 'running':
        return 'bg-green-500';
      case 'stopping':
        return 'bg-yellow-500';
      case 'completed':
        return 'bg-gray-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  }, []);

  // Determine if start button should be disabled
  const isStartDisabled = 
    startMutation.isPending || 
    stopMutation.isPending || 
    (statusData?.status === 'running' || statusData?.status === 'starting');

  // Get active sessions
  const sessions = sessionsData || [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Paper Trading</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Configuration Form */}
          <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
            <h3 className="text-lg font-medium mb-4">Configuration</h3>
            
            <div className="mb-4">
              <h3 className="text-lg font-medium mb-2">Trading Configuration</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Configuration</label>
                  <Select
                    value={configPath}
                    onChange={(e) => setConfigPath(e.target.value)}
                    disabled={startMutation.isPending}
                  >
                    {availableConfigs.map((config) => (
                      <option key={config} value={config}>{config}</option>
                    ))}
                  </Select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Duration (minutes)</label>
                  <Input
                    type="number"
                    value={duration}
                    onChange={(e) => setDuration(parseInt(e.target.value) || 60)}
                    min={1}
                    max={1440}
                    disabled={startMutation.isPending}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Update Interval (minutes)</label>
                  <Input
                    type="number"
                    value={interval}
                    onChange={(e) => setInterval(parseInt(e.target.value) || 1)}
                    min={1}
                    max={60}
                    disabled={startMutation.isPending}
                  />
                </div>
              </div>
              
              <div className="mt-6 flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-4">
                <div>
                  <h4 className="text-md font-medium mb-2">Standard Mode</h4>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                    Start paper trading with manual oversight and control
                  </p>
                  <Button
                    onClick={handleStartPaperTrading}
                    disabled={isStartDisabled}
                    variant="secondary"
                  >
                    {startMutation.isPending ? 'Starting...' : 'Start Paper Trading'}
                  </Button>
                </div>
                
                <Divider orientation="vertical" className="hidden sm:block" />
                <Divider className="sm:hidden" />
                
                <div>
                  <h4 className="text-md font-medium mb-2">Autonomous Mode</h4>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                    Start fully autonomous trading with AI-driven decisions
                  </p>
                  <Tooltip content="Trading agent will operate independently with full decision-making autonomy">
                    <div>
                      <AutonomousTradingButton
                        configPath={configPath}
                        duration={duration}
                        interval={interval}
                      />
                    </div>
                  </Tooltip>
                </div>
              </div>
            </div>
            
            {/* Status */}
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
              <h3 className="text-lg font-medium mb-4">Current Status</h3>
              
              {statusLoading ? (
                <div className="flex justify-center items-center h-32">
                  <Spinner size="lg" />
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-2">
                    <div>Status:</div>
                    <div>
                      <Badge className={`${getStatusColor(statusData?.status || 'idle')} text-white`}>
                        {statusData?.status || 'idle'}
                      </Badge>
                    </div>
                    
                    {statusData?.uptime_seconds && (
                      <>
                        <div>Uptime:</div>
                        <div>
                          {Math.floor(statusData.uptime_seconds / 60)} minutes, {statusData.uptime_seconds % 60} seconds
                        </div>
                      </>
                    )}
                    
                    {statusData?.symbols && statusData.symbols.length > 0 && (
                      <>
                        <div>Symbols:</div>
                        <div>{statusData.symbols.join(', ')}</div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Status */}
          <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
            <h3 className="text-lg font-medium mb-4">Current Status</h3>
            
            {statusLoading ? (
              <div className="flex justify-center items-center h-32">
                <Spinner size="lg" />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-2">
                  <div>Status:</div>
                  <div>
                    <Badge className={`${getStatusColor(statusData?.status || 'idle')} text-white`}>
                      {statusData?.status || 'idle'}
                    </Badge>
                  </div>
                  
                  {statusData?.uptime_seconds && (
                    <>
                      <div>Uptime:</div>
                      <div>
                        {Math.floor(statusData.uptime_seconds / 60)} minutes, {statusData.uptime_seconds % 60} seconds
                      </div>
                    </>
                  )}
                  
                  {statusData?.symbols && statusData.symbols.length > 0 && (
                    <>
                      <div>Symbols:</div>
                      <div>{statusData.symbols.join(', ')}</div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Active Sessions */}
        <div className="mt-8">
          <h3 className="text-lg font-medium mb-4">Active Sessions</h3>
          
          {sessions.length === 0 ? (
            <div className="text-center p-8 bg-gray-50 dark:bg-gray-800 rounded-md">
              <p className="text-gray-500 dark:text-gray-400">No active paper trading sessions</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Session ID
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Start Time
                    </th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200">
                  {sessions.map((session: PaperTradingSession) => (
                    <tr key={session.session_id}>
                      <td className="px-4 py-2 whitespace-nowrap">
                        {session.session_id.substring(0, 8)}...
                      </td>
                      <td className="px-4 py-2 whitespace-nowrap">
                        <Badge className={`${getStatusColor(session.status)} text-white`}>
                          {session.status}
                        </Badge>
                      </td>
                      <td className="px-4 py-2 whitespace-nowrap">
                        {session.start_time 
                          ? new Date(session.start_time).toLocaleString() 
                          : 'N/A'}
                      </td>
                      <td className="px-4 py-2 whitespace-nowrap">
                        {session.status === 'running' && (
                          <Button
                            variant="danger"
                            size="sm"
                            onClick={() => handleStopPaperTrading(session.session_id)}
                            disabled={stopMutation.isPending}
                            className="mr-2"
                          >
                            {stopMutation.isPending ? 'Stopping...' : 'Stop'}
                          </Button>
                        )}
                        <Link to={`/paper-trading/${session.session_id}`}>
                          <Button
                            variant={selectedSessionId === session.session_id ? "primary" : "secondary"}
                            size="sm"
                          >
                            Details
                          </Button>
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default PaperTradingPanel;