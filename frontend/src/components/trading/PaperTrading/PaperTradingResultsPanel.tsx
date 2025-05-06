import React from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '../../../components/ui/Card';
import { Spinner } from '../../../components/common/Spinner';
import { Badge } from '../../../components/ui/Badge';
import { paperTradingApi } from '../../../api/paperTrading';

const PaperTradingResultsPanel: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  
  // Fetch session details
  const {
    data: sessionData,
    isLoading: sessionLoading,
    error: sessionError
  } = useQuery({
    queryKey: ['paperTradingSession', sessionId],
    queryFn: () => sessionId ? paperTradingApi.getSessionDetails(sessionId) : null,
    enabled: !!sessionId,
    refetchInterval: 5000 // Refetch every 5 seconds
  });

  // Fetch results if session is completed
  const {
    data: resultsData,
    isLoading: resultsLoading,
    error: resultsError
  } = useQuery({
    queryKey: ['paperTradingResults', sessionId],
    queryFn: () => sessionId ? paperTradingApi.getResults(sessionId) : null,
    enabled: !!sessionId && (sessionData?.status === 'completed' || sessionData?.status === 'error')
  });

  // Status badge color
  const getStatusColor = (status: string) => {
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
  };

  // Loading state
  if (sessionLoading) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <Spinner size="lg" className="mx-auto" />
          <p className="mt-4 text-gray-500">Loading session details...</p>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (sessionError) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <div className="text-red-500 mb-2">
            <svg className="w-12 h-12 mx-auto" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
          </div>
          <h3 className="text-lg font-medium">Error Loading Session</h3>
          <p className="mt-2 text-gray-500">Failed to load paper trading session details.</p>
        </CardContent>
      </Card>
    );
  }

  // No session found
  if (!sessionData) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <p className="text-gray-500">No session found with ID: {sessionId}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          <span>Paper Trading Results</span>
          {sessionData?.status && (
            <Badge className={`${getStatusColor(sessionData.status)} text-white`}>
              {sessionData.status}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Session Details */}
        <div className="mb-6">
          <h3 className="font-medium mb-2">Session Details</h3>
          <div className="grid grid-cols-2 gap-2">
            <div>Status:</div>
            <div>
              <Badge className={`${getStatusColor(sessionData?.status)} text-white`}>
                {sessionData?.status || 'Unknown'}
              </Badge>
            </div>
            
            <div>Started:</div>
            <div>
              {sessionData?.start_time 
                ? new Date(sessionData.start_time).toLocaleString() 
                : 'N/A'}
            </div>
            
            <div>Uptime:</div>
            <div>
              {sessionData?.uptime_seconds 
                ? `${Math.floor(sessionData.uptime_seconds / 60)} minutes, ${sessionData.uptime_seconds % 60} seconds` 
                : 'N/A'}
            </div>
            
            <div>Symbols:</div>
            <div>{sessionData?.symbols?.join(', ') || 'None'}</div>
          </div>
        </div>
        
        {/* Results */}
        {(sessionData?.status === 'completed' || sessionData?.status === 'error') && (
          <div>
            <h3 className="font-medium mb-2">Performance Metrics</h3>
            {resultsLoading ? (
              <div className="flex justify-center items-center h-32">
                <Spinner size="lg" />
              </div>
            ) : resultsError ? (
              <div className="text-center p-4 bg-red-50 dark:bg-red-900 dark:bg-opacity-20 rounded-md">
                <p className="text-red-500">Error loading results</p>
              </div>
            ) : !resultsData ? (
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                <p className="text-gray-500">No results available</p>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="grid grid-cols-2 gap-2">
                    <div>Total Return:</div>
                    <div className={resultsData.performance_metrics?.total_return >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {resultsData.performance_metrics?.total_return 
                        ? `${(resultsData.performance_metrics.total_return * 100).toFixed(2)}%` 
                        : 'N/A'}
                    </div>
                    
                    <div>Sharpe Ratio:</div>
                    <div>
                      {resultsData.performance_metrics?.sharpe_ratio 
                        ? resultsData.performance_metrics.sharpe_ratio.toFixed(2) 
                        : 'N/A'}
                    </div>
                    
                    <div>Max Drawdown:</div>
                    <div className="text-red-600">
                      {resultsData.performance_metrics?.max_drawdown 
                        ? `${(resultsData.performance_metrics.max_drawdown * 100).toFixed(2)}%` 
                        : 'N/A'}
                    </div>
                    
                    <div>Win Rate:</div>
                    <div>
                      {resultsData.performance_metrics?.win_rate 
                        ? `${(resultsData.performance_metrics.win_rate * 100).toFixed(2)}%` 
                        : 'N/A'}
                    </div>
                  </div>
                
                  <div className="col-span-2">
                    <h4 className="font-medium mb-2">Trades ({resultsData.trades?.length || 0})</h4>
                    {resultsData.trades && resultsData.trades.length > 0 ? (
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead>
                            <tr>
                              <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Symbol
                              </th>
                              <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Side
                              </th>
                              <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Price
                              </th>
                              <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Size
                              </th>
                              <th className="px-2 py-1 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Timestamp
                              </th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-200">
                            {resultsData.trades.slice(0, 10).map((trade: any, index: number) => (
                              <tr key={index}>
                                <td className="px-2 py-1 whitespace-nowrap">{trade.symbol}</td>
                                <td className="px-2 py-1 whitespace-nowrap">
                                  <Badge className={trade.side === 'buy' ? 'bg-green-500 text-white' : 'bg-red-500 text-white'}>
                                    {trade.side}
                                  </Badge>
                                </td>
                                <td className="px-2 py-1 whitespace-nowrap">{trade.price}</td>
                                <td className="px-2 py-1 whitespace-nowrap">{trade.size}</td>
                                <td className="px-2 py-1 whitespace-nowrap">
                                  {new Date(trade.timestamp).toLocaleString()}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {resultsData.trades.length > 10 && (
                          <div className="text-center text-sm text-gray-500 mt-2">
                            Showing 10 of {resultsData.trades.length} trades
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                        <p className="text-gray-500">No trades available</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Current Status (if running) */}
        {sessionData?.status === 'running' && (
          <div className="text-center p-4">
            <p className="text-gray-500">
              Paper trading session is currently running.
              <br />
              Results will be available when the session completes.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default PaperTradingResultsPanel;