import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '../../../components/ui/Card';
import { Button } from '../../../components/ui/Button';
import { Checkbox } from '../../../components/ui/Checkbox';
import { Spinner } from '../../../components/common/Spinner';
import { paperTradingApi } from '../../../api/paperTrading';
import { 
  ExportFormat, 
  ExportOptions, 
  DEFAULT_EXPORT_OPTIONS,
  PaperTradingExportService 
} from '../../../services/PaperTradingExportService';
import { useNotification } from '../../../components/common/NotificationSystem';

const PaperTradingExportPanel: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const { showNotification } = useNotification();
  const [exportOptions, setExportOptions] = useState<ExportOptions>(DEFAULT_EXPORT_OPTIONS);
  
  // Fetch session details and results if sessionId is provided
  const { 
    data: sessionData,
    isLoading: sessionLoading,
    error: sessionError
  } = useQuery({
    queryKey: ['paperTradingSession', sessionId],
    queryFn: () => sessionId ? paperTradingApi.getSessionDetails(sessionId) : null,
    enabled: !!sessionId
  });
  
  const {
    data: resultsData,
    isLoading: resultsLoading,
    error: resultsError
  } = useQuery({
    queryKey: ['paperTradingResults', sessionId],
    queryFn: () => sessionId ? paperTradingApi.getResults(sessionId) : null,
    enabled: !!sessionId && (sessionData?.status === 'completed' || sessionData?.status === 'error')
  });
  
  const {
    data: alertsData,
    isLoading: alertsLoading,
    error: alertsError
  } = useQuery({
    queryKey: ['paperTradingAlerts', sessionId],
    queryFn: () => sessionId ? paperTradingApi.getSessionAlerts(sessionId) : null,
    enabled: !!sessionId && exportOptions.includeAlerts
  });
  
  // Handle export format change
  const handleFormatChange = (format: ExportFormat) => {
    setExportOptions(prev => ({
      ...prev,
      format
    }));
  };
  
  // Handle checkbox change
  const handleCheckboxChange = (key: keyof Omit<ExportOptions, 'format' | 'dateRange'>) => {
    setExportOptions(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };
  
  // Handle export button click
  const handleExport = () => {
    if (!sessionId || !resultsData) {
      showNotification({
        type: 'error',
        title: 'Export Error',
        message: 'No data available to export'
      });
      return;
    }
    
    try {
      // Combine all data for export
      const exportData = {
        ...resultsData,
        session_id: sessionId,
        start_time: sessionData?.start_time,
        end_time: new Date().toISOString(),
        duration_minutes: sessionData?.uptime_seconds ? Math.floor(sessionData.uptime_seconds / 60) : 0,
        symbols: sessionData?.symbols || [],
        alerts: alertsData?.alerts || []
      };
      
      // Export the data
      PaperTradingExportService.exportResults(exportData, exportOptions);
      
      // Show success notification
      showNotification({
        type: 'success',
        title: 'Export Successful',
        message: `Paper trading results exported in ${exportOptions.format.toUpperCase()} format`
      });
    } catch (error) {
      console.error('Export error:', error);
      showNotification({
        type: 'error',
        title: 'Export Error',
        message: 'Failed to export paper trading results'
      });
    }
  };
  
  // Check if export is available
  const isExportAvailable = !!sessionId && 
    !!resultsData && 
    (sessionData?.status === 'completed' || sessionData?.status === 'error');
  
  // Loading state
  const isLoading = sessionLoading || resultsLoading || (exportOptions.includeAlerts && alertsLoading);
  
  // Error state
  const hasError = !!sessionError || !!resultsError || (exportOptions.includeAlerts && !!alertsError);
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Export Results</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex justify-center p-4">
            <Spinner />
          </div>
        ) : hasError ? (
          <p className="text-red-500 text-center p-4">Error loading data for export</p>
        ) : !isExportAvailable ? (
          <p className="text-gray-500 text-center p-4">
            {!sessionId 
              ? 'Select a session to export results' 
              : 'Results are only available for completed sessions'}
          </p>
        ) : (
          <div className="space-y-4">
            {/* Export Format */}
            <div>
              <h3 className="text-sm font-medium mb-2">Export Format</h3>
              <div className="flex space-x-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    className="form-radio"
                    checked={exportOptions.format === ExportFormat.CSV}
                    onChange={() => handleFormatChange(ExportFormat.CSV)}
                  />
                  <span>CSV</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    className="form-radio"
                    checked={exportOptions.format === ExportFormat.JSON}
                    onChange={() => handleFormatChange(ExportFormat.JSON)}
                  />
                  <span>JSON</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer opacity-50">
                  <input
                    type="radio"
                    className="form-radio"
                    disabled
                    checked={exportOptions.format === ExportFormat.PDF}
                    onChange={() => handleFormatChange(ExportFormat.PDF)}
                  />
                  <span>PDF (Coming Soon)</span>
                </label>
              </div>
            </div>
            
            {/* Export Options */}
            <div>
              <h3 className="text-sm font-medium mb-2">Include Data</h3>
              <div className="space-y-2">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <Checkbox 
                    checked={exportOptions.includeMetrics}
                    onChange={() => handleCheckboxChange('includeMetrics')}
                  />
                  <span>Performance Metrics</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <Checkbox 
                    checked={exportOptions.includeTrades}
                    onChange={() => handleCheckboxChange('includeTrades')}
                  />
                  <span>Trades</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <Checkbox 
                    checked={exportOptions.includePortfolioHistory}
                    onChange={() => handleCheckboxChange('includePortfolioHistory')}
                  />
                  <span>Portfolio History</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <Checkbox 
                    checked={exportOptions.includeAlerts}
                    onChange={() => handleCheckboxChange('includeAlerts')}
                  />
                  <span>Alerts</span>
                </label>
              </div>
            </div>
            
            {/* Export Button */}
            <div className="pt-2">
              <Button 
                onClick={handleExport}
                disabled={!isExportAvailable}
                className="w-full"
              >
                Export Results
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default PaperTradingExportPanel;