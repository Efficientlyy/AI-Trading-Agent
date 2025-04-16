import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Cell } from 'recharts';
import { BacktestMetrics } from './BacktestingInterface';

export interface OptimizationResult {
  parameters: Record<string, any>;
  metrics: BacktestMetrics;
}

export interface StrategyOptimizerProps {
  symbol: string;
  strategy: string;
  availableParameters: ParameterConfig[];
  onRunOptimization?: (params: OptimizationParams) => void;
  optimizationResults?: OptimizationResult[];
  isLoading?: boolean;
}

export interface OptimizationParams {
  symbol: string;
  strategy: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  targetMetric: keyof BacktestMetrics;
  parameters: ParameterConfig[];
}

export interface ParameterConfig {
  name: string;
  min: number;
  max: number;
  step: number;
  currentValue: number;
  description?: string;
}

const StrategyOptimizer: React.FC<StrategyOptimizerProps> = ({
  symbol,
  strategy,
  availableParameters = [],
  onRunOptimization,
  optimizationResults = [],
  isLoading = false
}) => {
  // Default parameter configurations based on strategy
  const getDefaultParameters = (): ParameterConfig[] => {
    switch (strategy) {
      case 'Moving Average Crossover':
        return [
          { name: 'fastPeriod', min: 5, max: 50, step: 1, currentValue: 10, description: 'Fast moving average period' },
          { name: 'slowPeriod', min: 20, max: 200, step: 5, currentValue: 50, description: 'Slow moving average period' },
          { name: 'signalPeriod', min: 3, max: 20, step: 1, currentValue: 9, description: 'Signal line period' }
        ];
      case 'RSI Oscillator':
        return [
          { name: 'period', min: 7, max: 30, step: 1, currentValue: 14, description: 'RSI calculation period' },
          { name: 'overbought', min: 60, max: 90, step: 1, currentValue: 70, description: 'Overbought threshold' },
          { name: 'oversold', min: 10, max: 40, step: 1, currentValue: 30, description: 'Oversold threshold' }
        ];
      case 'MACD Crossover':
        return [
          { name: 'fastPeriod', min: 8, max: 20, step: 1, currentValue: 12, description: 'Fast EMA period' },
          { name: 'slowPeriod', min: 20, max: 40, step: 1, currentValue: 26, description: 'Slow EMA period' },
          { name: 'signalPeriod', min: 5, max: 15, step: 1, currentValue: 9, description: 'Signal line period' }
        ];
      case 'Bollinger Breakout':
        return [
          { name: 'period', min: 10, max: 50, step: 1, currentValue: 20, description: 'Calculation period' },
          { name: 'standardDeviations', min: 1, max: 4, step: 0.1, currentValue: 2, description: 'Number of standard deviations' }
        ];
      case 'Sentiment-Based':
        return [
          { name: 'sentimentThreshold', min: 0.1, max: 0.9, step: 0.05, currentValue: 0.6, description: 'Sentiment signal threshold' },
          { name: 'lookbackPeriod', min: 1, max: 30, step: 1, currentValue: 7, description: 'Sentiment lookback period (days)' }
        ];
      default:
        return [];
    }
  };
  
  // State for optimization parameters
  const [parameters, setParameters] = useState<ParameterConfig[]>(() => {
    return availableParameters.length > 0 ? availableParameters : getDefaultParameters();
  });
  
  const [optimizationParams, setOptimizationParams] = useState<OptimizationParams>(() => {
    const defaultParams = availableParameters.length > 0 ? availableParameters : getDefaultParameters();
    return {
      symbol,
      strategy,
      startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 1)).toISOString().split('T')[0], // Default to 1 year ago
      endDate: new Date().toISOString().split('T')[0], // Default to today
      initialCapital: 10000,
      targetMetric: 'sharpeRatio',
      parameters: defaultParams
    };
  });
  
  // State for visualization options
  const [visualizationType, setVisualizationType] = useState<'table' | 'scatter' | 'radar'>('table');
  const [selectedParameter, setSelectedParameter] = useState<string>(() => {
    const defaultParams = availableParameters.length > 0 ? availableParameters : getDefaultParameters();
    return defaultParams.length > 0 ? defaultParams[0].name : '';
  });
  
  // Update parameters when strategy changes
  useEffect(() => {
    if (availableParameters.length === 0) {
      const newParams = getDefaultParameters();
      setParameters(newParams);
      if (newParams.length > 0 && !selectedParameter) {
        setSelectedParameter(newParams[0].name);
      }
    }
  }, [strategy, availableParameters, selectedParameter]);
  
  // Update optimization params when parameters change
  useEffect(() => {
    setOptimizationParams(prev => ({
      ...prev,
      symbol,
      strategy,
      parameters
    }));
  }, [symbol, strategy, parameters]);
  
  // Handle parameter range changes
  const handleParameterChange = (index: number, field: keyof ParameterConfig, value: any) => {
    const updatedParameters = [...parameters];
    updatedParameters[index] = {
      ...updatedParameters[index],
      [field]: typeof value === 'string' ? parseFloat(value) : value
    };
    setParameters(updatedParameters);
  };
  
  // Handle date and capital changes
  const handleInputChange = (field: keyof OptimizationParams, value: any) => {
    setOptimizationParams(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Run optimization
  const handleRunOptimization = () => {
    if (onRunOptimization) {
      onRunOptimization(optimizationParams);
    }
  };
  
  // Format parameter value for display
  const formatParameterValue = (value: number): string => {
    return value % 1 === 0 ? value.toString() : value.toFixed(2);
  };
  
  // Prepare data for parameter impact visualization
  const prepareParameterImpactData = () => {
    if (!selectedParameter || optimizationResults.length === 0) return [];
    
    return optimizationResults.map(result => ({
      parameterValue: result.parameters[selectedParameter],
      metricValue: result.metrics[optimizationParams.targetMetric],
      size: 20, // Size for scatter plot
      result // Keep the full result for reference
    }));
  };
  
  // Prepare data for radar chart
  const prepareRadarData = () => {
    if (optimizationResults.length === 0) return [];
    
    // Take top 5 results
    const topResults = optimizationResults.slice(0, 5);
    
    // Create radar data structure
    const radarData = parameters.map(param => {
      const dataPoint: any = { parameter: param.name };
      
      topResults.forEach((result, index) => {
        // Normalize parameter value to 0-100 scale
        const normalizedValue = ((result.parameters[param.name] - param.min) / (param.max - param.min)) * 100;
        dataPoint[`Result ${index + 1}`] = normalizedValue;
      });
      
      return dataPoint;
    });
    
    return radarData;
  };
  
  // Get color based on metric value
  const getMetricColor = (value: number, metric: keyof BacktestMetrics): string => {
    // For drawdown, lower is better
    if (metric === 'maxDrawdown') {
      if (value < 10) return '#10B981'; // Green
      if (value < 20) return '#F59E0B'; // Yellow
      return '#EF4444'; // Red
    }
    
    // For other metrics, higher is better
    if (metric === 'sharpeRatio') {
      if (value > 1.5) return '#10B981'; // Green
      if (value > 1) return '#F59E0B'; // Yellow
      return '#EF4444'; // Red
    }
    
    if (metric === 'winRate') {
      if (value > 60) return '#10B981'; // Green
      if (value > 50) return '#F59E0B'; // Yellow
      return '#EF4444'; // Red
    }
    
    if (metric.includes('Return')) {
      if (value > 20) return '#10B981'; // Green
      if (value > 10) return '#F59E0B'; // Yellow
      return '#EF4444'; // Red
    }
    
    // Default
    return '#4F46E5'; // Indigo
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Strategy Optimizer</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Configuration Panel */}
        <div className="md:col-span-1 space-y-4">
          <div>
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Configuration</h3>
            
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Symbol
                </label>
                <div className="text-sm font-medium text-gray-900 dark:text-gray-100 px-3 py-2 bg-gray-100 dark:bg-gray-800 rounded-md">
                  {symbol}
                </div>
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Strategy
                </label>
                <div className="text-sm font-medium text-gray-900 dark:text-gray-100 px-3 py-2 bg-gray-100 dark:bg-gray-800 rounded-md">
                  {strategy}
                </div>
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Start Date
                </label>
                <input
                  type="date"
                  value={optimizationParams.startDate}
                  onChange={(e) => handleInputChange('startDate', e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                />
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  End Date
                </label>
                <input
                  type="date"
                  value={optimizationParams.endDate}
                  onChange={(e) => handleInputChange('endDate', e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                />
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Initial Capital
                </label>
                <input
                  type="number"
                  min="1000"
                  step="1000"
                  value={optimizationParams.initialCapital}
                  onChange={(e) => handleInputChange('initialCapital', Number(e.target.value))}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                />
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Optimization Target
                </label>
                <select
                  value={optimizationParams.targetMetric}
                  onChange={(e) => handleInputChange('targetMetric', e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                >
                  <option value="sharpeRatio">Sharpe Ratio</option>
                  <option value="totalReturn">Total Return</option>
                  <option value="maxDrawdown">Max Drawdown</option>
                  <option value="winRate">Win Rate</option>
                  <option value="profitFactor">Profit Factor</option>
                </select>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Parameter Ranges</h3>
            
            <div className="space-y-4">
              {parameters.map((param, index) => (
                <div key={param.name} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                      {param.name}
                    </label>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {param.description}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Min
                      </label>
                      <input
                        type="number"
                        step={param.step}
                        value={param.min}
                        onChange={(e) => handleParameterChange(index, 'min', e.target.value)}
                        className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Max
                      </label>
                      <input
                        type="number"
                        step={param.step}
                        value={param.max}
                        onChange={(e) => handleParameterChange(index, 'max', e.target.value)}
                        className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Step
                      </label>
                      <input
                        type="number"
                        step={param.step < 1 ? 0.01 : 1}
                        value={param.step}
                        onChange={(e) => handleParameterChange(index, 'step', e.target.value)}
                        className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <button
              type="button"
              onClick={handleRunOptimization}
              disabled={isLoading}
              className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              {isLoading ? 'Running Optimization...' : 'Run Optimization'}
            </button>
          </div>
        </div>
        
        {/* Results Panel */}
        <div className="md:col-span-2">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Optimization Results
            </h3>
            
            {optimizationResults.length > 0 && (
              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={() => setVisualizationType('table')}
                  className={`px-2 py-1 text-xs rounded-md ${
                    visualizationType === 'table' 
                      ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-200' 
                      : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-200'
                  }`}
                >
                  Table
                </button>
                <button
                  type="button"
                  onClick={() => setVisualizationType('scatter')}
                  className={`px-2 py-1 text-xs rounded-md ${
                    visualizationType === 'scatter' 
                      ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-200' 
                      : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-200'
                  }`}
                >
                  Parameter Impact
                </button>
                <button
                  type="button"
                  onClick={() => setVisualizationType('radar')}
                  className={`px-2 py-1 text-xs rounded-md ${
                    visualizationType === 'radar' 
                      ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-200' 
                      : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-200'
                  }`}
                >
                  Radar
                </button>
              </div>
            )}
          </div>
          
          {isLoading ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
            </div>
          ) : optimizationResults.length === 0 ? (
            <div className="flex flex-col justify-center items-center h-64 text-gray-500 dark:text-gray-400">
              <svg className="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-sm">Run optimization to see results</p>
            </div>
          ) : (
            <div className="space-y-4">
              {visualizationType === 'table' && (
                <div className="border rounded-md overflow-hidden" style={{ display: visualizationType === 'table' ? 'block' : 'none' }}>
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-800">
                      <tr>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Rank
                        </th>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Parameters
                        </th>
                        <th scope="col" className="px-3 py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          {optimizationParams.targetMetric.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                      {optimizationResults.map((result, index) => (
                        <tr key={index} className={index === 0 ? 'bg-green-50 dark:bg-green-900 bg-opacity-30' : ''}>
                          <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                            {index + 1}
                          </td>
                          <td className="px-3 py-2 text-sm text-gray-500 dark:text-gray-400">
                            <div className="space-y-1">
                              {Object.entries(result.parameters).map(([key, value]) => (
                                <div key={key} className="text-xs">
                                  <span className="font-medium">{key}:</span> {formatParameterValue(value as number)}
                                </div>
                              ))}
                            </div>
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-right font-medium text-gray-900 dark:text-gray-100">
                            {optimizationParams.targetMetric === 'maxDrawdown' 
                              ? `-${result.metrics[optimizationParams.targetMetric]}%` 
                              : optimizationParams.targetMetric.includes('Return') 
                                ? `${result.metrics[optimizationParams.targetMetric]}%`
                                : result.metrics[optimizationParams.targetMetric]}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              
              {visualizationType === 'scatter' && (
                <div style={{ display: visualizationType === 'scatter' ? 'block' : 'none' }}>
                  <div className="mb-2 flex justify-between items-center">
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                      Parameter:
                    </label>
                    <select
                      value={selectedParameter}
                      onChange={(e) => setSelectedParameter(e.target.value)}
                      className="ml-2 px-2 py-1 text-xs border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                    >
                      {parameters.map(param => (
                        <option key={param.name} value={param.name}>{param.name}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div className="h-64 border rounded-md p-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart
                        margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                          type="number" 
                          dataKey="parameterValue" 
                          name={selectedParameter} 
                          domain={['dataMin', 'dataMax']}
                          label={{ value: selectedParameter, position: 'bottom', offset: 0 }}
                        />
                        <YAxis 
                          type="number" 
                          dataKey="metricValue" 
                          name={optimizationParams.targetMetric} 
                          label={{ 
                            value: optimizationParams.targetMetric, 
                            angle: -90, 
                            position: 'left' 
                          }}
                        />
                    
                        <Tooltip 
                          cursor={{ strokeDasharray: '3 3' }}
                          formatter={(value: any, name: string) => {
                            if (name === selectedParameter) return [value, name];
                            if (name === optimizationParams.targetMetric) {
                              return [
                                optimizationParams.targetMetric === 'maxDrawdown' || optimizationParams.targetMetric.includes('Return')
                                  ? `${value}%`
                                  : value,
                                name.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())
                              ];
                            }
                            return [value, name];
                          }}
                        />
                        <Scatter 
                          name="Parameters" 
                          data={prepareParameterImpactData()} 
                          fill="#8884d8"
                        >
                          {prepareParameterImpactData().map((entry, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={getMetricColor(entry.metricValue, optimizationParams.targetMetric)} 
                            />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
              
              {visualizationType === 'radar' && (
                <div className="h-80 border rounded-md p-2" style={{ display: visualizationType === 'radar' ? 'block' : 'none' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart outerRadius={90} data={prepareRadarData()}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="parameter" />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} />
                      {optimizationResults.slice(0, 5).map((_, index) => (
                        <Radar
                          key={`radar-${index}`}
                          name={`Result ${index + 1}`}
                          dataKey={`Result ${index + 1}`}
                          stroke={[
                            '#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'
                          ][index]}
                          fill={[
                            '#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'
                          ][index]}
                          fillOpacity={0.2}
                        />
                      ))}
                      <Legend />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              )}
              
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => {
                    if (optimizationResults.length > 0) {
                      // Apply the best parameters
                      const bestParams = optimizationResults[0].parameters;
                      const updatedParameters = parameters.map(param => ({
                        ...param,
                        currentValue: bestParams[param.name] || param.currentValue
                      }));
                      
                      setParameters(updatedParameters);
                      setOptimizationParams(prev => ({
                        ...prev,
                        parameters: updatedParameters
                      }));
                    }
                  }}
                  className="py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                >
                  Apply Best Parameters
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StrategyOptimizer;
