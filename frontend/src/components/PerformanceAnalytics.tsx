import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Divider,
  CircularProgress,
  useTheme,
  Paper
} from '@mui/material';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, Cell, Area, AreaChart
} from 'recharts';
import apiClient from '../api/apiClient';
import { formatCurrency, formatPercent } from '../utils/formatters';

interface PerformanceMetric {
  timestamp: string;
  value: number;
  benchmark?: number;
}

interface MetricSummary {
  current: number;
  change: number;
  changePercentage: number;
}

interface PerformanceAnalyticsProps {
  timeframe?: string;
  assetId?: string;
  strategyId?: string;
  refreshInterval?: number;
}

/**
 * Advanced performance analytics component for tracking key metrics.
 * 
 * Features:
 * - Real-time performance tracking
 * - Benchmark comparison
 * - Strategy attribution analysis
 * - Risk-adjusted return metrics
 */
const PerformanceAnalytics: React.FC<PerformanceAnalyticsProps> = ({
  timeframe = '1M',
  assetId,
  strategyId,
  refreshInterval = 60000  // 1 minute default
}) => {
  const theme = useTheme();
  const [loading, setLoading] = useState<boolean>(true);
  const [returnsData, setReturnsData] = useState<PerformanceMetric[]>([]);
  const [drawdownData, setDrawdownData] = useState<PerformanceMetric[]>([]);
  const [volatilityData, setVolatilityData] = useState<PerformanceMetric[]>([]);
  const [metrics, setMetrics] = useState<Record<string, MetricSummary>>({});
  const [sharpeRatio, setSharpeRatio] = useState<number>(0);
  const [tradesData, setTradesData] = useState<any[]>([]);
  const [strategyAttribution, setStrategyAttribution] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Chart colors
  const colors = {
    primary: theme.palette.primary.main,
    secondary: theme.palette.secondary.main,
    success: theme.palette.success.main,
    error: theme.palette.error.main,
    warning: theme.palette.warning.main,
    info: theme.palette.info.main,
    benchmark: theme.palette.mode === 'dark' ? '#8884d8' : '#82ca9d',
  };

  // Fetch performance data
  const fetchPerformanceData = async () => {
    try {
      setLoading(true);
      
      // Build query parameters
      const params = new URLSearchParams();
      params.append('timeframe', timeframe);
      if (assetId) params.append('assetId', assetId);
      if (strategyId) params.append('strategyId', strategyId);
      
      // Fetch returns data
      const returnsResponse = await apiClient.get(`/analytics/returns?${params.toString()}`);
      setReturnsData(returnsResponse.data || []);
      
      // Fetch drawdown data
      const drawdownResponse = await apiClient.get(`/analytics/drawdown?${params.toString()}`);
      setDrawdownData(drawdownResponse.data || []);
      
      // Fetch volatility data
      const volatilityResponse = await apiClient.get(`/analytics/volatility?${params.toString()}`);
      setVolatilityData(volatilityResponse.data || []);
      
      // Fetch Sharpe ratio data
      const sharpeResponse = await apiClient.get(`/analytics/sharpe?${params.toString()}`);
      const sharpeData = sharpeResponse.data || { ratio: 0 };
      setSharpeRatio(sharpeData.ratio);
      
      // Fetch trades data
      const tradesResponse = await apiClient.get(`/analytics/trades?${params.toString()}`);
      setTradesData(tradesResponse.data || []);
      
      // Fetch summary metrics
      const metricsResponse = await apiClient.get(`/analytics/summary?${params.toString()}`);
      setMetrics(metricsResponse.data || {});
      
      // Fetch strategy attribution
      const attributionResponse = await apiClient.get(`/analytics/attribution?${params.toString()}`);
      setStrategyAttribution(attributionResponse.data || []);
      
      setError(null);
    } catch (err) {
      console.error('Error fetching performance data:', err);
      setError('Failed to load performance data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Fetch data initially and set up refresh interval
  useEffect(() => {
    fetchPerformanceData();
    
    // Set up refresh interval
    const intervalId = setInterval(fetchPerformanceData, refreshInterval);
    
    // Clean up on unmount
    return () => clearInterval(intervalId);
  }, [timeframe, assetId, strategyId, refreshInterval]);

  // Render loading state
  if (loading && !returnsData.length) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  // Render error state
  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Total Return" 
            value={metrics.return?.current || 0} 
            change={metrics.return?.change || 0}
            changePercentage={metrics.return?.changePercentage || 0}
            format="percentage"
            color={colors.primary}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Sharpe Ratio" 
            value={metrics.sharpe?.current || 0} 
            change={metrics.sharpe?.change || 0}
            changePercentage={metrics.sharpe?.changePercentage || 0}
            format="number"
            color={colors.secondary}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Max Drawdown" 
            value={metrics.drawdown?.current || 0} 
            change={metrics.drawdown?.change || 0}
            changePercentage={metrics.drawdown?.changePercentage || 0}
            format="percentage"
            color={colors.error}
            invertChange={true}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="Volatility" 
            value={metrics.volatility?.current || 0} 
            change={metrics.volatility?.change || 0}
            changePercentage={metrics.volatility?.changePercentage || 0}
            format="percentage"
            color={colors.warning}
            invertChange={true}
          />
        </Grid>
      </Grid>

      {/* Performance Charts */}
      <Grid container spacing={3}>
        {/* Returns Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Cumulative Return
              </Typography>
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <AreaChart
                    data={returnsData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    />
                    <YAxis 
                      tickFormatter={(tick) => formatPercent(tick)}
                    />
                    <Tooltip 
                      formatter={(value: any) => [formatPercent(value), 'Return']}
                      labelFormatter={(label) => new Date(label).toLocaleString()}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      name="Portfolio" 
                      stroke={colors.primary} 
                      fill={colors.primary}
                      fillOpacity={0.2}
                    />
                    {returnsData.length > 0 && returnsData[0].benchmark !== undefined && (
                      <Area 
                        type="monotone" 
                        dataKey="benchmark" 
                        name="Benchmark" 
                        stroke={colors.benchmark} 
                        fill={colors.benchmark}
                        fillOpacity={0.1}
                      />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </Grid>

        {/* Strategy Attribution */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Strategy Attribution
              </Typography>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie
                      data={strategyAttribution}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    >
                      {strategyAttribution.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.color || Object.values(colors)[index % Object.values(colors).length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip 
                      formatter={(value: any) => formatPercent(value)}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </Grid>

        {/* Drawdown Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Drawdown
              </Typography>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <AreaChart
                    data={drawdownData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    />
                    <YAxis 
                      tickFormatter={(tick) => formatPercent(tick)}
                    />
                    <Tooltip 
                      formatter={(value: any) => [formatPercent(value), 'Drawdown']}
                      labelFormatter={(label) => new Date(label).toLocaleString()}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      name="Drawdown" 
                      stroke={colors.error} 
                      fill={colors.error}
                      fillOpacity={0.2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </Grid>

        {/* Volatility Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Rolling Volatility
              </Typography>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <LineChart
                    data={volatilityData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    />
                    <YAxis 
                      tickFormatter={(tick) => formatPercent(tick)}
                    />
                    <Tooltip 
                      formatter={(value: any) => [formatPercent(value), 'Volatility']}
                      labelFormatter={(label) => new Date(label).toLocaleString()}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      name="Portfolio" 
                      stroke={colors.warning} 
                      dot={false}
                    />
                    {volatilityData.length > 0 && volatilityData[0].benchmark !== undefined && (
                      <Line 
                        type="monotone" 
                        dataKey="benchmark" 
                        name="Benchmark" 
                        stroke={colors.benchmark} 
                        dot={false}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

// Metric Card Component
interface MetricCardProps {
  title: string;
  value: number;
  change: number;
  changePercentage: number;
  format: 'percentage' | 'currency' | 'number';
  color: string;
  invertChange?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changePercentage,
  format,
  color,
  invertChange = false
}) => {
  // Format value based on type
  const formatValue = (val: number): string => {
    if (format === 'percentage') return formatPercent(val);
    if (format === 'currency') return formatCurrency(val);
    return val.toFixed(2);
  };

  // Determine if change is positive (considering inversion)
  const isPositive = invertChange ? change < 0 : change > 0;
  const changeColor = change === 0 ? 'text.secondary' : (isPositive ? 'success.main' : 'error.main');

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="subtitle2" color="text.secondary">
          {title}
        </Typography>
        <Typography variant="h4" sx={{ color, my: 1 }}>
          {formatValue(value)}
        </Typography>
        <Typography variant="body2" sx={{ color: changeColor }}>
          {change > 0 && '+'}{formatValue(change)} ({changePercentage > 0 && '+'}{formatPercent(changePercentage)})
        </Typography>
      </CardContent>
    </Card>
  );
};

export default PerformanceAnalytics;
