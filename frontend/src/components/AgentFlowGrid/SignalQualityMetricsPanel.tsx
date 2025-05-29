import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  LinearProgress,
  Skeleton,
  ButtonGroup,
  Button,
  Tooltip,
  CircularProgress,
  useTheme,
  useMediaQuery
} from '@mui/material';
import { 
  Radar, 
  Bar,
  Doughnut 
} from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Filler,
  Tooltip as ChartTooltip,
  Legend
} from 'chart.js';
import sentimentAnalyticsService, { 
  SignalQualityMetrics, 
  TimeFrame 
} from '../../api/sentimentAnalyticsService';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

// Register required Chart.js components
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Filler,
  ChartTooltip,
  Legend
);

interface SignalQualityMetricsPanelProps {
  agentId: string;
  selectedSymbol: string | null;
}

// Helper tooltips for metrics explanation
const metricTooltips = {
  accuracy: "Percentage of sentiment signals that correctly predicted market direction",
  precision: "Ability to identify positive signals correctly (true positives / (true positives + false positives))",
  recall: "Ability to find all positive signals (true positives / (true positives + false negatives))",
  f1_score: "Harmonic mean of precision and recall - balances both metrics",
  success_rate: "Percentage of signals that resulted in profitable trades",
  average_confidence: "Average confidence level reported for all predictions"
};

const SignalQualityMetricsPanel: React.FC<SignalQualityMetricsPanelProps> = ({
  agentId,
  selectedSymbol
}) => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.between('sm', 'md'));
  
  const [timeframe, setTimeframe] = useState<TimeFrame>('30d');
  const [metricsData, setMetricsData] = useState<SignalQualityMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Memoized fetch function to avoid unnecessary recreations
  const fetchMetricsData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await sentimentAnalyticsService.getSignalQualityMetrics(
        agentId,
        timeframe
      );
      setMetricsData(data);
    } catch (err) {
      console.error('Error fetching signal quality metrics:', err);
      setError('Failed to load signal quality metrics');
    } finally {
      setIsLoading(false);
    }
  }, [agentId, timeframe]);
  
  // Fetch metrics data when dependencies change
  useEffect(() => {
    fetchMetricsData();
  }, [fetchMetricsData]);
  
  // Memoized panel title based on selected symbol
  const panelTitle = useMemo(() => {
    return selectedSymbol 
      ? `${selectedSymbol} Signal Quality Metrics` 
      : 'Overall Signal Quality Metrics';
  }, [selectedSymbol]);
  
  // Memoized radar chart data
  const radarChartData = useMemo(() => {
    if (!metricsData) return null;
    
    return {
      labels: [
        'Accuracy', 
        'Precision', 
        'Recall', 
        'F1 Score', 
        'Success Rate', 
        'Confidence'
      ],
      datasets: [
        {
          label: 'Quality Metrics',
          data: [
            metricsData.overall_accuracy,
            metricsData.overall_precision,
            metricsData.overall_recall,
            metricsData.overall_f1_score,
            metricsData.total_successful_signals / metricsData.total_signals,
            selectedSymbol && metricsData.performance_by_symbol[selectedSymbol]
              ? metricsData.performance_by_symbol[selectedSymbol].avg_confidence
              : Object.values(metricsData.performance_by_symbol).reduce((acc, curr) => acc + curr.avg_confidence, 0) / 
                Object.keys(metricsData.performance_by_symbol).length
          ],
          backgroundColor: 'rgba(76, 175, 80, 0.2)',
          borderColor: 'rgba(76, 175, 80, 0.8)',
          borderWidth: 2,
          pointBackgroundColor: '#4caf50',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#4caf50',
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    };
  }, [metricsData, selectedSymbol]);
  
  // Memoized symbol performance bar chart data
  const symbolPerformanceData = useMemo(() => {
    if (!metricsData) return null;
    
    // If specific symbol is selected, only show that symbol vs overall
    const symbols = selectedSymbol
      ? [selectedSymbol] 
      : Object.keys(metricsData.performance_by_symbol).slice(0, 6);
      
    return {
      labels: symbols,
      datasets: [
        {
          label: 'Accuracy',
          data: symbols.map(sym => metricsData.performance_by_symbol[sym].accuracy),
          backgroundColor: '#4caf50',
          barThickness: isMobile ? 15 : 25
        },
        {
          label: 'Confidence',
          data: symbols.map(sym => metricsData.performance_by_symbol[sym].avg_confidence),
          backgroundColor: '#2196f3',
          barThickness: isMobile ? 15 : 25
        }
      ]
    };
  }, [metricsData, selectedSymbol, isMobile]);
  
  // Memoized signal distribution doughnut chart data
  const signalDistributionData = useMemo(() => {
    if (!metricsData) return null;
    
    // Calculate success/failure rate
    const successfulSignals = metricsData.total_successful_signals;
    const unsuccessfulSignals = metricsData.total_signals - metricsData.total_successful_signals;
    
    return {
      labels: ['Successful Signals', 'Unsuccessful Signals'],
      datasets: [
        {
          data: [successfulSignals, unsuccessfulSignals],
          backgroundColor: ['#4caf50', '#f44336'],
          borderColor: isDarkMode ? 'rgba(30,30,30,0.8)' : '#fff',
          borderWidth: 2,
          hoverBackgroundColor: ['rgba(76, 175, 80, 0.8)', 'rgba(244, 67, 54, 0.8)'],
          hoverBorderColor: isDarkMode ? 'rgba(30,30,30,1)' : '#fff',
          hoverBorderWidth: 3
        }
      ]
    };
  }, [metricsData, isDarkMode]);
  
  // Memoized radar chart options
  const radarOptions = useMemo(() => {
    return {
      scales: {
        r: {
          // Specify the correct scale type for the radar chart
          type: 'radialLinear' as const,
          beginAtZero: true,
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.2,
            display: false,
            backdropColor: 'transparent'
          },
          grid: {
            color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
          },
          angleLines: {
            color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
          },
          pointLabels: {
            color: isDarkMode ? '#ddd' : '#333',
            font: {
              size: isMobile ? 10 : 12
            }
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const value = context.raw;
              return `${context.label}: ${(value * 100).toFixed(1)}%`;
            }
          },
          backgroundColor: isDarkMode ? 'rgba(25, 25, 25, 0.8)' : 'rgba(255, 255, 255, 0.8)',
          titleColor: isDarkMode ? '#fff' : '#333',
          bodyColor: isDarkMode ? '#ddd' : '#555',
          borderColor: isDarkMode ? 'rgba(200, 200, 200, 0.2)' : 'rgba(0, 0, 0, 0.1)',
          borderWidth: 1
        }
      },
      maintainAspectRatio: false,
      responsive: true
    };
  }, [isDarkMode, isMobile]);
  
  // Memoized bar chart options
  const barOptions = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top' as const,
          labels: {
            color: isDarkMode ? '#ddd' : '#333',
            boxWidth: 15,
            padding: 10
          }
        },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const value = context.raw;
              return `${context.dataset.label}: ${(value * 100).toFixed(1)}%`;
            }
          },
          backgroundColor: isDarkMode ? 'rgba(25, 25, 25, 0.8)' : 'rgba(255, 255, 255, 0.8)',
          titleColor: isDarkMode ? '#fff' : '#333',
          bodyColor: isDarkMode ? '#ddd' : '#555',
          borderColor: isDarkMode ? 'rgba(200, 200, 200, 0.2)' : 'rgba(0, 0, 0, 0.1)',
          borderWidth: 1
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          // Specify the correct scale type for Chart.js
          type: 'linear' as const,
          ticks: {
            // Use proper callback signature for Chart.js
            callback: function(tickValue: string | number) {
              return `${(Number(tickValue) * 100).toFixed(0)}%`;
            },
            color: isDarkMode ? '#aaa' : '#666'
          },
          grid: {
            color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'
          }
        },
        x: {
          ticks: {
            color: isDarkMode ? '#aaa' : '#666'
          },
          grid: {
            color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)'
          }
        }
      }
    };
  }, [isDarkMode]);
  
  // Memoized doughnut chart options
  const doughnutOptions = useMemo(() => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom' as const,
          labels: {
            padding: 15,
            color: isDarkMode ? '#ddd' : '#333'
          }
        },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const total = context.dataset.data.reduce((acc: number, data: number) => acc + data, 0);
              const value = context.raw;
              const percentage = ((value / total) * 100).toFixed(1);
              return `${context.label}: ${value} (${percentage}%)`;
            }
          },
          backgroundColor: isDarkMode ? 'rgba(25, 25, 25, 0.8)' : 'rgba(255, 255, 255, 0.8)',
          titleColor: isDarkMode ? '#fff' : '#333',
          bodyColor: isDarkMode ? '#ddd' : '#555',
          borderColor: isDarkMode ? 'rgba(200, 200, 200, 0.2)' : 'rgba(0, 0, 0, 0.1)',
          borderWidth: 1
        }
      }
    };
  }, [isDarkMode]);
  
  // Memoized timeframe change handler
  const handleTimeframeChange = useCallback((newTimeframe: TimeFrame) => {
    setTimeframe(newTimeframe);
  }, []);
  
  // Helper function to format percentage
  const formatPercentage = useCallback((value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  }, []);
  
  // Helper function to get color based on metric value
  const getMetricColor = useCallback((value: number) => {
    if (value >= 0.75) return '#4caf50';
    if (value >= 0.5) return '#ff9800';
    return '#f44336';
  }, []);
  
  // Helper function to create success rate calculation
  const getSuccessRate = useCallback((metrics: SignalQualityMetrics) => {
    return metrics.total_successful_signals / metrics.total_signals;
  }, []);

  // Generate metrics for display
  const metrics = useMemo(() => {
    if (!metricsData) return [];
    
    return [
      { 
        key: 'accuracy', 
        label: 'Accuracy', 
        value: metricsData.overall_accuracy,
        tooltip: metricTooltips.accuracy
      },
      { 
        key: 'precision', 
        label: 'Precision', 
        value: metricsData.overall_precision,
        tooltip: metricTooltips.precision
      },
      { 
        key: 'recall', 
        label: 'Recall', 
        value: metricsData.overall_recall,
        tooltip: metricTooltips.recall
      },
      { 
        key: 'f1_score', 
        label: 'F1 Score', 
        value: metricsData.overall_f1_score,
        tooltip: metricTooltips.f1_score
      },
      { 
        key: 'success_rate', 
        label: 'Success Rate', 
        value: getSuccessRate(metricsData),
        tooltip: metricTooltips.success_rate
      },
      { 
        key: 'confidence', 
        label: 'Avg. Confidence', 
        value: selectedSymbol && metricsData.performance_by_symbol[selectedSymbol]
          ? metricsData.performance_by_symbol[selectedSymbol].avg_confidence
          : Object.values(metricsData.performance_by_symbol).reduce((acc, curr) => acc + curr.avg_confidence, 0) / 
            Object.keys(metricsData.performance_by_symbol).length,
        tooltip: metricTooltips.average_confidence
      }
    ];
  }, [metricsData, selectedSymbol, getSuccessRate]);

  return (
    <Paper 
      elevation={1}
      sx={{ 
        p: { xs: 2, md: 3 }, 
        borderRadius: 2,
        bgcolor: isDarkMode ? 'rgba(30, 30, 30, 0.7)' : '#fff',
        boxShadow: isDarkMode 
          ? '0 4px 20px rgba(0,0,0,0.2)' 
          : '0 2px 10px rgba(0,0,0,0.05)'
      }}
    >
      {isLoading ? (
        // Loading state
        <Box sx={{ p: 4 }}>
          <Skeleton variant="rectangular" height={40} width="60%" sx={{ mb: 3 }} />
          <Skeleton variant="rectangular" height={30} width="40%" sx={{ mb: 2 }} />
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Skeleton variant="rectangular" height={300} />
            </Grid>
            <Grid item xs={12} md={4}>
              <Skeleton variant="rectangular" height={300} />
            </Grid>
            <Grid item xs={12}>
              <Skeleton variant="rectangular" height={250} />
            </Grid>
          </Grid>
        </Box>
      ) : error ? (
        // Error state
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h5" color="error" gutterBottom>
            Error Loading Metrics
          </Typography>
          <Typography variant="body1">
            {error}
          </Typography>
          <Button 
            variant="contained" 
            sx={{ mt: 2 }} 
            onClick={() => fetchMetricsData()}
          >
            Retry
          </Button>
        </Box>
      ) : metricsData && (
        // Content when data is loaded
        <>
          <Box sx={{ display: 'flex', flexDirection: 'column', mb: 3 }}>
            {/* Header with title and timeframe selection */}
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: { xs: 'flex-start', sm: 'center' },
              flexDirection: { xs: 'column', sm: 'row' },
              mb: 3,
              gap: 2
            }}>
              <Typography variant={isMobile ? "h6" : "h5"} fontWeight={500}>
                {panelTitle}
              </Typography>
              
              <ButtonGroup size={isMobile ? "small" : "medium"} variant="outlined">
                <Button 
                  onClick={() => handleTimeframeChange('7d')}
                  variant={timeframe === '7d' ? 'contained' : 'outlined'}
                >
                  7D
                </Button>
                <Button 
                  onClick={() => handleTimeframeChange('30d')}
                  variant={timeframe === '30d' ? 'contained' : 'outlined'}
                >
                  30D
                </Button>
                <Button 
                  onClick={() => handleTimeframeChange('90d')}
                  variant={timeframe === '90d' ? 'contained' : 'outlined'}
                >
                  90D
                </Button>
              </ButtonGroup>
            </Box>
            
            {/* Summary statistics */}
            <Box sx={{ mb: 3 }}>
              <Grid container spacing={3}>
                {/* Total signals statistic */}
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ 
                    p: 2,
                    borderRadius: 2,
                    bgcolor: isDarkMode ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)',
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                      {selectedSymbol ? `${selectedSymbol} Signals` : 'Total Signals'}
                    </Typography>
                    <Typography variant="h4" sx={{ fontWeight: 500, color: 'primary.main' }}>
                      {selectedSymbol && metricsData.performance_by_symbol[selectedSymbol] 
                        ? metricsData.performance_by_symbol[selectedSymbol].signal_count 
                        : metricsData.total_signals}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      in the past {timeframe}
                    </Typography>
                  </Box>
                </Grid>
                
                {/* Key metrics */}
                {metrics.slice(0, 3).map(metric => (
                  <Grid item xs={12} sm={6} md={3} key={metric.key}>
                    <Box sx={{ 
                      p: 2,
                      borderRadius: 2,
                      bgcolor: isDarkMode ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)',
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body2" color="textSecondary">
                          {metric.label}
                        </Typography>
                        <Tooltip title={metric.tooltip} arrow>
                          <InfoOutlinedIcon sx={{ ml: 0.5, fontSize: 16, opacity: 0.6 }} />
                        </Tooltip>
                      </Box>
                      <Typography variant="h4" sx={{ 
                        fontWeight: 500, 
                        color: getMetricColor(metric.value)
                      }}>
                        {formatPercentage(metric.value)}
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Box>
            
            {/* Metrics progress bars for all metrics */}
            <Box sx={{ mb: 4 }}>
              <Typography variant="subtitle2" sx={{ mb: 2, opacity: 0.8 }}>
                Signal Quality Metrics
              </Typography>
              
              <Grid container spacing={2}>
                {metrics.map(metric => (
                  <Grid item xs={12} key={metric.key}>
                    <Box>
                      <Box sx={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        mb: 0.5,
                        alignItems: 'center'
                      }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Typography variant="body2" sx={{ mr: 0.5 }}>
                            {metric.label}
                          </Typography>
                          <Tooltip title={metric.tooltip} arrow>
                            <InfoOutlinedIcon sx={{ fontSize: 16, color: 'primary.main', opacity: 0.7 }} />
                          </Tooltip>
                        </Box>
                        <Typography variant="body2" fontWeight={500} sx={{ 
                          color: getMetricColor(metric.value)
                        }}>
                          {formatPercentage(metric.value)}
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={metric.value * 100}
                        sx={{ 
                          height: 8, 
                          borderRadius: 1,
                          bgcolor: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                          '& .MuiLinearProgress-bar': {
                            bgcolor: getMetricColor(metric.value)
                          }
                        }} 
                      />
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Box>
          </Box>
          
          {/* Charts section */}
          <Grid container spacing={3}>
            {/* Radar Chart */}
            <Grid item xs={12} md={6}>
              <Box sx={{ height: 320 }}>
                <Typography variant="subtitle2" gutterBottom sx={{ opacity: 0.8 }}>
                  Metrics Overview
                </Typography>
                {radarChartData && (
                  <Radar 
                    data={radarChartData} 
                    options={radarOptions} 
                    height={isMobile ? 280 : 300}
                  />
                )}
              </Box>
            </Grid>
            
            {/* Distribution Chart */}
            <Grid item xs={12} md={6}>
              <Box sx={{ height: 320 }}>
                <Typography variant="subtitle2" gutterBottom sx={{ opacity: 0.8 }}>
                  Signal Distribution
                </Typography>
                {signalDistributionData && (
                  <Box sx={{ height: 300, position: 'relative' }}>
                    <Doughnut
                      data={signalDistributionData}
                      options={doughnutOptions}
                    />
                    <Box sx={{ 
                      position: 'absolute', 
                      top: '50%', 
                      left: '50%', 
                      transform: 'translate(-50%, -65%)',
                      textAlign: 'center'
                    }}>
                      <Typography variant="body2" sx={{ color: isDarkMode ? '#aaa' : '#666' }}>
                        Success Rate
                      </Typography>
                      <Typography variant="h5" fontWeight={600} sx={{ 
                        color: getMetricColor(getSuccessRate(metricsData))
                      }}>
                        {formatPercentage(getSuccessRate(metricsData))}
                      </Typography>
                    </Box>
                  </Box>
                )}
              </Box>
            </Grid>
            
            {/* Symbol Performance Chart */}
            <Grid item xs={12}>
              <Box sx={{ height: isMobile ? 280 : 350 }}>
                <Typography variant="subtitle2" gutterBottom sx={{ opacity: 0.8 }}>
                  {selectedSymbol ? 'Symbol Performance' : 'Performance by Symbol'}
                </Typography>
                {symbolPerformanceData && (
                  <Bar
                    data={symbolPerformanceData}
                    options={barOptions}
                    height={isMobile ? 260 : 330}
                  />
                )}
              </Box>
            </Grid>
          </Grid>
        </>
      )}
    </Paper>
  );
};

// Wrap with React.memo to prevent unnecessary re-renders
export default React.memo(SignalQualityMetricsPanel);
