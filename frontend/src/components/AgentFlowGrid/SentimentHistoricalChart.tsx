import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Box, Paper, Typography, Skeleton, ButtonGroup, Button, useTheme, useMediaQuery } from '@mui/material';
import { Line } from 'react-chartjs-2';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  Title, 
  Tooltip, 
  Legend, 
  Filler,
  ChartData,
  ChartOptions
} from 'chart.js';
import { format, parseISO } from 'date-fns';
import sentimentAnalyticsService, { 
  SentimentHistoricalData, 
  TimeFrame, 
  SentimentDataPoint 
} from '../../api/sentimentAnalyticsService';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface Props {
  agentId: string;
  symbol: string | null;
  height?: number;
}

const SentimentHistoricalChart: React.FC<Props> = ({ 
  agentId, 
  symbol, 
  height = 400
}) => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const [timeframe, setTimeframe] = useState<TimeFrame>('24h');
  const [chartData, setChartData] = useState<SentimentHistoricalData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch historical sentiment data
  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
      setChartData(data);
    } catch (err) {
      setError('Failed to load sentiment data. Please try again later.');
      console.error('Error fetching sentiment data:', err);
    } finally {
      setIsLoading(false);
    }
  }, [agentId, symbol, timeframe]);

  // Load data on initial render and when dependencies change
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Memoized timeframe change handler
  const handleTimeframeChange = useCallback((newTimeframe: TimeFrame) => {
    setTimeframe(newTimeframe);
  }, []);

  // Memoized chart data formatting
  const formattedChartData = useMemo((): ChartData<'line'> => {
    if (!chartData || !chartData.dataPoints.length) return {
      labels: [],
      datasets: []
    };

    return {
      labels: chartData.dataPoints.map((item: SentimentDataPoint) => {
        const date = parseISO(item.timestamp);
        return timeframe === '24h' 
          ? format(date, 'HH:mm') 
          : timeframe === '7d' 
            ? format(date, 'EEE HH:mm')
            : format(date, 'MMM dd');
      }),
      datasets: [
        {
          label: 'Sentiment',
          data: chartData.dataPoints.map(item => item.sentiment_score),
          borderColor: '#4caf50',
          backgroundColor: 'rgba(76, 175, 80, 0.1)',
          borderWidth: 2,
          pointBackgroundColor: '#4caf50',
          pointBorderColor: '#fff',
          pointRadius: isMobile ? 1 : 3,
          pointHoverRadius: isMobile ? 3 : 5,
          tension: 0.3,
          fill: true,
          yAxisID: 'y'
        },
        {
          label: 'Confidence',
          data: chartData.dataPoints.map(item => item.confidence),
          borderColor: '#2196f3',
          backgroundColor: 'rgba(33, 150, 243, 0.1)',
          borderWidth: 2,
          borderDash: [5, 5],
          pointBackgroundColor: '#2196f3',
          pointBorderColor: '#fff',
          pointRadius: isMobile ? 1 : 2,
          pointHoverRadius: isMobile ? 2 : 4,
          tension: 0.2,
          fill: false,
          yAxisID: 'y1'
        }
      ]
    };
  }, [chartData, timeframe, isMobile]);

  // Memoized chart options
  const chartOptions = useMemo((): ChartOptions<'line'> => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            color: isDarkMode ? '#eee' : '#333',
          }
        },
        tooltip: {
          backgroundColor: isDarkMode ? 'rgba(25, 25, 25, 0.8)' : 'rgba(255, 255, 255, 0.8)',
          titleColor: isDarkMode ? '#fff' : '#333',
          bodyColor: isDarkMode ? '#ddd' : '#555',
          borderColor: isDarkMode ? 'rgba(200, 200, 200, 0.2)' : 'rgba(0, 0, 0, 0.1)',
          borderWidth: 1,
          padding: 10,
          displayColors: true,
          usePointStyle: true,
        },
      },
      scales: {
        x: {
          display: true,
          grid: {
            color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
          },
          ticks: {
            color: isDarkMode ? '#aaa' : '#666',
            maxRotation: 45,
            minRotation: 0,
            autoSkipPadding: 10,
            maxTicksLimit: isMobile ? 5 : 10,
          }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          min: -1,
          max: 1,
          grid: {
            color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)',
          },
          ticks: {
            color: isDarkMode ? '#aaa' : '#666',
          },
          title: {
            display: true,
            text: 'Sentiment Score',
            color: isDarkMode ? '#aaa' : '#666',
          }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          min: 0,
          max: 1,
          grid: {
            drawOnChartArea: false,
          },
          ticks: {
            color: isDarkMode ? '#aaa' : '#666',
          },
          title: {
            display: true,
            text: 'Confidence',
            color: isDarkMode ? '#aaa' : '#666',
          }
        },
      },
    };
  }, [isDarkMode, isMobile]);

  return (
    <Paper 
      elevation={1}
      sx={{ 
        p: 2, 
        mb: 3, 
        borderRadius: 2,
        bgcolor: isDarkMode ? 'rgba(0, 0, 0, 0.2)' : '#fff',
        boxShadow: isDarkMode ? '0 4px 12px rgba(0,0,0,0.15)' : '0 2px 8px rgba(0,0,0,0.05)'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant={isMobile ? 'subtitle1' : 'h6'} sx={{ fontWeight: 'medium' }}>
          {symbol 
            ? `${symbol} Sentiment Trend` 
            : 'Overall Sentiment Trend'
          }
        </Typography>
        
        <ButtonGroup size={isMobile ? 'small' : 'medium'} variant="outlined">
          <Button 
            onClick={() => handleTimeframeChange('24h')}
            variant={timeframe === '24h' ? 'contained' : 'outlined'}
          >
            24H
          </Button>
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
      
      {isLoading ? (
        <Box sx={{ height: height, width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Skeleton variant="rectangular" width="100%" height={height} animation="wave" />
        </Box>
      ) : error ? (
        <Box sx={{ height: height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="body1" color="error">
            {error}
          </Typography>
        </Box>
      ) : (
        <Box sx={{ height: height }}>
          <Line data={formattedChartData} options={chartOptions} height={height} />
        </Box>
      )}
      
      {chartData && !isLoading && !error && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2, flexWrap: 'wrap' }}>
          <Typography variant="caption" sx={{ color: isDarkMode ? '#aaa' : '#666' }}>
            {chartData.dataPoints.length} data points
          </Typography>
          <Typography variant="caption" sx={{ color: isDarkMode ? '#aaa' : '#666' }}>
            Avg. Sentiment: {chartData.metadata.average_sentiment.toFixed(2)} | Avg. Confidence: {chartData.metadata.average_confidence.toFixed(2)}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

// Wrap the component with React.memo to avoid unnecessary re-renders
export default React.memo(SentimentHistoricalChart);
