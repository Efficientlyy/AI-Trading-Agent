import React, { useMemo } from 'react';
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
  ChartOptions
} from 'chart.js';
import { format } from 'date-fns';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export interface SentimentDataPoint {
  timestamp: string;
  sentiment: number;
  confidence?: number;
  source?: string;
  symbol?: string;
}

interface SentimentTrendChartProps {
  data: SentimentDataPoint[];
  symbol?: string;
  title?: string;
  height?: number;
  width?: string;
  showConfidence?: boolean;
  timeRange?: '1d' | '1w' | '1m' | '3m';
}

/**
 * Component that visualizes sentiment trend over time
 */
const SentimentTrendChart: React.FC<SentimentTrendChartProps> = ({
  data,
  symbol,
  title = 'Sentiment Trend',
  height = 300,
  width = '100%',
  showConfidence = true,
  timeRange = '1w'
}) => {
  // Process and prepare chart data
  const chartData = useMemo(() => {
    // Sort data by timestamp
    const sortedData = [...data].sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    // Extract dates and sentiment values
    const labels = sortedData.map(item => 
      format(new Date(item.timestamp), 'MMM d, HH:mm')
    );
    
    const sentimentValues = sortedData.map(item => item.sentiment);
    
    // Generate confidence bands if available and requested
    const confidenceValues = showConfidence 
      ? sortedData.map(item => item.confidence || 0.5) 
      : [];
    
    const upperBand = showConfidence 
      ? sentimentValues.map((val, i) => 
          Math.min(1, val + (1 - (confidenceValues[i] || 0.5)))
        ) 
      : [];
    
    const lowerBand = showConfidence 
      ? sentimentValues.map((val, i) => 
          Math.max(-1, val - (1 - (confidenceValues[i] || 0.5)))
        ) 
      : [];
    
    return {
      labels,
      datasets: [
        {
          label: `${symbol} Sentiment`,
          data: sentimentValues,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.3,
          fill: false
        },
        ...(showConfidence ? [
          {
            label: 'Upper Confidence',
            data: upperBand,
            borderColor: 'rgba(75, 192, 192, 0.2)',
            backgroundColor: 'rgba(75, 192, 192, 0)',
            tension: 0.3,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
          },
          {
            label: 'Lower Confidence',
            data: lowerBand,
            borderColor: 'rgba(75, 192, 192, 0.2)',
            backgroundColor: 'rgba(75, 192, 192, 0)',
            tension: 0.3,
            borderDash: [5, 5],
            fill: '+1', // Fill between upper and lower bands
            pointRadius: 0
          }
        ] : [])
      ]
    };
  }, [data, symbol, showConfidence]);

  // Chart options
  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: 'rgba(255, 255, 255, 0.8)'
        }
      },
      title: {
        display: true,
        text: title
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const dataIndex = context.dataIndex;
            const datasetIndex = context.datasetIndex;
            
            if (datasetIndex === 0) {
              const sentimentValue = data[dataIndex]?.sentiment || 0;
              const confidenceValue = data[dataIndex]?.confidence || 0;
              const source = data[dataIndex]?.source || 'Unknown';
              
              return [
                `Sentiment: ${sentimentValue.toFixed(2)}`,
                `Confidence: ${(confidenceValue * 100).toFixed(0)}%`,
                `Source: ${source}`
              ];
            }
            
            return '';
          }
        }
      }
    },
    scales: {
      y: {
        min: -1,
        max: 1,
        ticks: {
          callback: function(value) {
            if (value === 1) return 'Very Positive';
            if (value === 0.5) return 'Positive';
            if (value === 0) return 'Neutral';
            if (value === -0.5) return 'Negative';
            if (value === -1) return 'Very Negative';
            return '';
          },
          color: 'rgba(255, 255, 255, 0.8)'
        },
        grid: {
          color: (context) => {
            if (context.tick.value === 0) {
              return 'rgba(255, 255, 255, 0.3)';
            }
            return 'rgba(255, 255, 255, 0.1)';
          }
        }
      },
      x: {
        ticks: {
          maxRotation: 45,
          minRotation: 45,
          color: 'rgba(255, 255, 255, 0.8)'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      }
    }
  };
  
  return (
    <div style={{ height, width }}>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default SentimentTrendChart;
