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

// Register ChartJS components if not already registered
if (!ChartJS.registry.getController('line')) {
  ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
  );
}

export interface AssetSentimentData {
  symbol: string;
  name?: string;
  color?: string;
  data: Array<{
    timestamp: string;
    sentiment: number;
    confidence?: number;
  }>;
}

interface AssetComparisonChartProps {
  assets: AssetSentimentData[];
  title?: string;
  height?: number;
  width?: string;
  showConfidence?: boolean;
  normalizeData?: boolean;
  timeRange?: string;
}

// Generate a color based on index to ensure consistent colors
const getDefaultColor = (index: number): string => {
  const colors = [
    'rgb(75, 192, 192)',    // Teal
    'rgb(255, 99, 132)',    // Pink
    'rgb(54, 162, 235)',    // Blue
    'rgb(255, 159, 64)',    // Orange
    'rgb(153, 102, 255)',   // Purple
    'rgb(255, 205, 86)',    // Yellow
    'rgb(201, 203, 207)'    // Grey
  ];
  return colors[index % colors.length];
};

/**
 * Component that visualizes sentiment comparison across multiple assets
 */
const AssetComparisonChart: React.FC<AssetComparisonChartProps> = ({
  assets,
  title = 'Asset Sentiment Comparison',
  height = 300,
  width = '100%',
  showConfidence = false,
  normalizeData = false,
  timeRange = '1w'
}) => {
  // Process and prepare chart data
  const chartData = useMemo(() => {
    // Find the common date range across all assets
    const allDates = new Set<string>();
    assets.forEach(asset => {
      asset.data.forEach(point => {
        // Convert to date string for comparison
        const dateStr = new Date(point.timestamp).toISOString().split('T')[0];
        allDates.add(dateStr);
      });
    });

    // Convert to array and sort chronologically
    const sortedDates = Array.from(allDates).sort();
    
    // Format for display
    const labels = sortedDates.map(dateStr => {
      const date = new Date(dateStr);
      return format(date, 'MMM d, yyyy');
    });
    
    // Create datasets for each asset
    const datasets = assets.map((asset, index) => {
      // Map data to the common date range
      const assetData = sortedDates.map(dateStr => {
        // Find data point for this date
        const point = asset.data.find(p => {
          const pointDate = new Date(p.timestamp).toISOString().split('T')[0];
          return pointDate === dateStr;
        });
        
        // Return sentiment value or null if no data for this date
        return point ? point.sentiment : null;
      });
      
      // If normalize is true, scale data to 0-1 range
      let processedData = assetData;
      if (normalizeData && assetData.some(d => d !== null)) {
        const validData = assetData.filter(d => d !== null) as number[];
        const min = Math.min(...validData);
        const max = Math.max(...validData);
        const range = max - min;
        
        if (range > 0) {
          processedData = assetData.map(d => 
            d !== null ? (d - min) / range : null
          );
        }
      }
      
      // Use provided color or generate one
      const color = asset.color || getDefaultColor(index);
      
      return {
        label: `${asset.symbol} ${asset.name ? `(${asset.name})` : ''}`,
        data: processedData,
        borderColor: color,
        backgroundColor: `${color.replace('rgb', 'rgba').replace(')', ', 0.5)')}`,
        tension: 0.3,
        fill: false,
        pointRadius: 3,
        pointHoverRadius: 5
      };
    });
    
    return { labels, datasets };
  }, [assets, normalizeData]);

  // Chart options
  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: title
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const datasetIndex = context.datasetIndex;
            const dataIndex = context.dataIndex;
            const asset = assets[datasetIndex];
            const value = context.raw as number;
            
            if (value === null) return `${asset.symbol}: No data`;
            
            // Get confidence if available
            const dateStr = new Date(chartData.labels[dataIndex] as string).toISOString().split('T')[0];
            const dataPoint = asset.data.find(p => {
              const pointDate = new Date(p.timestamp).toISOString().split('T')[0];
              return pointDate === dateStr;
            });
            
            const confidence = dataPoint?.confidence !== undefined 
              ? `Confidence: ${(dataPoint.confidence * 100).toFixed(0)}%` 
              : '';
            
            const sentimentText = normalizeData 
              ? `Relative Sentiment: ${value.toFixed(2)}`
              : `Sentiment: ${value.toFixed(2)}`;
            
            return confidence ? [sentimentText, confidence] : sentimentText;
          }
        }
      }
    },
    scales: {
      y: {
        title: {
          display: true,
          text: normalizeData ? 'Relative Sentiment' : 'Sentiment Score'
        },
        ticks: {
          callback: function(value) {
            if (!normalizeData) {
              if (value === 1) return 'Very Positive';
              if (value === 0.5) return 'Positive';
              if (value === 0) return 'Neutral';
              if (value === -0.5) return 'Negative';
              if (value === -1) return 'Very Negative';
            }
            return value;
          }
        },
        grid: {
          color: (context) => {
            if (context.tick.value === 0) {
              return 'rgba(0, 0, 0, 0.3)';
            }
            return 'rgba(0, 0, 0, 0.1)';
          }
        }
      },
      x: {
        ticks: {
          maxRotation: 45,
          minRotation: 45
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

export default AssetComparisonChart;
