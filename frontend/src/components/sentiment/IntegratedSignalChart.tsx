import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export interface IntegratedSignalData {
  symbol: string;
  technicalSignal: number;
  sentimentSignal: number;
  combinedSignal: number;
  technicalConfidence?: number;
  sentimentConfidence?: number;
  combinedConfidence?: number;
  signalType?: string;
}

interface IntegratedSignalChartProps {
  data: IntegratedSignalData[];
  title?: string;
  height?: number;
  width?: string;
  showConfidence?: boolean;
}

/**
 * Component that visualizes the comparison of technical and sentiment signals
 * and how they combine to form the integrated signal
 */
const IntegratedSignalChart: React.FC<IntegratedSignalChartProps> = ({
  data,
  title = 'Integrated Trading Signals',
  height = 300,
  width = '100%',
  showConfidence = true
}) => {
  // Prepare chart data
  const labels = data.map(item => item.symbol);
  
  const chartData = {
    labels,
    datasets: [
      {
        label: 'Technical Signal',
        data: data.map(item => item.technicalSignal),
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      },
      {
        label: 'Sentiment Signal',
        data: data.map(item => item.sentimentSignal),
        backgroundColor: 'rgba(255, 206, 86, 0.8)',
        borderColor: 'rgba(255, 206, 86, 1)',
        borderWidth: 1
      },
      {
        label: 'Combined Signal',
        data: data.map(item => item.combinedSignal),
        backgroundColor: 'rgba(75, 192, 192, 0.8)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      }
    ]
  };

  // Chart options
  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y' as const,
    plugins: {
      legend: {
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
          afterLabel: (context) => {
            const dataIndex = context.dataIndex;
            const datasetIndex = context.datasetIndex;
            
            if (!showConfidence) return '';
            
            const item = data[dataIndex];
            
            if (datasetIndex === 0 && item.technicalConfidence) {
              return `Confidence: ${(item.technicalConfidence * 100).toFixed(0)}%`;
            }
            
            if (datasetIndex === 1 && item.sentimentConfidence) {
              return `Confidence: ${(item.sentimentConfidence * 100).toFixed(0)}%`;
            }
            
            if (datasetIndex === 2 && item.combinedConfidence) {
              return [
                `Confidence: ${(item.combinedConfidence * 100).toFixed(0)}%`,
                `Signal Type: ${item.signalType || 'Neutral'}`
              ];
            }
            
            return '';
          }
        }
      }
    },
    scales: {
      x: {
        min: -1,
        max: 1,
        ticks: {
          callback: function(value) {
            if (value === 1) return 'Strong Buy';
            if (value === 0.5) return 'Buy';
            if (value === 0) return 'Neutral';
            if (value === -0.5) return 'Sell';
            if (value === -1) return 'Strong Sell';
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
      }
    }
  };
  
  return (
    <div style={{ height, width }}>
      <Bar data={chartData} options={options} />
    </div>
  );
};

export default IntegratedSignalChart;
