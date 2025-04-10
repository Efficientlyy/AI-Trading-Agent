import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';
import { OHLCV } from '../../types';

interface SimpleChartProps {
  data: OHLCV[] | null;
  symbol: string;
}

const SimpleChart: React.FC<SimpleChartProps> = ({ data, symbol }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;
    
    // Store ref value to avoid React hooks exhaustive-deps warning
    const chartContainer = chartContainerRef.current;
    
    // Clear previous chart
    chartContainer.innerHTML = '';
    
    // Create chart
    const chart = createChart(chartContainer, {
      width: chartContainer.clientWidth,
      height: 400,
      layout: {
        backgroundColor: '#1E1E1E',
        textColor: '#D9D9D9',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });
    
    // Create candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26A69A',
      downColor: '#EF5350',
      borderVisible: false,
      wickUpColor: '#26A69A',
      wickDownColor: '#EF5350',
    });
    
    // Format data
    const formattedData = data.map(item => ({
      time: Math.floor(new Date(item.timestamp).getTime() / 1000),
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }));
    
    // Set data
    candleSeries.setData(formattedData);
    
    // Fit content
    chart.timeScale().fitContent();
    
    // Handle resize
    const handleResize = () => {
      chart.applyOptions({ width: chartContainer.clientWidth });
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      // Instead of chart.remove(), we'll just clear the container
      chartContainer.innerHTML = '';
    };
  }, [data]);
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        {symbol} Chart
      </h3>
      <div ref={chartContainerRef} className="w-full h-[400px]" />
    </div>
  );
};

export default SimpleChart;
