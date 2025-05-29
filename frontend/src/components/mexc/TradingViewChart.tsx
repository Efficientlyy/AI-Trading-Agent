import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, useTheme } from '@mui/material';

// Extend the Window interface to include TradingView
declare global {
  interface Window {
    TradingView?: any;
  }
}

interface TradingViewChartProps {
  symbol: string;
  timeframe: string;
  data?: any[];
}

const TradingViewChart: React.FC<TradingViewChartProps> = ({ symbol, timeframe, data }) => {
  const theme = useTheme();
  const textColor = theme.palette.mode === 'dark' ? '#f8fafc' : theme.palette.text.primary;
  const secondaryTextColor = theme.palette.mode === 'dark' ? '#94a3b8' : theme.palette.text.secondary;
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);

  useEffect(() => {
    // Clean up any existing chart
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
      chartInstanceRef.current = null;
    }
    
    // Check if TradingView library is available
    if (window.TradingView && chartContainerRef.current) {
      const formattedSymbol = symbol.replace('/', '');
      
      // Create a lightweight chart if data is available
      try {
        chartInstanceRef.current = new window.TradingView.widget({
          container_id: chartContainerRef.current.id,
          symbol: `MEXC:${formattedSymbol}`,
          interval: timeframe,
          timezone: "Etc/UTC",
          theme: "dark",
          style: "1",
          locale: "en",
          toolbar_bg: "#f1f3f6",
          enable_publishing: false,
          hide_side_toolbar: false,
          allow_symbol_change: true,
          save_image: false,
          studies: ["RSI@tv-basicstudies", "MAExp@tv-basicstudies", "MACD@tv-basicstudies"],
          show_popup_button: true,
          popup_width: "1000",
          popup_height: "650",
          autosize: true,
        });
      } catch (error) {
        console.error('Error creating TradingView chart:', error);
      }
    } else {
      // If TradingView is not available, we'll use a placeholder
      console.warn('TradingView library not available, using placeholder chart');
    }
    
    // Clean up on unmount
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.remove();
      }
    };
  }, [symbol, timeframe]);

  // If there's no data, show a fallback chart with mock data
  if (!data || data.length === 0) {
    return (
      <Box sx={{ height: '100%', width: '100%', position: 'relative' }}>
        <Box
          sx={{
            height: '100%',
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            bgcolor: 'background.paper',
          }}
          className="chart-placeholder"
        >
          <Typography variant="h5" className="chart-data-message" sx={{ mb: 2, color: textColor }}>
            Chart data not available
          </Typography>
          <Typography variant="body2" className="chart-data-message" sx={{ color: secondaryTextColor }}>
            Using TradingView public chart for {symbol}
          </Typography>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <div id="tradingview_chart_container" ref={chartContainerRef} style={{ width: '100%', height: '100%' }}></div>
    </Box>
  );
};

export default TradingViewChart;