import React, { useState, lazy, Suspense } from 'react';
import { Box, CircularProgress, Tab, Tabs, Typography, useTheme, useMediaQuery } from '@mui/material';
import EqualizerIcon from '@mui/icons-material/Equalizer';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import SymbolFilterControl from './SymbolFilterControl';

// Lazy load components for performance optimization
const SentimentHistoricalChart = lazy(() => import('./SentimentHistoricalChart'));
const SignalQualityMetricsPanel = lazy(() => import('./SignalQualityMetricsPanel'));

interface AdvancedAnalyticsTabProps {
  agentId: string;
}

// TabPanel component for tab content
const TabPanel: React.FC<{
  children?: React.ReactNode;
  index: number;
  value: number;
}> = ({ children, value, index }) => {
  return (
    <div role="tabpanel" hidden={value !== index} style={{ width: '100%' }}>
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
};

/**
 * Advanced Analytics Tab Component
 * 
 * Displays advanced sentiment analysis features with optimized rendering using:
 * - React.memo for component memoization
 * - Lazy loading for heavy components
 * - State management to maintain selected symbol across tabs
 */
const AdvancedAnalyticsTab: React.FC<AdvancedAnalyticsTabProps> = ({ agentId }) => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const [tabIndex, setTabIndex] = useState(0);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  
  // Memoized tab change handler
  const handleTabChange = React.useCallback((event: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
  }, []);

  // Memoized symbol change handler
  const handleSymbolChange = React.useCallback((symbol: string | null) => {
    setSelectedSymbol(symbol);
  }, []);

  return (
    <Box sx={{ width: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Advanced Sentiment Analytics
      </Typography>
      
      {/* Symbol filter control component */}
      <SymbolFilterControl 
        agentId={agentId}
        selectedSymbol={selectedSymbol}
        onSymbolChange={handleSymbolChange}
      />
      
      {/* Tabs for different analysis views */}
      <Tabs 
        value={tabIndex} 
        onChange={handleTabChange}
        variant={isMobile ? "fullWidth" : "standard"}
        sx={{
          borderBottom: 1,
          borderColor: isDarkMode ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)',
          mb: 2
        }}
      >
        <Tab 
          icon={<ShowChartIcon />} 
          label={isMobile ? undefined : "Historical Trends"} 
          iconPosition={isMobile ? "top" : "start"}
        />
        <Tab 
          icon={<EqualizerIcon />} 
          label={isMobile ? undefined : "Signal Quality Metrics"} 
          iconPosition={isMobile ? "top" : "start"}
        />
      </Tabs>

      {/* Historical sentiment visualization */}
      <TabPanel value={tabIndex} index={0}>
        <Suspense fallback={
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        }>
          <SentimentHistoricalChart 
            agentId={agentId} 
            symbol={selectedSymbol} 
            height={400}
          />
        </Suspense>
      </TabPanel>

      {/* Signal quality metrics visualization */}
      <TabPanel value={tabIndex} index={1}>
        <Suspense fallback={
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        }>
          <SignalQualityMetricsPanel
            agentId={agentId}
            selectedSymbol={selectedSymbol}
          />
        </Suspense>
      </TabPanel>
    </Box>
  );
};

// Export a memoized version of the component to prevent unnecessary re-renders
export default React.memo(AdvancedAnalyticsTab);
