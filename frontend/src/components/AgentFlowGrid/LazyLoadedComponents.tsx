import React, { Suspense, lazy } from 'react';
import { Box, CircularProgress, useTheme, useMediaQuery } from '@mui/material';

// Lazy loaded components
const SentimentHistoricalChart = lazy(() => import('./SentimentHistoricalChart'));
const SignalQualityMetricsPanel = lazy(() => import('./SignalQualityMetricsPanel'));
const SymbolFilterControl = lazy(() => import('./SymbolFilterControl'));
const PipelineFlowAnimation = lazy(() => import('./PipelineFlowAnimation'));

// Loading fallback component
export const LoadingFallback: React.FC = () => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  
  return (
    <Box 
      sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '200px',
        bgcolor: isDarkMode ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
        borderRadius: 1
      }}
    >
      <CircularProgress size={40} />
    </Box>
  );
};

// Lazy loaded Pipeline Tab component
export const LazyPipelineTab: React.FC<{
  wsData: any;
  wsConnected: boolean;
  pipeline_status: string;
  handleComponentClick: (componentName: string) => void;
  lastUpdate: string;
  pipelineLatency: number;
}> = ({ wsData, wsConnected, pipeline_status, handleComponentClick, lastUpdate, pipelineLatency }) => {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <PipelineFlowAnimation 
        components={wsData?.components?.map((comp: any) => ({
          name: comp.name,
          metrics: comp.metrics,
          status: comp.status as 'online' | 'offline' | 'error' | 'processing',
          isActive: comp.is_active
        })) || []} 
        flowActive={pipeline_status === 'running'}
        onComponentClick={handleComponentClick}
      />
    </Suspense>
  );
};

// Lazy loaded Advanced Analytics Tab content
export const LazyAdvancedAnalyticsTab: React.FC<{
  agentId: string;
  selectedSymbol: string | null;
  setSelectedSymbol: (symbol: string | null) => void;
}> = ({ agentId, selectedSymbol, setSelectedSymbol }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  return (
    <Box sx={{ p: isMobile ? 1 : 2 }}>
      <Suspense fallback={<LoadingFallback />}>
        <SymbolFilterControl 
          agentId={agentId}
          selectedSymbol={selectedSymbol}
          onSymbolChange={setSelectedSymbol}
        />
      </Suspense>
      
      <Box sx={{ mb: 3 }}>
        <Suspense fallback={<LoadingFallback />}>
          <SentimentHistoricalChart 
            agentId={agentId}
            symbol={selectedSymbol}
            height={isMobile ? 250 : 350}
          />
        </Suspense>
      </Box>
      
      <Suspense fallback={<LoadingFallback />}>
        <SignalQualityMetricsPanel 
          agentId={agentId}
          selectedSymbol={selectedSymbol}
        />
      </Suspense>
    </Box>
  );
};
