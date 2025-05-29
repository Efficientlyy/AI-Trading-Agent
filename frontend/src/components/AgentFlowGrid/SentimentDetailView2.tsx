import React, { useState, useEffect, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Box,
  Tabs,
  Tab,
  Typography,
  Paper,
  Grid,
  Chip,
  LinearProgress,
  CircularProgress,
  Button,
  Fade,
  Tooltip
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CloudQueueIcon from '@mui/icons-material/CloudQueue';
import RefreshIcon from '@mui/icons-material/Refresh';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import MemoryIcon from '@mui/icons-material/Memory';
import DataUsageIcon from '@mui/icons-material/DataUsage';
import StorageIcon from '@mui/icons-material/Storage';
import { Agent as AgentType } from '../../context/SystemControlContext';
import PipelineFlowAnimation from './PipelineFlowAnimation';
import PipelineComponentDetail from './PipelineComponentDetail';
import useSentimentPipelineSocket from '../../hooks/useSentimentPipelineSocket';

// Interface for sentiment signals
interface SentimentSignal {
  symbol: string;
  analysis_time: string;
  sentiment_score: number;
  signal_strength: number;
  confidence: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  suggested_action: 'buy' | 'sell' | 'hold';
}

// Interface for the component props
interface SentimentDetailViewProps {
  open: boolean;
  onClose: () => void;
  agent: AgentType;
}

// Service for sentiment pipeline data
const sentimentPipelineService = {
  getPipelineData: async (agentId: string) => {
    // Mock data until API is implemented
    return {
      pipeline_latency: 0.35,
      components: {
        alpha_vantage_client: {
          status: 'online',
          api_calls: 128,
          error_rate: 0.02,
          last_update: new Date().toISOString()
        },
        sentiment_processor: {
          status: 'online',
          processed_count: 1342,
          avg_processing_time: 45.8
        },
        signal_generator: {
          status: 'online',
          total_signals: 248,
          configuration: {
            threshold: 0.15,
            time_window: 30
          }
        },
        cache_manager: {
          status: 'online',
          cache_size: 512,
          hit_ratio: 0.85,
          ttl: 3600
        }
      },
      global_metrics: {
        total_signals: 248,
        success_rate: 0.78,
        avg_sentiment_score: 0.23
      }
    };
  },
  getLatestSignals: async (agentId: string, limit: number) => {
    // Mock data until API is implemented
    return {
      signals: [
        {
          symbol: 'BTC',
          analysis_time: new Date().toISOString(),
          sentiment_score: 0.42,
          signal_strength: 0.65,
          confidence: 0.78,
          trend: 'bullish' as 'bullish',
          suggested_action: 'buy' as 'buy'
        },
        {
          symbol: 'ETH',
          analysis_time: new Date().toISOString(),
          sentiment_score: 0.28,
          signal_strength: 0.55,
          confidence: 0.72,
          trend: 'bullish' as 'bullish',
          suggested_action: 'hold' as 'hold'
        },
        {
          symbol: 'XRP',
          analysis_time: new Date().toISOString(),
          sentiment_score: -0.18,
          signal_strength: 0.45,
          confidence: 0.62,
          trend: 'bearish' as 'bearish',
          suggested_action: 'sell' as 'sell'
        }
      ] as SentimentSignal[]
    };
  },
  getMonitoredSymbols: async (agentId: string) => {
    // Mock data until API is implemented
    return {
      symbols: ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT'] as string[]
    };
  }
};
// Component for detailed visualization of sentiment analysis process
const SentimentDetailView: React.FC<SentimentDetailViewProps> = ({ agent, open, onClose }) => {
  const [tabValue, setTabValue] = useState(0);
  
  // State for traditional API-based data fetching
  const [pipelineData, setPipelineData] = useState<any>({
    latency: 0,
    alphaVantageClient: {
      status: 'unknown',
      apiCalls: 0,
      errorRate: 0,
      lastUpdate: '',
    },
    sentimentProcessor: {
      status: 'unknown',
      processedCount: 0,
      avgProcessingTime: 0,
    },
    signalGenerator: {
      status: 'unknown',
      totalSignals: 0,
      configuration: {
        threshold: 0,
        timeWindow: 0,
      },
    },
    cacheManager: {
      status: 'unknown',
      cacheSize: 0,
      hitRatio: 0,
    },
    globalMetrics: {
      total_signals: 0,
      success_rate: 0,
      avg_sentiment_score: 0,
    }
  });
  const [latestSignals, setLatestSignals] = useState<SentimentSignal[]>([]);
  const [monitoredSymbols, setMonitoredSymbols] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Real-time websocket connection state
  const [liveDataEnabled, setLiveDataEnabled] = useState<boolean>(true);
  const [selectedComponent, setSelectedComponent] = useState<any | null>(null);
  const [showComponentDetail, setShowComponentDetail] = useState<boolean>(false);
  
  // Custom hook for real-time sentiment pipeline updates
  const {
    isConnected: wsConnected,
    pipelineData: wsData,
    error: wsError,
    loading: wsLoading,
    refreshData: refreshWsData
  } = useSentimentPipelineSocket(agent.agent_id);
  
  // Helper function to map pipeline data from API response
  const mapPipelineData = (data: any): any => {
    return {
      latency: data.pipeline_latency || 0,
      alphaVantageClient: {
        status: data.components?.alpha_vantage_client?.status || 'offline',
        apiCalls: data.components?.alpha_vantage_client?.api_calls || 0,
        errorRate: data.components?.alpha_vantage_client?.error_rate || 0,
        lastUpdate: data.components?.alpha_vantage_client?.last_update || 'N/A',
      },
      sentimentProcessor: {
        status: data.components?.sentiment_processor?.status || 'offline',
        processedCount: data.components?.sentiment_processor?.processed_count || 0,
        avgProcessingTime: data.components?.sentiment_processor?.avg_processing_time || 0,
      },
      signalGenerator: {
        status: data.components?.signal_generator?.status || 'offline',
        totalSignals: data.components?.signal_generator?.total_signals || 0,
        configuration: {
          threshold: data.components?.signal_generator?.configuration?.threshold || 0,
          timeWindow: data.components?.signal_generator?.configuration?.time_window || 0,
        },
      },
      cacheManager: {
        status: data.components?.cache_manager?.status || 'offline',
        cacheSize: data.components?.cache_manager?.cache_size || 0,
        hitRatio: data.components?.cache_manager?.hit_ratio || 0,
        ttl: data.components?.cache_manager?.ttl,
      },
      globalMetrics: {
        total_signals: data.global_metrics?.total_signals || 0,
        success_rate: data.global_metrics?.success_rate || 0,
        avg_sentiment_score: data.global_metrics?.avg_sentiment_score || 0,
      },
    };
  };

  // Handle clicking on a pipeline component
  const handleComponentClick = useCallback((componentName: string) => {
    if (!wsData) return;
    
    // Find the clicked component in the WebSocket data
    const component = wsData.components.find((comp: any) => 
      comp.name.toLowerCase().replace(/ /g, '_') === componentName.toLowerCase().replace(/ /g, '_')
    );
    
    if (component) {
      setSelectedComponent(component);
      setShowComponentDetail(true);
    }
  }, [wsData]);
  
  // Toggle live data updates
  const toggleLiveData = useCallback(() => {
    setLiveDataEnabled(prev => !prev);
  }, []);
  
  // Convert WebSocket data to component format for the flow animation
  const getPipelineComponents = useCallback(() => {
    if (!wsData) return [];
    
    return wsData.components.map((comp: any) => ({
      name: comp.name,
      metrics: comp.metrics,
      status: comp.status as 'online' | 'offline' | 'error' | 'processing',
      isActive: comp.is_active
    }));
  }, [wsData]);
  
  // Refresh data manually
  const handleRefreshData = useCallback(() => {
    if (liveDataEnabled) {
      refreshWsData();
    } else {
      fetchData();
    }
  }, [liveDataEnabled, refreshWsData]);

  // Fetch data on component mount
  useEffect(() => {
    if (open) {
      fetchData();
    }
  }, [open, agent.agent_id]);

  // Fetch all required data
  const fetchData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      await Promise.all([
        fetchPipelineData(),
        fetchLatestSignals(),
        fetchMonitoredSymbols()
      ]);
      setIsLoading(false);
    } catch (err) {
      console.error('Error fetching sentiment data:', err);
      setIsLoading(false);
      setError('Failed to fetch sentiment data');
    }
  };

  // Fetch pipeline data
  const fetchPipelineData = async () => {
    try {
      const data = await sentimentPipelineService.getPipelineData(agent.agent_id);
      setPipelineData(mapPipelineData(data));
    } catch (err) {
      console.error('Error fetching pipeline data:', err);
      // Fallback to metrics from agent object if API fails
      setPipelineData({
        ...pipelineData,
        globalMetrics: {
          total_signals: agent.metrics?.total_signals_generated || 0,
          success_rate: agent.metrics?.success_rate || 0,
          avg_sentiment_score: agent.metrics?.avg_sentiment_score || 0,
        }
      });
      throw err;
    }
  };

  // Fetch latest signals
  const fetchLatestSignals = async () => {
    try {
      const data = await sentimentPipelineService.getLatestSignals(agent.agent_id, 10);
      setLatestSignals(data.signals || []);
    } catch (err) {
      console.error('Error fetching latest signals:', err);
      throw err;
    }
  };

  // Fetch monitored symbols
  const fetchMonitoredSymbols = async () => {
    try {
      const data = await sentimentPipelineService.getMonitoredSymbols(agent.agent_id);
      setMonitoredSymbols(data.symbols || []);
    } catch (err) {
      console.error('Error fetching monitored symbols:', err);
      setMonitoredSymbols(agent.symbols || []);
      throw err;
    }
  };

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Helper function to map crypto symbols to topics
  const mapSymbolToTopic = (symbol: string): string => {
    const topicMap: {[key: string]: string} = {
      "BTC": "bitcoin",
      "ETH": "ethereum",
      "XRP": "ripple",
      "SOL": "solana",
      "ADA": "cardano",
      "DOT": "polkadot",
      "DOGE": "dogecoin",
      "AVAX": "avalanche",
      "MATIC": "polygon"
    };
    
    return topicMap[symbol] || "crypto";
  };
  return (
    <>
      {/* Component Detail Dialog */}
      <PipelineComponentDetail
        component={selectedComponent}
        open={showComponentDetail}
        onClose={() => setShowComponentDetail(false)}
      />
      
      <Dialog
        open={open}
        onClose={onClose}
        maxWidth="lg"
        fullWidth
        PaperProps={{
          sx: { bgcolor: '#121212', color: '#fff', minHeight: '70vh' }
        }}
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h6">Sentiment Analysis Pipeline</Typography>
              <Chip 
                label={liveDataEnabled ? 'Live Data' : 'Static Data'}
                color={liveDataEnabled ? (wsConnected ? 'success' : 'warning') : 'default'}
                size="small"
                icon={liveDataEnabled ? <AutorenewIcon /> : undefined}
                sx={{ 
                  animation: liveDataEnabled && wsConnected ? 'pulse 2s infinite' : 'none',
                  '@keyframes pulse': {
                    '0%': { boxShadow: '0 0 0 0 rgba(76, 175, 80, 0.4)' },
                    '70%': { boxShadow: '0 0 0 10px rgba(76, 175, 80, 0)' },
                    '100%': { boxShadow: '0 0 0 0 rgba(76, 175, 80, 0)' }
                  }
                }}
                onClick={toggleLiveData}
              />
            </Box>
            <Box>
              <Tooltip title="Refresh Data">
                <IconButton 
                  onClick={handleRefreshData}
                  sx={{ color: 'white', mr: 1 }}
                  disabled={isLoading || wsLoading}
                >
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              <IconButton onClick={onClose} sx={{ color: 'white' }}>
                <CloseIcon />
              </IconButton>
            </Box>
          </Box>
        </DialogTitle>
        
        <DialogContent dividers>
          <Box sx={{ mb: 3 }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              textColor="primary"
              indicatorColor="primary"
              sx={{
                '& .MuiTabs-indicator': {
                  backgroundColor: '#4caf50',
                },
                '& .MuiTab-root': {
                  color: '#aaa',
                  '&.Mui-selected': {
                    color: '#4caf50',
                  },
                },
              }}
            >
              <Tab label="Pipeline Visualization" />
              <Tab label="Signals & Metrics" />
              <Tab label="Data Sources" />
            </Tabs>
          </Box>
          
          {isLoading && !liveDataEnabled ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography color="error">{error}</Typography>
              <Typography variant="body2" sx={{ mt: 1, color: '#aaa' }}>
                Showing available cached metrics instead.
              </Typography>
            </Box>
          ) : (
            <>
              {/* Pipeline Visualization Tab */}
              {tabValue === 0 && (
                <Box sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>Sentiment Pipeline Components</Typography>
                  <Typography variant="body2" sx={{ mb: 2, color: '#aaa' }}>
                    Real-time status and metrics for each component in the sentiment analysis pipeline.
                  </Typography>
                  
                  {/* Show the WebSocket-based interactive visualization if live data is enabled */}
                  {liveDataEnabled ? (
                    wsLoading ? (
                      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                        <CircularProgress />
                      </Box>
                    ) : wsError ? (
                      <Paper sx={{ p: 2, bgcolor: 'rgba(255,0,0,0.1)', color: '#ff6b6b', borderRadius: 1 }}>
                        <Typography>{wsError}</Typography>
                        <Button 
                          variant="outlined" 
                          color="error" 
                          size="small" 
                          sx={{ mt: 1 }}
                          onClick={() => setLiveDataEnabled(false)}
                        >
                          Switch to Static Data
                        </Button>
                      </Paper>
                    ) : wsData ? (
                      <Box>
                        {/* Animated Pipeline Flow Visualization */}
                        <Paper sx={{ p: 2, mb: 2, bgcolor: '#1a2035', borderRadius: 2 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="subtitle1">
                              Pipeline Flow
                              <Typography component="span" variant="caption" sx={{ ml: 1, color: '#4caf50' }}>
                                {wsData.pipeline_status === 'running' ? 'Active' : 
                                 wsData.pipeline_status === 'error' ? 'Error' : 'Inactive'}
                              </Typography>
                            </Typography>
                            <Tooltip title="Pipeline components are clickable for detailed insights">
                              <IconButton size="small">
                                <InfoOutlinedIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          
                          {/* Interactive Flow Animation Component */}
                          <PipelineFlowAnimation 
                            components={getPipelineComponents()} 
                            flowActive={wsData.pipeline_status === 'running'}
                            onComponentClick={handleComponentClick}
                          />
                          
                          {/* Last Update Information */}
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                            <Typography variant="caption" sx={{ color: '#aaa' }}>
                              Last updated: {new Date(wsData.last_update).toLocaleString()}
                            </Typography>
                            <Typography variant="caption" sx={{ color: '#aaa' }}>
                              Pipeline latency: {wsData.pipeline_latency.toFixed(3)}s
                            </Typography>
                          </Box>
                        </Paper>
                        
                        {/* Global Metrics */}
                        <Paper sx={{ p: 2, bgcolor: '#1a2035', borderRadius: 2 }}>
                          <Typography variant="subtitle1" gutterBottom>Pipeline Performance</Typography>
                          <Grid container spacing={3}>
                            {Object.entries(wsData.global_metrics).map(([key, value]) => {
                              // Format keys from snake_case to Title Case
                              const formattedKey = key
                                .split('_')
                                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                                .join(' ');
                              
                              // Format value based on type
                              let displayValue = value;
                              if (typeof value === 'number') {
                                // Check if it's likely a percentage
                                if (key.includes('rate') || key.includes('ratio') || key.includes('accuracy')) {
                                  if (value <= 1) {
                                    displayValue = `${(value * 100).toFixed(1)}%`;
                                  } else {
                                    displayValue = `${value.toFixed(1)}%`;
                                  }
                                } 
                                // Check if it's a decimal that should be formatted
                                else if (value % 1 !== 0) {
                                  displayValue = value.toFixed(2);
                                }
                              }
                              
                              return (
                                <Grid item xs={6} sm={3} key={key}>
                                  <Typography variant="body2" sx={{ color: '#aaa' }}>{formattedKey}</Typography>
                                  <Typography variant="h6">{displayValue}</Typography>
                                </Grid>
                              );
                            })}
                          </Grid>
                        </Paper>
                      </Box>
                    ) : (
                      <Box sx={{ textAlign: 'center', p: 4 }}>
                        <Typography>No live data available</Typography>
                        <Button 
                          variant="outlined" 
                          onClick={refreshWsData} 
                          sx={{ mt: 2 }}
                          startIcon={<RefreshIcon />}
                        >
                          Refresh
                        </Button>
                      </Box>
                    )
                  ) : (
                    // Traditional static API-based visualization
                    <Grid container spacing={2}>
                      {/* Alpha Vantage Client */}
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: '#1a2035' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <CloudQueueIcon sx={{ mr: 1, color: '#64b5f6' }} />
                            <Typography variant="subtitle1">Alpha Vantage Client</Typography>
                            <Chip 
                              label={pipelineData.alphaVantageClient.status} 
                              size="small"
                              color={pipelineData.alphaVantageClient.status === 'online' ? 'success' : 'error'}
                              sx={{ ml: 1, height: 20 }}
                            />
                          </Box>
                          <Typography variant="body2" sx={{ color: '#aaa', mb: 1 }}>
                            Last update: {pipelineData.alphaVantageClient.lastUpdate || 'N/A'}
                          </Typography>
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="body2" sx={{ mb: 0.5 }}>
                              API calls: {pipelineData.alphaVantageClient.apiCalls} | Error rate: {(pipelineData.alphaVantageClient.errorRate * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        </Paper>
                      </Grid>
                      
                      {/* Sentiment Processor */}
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: '#1a2035' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <MemoryIcon sx={{ mr: 1, color: '#64b5f6' }} />
                            <Typography variant="subtitle1">Sentiment Processor</Typography>
                            <Chip 
                              label={pipelineData.sentimentProcessor.status} 
                              size="small"
                              color={pipelineData.sentimentProcessor.status === 'online' ? 'success' : 'error'}
                              sx={{ ml: 1, height: 20 }}
                            />
                          </Box>
                          <Typography variant="body2" sx={{ color: '#aaa', mb: 1 }}>
                            Processed: {pipelineData.sentimentProcessor.processedCount} items
                          </Typography>
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="body2" sx={{ mb: 0.5 }}>
                              Avg. processing time: {pipelineData.sentimentProcessor.avgProcessingTime.toFixed(2)}ms
                            </Typography>
                          </Box>
                        </Paper>
                      </Grid>
                      
                      {/* Signal Generator */}
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: '#1a2035' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <DataUsageIcon sx={{ mr: 1, color: '#64b5f6' }} />
                            <Typography variant="subtitle1">Signal Generator</Typography>
                            <Chip 
                              label={pipelineData.signalGenerator.status} 
                              size="small"
                              color={pipelineData.signalGenerator.status === 'online' ? 'success' : 'error'}
                              sx={{ ml: 1, height: 20 }}
                            />
                          </Box>
                          <Typography variant="body2" sx={{ color: '#aaa', mb: 1 }}>
                            Total signals: {pipelineData.signalGenerator.totalSignals}
                          </Typography>
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="body2" sx={{ mb: 0.5 }}>
                              Config: Threshold {pipelineData.signalGenerator.configuration.threshold} / Window {pipelineData.signalGenerator.configuration.timeWindow}min
                            </Typography>
                          </Box>
                        </Paper>
                      </Grid>
                      
                      {/* Cache Manager */}
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: '#1a2035' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <StorageIcon sx={{ mr: 1, color: '#64b5f6' }} />
                            <Typography variant="subtitle1">Cache Manager</Typography>
                            <Chip 
                              label={pipelineData.cacheManager.status} 
                              size="small"
                              color={pipelineData.cacheManager.status === 'online' ? 'success' : 'error'}
                              sx={{ ml: 1, height: 20 }}
                            />
                          </Box>
                          <Typography variant="body2" sx={{ color: '#aaa', mb: 1 }}>
                            Cache size: {pipelineData.cacheManager.cacheSize} items
                          </Typography>
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="body2" sx={{ mb: 0.5 }}>
                              Hit ratio: {(pipelineData.cacheManager.hitRatio * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        </Paper>
                      </Grid>
                      
                      {/* Pipeline Performance */}
                      <Grid item xs={12}>
                        <Paper sx={{ p: 2, bgcolor: '#1a2035' }}>
                          <Typography variant="subtitle1" gutterBottom>Pipeline Performance</Typography>
                          <Grid container spacing={2}>
                            <Grid item xs={6} sm={3}>
                              <Typography variant="body2" sx={{ color: '#aaa' }}>Processing Latency</Typography>
                              <Typography variant="h6">{pipelineData.latency.toFixed(2)}s</Typography>
                            </Grid>
                            <Grid item xs={6} sm={3}>
                              <Typography variant="body2" sx={{ color: '#aaa' }}>Total Signals</Typography>
                              <Typography variant="h6">{pipelineData.globalMetrics.total_signals}</Typography>
                            </Grid>
                            <Grid item xs={6} sm={3}>
                              <Typography variant="body2" sx={{ color: '#aaa' }}>Success Rate</Typography>
                              <Typography variant="h6">{(pipelineData.globalMetrics.success_rate * 100).toFixed(1)}%</Typography>
                            </Grid>
                            <Grid item xs={6} sm={3}>
                              <Typography variant="body2" sx={{ color: '#aaa' }}>Avg Score</Typography>
                              <Typography variant="h6">{pipelineData.globalMetrics.avg_sentiment_score.toFixed(2)}</Typography>
                            </Grid>
                          </Grid>
                        </Paper>
                      </Grid>
                    </Grid>
                  )}
                </Box>
              )}
              
              {/* Signals & Metrics Tab */}
              {tabValue === 1 && (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Latest Sentiment Signals
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 3, color: '#aaa' }}>
                    Most recent sentiment analysis signals and indicators from market data.
                  </Typography>
                  
                  {latestSignals.length > 0 ? (
                    <Grid container spacing={2}>
                      {latestSignals.slice(0, 6).map((signal, index) => (
                        <Grid item xs={12} sm={6} md={4} key={index}>
                          <Paper sx={{ p: 2, bgcolor: '#1a2035', borderRadius: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Box>
                                <Typography variant="subtitle2" sx={{ color: '#fff' }}>
                                  {signal.symbol}
                                </Typography>
                                <Typography variant="caption" sx={{ color: '#aaa' }}>
                                  {new Date(signal.analysis_time).toLocaleString()}
                                </Typography>
                              </Box>
                              <Chip
                                label={signal.trend}
                                size="small"
                                color={
                                  signal.trend === 'bullish' ? 'success' :
                                  signal.trend === 'bearish' ? 'error' : 'default'
                                }
                              />
                            </Box>
                            
                            <Box sx={{ mb: 1 }}>
                              <Typography variant="body2" sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Sentiment Score:</span>
                                <span style={{ 
                                  color: signal.sentiment_score > 0.2 ? '#4caf50' : 
                                         signal.sentiment_score < -0.2 ? '#f44336' : '#ffb74d'
                                }}>
                                  {signal.sentiment_score.toFixed(2)}
                                </span>
                              </Typography>
                              <Typography variant="body2" sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Signal Strength:</span>
                                <span>{(signal.signal_strength * 100).toFixed(1)}%</span>
                              </Typography>
                              <Typography variant="body2" sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>Confidence:</span>
                                <span>{(signal.confidence * 100).toFixed(1)}%</span>
                              </Typography>
                            </Box>
                            
                            <Box>
                              <Chip
                                label={signal.suggested_action}
                                size="small"
                                sx={{ 
                                  bgcolor: 
                                    signal.suggested_action === 'buy' ? 'rgba(76, 175, 80, 0.2)' : 
                                    signal.suggested_action === 'sell' ? 'rgba(244, 67, 54, 0.2)' : 
                                    'rgba(255, 183, 77, 0.2)',
                                  color: 
                                    signal.suggested_action === 'buy' ? '#4caf50' : 
                                    signal.suggested_action === 'sell' ? '#f44336' : 
                                    '#ffb74d',
                                  width: '100%',
                                  justifyContent: 'space-between'
                                }}
                              />
                            </Box>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  ) : (
                    <Paper sx={{ p: 3, textAlign: 'center', bgcolor: 'rgba(0,0,0,0.2)' }}>
                      <Typography variant="body1" sx={{ color: '#aaa' }}>
                        No sentiment signals available
                      </Typography>
                    </Paper>
                  )}
                </Box>
              )}
              
              {/* Data Sources Tab */}
              {tabValue === 2 && (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Data Sources
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 3, color: '#aaa' }}>
                    Source providers and monitored symbols for sentiment analysis.
                  </Typography>
                  
                  {isLoading ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                      <CircularProgress />
                    </Box>
                  ) : (
                    <Paper sx={{ p: 2, bgcolor: '#1e2638', borderRadius: 2 }}>
                      <Typography variant="subtitle2" sx={{ mb: 2, color: '#64b5f6' }}>
                        Monitored Symbols
                      </Typography>
                      
                      <Grid container spacing={1}>
                        {monitoredSymbols.length > 0 ? monitoredSymbols.map((symbol, index) => (
                          <Grid item key={index}>
                            <Chip 
                              label={symbol}
                              sx={{ 
                                bgcolor: 'rgba(33, 150, 243, 0.1)', 
                                color: '#64b5f6',
                                '&:hover': { bgcolor: 'rgba(33, 150, 243, 0.2)' } 
                              }}
                            />
                          </Grid>
                        )) : (
                          <Grid item xs={12}>
                            <Typography variant="body2" sx={{ color: '#aaa', fontStyle: 'italic' }}>
                              No symbols being monitored
                            </Typography>
                          </Grid>
                        )}
                      </Grid>
                    </Paper>
                  )}
                </Box>
              )}
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
};

export default SentimentDetailView;
