import {
  Add as AddIcon,
  CheckCircle as CheckCircleIcon,
  Edit as EditIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  HourglassEmpty as PendingIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  NetworkCheck as NetworkCheckIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon
} from '@mui/icons-material';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  IconButton,
  Paper,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography
} from '@mui/material';
import React, { useState, useEffect } from 'react';
import { Agent, useSystemControl } from '../../context/SystemControlContext';
import { useLLMOversight } from '../../context/LLMOversightContext';
import AgentConfigurationDialog from './AgentConfigurationDialog';
import { oversightClient } from '../../api/oversightClient';

// Dark theme color constants
const darkBg = 'rgba(30, 34, 45, 0.9)';
const darkPaperBg = 'rgba(45, 50, 65, 0.8)';
const darkText = '#ffffff'; // White text for maximum visibility
const darkSecondaryText = 'rgba(255, 255, 255, 0.7)';
const darkBorder = 'rgba(255, 255, 255, 0.1)';

// Define interfaces for LLM metrics
interface LLMMetrics {
  model_statistics: {
    avg_confidence: number;
    avg_response_time_ms: number;
    total_decisions_evaluated: number;
    intervention_rate: number;
    token_usage: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
  };
  health_indicators: {
    current_latency_ms: number;
    status: string;
    uptime_percentage: number;
    last_communication: string;
    connection_failures: number;
  };
  recent_alerts: Array<{
    id: string;
    timestamp: string;
    severity: string;
    message: string;
    resolved: boolean;
  }>;
  performance_trends: Array<{
    date: string;
    accuracy: number;
    latency_ms: number;
  }>;
}

interface RecentAnalysis {
  id: string;
  timestamp: string;
  symbol: string;
  decision: string;
  confidence: number;
  override: boolean;
  reasoning: string;
  result: string;
}

interface LLMDialogState {
  open: boolean;
  view: 'metrics' | 'analyses' | 'settings' | 'test';
}

const AgentStatusGrid: React.FC = () => {
  const { agents, isLoading, startAgent, stopAgent } = useSystemControl();
  const { status, checkConnection } = useLLMOversight();
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [agentToEdit, setAgentToEdit] = useState<Agent | null>(null);
  const [llmMetrics, setLLMMetrics] = useState<LLMMetrics | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<RecentAnalysis[]>([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.75);
  const [isAdjustingThreshold, setIsAdjustingThreshold] = useState(false);
  const [connectionTestResult, setConnectionTestResult] = useState<any>(null);
  const [llmDialogState, setLLMDialogState] = useState<LLMDialogState>({
    open: false,
    view: 'metrics'
  });
  
  // Load LLM metrics when an LLM Oversight agent is selected
  useEffect(() => {
    if (selectedAgent && isLLMOversightAgent(selectedAgent.agent_id) && detailsOpen) {
      fetchLLMMetrics();
      fetchRecentAnalyses();
    }
  }, [selectedAgent, detailsOpen]);
  
  // Fetch LLM-specific metrics 
  const fetchLLMMetrics = async () => {
    try {
      const metrics = await oversightClient.getLLMMetrics();
      setLLMMetrics(metrics);
    } catch (error) {
      console.error('Error fetching LLM metrics:', error);
    }
  };
  
  // Fetch recent analyses
  const fetchRecentAnalyses = async () => {
    try {
      const result = await oversightClient.getRecentAnalyses(5);
      setRecentAnalyses(result.analyses || []);
    } catch (error) {
      console.error('Error fetching recent analyses:', error);
    }
  };
  
  // Test LLM connection
  const testLLMConnection = async () => {
    try {
      setConnectionTestResult({ status: 'testing' });
      const result = await oversightClient.testConnection();
      setConnectionTestResult(result);
    } catch (error) {
      console.error('Error testing LLM connection:', error);
      setConnectionTestResult({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      });
    }
  };
  
  // Update confidence threshold
  const updateConfidenceThreshold = async () => {
    try {
      setIsAdjustingThreshold(true);
      await oversightClient.adjustConfidenceThreshold(confidenceThreshold);
      await fetchLLMMetrics(); // Refresh metrics after update
      setIsAdjustingThreshold(false);
    } catch (error) {
      console.error('Error updating confidence threshold:', error);
      setIsAdjustingThreshold(false);
    }
  };
  
  // Open LLM Dialog 
  const openLLMDialog = (view: 'metrics' | 'analyses' | 'settings' | 'test') => {
    setLLMDialogState({
      open: true,
      view
    });
  };
  
  // Close LLM Dialog
  const closeLLMDialog = () => {
    setLLMDialogState({
      ...llmDialogState,
      open: false
    });
  };

  // Get status color based on agent status
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'stopped':
        return 'error';
      case 'error':
        return 'error';
      case 'initializing':
        return 'warning';
      default:
        return 'default';
    }
  };

  // Get status icon based on agent status
  const getStatusIcon = (status: string): React.ReactElement | undefined => {
    switch (status) {
      case 'running':
        return <CheckCircleIcon fontSize="small" />;
      case 'stopped':
        return <StopIcon fontSize="small" />;
      case 'error':
        return <ErrorIcon fontSize="small" />;
      case 'initializing':
        return <PendingIcon fontSize="small" />;
      default:
        return undefined;
    }
  };

  // Format date to readable format
  const formatDate = (dateString: string | undefined) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  // Open agent details dialog
  const handleOpenDetails = (agent: Agent) => {
    setSelectedAgent(agent);
    setDetailsOpen(true);
  };

  // Close agent details dialog
  const handleCloseDetails = () => {
    setDetailsOpen(false);
  };

  // Open agent configuration dialog for creating a new agent
  const handleOpenCreateDialog = () => {
    setAgentToEdit(null);
    setConfigDialogOpen(true);
  };

  // Open agent configuration dialog for editing an existing agent
  const handleOpenEditDialog = (agent: Agent) => {
    setAgentToEdit(agent);
    setConfigDialogOpen(true);
  };

  // Close agent configuration dialog
  const handleCloseConfigDialog = () => {
    setConfigDialogOpen(false);
    setAgentToEdit(null);
  };

  // Check if an agent is the LLM Oversight Agent
  const isLLMOversightAgent = (agentId: string): boolean => {
    return agentId === 'llm_oversight_agent';
  };

  // Render agent type cell with special styling for LLM Oversight
  const renderAgentTypeCell = (agent: Agent) => {
    if (isLLMOversightAgent(agent.agent_id)) {
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 0.5 }}>
          <Chip
            label={agent.type}
            size="small"
            color="primary"
            variant="filled"
            sx={{
              bgcolor: 'rgba(75, 0, 130, 0.9)',
              color: '#ffffff',
              fontWeight: 'bold',
            }}
          />
          <Chip
            label={status.isConnected ? 'Connected' : 'Disconnected'}
            size="small"
            color={status.isConnected ? 'success' : 'error'}
            variant="outlined"
            sx={{ fontSize: '0.6rem' }}
          />
        </Box>
      );
    }
    return agent.type;
  };

  // Render agent strategy cell with special text for LLM Oversight
  const renderAgentStrategyCell = (agent: Agent) => {
    if (isLLMOversightAgent(agent.agent_id)) {
      return 'LLM-Based Decision Oversight';
    }
    return agent.strategy || 'N/A';
  };

  return (
    <Card elevation={3} sx={{ mb: 3, bgcolor: darkBg, color: darkText, borderRadius: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="div" sx={{ fontWeight: 'bold', color: darkText }}>
            Trading Agents
          </Typography>

          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={handleOpenCreateDialog}
            size="small"
          >
            Create Agent
          </Button>
        </Box>

        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : agents.length === 0 ? (
          <Box sx={{ textAlign: 'center', p: 3, bgcolor: darkPaperBg, borderRadius: 1 }}>
            <Typography variant="body1" sx={{ color: darkSecondaryText }}>
              No trading agents available
            </Typography>
          </Box>
        ) : (
          <TableContainer component={Paper} variant="outlined" sx={{ bgcolor: darkPaperBg, borderColor: darkBorder }}>
            <Table sx={{ '& .MuiTableCell-root': { color: darkText, borderColor: darkBorder } }}>
              <TableHead sx={{ '& .MuiTableCell-head': { fontWeight: 'bold', color: darkText, bgcolor: 'rgba(30, 34, 45, 0.95)' } }}>
                <TableRow>
                  <TableCell>Agent</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Strategy</TableCell>
                  <TableCell>Symbols</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Last Active</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {agents.map((agent) => (
                  <TableRow 
                    key={agent.agent_id} 
                    hover
                    sx={isLLMOversightAgent(agent.agent_id) ? { 
                      bgcolor: 'rgba(75, 0, 130, 0.1)'
                    } : {}}
                  >
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        {agent.name}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {agent.agent_id}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {renderAgentTypeCell(agent)}
                    </TableCell>
                    <TableCell>{renderAgentStrategyCell(agent)}</TableCell>
                    <TableCell>
                      {agent.symbols && agent.symbols.length > 0 ? (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {agent.symbols?.slice(0, 2).map((symbol) => (
                            <Chip
                              key={symbol}
                              label={symbol}
                              size="small"
                              variant="outlined"
                              sx={{
                                bgcolor: 'rgba(66, 66, 66, 0.8)',
                                color: darkText,
                                fontSize: '0.7rem',
                              }}
                            />
                          ))}
                          {agent.symbols && agent.symbols.length > 2 && (
                            <Chip
                              label={`+${agent.symbols.length - 2}`}
                              size="small"
                              variant="outlined"
                              sx={{
                                bgcolor: 'rgba(66, 66, 66, 0.8)',
                                color: darkText,
                                fontSize: '0.7rem',
                              }}
                            />
                          )}
                        </Box>
                      ) : (
                        <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                          No symbols
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={getStatusIcon(agent.status)}
                        label={agent.status.toUpperCase()}
                        color={getStatusColor(agent.status) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{formatDate(agent.last_updated)}</TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            onClick={() => handleOpenDetails(agent)}
                          >
                            <InfoIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>

                        {isLLMOversightAgent(agent.agent_id) ? (
                          <>
                            <Tooltip title="View Recent Analyses">
                              <IconButton
                                size="small"
                                color="primary"
                                onClick={() => {
                                  setSelectedAgent(agent);
                                  openLLMDialog('analyses');
                                }}
                              >
                                <AssessmentIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Test Connection">
                              <IconButton
                                size="small"
                                color="info"
                                onClick={() => {
                                  setSelectedAgent(agent);
                                  openLLMDialog('test');
                                  testLLMConnection();
                                }}
                              >
                                <NetworkCheckIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Adjust Confidence Threshold">
                              <IconButton
                                size="small"
                                color="secondary"
                                onClick={() => {
                                  setSelectedAgent(agent);
                                  openLLMDialog('settings');
                                }}
                              >
                                <SettingsIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </>
                        ) : (
                          <Tooltip title="Edit Agent">
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => handleOpenEditDialog(agent)}
                            >
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}

                        {agent.status === 'running' ? (
                          <Tooltip title="Stop Agent">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => stopAgent(agent.agent_id)}
                            >
                              <StopIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        ) : (
                          <Tooltip title="Start Agent">
                            <IconButton
                              size="small"
                              color="success"
                              onClick={() => startAgent(agent.agent_id)}
                              disabled={agent.status === 'error'}
                            >
                              <PlayIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {/* Agent Configuration Dialog */}
        <AgentConfigurationDialog
          open={configDialogOpen}
          onClose={handleCloseConfigDialog}
          editAgent={agentToEdit}
        />
        
        {/* LLM Oversight Dialog - For metrics, analyses, settings, and connection testing */}
        <Dialog
          open={llmDialogState.open}
          onClose={closeLLMDialog}
          maxWidth="md"
          fullWidth
          PaperProps={{
            style: {
              backgroundColor: darkBg,
              color: darkText,
              borderRadius: 8
            }
          }}
        >
          <DialogTitle sx={{ 
            borderBottom: `1px solid ${darkBorder}`,
            bgcolor: 'rgba(75, 0, 130, 0.6)'
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                {llmDialogState.view === 'metrics' && 'LLM Oversight Metrics'}
                {llmDialogState.view === 'analyses' && 'Recent Decision Analyses'}
                {llmDialogState.view === 'settings' && 'Oversight Settings'}
                {llmDialogState.view === 'test' && 'Connection Test'}
                <Typography variant="caption" display="block" sx={{ mt: 0.5, color: 'rgba(255, 255, 255, 0.8)' }}>
                  {status.provider} - {status.model}
                </Typography>
              </Typography>
              <Box>
                <Button 
                  variant={llmDialogState.view === 'metrics' ? 'contained' : 'outlined'} 
                  size="small" 
                  onClick={() => setLLMDialogState({...llmDialogState, view: 'metrics'})}
                  sx={{ mr: 1 }}
                >
                  Metrics
                </Button>
                <Button 
                  variant={llmDialogState.view === 'analyses' ? 'contained' : 'outlined'} 
                  size="small" 
                  onClick={() => setLLMDialogState({...llmDialogState, view: 'analyses'})}
                  sx={{ mr: 1 }}
                >
                  Analyses
                </Button>
                <Button 
                  variant={llmDialogState.view === 'settings' ? 'contained' : 'outlined'} 
                  size="small" 
                  onClick={() => setLLMDialogState({...llmDialogState, view: 'settings'})}
                  sx={{ mr: 1 }}
                >
                  Settings
                </Button>
              </Box>
            </Box>
          </DialogTitle>
          
          <DialogContent dividers sx={{ bgcolor: darkPaperBg, borderColor: darkBorder, p: 3 }}>
            {/* Metrics View */}
            {llmDialogState.view === 'metrics' && (
              <Grid container spacing={3}>
                {/* Model Statistics */}
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'rgba(40, 44, 55, 0.8)', borderColor: darkBorder }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold', color: darkText }}>
                      Model Performance
                    </Typography>
                    {llmMetrics ? (
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkBg, borderColor: darkBorder }}>
                            <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                              Avg. Confidence
                            </Typography>
                            <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                              {(llmMetrics.model_statistics.avg_confidence * 100).toFixed(1)}%
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6}>
                          <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkBg, borderColor: darkBorder }}>
                            <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                              Response Time
                            </Typography>
                            <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                              {llmMetrics.model_statistics.avg_response_time_ms}ms
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6}>
                          <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkBg, borderColor: darkBorder }}>
                            <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                              Decisions Evaluated
                            </Typography>
                            <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                              {llmMetrics.model_statistics.total_decisions_evaluated}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={6}>
                          <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkBg, borderColor: darkBorder }}>
                            <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                              Intervention Rate
                            </Typography>
                            <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                              {(llmMetrics.model_statistics.intervention_rate * 100).toFixed(1)}%
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    ) : (
                      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                        <CircularProgress />
                      </Box>
                    )}
                  </Paper>
                </Grid>
                
                {/* Health indicators */}
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'rgba(40, 44, 55, 0.8)', borderColor: darkBorder }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold', color: darkText }}>
                      System Health
                    </Typography>
                    {llmMetrics ? (
                      <>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Box sx={{ position: 'relative', display: 'inline-flex', mr: 2 }}>
                            <CircularProgress 
                              variant="determinate" 
                              value={llmMetrics.health_indicators.uptime_percentage} 
                              color={llmMetrics.health_indicators.status === 'healthy' ? 'success' : 'warning'}
                            />
                            <Box
                              sx={{
                                top: 0,
                                left: 0,
                                bottom: 0,
                                right: 0,
                                position: 'absolute',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                              }}
                            >
                              <Typography
                                variant="caption"
                                component="div"
                                color="text.secondary"
                              >
                                {`${Math.round(llmMetrics.health_indicators.uptime_percentage)}%`}
                              </Typography>
                            </Box>
                          </Box>
                          <Box>
                            <Typography variant="body2" sx={{ color: darkText }}>
                              Status: <Chip 
                                label={llmMetrics.health_indicators.status.toUpperCase()} 
                                size="small" 
                                color={llmMetrics.health_indicators.status === 'healthy' ? 'success' : 'warning'}
                              />
                            </Typography>
                            <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                              Current Latency: {llmMetrics.health_indicators.current_latency_ms}ms
                            </Typography>
                          </Box>
                        </Box>
                        
                        {/* Connection failures */}
                        <Typography variant="body2" sx={{ color: darkText, mb: 1 }}>
                          Connection Failures: {llmMetrics.health_indicators.connection_failures}
                        </Typography>
                        
                        <Typography variant="body2" sx={{ color: darkText }}>
                          Last Communication: {new Date(llmMetrics.health_indicators.last_communication).toLocaleString()}
                        </Typography>
                      </>
                    ) : (
                      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                        <CircularProgress />
                      </Box>
                    )}
                  </Paper>
                </Grid>
                
                {/* Recent alerts */}
                <Grid item xs={12}>
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'rgba(40, 44, 55, 0.8)', borderColor: darkBorder }}>
                    <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 'bold', color: darkText }}>
                      Recent Alerts
                    </Typography>
                    {llmMetrics ? (
                      llmMetrics.recent_alerts.length > 0 ? (
                        <TableContainer>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Severity</TableCell>
                                <TableCell>Message</TableCell>
                                <TableCell>Time</TableCell>
                                <TableCell>Status</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {llmMetrics.recent_alerts.map((alert) => (
                                <TableRow key={alert.id}>
                                  <TableCell>
                                    <Chip 
                                      label={alert.severity} 
                                      size="small" 
                                      color={
                                        alert.severity === 'critical' ? 'error' :
                                        alert.severity === 'warning' ? 'warning' : 'info'
                                      }
                                    />
                                  </TableCell>
                                  <TableCell>{alert.message}</TableCell>
                                  <TableCell>{new Date(alert.timestamp).toLocaleString()}</TableCell>
                                  <TableCell>
                                    <Chip 
                                      label={alert.resolved ? 'Resolved' : 'Active'} 
                                      size="small" 
                                      color={alert.resolved ? 'success' : 'error'}
                                      variant="outlined"
                                    />
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      ) : (
                        <Typography variant="body2" sx={{ color: darkSecondaryText, textAlign: 'center', p: 2 }}>
                          No recent alerts
                        </Typography>
                      )
                    ) : (
                      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                        <CircularProgress />
                      </Box>
                    )}
                  </Paper>
                </Grid>
              </Grid>
            )}
            
            {/* Analyses View */}
            {llmDialogState.view === 'analyses' && (
              <>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: darkText }}>
                    Recent Decision Analyses
                  </Typography>
                  <Button 
                    size="small" 
                    variant="outlined" 
                    startIcon={<RefreshIcon />}
                    onClick={fetchRecentAnalyses}
                  >
                    Refresh
                  </Button>
                </Box>
                
                {recentAnalyses.length > 0 ? (
                  <TableContainer component={Paper} variant="outlined" sx={{ bgcolor: darkPaperBg, borderColor: darkBorder }}>
                    <Table sx={{ '& .MuiTableCell-root': { color: darkText, borderColor: darkBorder } }}>
                      <TableHead sx={{ '& .MuiTableCell-head': { fontWeight: 'bold', color: darkText, bgcolor: 'rgba(40, 44, 55, 0.95)' } }}>
                        <TableRow>
                          <TableCell>Symbol</TableCell>
                          <TableCell>Decision</TableCell>
                          <TableCell>Confidence</TableCell>
                          <TableCell>Override</TableCell>
                          <TableCell>Result</TableCell>
                          <TableCell>Time</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {recentAnalyses.map((analysis) => (
                          <TableRow key={analysis.id}>
                            <TableCell>{analysis.symbol}</TableCell>
                            <TableCell>
                              <Chip 
                                label={analysis.decision} 
                                size="small" 
                                color={
                                  analysis.decision === 'BUY' ? 'success' :
                                  analysis.decision === 'SELL' ? 'error' : 'default'
                                }
                              />
                            </TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <Box sx={{ position: 'relative', display: 'inline-flex', mr: 1, width: 30, height: 30 }}>
                                  <CircularProgress 
                                    variant="determinate" 
                                    value={analysis.confidence * 100} 
                                    size={30}
                                    thickness={8}
                                    color={
                                      analysis.confidence > 0.8 ? 'success' :
                                      analysis.confidence > 0.6 ? 'info' : 'warning'
                                    }
                                  />
                                </Box>
                                {(analysis.confidence * 100).toFixed(0)}%
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={analysis.override ? 'Yes' : 'No'} 
                                size="small" 
                                color={analysis.override ? 'warning' : 'success'}
                                variant="outlined"
                              />
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={analysis.result} 
                                size="small" 
                                color={
                                  analysis.result === 'profitable' ? 'success' :
                                  analysis.result === 'loss' ? 'error' : 'default'
                                }
                                variant="outlined"
                              />
                            </TableCell>
                            <TableCell>{new Date(analysis.timestamp).toLocaleString()}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Typography variant="body2" sx={{ color: darkSecondaryText, textAlign: 'center', p: 3 }}>
                    No recent analyses available
                  </Typography>
                )}
                
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: darkText, mb: 1 }}>
                    Understanding Override Decisions
                  </Typography>
                  <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                    The LLM oversight agent reviews all trading decisions and may override them based on various factors including current market conditions, risk assessment, and pattern recognition. High confidence scores generally indicate more reliable analyses.
                  </Typography>
                </Box>
              </>
            )}
            
            {/* Settings View */}
            {llmDialogState.view === 'settings' && (
              <>
                <Typography variant="subtitle1" sx={{ mb: 3, fontWeight: 'bold', color: darkText }}>
                  Model Parameters
                </Typography>
                
                <Paper variant="outlined" sx={{ p: 3, mb: 3, bgcolor: 'rgba(40, 44, 55, 0.8)', borderColor: darkBorder }}>
                  <Typography variant="subtitle2" sx={{ mb: 1, color: darkText }}>
                    Confidence Threshold
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, color: darkSecondaryText }}>
                    Set the minimum confidence level required for the LLM to override a trading decision. Higher values result in fewer interventions.
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ width: '100%', mr: 2 }}>
                      <Slider
                        value={confidenceThreshold}
                        min={0}
                        max={1}
                        step={0.01}
                        onChange={(_: Event, value: number | number[]) => setConfidenceThreshold(value as number)}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(value: number) => `${(value * 100).toFixed(0)}%`}
                        disabled={isAdjustingThreshold}
                      />
                    </Box>
                    <Typography variant="body2" sx={{ minWidth: 60, color: darkText }}>
                      {(confidenceThreshold * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                  
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={updateConfidenceThreshold}
                    disabled={isAdjustingThreshold}
                    startIcon={isAdjustingThreshold ? <CircularProgress size={20} /> : <SaveIcon />}
                    fullWidth
                  >
                    {isAdjustingThreshold ? 'Updating...' : 'Save Threshold'}
                  </Button>
                </Paper>
                
                <Paper variant="outlined" sx={{ p: 3, bgcolor: 'rgba(40, 44, 55, 0.8)', borderColor: darkBorder }}>
                  <Typography variant="subtitle2" sx={{ mb: 1, color: darkText }}>
                    Connection Information
                  </Typography>
                  <Grid container spacing={2} sx={{ mb: 2 }}>
                    <Grid item xs={6}>
                      <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                        Provider
                      </Typography>
                      <Typography variant="body1" sx={{ color: darkText }}>
                        {status.provider}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                        Model
                      </Typography>
                      <Typography variant="body1" sx={{ color: darkText }}>
                        {status.model}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                        Status
                      </Typography>
                      <Chip 
                        label={status.isConnected ? 'Connected' : 'Disconnected'} 
                        size="small" 
                        color={status.isConnected ? 'success' : 'error'}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                        Health
                      </Typography>
                      <Chip 
                        label={status.health} 
                        size="small" 
                        color={
                          status.health === 'healthy' ? 'success' :
                          status.health === 'unhealthy' ? 'error' : 'warning'
                        }
                      />
                    </Grid>
                  </Grid>
                  
                  <Button
                    variant="outlined"
                    color="primary"
                    onClick={() => {
                      checkConnection();
                      testLLMConnection();
                    }}
                    startIcon={<RefreshIcon />}
                    fullWidth
                  >
                    Refresh Connection Status
                  </Button>
                </Paper>
              </>
            )}
            
            {/* Test Connection View */}
            {llmDialogState.view === 'test' && (
              <>
                <Typography variant="subtitle1" sx={{ mb: 3, fontWeight: 'bold', color: darkText }}>
                  OpenRouter Connection Test
                </Typography>
                
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={testLLMConnection}
                    startIcon={connectionTestResult?.status === 'testing' ? <CircularProgress size={20} /> : <NetworkCheckIcon />}
                    disabled={connectionTestResult?.status === 'testing'}
                  >
                    {connectionTestResult?.status === 'testing' ? 'Testing...' : 'Test Connection'}
                  </Button>
                </Box>
                
                {connectionTestResult && connectionTestResult.status !== 'testing' && (
                  <Paper variant="outlined" sx={{ p: 3, bgcolor: 'rgba(40, 44, 55, 0.8)', borderColor: darkBorder }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Box sx={{ mr: 2 }}>
                        {connectionTestResult.success ? (
                          <CheckCircleIcon color="success" fontSize="large" />
                        ) : (
                          <ErrorIcon color="error" fontSize="large" />
                        )}
                      </Box>
                      <Box>
                        <Typography variant="h6" sx={{ color: darkText }}>
                          {connectionTestResult.success ? 'Connection Successful' : 'Connection Failed'}
                        </Typography>
                        <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                          {connectionTestResult.success 
                            ? `Connected to ${connectionTestResult.provider || 'OpenRouter'}` 
                            : connectionTestResult.error || 'Unknown error'}
                        </Typography>
                      </Box>
                    </Box>
                    
                    {connectionTestResult.success && (
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                            Model
                          </Typography>
                          <Typography variant="body1" sx={{ color: darkText }}>
                            {connectionTestResult.model || 'Unknown'}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                            Latency
                          </Typography>
                          <Typography variant="body1" sx={{ color: darkText }}>
                            {connectionTestResult.latency_ms}ms
                          </Typography>
                        </Grid>
                        <Grid item xs={12}>
                          <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                            Timestamp
                          </Typography>
                          <Typography variant="body1" sx={{ color: darkText }}>
                            {new Date(connectionTestResult.timestamp).toLocaleString()}
                          </Typography>
                        </Grid>
                      </Grid>
                    )}
                  </Paper>
                )}
                
                <Box sx={{ mt: 3 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: darkText, mb: 1 }}>
                    Connection Requirements
                  </Typography>
                  <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                    The LLM oversight integration requires a valid OpenRouter API key with access to language models such as GPT-4 or Claude. Ensure your API key has sufficient token allocation for production use. Connection issues may affect the system's ability to analyze trading decisions.
                  </Typography>
                </Box>
              </>
            )}
          </DialogContent>
          
          <DialogActions sx={{ bgcolor: darkPaperBg, borderTop: `1px solid ${darkBorder}` }}>
            <Button onClick={closeLLMDialog} sx={{ color: darkText }}>Close</Button>
          </DialogActions>
        </Dialog>

        {/* Agent Details Dialog */}
        <Dialog
          open={detailsOpen}
          onClose={handleCloseDetails}
          maxWidth="md"
          fullWidth
          PaperProps={{
            style: {
              backgroundColor: darkBg,
              color: darkText,
              borderRadius: 8
            }
          }}
        >
          {selectedAgent && (
            <>
              <DialogTitle sx={{ 
                borderBottom: `1px solid ${darkBorder}`,
                bgcolor: isLLMOversightAgent(selectedAgent.agent_id) ? 'rgba(75, 0, 130, 0.6)' : 'inherit'
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                    Agent Details: {selectedAgent.name}
                    {isLLMOversightAgent(selectedAgent.agent_id) && (
                      <Typography variant="caption" display="block" sx={{ mt: 0.5, color: 'rgba(255, 255, 255, 0.8)' }}>
                        Powered by OpenRouter LLM Integration
                      </Typography>
                    )}
                  </Typography>
                  <Chip
                    label={selectedAgent.status.toUpperCase()}
                    color={getStatusColor(selectedAgent.status) as any}
                    size="small"
                    icon={getStatusIcon(selectedAgent.status)}
                    sx={{ fontWeight: 'bold' }}
                  />
                </Box>
              </DialogTitle>
              <DialogContent dividers sx={{ bgcolor: darkPaperBg, borderColor: darkBorder }}>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" sx={{ color: darkText, fontWeight: 'bold' }}>
                      Configuration
                    </Typography>
                    <Typography variant="body1" sx={{ color: darkText, mb: 1 }}>
                      {selectedAgent.agent_id}
                    </Typography>

                    <Typography variant="subtitle2" sx={{ color: darkText, fontWeight: 'bold', mt: 2 }}>
                      Type
                    </Typography>
                    <Typography variant="body1" sx={{ color: darkText, mb: 1 }}>
                      {isLLMOversightAgent(selectedAgent.agent_id) ? (
                        <Chip
                          label={selectedAgent.type}
                          size="small"
                          color="primary"
                          variant="filled"
                          sx={{
                            bgcolor: 'rgba(75, 0, 130, 0.9)',
                            color: '#ffffff',
                            fontWeight: 'bold',
                          }}
                        />
                      ) : (
                        selectedAgent.type
                      )}
                    </Typography>

                    <Typography variant="subtitle2" sx={{ color: darkText, fontWeight: 'bold', mt: 2 }}>
                      Strategy
                    </Typography>
                    <Typography variant="body1" sx={{ color: darkText, mb: 1 }}>
                      {renderAgentStrategyCell(selectedAgent)}
                    </Typography>

                    <Typography variant="subtitle2" sx={{ color: darkText, fontWeight: 'bold', mt: 2 }}>
                      Last Active
                    </Typography>
                    <Typography variant="body1" sx={{ color: darkText, mb: 1 }}>
                      {formatDate(selectedAgent.last_updated)}
                    </Typography>
                    
                    {isLLMOversightAgent(selectedAgent.agent_id) && (
                      <>
                        <Typography variant="subtitle2" sx={{ color: darkText, fontWeight: 'bold', mt: 2 }}>
                          Provider
                        </Typography>
                        <Typography variant="body1" sx={{ color: darkText, mb: 1 }}>
                          OpenRouter
                        </Typography>
                      </>
                    )}
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" sx={{ color: darkText, fontWeight: 'bold' }}>
                      Trading Symbols
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                      {selectedAgent.symbols && selectedAgent.symbols.length > 0 ? (
                        selectedAgent.symbols.map((symbol) => (
                          <Chip
                            key={symbol}
                            label={symbol}
                            size="small"
                            variant="outlined"
                            sx={{
                              bgcolor: 'rgba(66, 66, 66, 0.8)',
                              color: darkText,
                              my: 0.5
                            }}
                          />
                        ))
                      ) : (
                        <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                          No symbols
                        </Typography>
                      )}
                    </Box>

                    {selectedAgent.metrics && (
                      selectedAgent.metrics.win_rate !== undefined || 
                      selectedAgent.metrics.profit_factor !== undefined || 
                      selectedAgent.metrics.avg_profit_loss !== undefined || 
                      selectedAgent.metrics.max_drawdown !== undefined
                    ) && (
                      <>
                        <Typography variant="subtitle2" sx={{ color: darkText, fontWeight: 'bold', mt: 2 }}>
                          Performance Metrics
                        </Typography>
                        <Grid container spacing={2} sx={{ mt: 0.5 }}>
                          {selectedAgent.metrics?.win_rate !== undefined && (
                            <Grid item xs={6}>
                              <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                  Win Rate
                                </Typography>
                                <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                                  {`${(selectedAgent.metrics.win_rate * 100).toFixed(1)}%`}
                                </Typography>
                              </Paper>
                            </Grid>
                          )}
                          {selectedAgent.metrics?.profit_factor !== undefined && (
                            <Grid item xs={6}>
                              <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                  Profit Factor
                                </Typography>
                                <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                                  {selectedAgent.metrics.profit_factor.toFixed(2)}
                                </Typography>
                              </Paper>
                            </Grid>
                          )}
                          {selectedAgent.metrics?.avg_profit_loss !== undefined && (
                            <Grid item xs={6}>
                              <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                  Avg Profit/Loss
                                </Typography>
                                <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
                                  {`${selectedAgent.metrics.avg_profit_loss > 0 ? '+' : ''}${selectedAgent.metrics.avg_profit_loss.toFixed(2)}%`}
                                </Typography>
                              </Paper>
                            </Grid>
                          )}
                          {selectedAgent.metrics?.max_drawdown !== undefined && (
                            <Grid item xs={6}>
                              <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                  Max Drawdown
                                </Typography>
                                <Typography variant="h6" sx={{ color: '#f44336', fontWeight: 'bold' }}>
                                  {`${selectedAgent.metrics.max_drawdown.toFixed(2)}%`}
                                </Typography>
                              </Paper>
                            </Grid>
                          )}
                        </Grid>
                      </>
                    )}
                  </Grid>
                </Grid>
              </DialogContent>
              <DialogActions sx={{ bgcolor: darkPaperBg, borderTop: `1px solid ${darkBorder}` }}>
                {selectedAgent.status === 'running' ? (
                  <Button
                    variant="contained"
                    color="error"
                    startIcon={<StopIcon />}
                    onClick={() => {
                      stopAgent(selectedAgent.agent_id);
                      handleCloseDetails();
                    }}
                  >
                    Stop Agent
                  </Button>
                ) : (
                  <Button
                    variant="contained"
                    color="success"
                    startIcon={<PlayIcon />}
                    onClick={() => {
                      startAgent(selectedAgent.agent_id);
                      handleCloseDetails();
                    }}
                    disabled={selectedAgent.status === 'error'}
                  >
                    Start Agent
                  </Button>
                )}
                <Button onClick={handleCloseDetails} sx={{ color: darkText }}>Close</Button>
              </DialogActions>
            </>
          )}
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default AgentStatusGrid;
