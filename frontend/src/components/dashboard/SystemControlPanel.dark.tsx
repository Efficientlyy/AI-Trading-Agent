import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Box,
  Chip,
  CircularProgress,
  Paper,
  Divider,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Alert,
  LinearProgress,
  Stack
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Settings as SettingsIcon,
  Pause as PauseIcon
} from '@mui/icons-material';
import { useSystemControl } from '../../context/SystemControlContext';

// Dark theme color constants
const darkBg = 'rgba(30, 34, 45, 0.9)';
const darkPaperBg = 'rgba(45, 50, 65, 0.8)';
const darkText = '#ffffff';
const darkSecondaryText = 'rgba(255, 255, 255, 0.7)';
const darkBorder = 'rgba(255, 255, 255, 0.1)';

const SystemControlPanel: React.FC = () => {
  const { 
    systemStatus, 
    isLoading, 
    agents, 
    sessions, 
    startAgent, 
    stopAgent 
  } = useSystemControl();
  
  // State for confirmation dialogs
  const [startAllDialogOpen, setStartAllDialogOpen] = useState(false);
  const [stopAllDialogOpen, setStopAllDialogOpen] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // Get status color based on system status
  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'online':
      case 'running':
        return 'success';
      case 'offline':
      case 'stopped':
        return 'error';
      case 'warning':
      case 'partial':
        return 'warning';
      default:
        return 'default';
    }
  };
  
  // Calculate system metrics
  const calculateMetrics = () => {
    const runningAgents = agents.filter(agent => agent.status === 'running').length;
    const totalAgents = agents.length;
    const agentUtilization = totalAgents > 0 ? (runningAgents / totalAgents) * 100 : 0;
    
    const activeSessions = sessions.filter(session => session.status === 'running').length;
    const totalSessions = sessions.length;
    const sessionUtilization = totalSessions > 0 ? (activeSessions / totalSessions) * 100 : 0;
    
    return {
      runningAgents,
      totalAgents,
      agentUtilization,
      activeSessions,
      totalSessions,
      sessionUtilization,
      systemLoad: Math.min(Math.max((runningAgents + activeSessions) / (totalAgents + totalSessions + 0.1) * 100, 0), 100) || 0
    };
  };
  
  const metrics = calculateMetrics();
  
  // Start all agents
  const handleStartAllAgents = async () => {
    setActionLoading(true);
    setActionError(null);
    setActionSuccess(null);
    
    try {
      // Filter out already running agents
      const agentsToStart = agents.filter(agent => agent.status !== 'running');
      
      if (agentsToStart.length === 0) {
        setActionSuccess('All agents are already running!');
        setStartAllDialogOpen(false);
        return;
      }
      
      // Start each agent sequentially
      for (const agent of agentsToStart) {
        await startAgent(agent.agent_id);
        // Small delay to prevent overwhelming the API
        await new Promise(resolve => setTimeout(resolve, 300));
      }
      
      setActionSuccess(`Started ${agentsToStart.length} agents successfully!`);
    } catch (error) {
      console.error('Error starting all agents:', error);
      setActionError('Failed to start all agents. Please try again.');
    } finally {
      setActionLoading(false);
      setStartAllDialogOpen(false);
    }
  };
  
  // Stop all agents
  const handleStopAllAgents = async () => {
    setActionLoading(true);
    setActionError(null);
    setActionSuccess(null);
    
    try {
      // Filter out already stopped agents
      const agentsToStop = agents.filter(agent => agent.status === 'running');
      
      if (agentsToStop.length === 0) {
        setActionSuccess('All agents are already stopped!');
        setStopAllDialogOpen(false);
        return;
      }
      
      // Stop each agent sequentially
      for (const agent of agentsToStop) {
        await stopAgent(agent.agent_id);
        // Small delay to prevent overwhelming the API
        await new Promise(resolve => setTimeout(resolve, 300));
      }
      
      setActionSuccess(`Stopped ${agentsToStop.length} agents successfully!`);
    } catch (error) {
      console.error('Error stopping all agents:', error);
      setActionError('Failed to stop all agents. Please try again.');
    } finally {
      setActionLoading(false);
      setStopAllDialogOpen(false);
    }
  };

  // Get status icon based on system status
  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'online':
      case 'running':
        return <CheckCircleIcon color="success" />;
      case 'offline':
      case 'stopped':
        return <ErrorIcon color="error" />;
      case 'warning':
      case 'partial':
        return <WarningIcon color="warning" />;
      default:
        return <WarningIcon color="disabled" />;
    }
  };

  // Determine overall system status
  const determineSystemStatus = (): string => {
    if (agents.length === 0 && sessions.length === 0) {
      return 'offline';
    }
    
    const runningAgents = agents.filter(agent => agent.status === 'running').length;
    const activeSessions = sessions.filter(session => session.status === 'running').length;
    
    if (runningAgents > 0 || activeSessions > 0) {
      if (runningAgents === agents.length && activeSessions === sessions.length) {
        return 'online';
      } else {
        return 'partial';
      }
    }
    
    return 'offline';
  };

  const systemStatusText = determineSystemStatus();

  return (
    <Card elevation={3} sx={{ mb: 3, bgcolor: darkBg, color: darkText, borderColor: darkBorder }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="div" sx={{ fontWeight: 'bold', color: darkText }}>
            System Control
          </Typography>
          
          <Box>
            <Tooltip title="System Settings">
              <IconButton size="small" sx={{ ml: 1, color: darkSecondaryText }}>
                <SettingsIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress sx={{ color: darkText }} />
          </Box>
        ) : (
          <>
            {actionSuccess && (
              <Alert severity="success" sx={{ mb: 2, bgcolor: 'rgba(46, 125, 50, 0.2)', color: '#81c784' }} onClose={() => setActionSuccess(null)}>
                {actionSuccess}
              </Alert>
            )}
            
            {actionError && (
              <Alert severity="error" sx={{ mb: 2, bgcolor: 'rgba(211, 47, 47, 0.2)', color: '#e57373' }} onClose={() => setActionError(null)}>
                {actionError}
              </Alert>
            )}
            
            <Grid container spacing={3}>
              {/* System Status Panel */}
              <Grid item xs={12} md={6}>
                <Paper elevation={1} sx={{ p: 2, height: '100%', bgcolor: darkPaperBg, color: darkText, borderColor: darkBorder }}>
                  <Typography variant="h6" sx={{ fontWeight: 'bold', color: darkText }} gutterBottom>
                    System Status
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                    {getStatusIcon(systemStatusText)}
                    <Typography variant="body1" sx={{ ml: 1, fontWeight: 'medium', color: darkText }}>
                      System Status
                    </Typography>
                    <Chip
                      label={systemStatusText.toUpperCase()}
                      color={getStatusColor(systemStatusText) as any}
                      size="small"
                      sx={{ ml: 2 }}
                    />
                  </Box>

                  <Box sx={{ mt: 3, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    <Button
                      variant="contained"
                      color="success"
                      startIcon={<PlayIcon />}
                      onClick={() => setStartAllDialogOpen(true)}
                      disabled={actionLoading || metrics.runningAgents === metrics.totalAgents || metrics.totalAgents === 0}
                    >
                      Start All Agents
                    </Button>
                    
                    <Button
                      variant="contained"
                      color="error"
                      startIcon={<StopIcon />}
                      onClick={() => setStopAllDialogOpen(true)}
                      disabled={actionLoading || metrics.runningAgents === 0}
                    >
                      Stop All Agents
                    </Button>
                    
                    <Button
                      variant="outlined"
                      color="warning"
                      startIcon={<PauseIcon />}
                      disabled={actionLoading || metrics.activeSessions === 0}
                      sx={{ borderColor: 'warning.main', color: 'warning.main' }}
                    >
                      Pause All Sessions
                    </Button>
                  </Box>
                </Paper>
              </Grid>

              {/* System Health Panel */}
              <Grid item xs={12} md={6}>
                <Paper elevation={1} sx={{ p: 2, height: '100%', bgcolor: darkPaperBg, color: darkText, borderColor: darkBorder }}>
                  <Typography variant="h6" sx={{ fontWeight: 'bold', color: darkText }} gutterBottom>
                    System Metrics
                  </Typography>
                  
                  <Stack spacing={2} sx={{ mt: 2 }}>
                    <Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'medium' }}>
                          Agent Utilization
                        </Typography>
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'bold' }}>
                          {metrics.runningAgents} / {metrics.totalAgents} agents running
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <Box sx={{ width: '100%', mr: 1 }}>
                          <LinearProgress 
                            variant="determinate" 
                            value={metrics.agentUtilization} 
                            color={metrics.agentUtilization > 80 ? "success" : "primary"}
                            sx={{ 
                              bgcolor: 'rgba(255, 255, 255, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                bgcolor: metrics.agentUtilization > 80 ? '#4caf50' : '#2196f3'
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'bold' }}>
                          {Math.round(metrics.agentUtilization)}%
                        </Typography>
                      </Box>
                    </Box>

                    <Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'medium' }}>
                          Session Utilization
                        </Typography>
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'bold' }}>
                          {metrics.activeSessions} / {metrics.totalSessions} sessions active
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <Box sx={{ width: '100%', mr: 1 }}>
                          <LinearProgress 
                            variant="determinate" 
                            value={metrics.sessionUtilization} 
                            color={metrics.sessionUtilization > 80 ? "success" : "primary"}
                            sx={{ 
                              bgcolor: 'rgba(255, 255, 255, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                bgcolor: metrics.sessionUtilization > 80 ? '#4caf50' : '#2196f3'
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'bold' }}>
                          {Math.round(metrics.sessionUtilization)}%
                        </Typography>
                      </Box>
                    </Box>

                    <Box>
                      <Typography variant="body2" sx={{ color: darkText, fontWeight: 'medium' }} gutterBottom>
                        System Load
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box sx={{ width: '100%', mr: 1 }}>
                          <LinearProgress 
                            variant="determinate" 
                            value={metrics.systemLoad} 
                            color={
                              metrics.systemLoad > 80 ? "error" : 
                              metrics.systemLoad > 60 ? "warning" : "success"
                            }
                            sx={{ 
                              bgcolor: 'rgba(255, 255, 255, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                bgcolor: metrics.systemLoad > 80 ? '#f44336' : 
                                        metrics.systemLoad > 60 ? '#ff9800' : '#4caf50'
                              }
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'bold' }}>
                          {Math.round(metrics.systemLoad)}%
                        </Typography>
                      </Box>
                    </Box>
                  </Stack>
                </Paper>
              </Grid>
              
              {/* Additional System Information */}
              <Grid item xs={12}>
                <Paper elevation={1} sx={{ p: 2, bgcolor: darkPaperBg, color: darkText, borderColor: darkBorder }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: darkText }} gutterBottom>
                    System Information
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ display: 'flex', alignItems: 'center', p: 1 }}>
                        <Box sx={{ 
                          width: 10, 
                          height: 10, 
                          borderRadius: '50%', 
                          bgcolor: metrics.runningAgents > 0 ? 'success.main' : 'error.main',
                          mr: 1 
                        }} />
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'medium' }}>
                          Trading Engine: <span style={{ fontWeight: 'bold' }}>{metrics.runningAgents > 0 ? 'Active' : 'Inactive'}</span>
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ display: 'flex', alignItems: 'center', p: 1 }}>
                        <Box sx={{ 
                          width: 10, 
                          height: 10, 
                          borderRadius: '50%', 
                          bgcolor: metrics.activeSessions > 0 ? 'success.main' : 'error.main',
                          mr: 1 
                        }} />
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'medium' }}>
                          Data Feed: <span style={{ fontWeight: 'bold' }}>{metrics.activeSessions > 0 ? 'Connected' : 'Disconnected'}</span>
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ display: 'flex', alignItems: 'center', p: 1 }}>
                        <Box sx={{ 
                          width: 10, 
                          height: 10, 
                          borderRadius: '50%', 
                          bgcolor: 'success.main',
                          mr: 1 
                        }} />
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'medium' }}>
                          Database: <span style={{ fontWeight: 'bold' }}>Connected</span>
                        </Typography>
                      </Box>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Box sx={{ display: 'flex', alignItems: 'center', p: 1 }}>
                        <Box sx={{ 
                          width: 10, 
                          height: 10, 
                          borderRadius: '50%', 
                          bgcolor: 'success.main',
                          mr: 1 
                        }} />
                        <Typography variant="body2" sx={{ color: darkText, fontWeight: 'medium' }}>
                          API Services: <span style={{ fontWeight: 'bold' }}>Online</span>
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>
            </Grid>
          </>
        )}
        
        {/* Confirmation Dialogs */}
        <Dialog
          open={startAllDialogOpen}
          onClose={() => setStartAllDialogOpen(false)}
          aria-labelledby="start-all-agents-dialog-title"
          PaperProps={{
            sx: { bgcolor: darkPaperBg, color: darkText }
          }}
        >
          <DialogTitle id="start-all-agents-dialog-title" sx={{ color: darkText }}>
            Start All Trading Agents
          </DialogTitle>
          <DialogContent>
            <DialogContentText sx={{ color: darkSecondaryText }}>
              This will start all currently inactive trading agents in the system. 
              Are you sure you want to proceed?
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setStartAllDialogOpen(false)} disabled={actionLoading} sx={{ color: darkSecondaryText }}>
              Cancel
            </Button>
            <Button 
              onClick={handleStartAllAgents} 
              color="primary" 
              variant="contained"
              disabled={actionLoading}
              startIcon={actionLoading ? <CircularProgress size={20} /> : <PlayIcon />}
            >
              Start All
            </Button>
          </DialogActions>
        </Dialog>
        
        <Dialog
          open={stopAllDialogOpen}
          onClose={() => setStopAllDialogOpen(false)}
          aria-labelledby="stop-all-agents-dialog-title"
          PaperProps={{
            sx: { bgcolor: darkPaperBg, color: darkText }
          }}
        >
          <DialogTitle id="stop-all-agents-dialog-title" sx={{ color: darkText }}>
            Stop All Trading Agents
          </DialogTitle>
          <DialogContent>
            <DialogContentText sx={{ color: darkSecondaryText }}>
              This will stop all currently running trading agents in the system. 
              Any open positions will be maintained but no new trades will be executed. 
              Are you sure you want to proceed?
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setStopAllDialogOpen(false)} disabled={actionLoading} sx={{ color: darkSecondaryText }}>
              Cancel
            </Button>
            <Button 
              onClick={handleStopAllAgents} 
              color="error" 
              variant="contained"
              disabled={actionLoading}
              startIcon={actionLoading ? <CircularProgress size={20} /> : <StopIcon />}
            >
              Stop All
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default SystemControlPanel;
