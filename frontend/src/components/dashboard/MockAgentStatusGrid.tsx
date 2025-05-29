import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Dialog,
  Grid,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Assessment as AssessmentIcon,
  NetworkCheck as NetworkCheckIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { mockAgents } from '../../api/mockData/mockAgents';

// Dark theme color constants
const darkBg = 'rgba(30, 34, 45, 0.9)';
const darkPaperBg = 'rgba(45, 50, 65, 0.8)';
const darkText = '#ffffff';
const darkSecondaryText = 'rgba(255, 255, 255, 0.7)';
const darkBorder = 'rgba(255, 255, 255, 0.1)';

const MockAgentStatusGrid: React.FC = () => {
  const [agents] = useState(mockAgents);
  const [selectedAgent, setSelectedAgent] = useState<any>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Check if an agent is the LLM Oversight Agent
  const isLLMOversightAgent = (agentId: string): boolean => {
    return agentId === 'llm_oversight_agent';
  };

  // Format date to readable format
  const formatDate = (dateString: string | undefined) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  // Open agent details dialog
  const handleOpenDetails = (agent: any) => {
    setSelectedAgent(agent);
    setDetailsOpen(true);
  };

  // Close agent details dialog
  const handleCloseDetails = () => {
    setDetailsOpen(false);
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
        return <StopIcon fontSize="small" />;
      case 'initializing':
        return <PlayIcon fontSize="small" />;
      default:
        return undefined;
    }
  };

  // Render agent type cell with special styling for LLM Oversight
  const renderAgentTypeCell = (agent: any) => {
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
            label="Connected"
            size="small"
            color="success"
            variant="outlined"
            sx={{ fontSize: '0.6rem' }}
          />
        </Box>
      );
    }
    return agent.type;
  };

  // Render agent strategy cell with special text for LLM Oversight
  const renderAgentStrategyCell = (agent: any) => {
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
        </Box>

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
                        {agent.symbols?.slice(0, 2).map((symbol: string) => (
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
                            >
                              <AssessmentIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Test Connection">
                            <IconButton
                              size="small"
                              color="info"
                            >
                              <NetworkCheckIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Adjust Confidence Threshold">
                            <IconButton
                              size="small"
                              color="secondary"
                            >
                              <SettingsIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </>
                      ) : null}

                      {agent.status === 'running' ? (
                        <Tooltip title="Stop Agent">
                          <IconButton
                            size="small"
                            color="error"
                          >
                            <StopIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      ) : (
                        <Tooltip title="Start Agent">
                          <IconButton
                            size="small"
                            color="success"
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
              <Typography variant="h6" sx={{ p: 2, borderBottom: `1px solid ${darkBorder}` }}>
                Agent Details: {selectedAgent.name}
                {isLLMOversightAgent(selectedAgent.agent_id) && (
                  <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    Powered by OpenRouter LLM Integration
                  </Typography>
                )}
              </Typography>
              <Box sx={{ p: 3 }}>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Configuration</Typography>
                    <Typography variant="body1">{selectedAgent.agent_id}</Typography>
                    
                    <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>Type</Typography>
                    <Typography variant="body1">
                      {isLLMOversightAgent(selectedAgent.agent_id) ? (
                        <Chip
                          label={selectedAgent.type}
                          size="small"
                          color="primary"
                          sx={{ bgcolor: 'rgba(75, 0, 130, 0.9)', color: '#fff' }}
                        />
                      ) : (
                        selectedAgent.type
                      )}
                    </Typography>
                    
                    <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>Strategy</Typography>
                    <Typography variant="body1">{renderAgentStrategyCell(selectedAgent)}</Typography>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Trading Symbols</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selectedAgent.symbols?.map((symbol: string) => (
                        <Chip
                          key={symbol}
                          label={symbol}
                          size="small"
                          variant="outlined"
                          sx={{ my: 0.5 }}
                        />
                      ))}
                    </Box>

                    {selectedAgent.metrics && (
                      <>
                        <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>Performance Metrics</Typography>
                        <Grid container spacing={2}>
                          {selectedAgent.metrics?.win_rate !== undefined && (
                            <Grid item xs={6}>
                              <Paper sx={{ p: 1.5 }}>
                                <Typography variant="caption">Win Rate</Typography>
                                <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                                  {`${(selectedAgent.metrics.win_rate * 100).toFixed(1)}%`}
                                </Typography>
                              </Paper>
                            </Grid>
                          )}
                          {selectedAgent.metrics?.profit_factor !== undefined && (
                            <Grid item xs={6}>
                              <Paper sx={{ p: 1.5 }}>
                                <Typography variant="caption">Profit Factor</Typography>
                                <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                                  {selectedAgent.metrics.profit_factor.toFixed(2)}
                                </Typography>
                              </Paper>
                            </Grid>
                          )}
                        </Grid>
                      </>
                    )}
                  </Grid>
                </Grid>
              </Box>
              <Box sx={{ p: 2, borderTop: `1px solid ${darkBorder}`, display: 'flex', justifyContent: 'flex-end' }}>
                <Button onClick={handleCloseDetails}>Close</Button>
              </Box>
            </>
          )}
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default MockAgentStatusGrid;
