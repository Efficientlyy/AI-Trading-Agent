import {
  Add as AddIcon,
  CheckCircle as CheckCircleIcon,
  Edit as EditIcon,
  Error as ErrorIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon
} from '@mui/icons-material';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
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
  Typography
} from '@mui/material';
import React, { useState } from 'react';
import { Session, useSystemControl } from '../../context/SystemControlContext';
import SessionConfigurationDialog from './SessionConfigurationDialog';

// Dark theme constants
const darkBg = '#1E1E1E';
const darkPaperBg = '#2D2D2D';
const darkBorder = '#444444';
const darkText = '#FFFFFF';
const darkSecondaryText = '#AAAAAA';

const SessionManagementPanel: React.FC = () => {
  const { sessions, isLoading, pauseSession, resumeSession } = useSystemControl();
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [expandedSession, setExpandedSession] = useState<string | null>(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [sessionToEdit, setSessionToEdit] = useState<Session | null>(null);

  // Get status color based on session status
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'stopped':
        return 'error';
      case 'paused':
        return 'warning';
      case 'completed':
        return 'info';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  // Get status icon based on session status
  const getStatusIcon = (status: string): React.ReactElement | undefined => {
    switch (status) {
      case 'running':
        return <CheckCircleIcon fontSize="small" />;
      case 'stopped':
        return <StopIcon fontSize="small" />;
      case 'paused':
        return <PauseIcon fontSize="small" />;
      case 'completed':
        return <CheckCircleIcon fontSize="small" />;
      case 'error':
        return <ErrorIcon fontSize="small" />;
      default:
        return undefined;
    }
  };

  // Format date to readable format
  const formatDate = (dateString: string | undefined) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  // Format uptime from seconds to readable format
  const formatUptime = (seconds: number | undefined) => {
    if (!seconds) return 'N/A';

    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  // Open session details dialog
  const handleOpenDetails = (session: Session) => {
    setSelectedSession(session);
    setDetailsOpen(true);
  };

  // Close session details dialog
  const handleCloseDetails = () => {
    setDetailsOpen(false);
  };

  // Toggle expanded session details
  const toggleExpand = (sessionId: string) => {
    if (expandedSession === sessionId) {
      setExpandedSession(null);
    } else {
      setExpandedSession(sessionId);
    }
  };

  // Open session configuration dialog for creating a new session
  const handleOpenCreateDialog = () => {
    setSessionToEdit(null);
    setConfigDialogOpen(true);
  };

  // Open session configuration dialog for editing an existing session
  const handleOpenEditDialog = (session: Session) => {
    setSessionToEdit(session);
    setConfigDialogOpen(true);
  };

  // Close session configuration dialog
  const handleCloseConfigDialog = () => {
    setConfigDialogOpen(false);
    setSessionToEdit(null);
  };

  return (
    <Card elevation={3} sx={{ mb: 3, bgcolor: '#111111', color: darkText, borderRadius: 2, border: '1px solid #555' }}> {/* Made bg even darker, added border for visibility */}
      <CardContent sx={{ bgcolor: 'transparent' }}> {/* Ensure CardContent is transparent to Card's bg */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="div" sx={{ fontWeight: 'bold', color: darkText }}>
            Trading Sessions
          </Typography>

          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={handleOpenCreateDialog}
            size="small"
            sx={{ color: '#FFFFFF', fontWeight: 'medium' }}
          >
            Create Session
          </Button>
        </Box>

        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : sessions.length === 0 ? (
          <Box sx={{ textAlign: 'center', p: 3 }}>
            <Typography variant="body1" sx={{ color: darkSecondaryText }}>
              No active paper trading sessions
            </Typography>
          </Box>
        ) : (
          <TableContainer component={Paper} variant="outlined" sx={{ bgcolor: darkPaperBg, borderColor: darkBorder }}>
            <Table sx={{ '& .MuiTableCell-root': { borderColor: darkBorder } }}>
              <TableHead sx={{ bgcolor: darkBg }}>
                <TableRow>
                  <TableCell width="40px" sx={{ color: darkText }}></TableCell>
                  <TableCell sx={{ color: darkText, fontWeight: 'bold' }}>Session ID</TableCell>
                  <TableCell sx={{ color: darkText, fontWeight: 'bold' }}>Symbols</TableCell>
                  <TableCell sx={{ color: darkText, fontWeight: 'bold' }}>Status</TableCell>
                  <TableCell sx={{ color: darkText, fontWeight: 'bold' }}>Start Time</TableCell>
                  <TableCell sx={{ color: darkText, fontWeight: 'bold' }}>Uptime</TableCell>
                  <TableCell align="right" sx={{ color: darkText, fontWeight: 'bold' }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sessions.map((session) => (
                  <React.Fragment key={session.session_id}>
                    <TableRow hover sx={{ '&:hover': { bgcolor: darkBg } }}>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => toggleExpand(session.session_id)}
                          sx={{ color: darkText }}
                        >
                          {expandedSession === session.session_id ? (
                            <ExpandLessIcon fontSize="small" />
                          ) : (
                            <ExpandMoreIcon fontSize="small" />
                          )}
                        </IconButton>
                      </TableCell>
                      <TableCell sx={{ color: darkText }}>
                        <Typography variant="body2" fontWeight="medium" sx={{ color: darkText }}>
                          {session.session_id}
                        </Typography>
                      </TableCell>
                      <TableCell sx={{ color: darkText }}>
                        {session.symbols && session.symbols.length > 0 ? (
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {session.symbols.slice(0, 2).map((symbol) => (
                              <Chip
                                key={symbol}
                                label={symbol}
                                size="small"
                                variant="outlined"
                                sx={{
                                  color: darkText,
                                  borderColor: darkBorder,
                                  '& .MuiChip-label': { color: darkText }
                                }}
                              />
                            ))}
                            {session.symbols.length > 2 && (
                              <Chip
                                label={`+${session.symbols.length - 2}`}
                                size="small"
                                sx={{
                                  color: darkText,
                                  borderColor: darkBorder,
                                  '& .MuiChip-label': { color: darkText }
                                }}
                                variant="outlined"
                              />
                            )}
                          </Box>
                        ) : (
                          'None'
                        )}
                      </TableCell>
                      <TableCell sx={{ color: darkText }}>
                        <Chip
                          icon={getStatusIcon(session.status)}
                          label={session.status.toUpperCase()}
                          color={getStatusColor(session.status) as any}
                          size="small"
                          sx={{ '& .MuiChip-label': { fontWeight: 'medium' } }}
                        />
                      </TableCell>
                      <TableCell sx={{ color: darkText }}>{formatDate(session.start_time)}</TableCell>
                      <TableCell sx={{ color: darkText }}>{formatUptime(session.uptime_seconds)}</TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              onClick={() => handleOpenDetails(session)}
                              sx={{ color: darkText }}
                            >
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>

                          <Tooltip title="Edit Session">
                            <IconButton
                              size="small"
                              onClick={() => handleOpenEditDialog(session)}
                              sx={{ color: '#90caf9' }} // Light blue color for edit
                            >
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>

                          {session.status === 'running' ? (
                            <Tooltip title="Pause Session">
                              <IconButton
                                size="small"
                                onClick={() => pauseSession(session.session_id)}
                                sx={{ color: '#ffb74d' }} // Amber color for warning
                              >
                                <PauseIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          ) : session.status === 'paused' ? (
                            <Tooltip title="Resume Session">
                              <IconButton
                                size="small"
                                onClick={() => resumeSession(session.session_id)}
                                sx={{ color: '#66bb6a' }} // Green color for success
                              >
                                <PlayIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          ) : null}
                        </Box>
                      </TableCell>
                    </TableRow>

                    {/* Expanded session details */}
                    <TableRow>
                      <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={7}>
                        <Collapse in={expandedSession === session.session_id} timeout="auto" unmountOnExit>
                          <Box sx={{ margin: 2 }}>
                            <Typography variant="subtitle2" gutterBottom component="div" sx={{ color: darkText, fontWeight: 'bold' }}>
                              Portfolio Summary
                            </Typography>

                            {session.current_portfolio ? (
                              <Grid container spacing={2} sx={{ mb: 2 }}>
                                <Grid item xs={12} sm={3}>
                                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                    <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                      Cash Balance
                                    </Typography>
                                    <Typography variant="h6" sx={{ color: darkText }}>
                                      ${session.current_portfolio.cash_balance?.toFixed(2) || '0.00'}
                                    </Typography>
                                  </Paper>
                                </Grid>
                                <Grid item xs={12} sm={3}>
                                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                    <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                      Total Value
                                    </Typography>
                                    <Typography variant="h6" sx={{ color: darkText }}>
                                      ${session.current_portfolio.total_value?.toFixed(2) || '0.00'}
                                    </Typography>
                                  </Paper>
                                </Grid>
                                <Grid item xs={12} sm={3}>
                                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                    <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                      Open Positions
                                    </Typography>
                                    <Typography variant="h6" sx={{ color: darkText }}>
                                      {session.current_portfolio.open_positions_count || 0}
                                    </Typography>
                                  </Paper>
                                </Grid>
                                <Grid item xs={12} sm={3}>
                                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                    <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                      P&L
                                    </Typography>
                                    <Typography
                                      variant="h6"
                                      sx={{
                                        color: session.current_portfolio.profit_loss > 0
                                          ? '#66bb6a' // Green for profit 
                                          : session.current_portfolio.profit_loss < 0
                                            ? '#f44336' // Red for loss
                                            : darkText
                                      }}
                                    >
                                      {session.current_portfolio.profit_loss > 0 ? '+' : ''}
                                      ${session.current_portfolio.profit_loss?.toFixed(2) || '0.00'}
                                    </Typography>
                                  </Paper>
                                </Grid>
                              </Grid>
                            ) : (
                              <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                                No portfolio data available
                              </Typography>
                            )}

                            {session.performance_metrics && (
                              <>
                                <Typography variant="subtitle2" gutterBottom component="div" sx={{ mt: 2, color: darkText, fontWeight: 'bold' }}>
                                  Performance Metrics
                                </Typography>
                                <Grid container spacing={2}>
                                  {Object.entries(session.performance_metrics).map(([key, value]) => (
                                    <Grid item xs={6} sm={3} key={key}>
                                      <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                                        <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                          {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                                        </Typography>
                                        <Typography variant="body1" sx={{ color: darkText }}>
                                          {typeof value === 'number'
                                            ? value.toFixed(2)
                                            : typeof value === 'string'
                                              ? value
                                              : String(value)}
                                        </Typography>
                                      </Paper>
                                    </Grid>
                                  ))}
                                </Grid>
                              </>
                            )}
                          </Box>
                        </Collapse>
                      </TableCell>
                    </TableRow>
                  </React.Fragment>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {/* Session Configuration Dialog */}
        <SessionConfigurationDialog
          open={configDialogOpen}
          onClose={handleCloseConfigDialog}
          editSession={sessionToEdit}
        />

        {/* Session Details Dialog */}
        <Dialog
          open={detailsOpen}
          onClose={handleCloseDetails}
          maxWidth="md"
          fullWidth
        >
          {selectedSession && (
            <>
              <DialogTitle sx={{ bgcolor: darkBg, color: darkText }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Typography variant="h6" sx={{ color: darkText }}>Session Details</Typography>
                  <Chip
                    label={selectedSession.status.toUpperCase()}
                    color={getStatusColor(selectedSession.status) as any}
                    size="small"
                    sx={{ '& .MuiChip-label': { fontWeight: 'medium' } }}
                  />
                </Box>
              </DialogTitle>
              <DialogContent dividers sx={{ bgcolor: darkPaperBg, borderColor: darkBorder }}>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" sx={{ color: darkSecondaryText }}>
                      Session ID
                    </Typography>
                    <Typography variant="body1" gutterBottom sx={{ color: darkText }}>
                      {selectedSession.session_id}
                    </Typography>

                    <Typography variant="subtitle2" sx={{ mt: 2, color: darkSecondaryText }}>
                      Start Time
                    </Typography>
                    <Typography variant="body1" gutterBottom sx={{ color: darkText }}>
                      {formatDate(selectedSession.start_time)}
                    </Typography>

                    <Typography variant="subtitle2" sx={{ mt: 2, color: darkSecondaryText }}>
                      Uptime
                    </Typography>
                    <Typography variant="body1" gutterBottom sx={{ color: darkText }}>
                      {formatUptime(selectedSession.uptime_seconds)}
                    </Typography>

                    <Typography variant="subtitle2" sx={{ mt: 2, color: darkSecondaryText }}>
                      Last Updated
                    </Typography>
                    <Typography variant="body1" gutterBottom sx={{ color: darkText }}>
                      {formatDate(selectedSession.last_updated)}
                    </Typography>
                  </Grid>

                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle2" sx={{ color: darkSecondaryText }}>
                      Trading Symbols
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                      {selectedSession.symbols && selectedSession.symbols.length > 0 ? (
                        selectedSession.symbols.map((symbol) => (
                          <Chip
                            key={symbol}
                            label={symbol}
                            size="small"
                            variant="outlined"
                            sx={{
                              color: darkText,
                              borderColor: darkBorder,
                              '& .MuiChip-label': { color: darkText }
                            }}
                          />
                        ))
                      ) : (
                        <Typography variant="body2" sx={{ color: darkText }}>None</Typography>
                      )}
                    </Box>

                    {selectedSession.current_portfolio && (
                      <>
                        <Typography variant="subtitle2" sx={{ mt: 2, color: darkSecondaryText }}>
                          Portfolio Summary
                        </Typography>
                        <Grid container spacing={2} sx={{ mt: 0.5 }}>
                          <Grid item xs={6}>
                            <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                              <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                Cash Balance
                              </Typography>
                              <Typography variant="h6" sx={{ color: darkText }}>
                                ${selectedSession.current_portfolio.cash_balance?.toFixed(2) || '0.00'}
                              </Typography>
                            </Paper>
                          </Grid>
                          <Grid item xs={6}>
                            <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                              <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                Total Value
                              </Typography>
                              <Typography variant="h6" sx={{ color: darkText }}>
                                ${selectedSession.current_portfolio.total_value?.toFixed(2) || '0.00'}
                              </Typography>
                            </Paper>
                          </Grid>
                          <Grid item xs={6}>
                            <Paper variant="outlined" sx={{ p: 1.5, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                              <Typography variant="caption" sx={{ color: darkSecondaryText }}>
                                P&L
                              </Typography>
                              <Typography
                                variant="h6"
                                sx={{
                                  color: selectedSession.current_portfolio.profit_loss > 0
                                    ? '#66bb6a' // Green for profit 
                                    : selectedSession.current_portfolio.profit_loss < 0
                                      ? '#f44336' // Red for loss
                                      : darkText
                                }}
                              >
                                {selectedSession.current_portfolio.profit_loss > 0 ? '+' : ''}
                                ${selectedSession.current_portfolio.profit_loss?.toFixed(2) || '0.00'}
                              </Typography>
                            </Paper>
                          </Grid>
                          <Grid item xs={6}>
                            <Paper variant="outlined" sx={{ p: 1.5 }}>
                              <Typography variant="caption" color="textSecondary">
                                Return
                              </Typography>
                              <Typography
                                variant="h6"
                                sx={{
                                  color: selectedSession.current_portfolio.return_percentage > 0
                                    ? '#66bb6a' // Green for profit 
                                    : selectedSession.current_portfolio.return_percentage < 0
                                      ? '#f44336' // Red for loss
                                      : darkText
                                }}
                              >
                                {selectedSession.current_portfolio.return_percentage > 0 ? '+' : ''}
                                {selectedSession.current_portfolio.return_percentage?.toFixed(2) || '0.00'}%
                              </Typography>
                            </Paper>
                          </Grid>
                        </Grid>
                      </>
                    )}
                  </Grid>
                </Grid>
              </DialogContent>
              <DialogActions sx={{ bgcolor: darkBg, borderTop: `1px solid ${darkBorder}` }}>
                {selectedSession.status === 'running' ? (
                  <Button
                    variant="contained"
                    sx={{ bgcolor: '#f57c00', '&:hover': { bgcolor: '#ef6c00' } }} // Amber color for warning
                    startIcon={<PauseIcon />}
                    onClick={() => {
                      pauseSession(selectedSession.session_id);
                      handleCloseDetails();
                    }}
                  >
                    Pause Session
                  </Button>
                ) : selectedSession.status === 'paused' ? (
                  <Button
                    variant="contained"
                    sx={{ bgcolor: '#43a047', '&:hover': { bgcolor: '#388e3c' } }} // Green color for success
                    startIcon={<PlayIcon />}
                    onClick={() => {
                      resumeSession(selectedSession.session_id);
                      handleCloseDetails();
                    }}
                  >
                    Resume Session
                  </Button>
                ) : null}
                <Button onClick={handleCloseDetails} sx={{ color: darkText }}>Close</Button>
              </DialogActions>
            </>
          )}
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default SessionManagementPanel;
