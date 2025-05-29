import React, { useContext } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Typography,
  IconButton,
  Box,
  Grid,
  Paper,
  Chip,
  Divider,
  useMediaQuery,
  Tooltip,
  useTheme,
  CircularProgress
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';

// Status indicator helper function
const getStatusColor = (status: string) => {
  switch(status?.toLowerCase()) {
    case 'online':
    case 'active':
    case 'running':
      return '#4caf50'; // Green for active/running components
    case 'warning':
    case 'degraded':
    case 'slow':
    case 'initializing':
      return '#ff9800'; // Yellow/orange for warning or degraded status
    case 'error':
    case 'offline':
    case 'stopped':
      return '#f44336'; // Red for error or stopped status
    default:
      return '#9e9e9e'; // Grey for unknown status
  }
};

// Status indicator icon helper function
const getStatusIcon = (status: string) => {
  switch(status?.toLowerCase()) {
    case 'online':
    case 'active':
    case 'running':
      return <CheckCircleOutlineIcon fontSize="small" />
    case 'warning':
    case 'degraded':
    case 'slow':
    case 'initializing':
      return <WarningAmberIcon fontSize="small" />
    case 'error':
    case 'offline':
    case 'stopped':
      return <ErrorOutlineIcon fontSize="small" />
    default:
      return null;
  }
};

// Extend the PipelineComponentData with additional properties
interface PipelineComponentData {
  name: string;
  status: string;
  metrics: Record<string, any>;
  is_active?: boolean;
  last_update?: string;
  logs?: string;
  errors?: string[];
}

interface PipelineComponentDetailProps {
  component: PipelineComponentData | null;
  open: boolean;
  onClose: () => void;
}

// Helper function to format keys from snake_case to Title Case
const formatKey = (key: string): string => {
  return key
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

// Helper function to format values based on their type
const formatValue = (value: any, key: string): string => {
  if (typeof value === 'number') {
    // Format percentages
    if (value >= 0 && value <= 1 && 
        ['rate', 'ratio', 'percentage', 'accuracy'].some(term => 
          key.toLowerCase().includes(term))) {
      return `${(value * 100).toFixed(2)}%`;
    }
    
    // Format timestamps (assume large numbers are timestamps)
    if (value > 1000000000000) {
      return new Date(value).toLocaleString();
    }
    
    // Format decimal numbers
    if (value % 1 !== 0) {
      return value.toFixed(4);
    }
  }
  
  // Format boolean values
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }
  
  // Return string representation for everything else
  return String(value);
};

// Status chip component
const StatusChip: React.FC<{ status: string }> = ({ status }) => {
  const getStatusColor = (): 'success' | 'error' | 'warning' | 'default' => {
    switch (status.toLowerCase()) {
      case 'online':
      case 'running':
      case 'active':
        return 'success';
      case 'error':
      case 'failed':
        return 'error';
      case 'processing':
      case 'initializing':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <Chip 
      label={status} 
      color={getStatusColor()} 
      size="small" 
      sx={{ fontWeight: 'bold', fontSize: '0.75rem' }}
    />
  );
};

const PipelineComponentDetail: React.FC<PipelineComponentDetailProps> = ({component, open, onClose}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));
  
  // Determine the grid size based on screen size
  const metricGridSize = isMobile ? 12 : isTablet ? 6 : 4;
  if (!component) {
    return null;
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { 
          bgcolor: theme.palette.mode === 'dark' ? '#121212' : '#fff', 
          color: theme.palette.mode === 'dark' ? '#fff' : '#333',
          borderRadius: 2,
          overflow: 'hidden'
        }
      }}
    >
      <DialogTitle sx={{ position: 'relative', pb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6">{component.name}</Typography>
            <Tooltip title={`Status: ${component.status}`}>
              <Chip 
                label={component.status} 
                size="small"
                icon={getStatusIcon(component.status) || undefined}
                sx={{ 
                  bgcolor: `${getStatusColor(component.status)}20`, 
                  color: getStatusColor(component.status),
                  fontWeight: 500,
                  '.MuiChip-icon': {
                    color: 'inherit'
                  }
                }} 
              />
            </Tooltip>
          </Box>
          <IconButton 
            onClick={onClose} 
            sx={{ 
              color: theme.palette.mode === 'dark' ? 'white' : '#333',
              '&:hover': { bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' }
            }}
          >
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        <Box sx={{ mb: 3 }}>
          <Paper sx={{ 
            p: 2, 
            bgcolor: theme.palette.mode === 'dark' ? '#1a2035' : '#f5f5f5', 
            borderRadius: 2,
            boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 2px 8px rgba(0,0,0,0.05)'
          }}>
            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
              Component Status
            </Typography>
            <Box sx={{ 
              display: 'flex', 
              flexWrap: 'wrap', 
              gap: 3,
              '& > div': {
                minWidth: isMobile ? '100%' : '140px'
              }
            }}>
              <Box>
                <Typography variant="body2" sx={{ color: theme.palette.mode === 'dark' ? '#aaa' : '#666', mb: 0.5 }}>
                  Status
                </Typography>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 1,
                  color: getStatusColor(component.status)
                }}>
                  {getStatusIcon(component.status)}
                  <Typography variant="body1" sx={{ fontWeight: 500 }}>{component.status}</Typography>
                </Box>
              </Box>
              
              {component.is_active !== undefined && (
                <Box>
                  <Typography variant="body2" sx={{ color: theme.palette.mode === 'dark' ? '#aaa' : '#666', mb: 0.5 }}>
                    Active
                  </Typography>
                  <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: 1,
                    color: component.is_active ? '#4caf50' : '#f44336'
                  }}>
                    {component.is_active ? 
                      <CheckCircleOutlineIcon fontSize="small" /> : 
                      <ErrorOutlineIcon fontSize="small" />
                    }
                    <Typography variant="body1" sx={{ fontWeight: 500 }}>
                      {component.is_active ? 'Yes' : 'No'}
                    </Typography>
                  </Box>
                </Box>
              )}
              
              {component.last_update && (
                <Box>
                  <Typography variant="body2" sx={{ color: theme.palette.mode === 'dark' ? '#aaa' : '#666', mb: 0.5 }}>
                    Last Update
                  </Typography>
                  <Typography variant="body1" sx={{ fontWeight: 500 }}>
                    {new Date(component.last_update).toLocaleString()}
                  </Typography>
                </Box>
              )}
            </Box>
          </Paper>
        </Box>
        
        <Divider sx={{ 
          my: 3, 
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' 
        }} />
        
        <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 500 }}>
          Component Metrics
        </Typography>
        
        {!component.metrics || Object.keys(component.metrics).length === 0 ? (
          <Paper sx={{ 
            p: 3, 
            bgcolor: theme.palette.mode === 'dark' ? '#1a2035' : '#f5f5f5', 
            borderRadius: 2, 
            textAlign: 'center',
            boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 2px 8px rgba(0,0,0,0.05)'
          }}>
            <Typography sx={{ color: theme.palette.mode === 'dark' ? '#aaa' : '#666', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
              <CircularProgress size={16} thickness={5} sx={{ opacity: 0.7 }} />
              No metrics available for this component
            </Typography>
          </Paper>
        ) : (
          <Grid container spacing={2}>
            {Object.entries(component.metrics).map(([key, value]) => {
              // Format keys from snake_case or camelCase to Title Case
              const formattedKey = key
                .replace(/_/g, ' ')
                .replace(/([A-Z])/g, ' $1')
                .split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
              
              // Format values based on type
              let displayValue = value;
              let isPercentage = false;
              let isPositive = false;
              
              if (typeof value === 'number') {
                // If it's a percentage value
                if (key.includes('rate') || key.includes('ratio') || key.includes('percent')) {
                  isPercentage = true;
                  isPositive = value > 0.7; // Consider high percentages as good
                  displayValue = `${(value * 100).toFixed(1)}%`;
                } 
                // If it's a time value
                else if (key.includes('time') || key.includes('duration')) {
                  // Format as appropriate time unit
                  if (value < 1000) {
                    displayValue = `${value.toFixed(2)}ms`;
                  } else if (value < 60000) {
                    displayValue = `${(value / 1000).toFixed(2)}s`;
                  } else {
                    displayValue = `${(value / 60000).toFixed(2)}min`;
                  }
                }
                // If it's a decimal that should be formatted
                else if (value % 1 !== 0) {
                  displayValue = value.toFixed(2);
                }
              } else if (typeof value === 'boolean') {
                displayValue = value ? 'Yes' : 'No';
                isPositive = value;
              }
              
              return (
                <Grid item xs={metricGridSize} sm={metricGridSize === 12 ? 6 : metricGridSize} md={4} key={key}>
                  <Paper sx={{ 
                    p: 2, 
                    bgcolor: theme.palette.mode === 'dark' ? '#1a2035' : '#f5f5f5', 
                    height: '100%',
                    borderRadius: 2,
                    transition: 'all 0.2s ease',
                    boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 2px 8px rgba(0,0,0,0.05)',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                    }
                  }}>
                    <Typography variant="body2" sx={{ 
                      color: theme.palette.mode === 'dark' ? '#aaa' : '#666', 
                      mb: 1 
                    }}>
                      {formattedKey}
                    </Typography>
                    <Typography variant="h6" sx={{
                      color: isPercentage ? (isPositive ? '#4caf50' : '#f44336') : 'inherit',
                      fontWeight: 500
                    }}>
                      {displayValue}
                    </Typography>
                  </Paper>
                </Grid>
              );
            })}
          </Grid>
        )}
        
        {component.logs && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 500 }}>
              Component Logs
            </Typography>
            <Paper sx={{ 
              p: 2, 
              bgcolor: theme.palette.mode === 'dark' ? '#1a2035' : '#f5f5f5', 
              borderRadius: 2,
              boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 2px 8px rgba(0,0,0,0.05)'
            }}>
              <Box sx={{ 
                bgcolor: theme.palette.mode === 'dark' ? '#121212' : '#f1f1f1', 
                p: 2, 
                borderRadius: 1, 
                maxHeight: 200, 
                overflowY: 'auto',
                fontFamily: 'monospace',
                fontSize: '0.85rem',
                whiteSpace: 'pre-wrap',
                color: theme.palette.mode === 'dark' ? '#e0e0e0' : '#333'
              }}>
                {component.logs}
              </Box>
            </Paper>
          </Box>
        )}
        
        {component.errors && Array.isArray(component.errors) && component.errors.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 2, color: '#f44336', fontWeight: 500 }}>
              Errors
            </Typography>
            <Paper sx={{ 
              p: 2, 
              bgcolor: theme.palette.mode === 'dark' ? '#1a2035' : '#f5f5f5', 
              borderRadius: 2,
              boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 2px 8px rgba(0,0,0,0.05)'
            }}>
              {component.errors.map((error: string, index: number) => (
                <Box key={index} sx={{ 
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(244, 67, 54, 0.1)' : 'rgba(244, 67, 54, 0.05)', 
                  p: 2, 
                  borderRadius: 1,
                  mb: index < (component.errors?.length || 0) - 1 ? 2 : 0,
                  color: '#f44336',
                  fontFamily: 'monospace',
                  fontSize: '0.85rem',
                  whiteSpace: 'pre-wrap',
                  border: '1px solid rgba(244, 67, 54, 0.2)'
                }}>
                  {error}
                </Box>
              ))}
            </Paper>
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default PipelineComponentDetail;
