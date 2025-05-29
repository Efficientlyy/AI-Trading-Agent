import React from 'react';
import { Box, styled, Paper, Typography, Chip, useTheme, useMediaQuery, Tooltip } from '@mui/material';
import { keyframes } from '@mui/system';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';

// Define animation keyframes
const flowAnimation = keyframes`
  0% {
    transform: translateX(0) scale(0.8);
    opacity: 0;
  }
  20% {
    opacity: 1;
  }
  80% {
    opacity: 1;
  }
  100% {
    transform: translateX(calc(100% - 10px)) scale(0.8);
    opacity: 0;
  }
`;

const pulseAnimation = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(66, 153, 225, 0.6);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(66, 153, 225, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(66, 153, 225, 0);
  }
`;

// Styled components
const FlowContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  height: '10px',
  width: '100%',
  marginTop: '12px',
  marginBottom: '12px',
  backgroundColor: 'rgba(30, 41, 59, 0.5)',
  borderRadius: '4px',
  overflow: 'hidden',
}));

const FlowParticle = styled(Box)<{ delay: number; active: boolean }>(({ delay, active, theme }) => ({
  position: 'absolute',
  width: '10px',
  height: '10px',
  backgroundColor: '#60a5fa',
  borderRadius: '50%',
  opacity: 0,
  animation: active ? `${flowAnimation} 2s ease-in-out infinite` : 'none',
  animationDelay: `${delay}ms`,
}));

const ComponentContainer = styled(Paper)<{ active: boolean; status: string }>(({ active, status, theme }) => ({
  padding: theme.spacing(1.5),
  backgroundColor: '#1a2035',
  borderRadius: '8px',
  position: 'relative',
  transition: 'all 0.3s ease',
  border: `2px solid ${
    status === 'online' ? '#10b981' : 
    status === 'offline' ? '#6b7280' : 
    status === 'error' ? '#ef4444' : 
    status === 'processing' ? '#3b82f6' : '#6b7280'
  }`,
  boxShadow: active ? `0 0 10px ${
    status === 'online' ? 'rgba(16, 185, 129, 0.6)' : 
    status === 'offline' ? 'rgba(107, 114, 128, 0.6)' : 
    status === 'error' ? 'rgba(239, 68, 68, 0.6)' : 
    status === 'processing' ? 'rgba(59, 130, 246, 0.6)' : 'rgba(107, 114, 128, 0.6)'
  }` : 'none',
  animation: active && status === 'processing' ? `${pulseAnimation} 2s infinite` : 'none',
  cursor: 'pointer',
  '&:hover': {
    transform: 'translateY(-3px)',
    boxShadow: `0 6px 12px ${
      status === 'online' ? 'rgba(16, 185, 129, 0.3)' : 
      status === 'offline' ? 'rgba(107, 114, 128, 0.3)' : 
      status === 'error' ? 'rgba(239, 68, 68, 0.3)' : 
      status === 'processing' ? 'rgba(59, 130, 246, 0.3)' : 'rgba(107, 114, 128, 0.3)'
    }`,
  },
}));

const StatusIndicator = styled(Box)<{ status: string }>(({ status, theme }) => ({
  position: 'absolute',
  top: '8px',
  right: '8px',
  width: '10px',
  height: '10px',
  borderRadius: '50%',
  backgroundColor: 
    status === 'online' ? '#10b981' : 
    status === 'offline' ? '#6b7280' : 
    status === 'error' ? '#ef4444' : 
    status === 'processing' ? '#3b82f6' : '#6b7280',
}));

// Props interface
interface PipelineComponentProps {
  name: string;
  metrics: Record<string, any>;
  status: 'online' | 'offline' | 'error' | 'processing';
  isActive: boolean;
  onClick?: () => void;
}

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
    case 'processing':
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
    case 'processing':
      return <WarningAmberIcon fontSize="small" />
    case 'error':
    case 'offline':
    case 'stopped':
      return <ErrorOutlineIcon fontSize="small" />
    default:
      return null;
  }
};

interface PipelineFlowAnimationProps {
  components: {
    name: string;
    status: 'online' | 'offline' | 'error' | 'processing';
    isActive: boolean;
    metrics?: Record<string, any>;
  }[];
  flowActive: boolean;
  onComponentClick?: (componentName: string) => void;
}

// Component that displays a pipeline component
const PipelineComponent: React.FC<PipelineComponentProps> = ({ 
  name, 
  metrics, 
  status, 
  isActive,
  onClick 
}) => {
  const displayMetrics = Object.entries(metrics)
    .filter(([key]) => key !== 'id' && key !== 'name')
    .map(([key, value]) => {
      // Format key from snake_case to Title Case
      const formattedKey = key
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
      
      return `${formattedKey}: ${value}`;
    })
    .join(' | ');

  return (
    <Tooltip title={displayMetrics} arrow placement="top">
      <ComponentContainer active={isActive} status={status} onClick={onClick}>
        <StatusIndicator status={status} />
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#fff' }}>
          {name}
        </Typography>
        <Typography variant="body2" sx={{ color: '#94a3b8', mt: 0.5, fontSize: '0.75rem' }}>
          {Object.entries(metrics).slice(0, 2).map(([key, value]) => {
            const formattedKey = key
              .split('_')
              .map(word => word.charAt(0).toUpperCase() + word.slice(1))
              .join(' ');
            
            return `${formattedKey}: ${value}`;
          }).join(' | ')}
        </Typography>
      </ComponentContainer>
    </Tooltip>
  );
};

// Main component that displays the pipeline with animations
const PipelineFlowAnimation: React.FC<PipelineFlowAnimationProps> = ({ 
  components, 
  flowActive,
  onComponentClick
}) => {
  // Create 5 particles with different delays
  const particleDelays = [0, 400, 800, 1200, 1600];

  return (
    <Box sx={{ mt: 2, width: '100%' }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {components.map((component, index) => (
          <React.Fragment key={component.name}>
            <PipelineComponent 
              name={component.name}
              metrics={component.metrics || {}}
              status={component.status}
              isActive={component.isActive}
              onClick={() => onComponentClick && onComponentClick(component.name)}
            />
            
            {/* Flow animation between components, except after the last component */}
            {index < components.length - 1 && (
              <FlowContainer>
                {particleDelays.map((delay, i) => (
                  <FlowParticle 
                    key={i} 
                    delay={delay} 
                    active={flowActive && components[index].isActive}
                  />
                ))}
              </FlowContainer>
            )}
          </React.Fragment>
        ))}
      </Box>
    </Box>
  );
};

export default PipelineFlowAnimation;
