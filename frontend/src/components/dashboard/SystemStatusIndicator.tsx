import React from 'react';
import { Box, Typography, Tooltip, CircularProgress } from '@mui/material';
import { SystemStatus } from '../../context/SystemControlContext';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import PauseCircleOutlineIcon from '@mui/icons-material/PauseCircleOutline';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

interface SystemStatusIndicatorProps {
  systemStatus: SystemStatus | null;
  isLoading: boolean;
}

const SystemStatusIndicator: React.FC<SystemStatusIndicatorProps> = ({ systemStatus, isLoading }) => {
  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CircularProgress size={20} sx={{ color: '#fff' }} />
        <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
          Loading system status...
        </Typography>
      </Box>
    );
  }

  if (!systemStatus) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <HelpOutlineIcon sx={{ color: '#aaa' }} />
        <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)' }}>
          System status unavailable
        </Typography>
      </Box>
    );
  }

  // Determine icon and color based on status
  let icon = <CheckCircleIcon sx={{ color: '#4caf50' }} />;
  let statusText = 'All systems operational';
  let bgColor = 'rgba(76, 175, 80, 0.1)';
  let borderColor = 'rgba(76, 175, 80, 0.3)';
  let tooltipText = 'All agents and sessions are running as expected';

  switch (systemStatus.status) {
    case 'running':
      icon = <CheckCircleIcon sx={{ color: '#4caf50' }} />;
      statusText = 'All systems operational';
      bgColor = 'rgba(76, 175, 80, 0.1)';
      borderColor = 'rgba(76, 175, 80, 0.3)';
      tooltipText = `${systemStatus.active_agents}/${systemStatus.total_agents} agents running • ${systemStatus.active_sessions}/${systemStatus.total_sessions} sessions active`;
      break;
    case 'partial':
      icon = <WarningIcon sx={{ color: '#ff9800' }} />;
      statusText = 'Partial system operation';
      bgColor = 'rgba(255, 152, 0, 0.1)';
      borderColor = 'rgba(255, 152, 0, 0.3)';
      tooltipText = `${systemStatus.active_agents}/${systemStatus.total_agents} agents running • ${systemStatus.active_sessions}/${systemStatus.total_sessions} sessions active`;
      break;
    case 'stopped':
      icon = <PauseCircleOutlineIcon sx={{ color: '#90caf9' }} />;
      statusText = 'System inactive';
      bgColor = 'rgba(144, 202, 249, 0.1)';
      borderColor = 'rgba(144, 202, 249, 0.3)';
      tooltipText = 'All agents and sessions are currently stopped';
      break;
    case 'error':
      icon = <ErrorIcon sx={{ color: '#f44336' }} />;
      statusText = 'System error detected';
      bgColor = 'rgba(244, 67, 54, 0.1)';
      borderColor = 'rgba(244, 67, 54, 0.3)';
      tooltipText = 'One or more critical components have encountered errors';
      break;
    default:
      break;
  }

  // Health metrics
  const cpuUsage = systemStatus.cpu_usage || systemStatus.health_metrics?.cpu_usage;
  const memoryUsage = systemStatus.memory_usage || systemStatus.health_metrics?.memory_usage;
  const diskUsage = systemStatus.disk_usage || systemStatus.health_metrics?.disk_usage;

  // Add health metrics to tooltip if available
  if (cpuUsage !== undefined || memoryUsage !== undefined || diskUsage !== undefined) {
    tooltipText += '\n\nHealth Metrics:';
    if (cpuUsage !== undefined) tooltipText += `\nCPU: ${cpuUsage}%`;
    if (memoryUsage !== undefined) tooltipText += `\nMemory: ${memoryUsage}%`;
    if (diskUsage !== undefined) tooltipText += `\nDisk: ${diskUsage}%`;
  }

  // Add uptime if available
  if (systemStatus.uptime_seconds !== undefined && systemStatus.start_time) {
    const hours = Math.floor(systemStatus.uptime_seconds / 3600);
    const minutes = Math.floor((systemStatus.uptime_seconds % 3600) / 60);
    const seconds = Math.floor(systemStatus.uptime_seconds % 60);
    
    const uptimeText = `${hours}h ${minutes}m ${seconds}s`;
    tooltipText += `\n\nUptime: ${uptimeText}\nStarted: ${new Date(systemStatus.start_time).toLocaleString()}`;
  }

  return (
    <Tooltip 
      title={<Typography style={{ whiteSpace: 'pre-line' }}>{tooltipText}</Typography>} 
      arrow
      placement="top"
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
          backgroundColor: bgColor,
          border: `1px solid ${borderColor}`,
          borderRadius: 1,
          px: 2,
          py: 1,
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          '&:hover': {
            filter: 'brightness(1.1)',
          }
        }}
      >
        {icon}
        <Box>
          <Typography 
            variant="body2" 
            sx={{ 
              fontWeight: 'bold', 
              color: '#fff',
              fontSize: '0.9rem'
            }}
          >
            {statusText}
          </Typography>
          
          <Typography
            variant="caption"
            sx={{
              color: 'rgba(255,255,255,0.7)',
              display: 'block',
              fontSize: '0.75rem'
            }}
          >
            {systemStatus.active_agents}/{systemStatus.total_agents} agents • {systemStatus.active_sessions}/{systemStatus.total_sessions} sessions
          </Typography>
        </Box>
      </Box>
    </Tooltip>
  );
};

export default SystemStatusIndicator;
