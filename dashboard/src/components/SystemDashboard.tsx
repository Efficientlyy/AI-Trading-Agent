import React from 'react';
import { Box, Typography } from '@mui/material';
import { SystemControlPanel } from './SystemControlPanel';
import AgentFlowGrid from './AgentFlowGrid/AgentFlowGrid';
import { useSystemControl } from '../context/SystemControlContext';

const SystemDashboard: React.FC = () => {
  const {
    systemStatus,
    startSystem,
    stopSystem,
    pauseAllSessions,
  } = useSystemControl();

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" sx={{ mb: 2, fontWeight: 'bold' }}>
        System Control Dashboard
      </Typography>
      <SystemControlPanel
        onPauseAll={pauseAllSessions}
        onResumeAll={startSystem}
        onStopAll={stopSystem}
        systemStatus={systemStatus?.status || 'unknown'}
      />
      <Box sx={{ mt: 3 }}>
        <Box sx={{ color: 'red', fontWeight: 'bold', fontSize: 22, mb: 2 }}>
          DEBUG: SystemDashboard (new grid version) is rendering!
        </Box>
        <AgentFlowGrid />
      </Box>
    </Box>
  );
};

export default SystemDashboard;


export default SystemDashboard;
