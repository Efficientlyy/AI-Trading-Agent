import { Box, Container, Typography } from '@mui/material';
import { ReactFlowProvider } from '@xyflow/react'; // Correctly added import
import React from 'react';
import { SystemControlProvider } from '../../context/SystemControlContext';
import AgentFlowGrid from '../AgentFlowGrid/AgentFlowGrid';
import ActivityFeed from './ActivityFeed';
import PerformanceMetricsPanel from './PerformanceMetricsPanel';
import SessionManagementPanel from './SessionManagementPanel';
import SystemControlPanel from './SystemControlPanel';

const SystemControlDashboard: React.FC = () => {
  return (
    <SystemControlProvider>
      <ReactFlowProvider> {/* Added ReactFlowProvider wrapper */}
        <Container maxWidth="xl" sx={{ bgcolor: '#121212', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ py: 3, flexGrow: 1, display: 'flex', flexDirection: 'column', bgcolor: '#171717' }}> {/* Added explicit dark bg to inner box */}
            <Typography variant="h4" component="h1" gutterBottom sx={{ color: '#fff' }}> {/* Ensure text is visible */}
              Paper Trading System Control
            </Typography>
            <Typography variant="subtitle1" color="textSecondary" paragraph sx={{ color: 'rgba(255,255,255,0.7)' }}> {/* Ensure text is visible */}
              Monitor and control the paper trading system, agents, and sessions
            </Typography>

            {/* System Control Panel */}
            <SystemControlPanel />

            {/* New Modular Agent Flow Grid with Visual Layout */}
            <Box sx={{ mt: 3, mb: 3, bgcolor: '#1e1e1e', p: 2, borderRadius: 2 }}> {/* Added dark bg */}
              <Typography variant="h6" component="h2" gutterBottom sx={{ color: '#fff' }}>
                Agent Flow Grid
              </Typography>
              <AgentFlowGrid />
            </Box>

            {/* Session Management Panel */}
            {/* Assuming SessionManagementPanel's root Card already has darkPaperBg */}
            <SessionManagementPanel />

            {/* Performance Metrics Panel */}
            {/* Assuming PerformanceMetricsPanel's root Card already has darkBg */}
            <PerformanceMetricsPanel />

            {/* Activity Feed */}
            {/* Assuming ActivityFeed's root Card already has darkPaperBg */}
            <ActivityFeed />
          </Box>
        </Container>
      </ReactFlowProvider> {/* Closing ReactFlowProvider wrapper */}
    </SystemControlProvider>
  );
};

export default SystemControlDashboard;
