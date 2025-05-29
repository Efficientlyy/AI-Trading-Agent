import { Box, Grid } from '@mui/material';
import React from 'react';
import { Agent } from '../../context/SystemControlContext';
import AgentCard from './AgentCard';

// A simpler, more reliable grid implementation without D3/React conflicts
interface AgentGridProps {
  agents: Agent[];
  onStartAgent: (agentId: string) => void;
  onStopAgent: (agentId: string) => void;
}

const AgentGrid: React.FC<AgentGridProps> = ({ agents, onStartAgent, onStopAgent }) => {
  return (
    <Box
      sx={{
        width: '100%',
        minHeight: '70vh',
        p: 2,
        background: '#151a24',
        borderRadius: 2,
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
        position: 'relative',
        overflow: 'auto',
        // Add grid pattern background for professional look
        backgroundImage: `
          linear-gradient(rgba(35, 43, 59, 0.3) 1px, transparent 1px),
          linear-gradient(90deg, rgba(35, 43, 59, 0.3) 1px, transparent 1px)
        `,
        backgroundSize: '20px 20px'
      }}
    >
      {/* Visual debug marker (can be removed later) */}
      <Box sx={{ color: 'red', fontWeight: 'bold', mb: 2 }}>
        Agent Grid is rendering!
      </Box>

      <Grid container spacing={3}>
        {agents.map(agent => (
          <Grid item xs={12} sm={6} md={4} key={agent.agent_id}>
            <AgentCard
              id={agent.agent_id}
              data={{
                agent: agent,
                onStart: onStartAgent,
                onStop: onStopAgent
              }}
              selected={false} // Ensure selected is passed
            />

            {/* Add visual connector arrows via CSS when needed */}
            {agents.indexOf(agent) < agents.length - 1 && (
              <Box
                sx={{
                  position: 'absolute',
                  display: { xs: 'none', md: 'block' }, // Hide on mobile
                  borderTop: '2px dashed #6ff',
                  width: '40px',
                  right: '-40px',
                  top: '50%',
                  '&::after': {
                    content: '""',
                    position: 'absolute',
                    right: 0,
                    top: '-6px',
                    width: 0,
                    height: 0,
                    borderTop: '5px solid transparent',
                    borderBottom: '5px solid transparent',
                    borderLeft: '8px solid #6ff',
                  }
                }}
              />
            )}
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default AgentGrid;