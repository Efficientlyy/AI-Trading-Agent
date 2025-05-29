import React from 'react';
import { Card, CardContent, Typography, Chip, Button, Box } from '@mui/material';
import { useSystemControl, Agent } from '../../context/SystemControlContext';
import AgentLogViewer from '../AgentLogViewer';

interface AgentCardProps {
  agent: Agent;
  onStart: (agentId: string) => void;
  onStop: (agentId: string) => void;
}

const statusColor = (status: string) => {
  switch (status) {
    case 'running': return 'success';
    case 'stopped': return 'default';
    case 'error': return 'error';
    case 'initializing': return 'warning';
    default: return 'default';
  }
};

const AgentCard: React.FC<AgentCardProps> = ({ agent, onStart, onStop }) => {
  return (
    <Card sx={{ minWidth: 300, maxWidth: 340, m: 1, background: '#181f2a', color: '#fff', borderRadius: 3, boxShadow: 4 }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>{agent.name}</Typography>
          <Chip label={agent.status} color={statusColor(agent.status)} size="small" />
        </Box>
        <Typography variant="subtitle2" sx={{ mt: 0.5, color: '#aaa' }}>{agent.type}</Typography>
        <Typography variant="body2" sx={{ mt: 0.5 }}>Last Active: {agent.last_active}</Typography>
        {agent.symbols && agent.symbols.length > 0 && (
          <Typography variant="body2" sx={{ mt: 0.5 }}>Symbols: {agent.symbols.join(', ')}</Typography>
        )}
        <Box sx={{ mt: 1, mb: 1 }}>
          {agent.status === 'running' ? (
            <Button variant="contained" color="error" size="small" onClick={() => onStop(agent.agent_id)}>Stop</Button>
          ) : (
            <Button variant="contained" color="success" size="small" onClick={() => onStart(agent.agent_id)}>Start</Button>
          )}
        </Box>
        {agent.performance_metrics && (
          <Box sx={{ mb: 1 }}>
            <Typography variant="caption" sx={{ color: '#6ff' }}>
              Win Rate: {agent.performance_metrics.win_rate ?? '-'} | Profit Factor: {agent.performance_metrics.profit_factor ?? '-'}
            </Typography><br />
            <Typography variant="caption" sx={{ color: '#6ff' }}>
              Avg PnL: {agent.performance_metrics.avg_profit_loss ?? '-'} | Max Drawdown: {agent.performance_metrics.max_drawdown ?? '-'}
            </Typography>
          </Box>
        )}
        <Box sx={{ bgcolor: '#232b3b', mt: 1, p: 1, borderRadius: 2, maxHeight: 100, overflow: 'auto' }}>
          <Typography variant="caption" sx={{ color: '#aaa' }}>Logs:</Typography>
          <AgentLogViewer agentId={agent.agent_id} autoRefresh={true} maxLines={6} />
        </Box>
      </CardContent>
    </Card>
  );
};

export default AgentCard;
