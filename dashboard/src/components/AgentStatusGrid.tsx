import React from 'react';
import AgentStatusGridBase from '../../../frontend/src/components/dashboard/AgentStatusGrid';
import { useSystemControl, Agent } from '../../../frontend/src/context/SystemControlContext';

interface DashboardAgentStatusGridProps {
  onSelectAgent: (agentId: string) => void;
  selectedAgentId: string;
  startAgent: (agentId: string) => void;
  stopAgent: (agentId: string) => void;
  isLoading: boolean;
}

const DashboardAgentStatusGrid: React.FC<DashboardAgentStatusGridProps> = ({
  onSelectAgent,
  selectedAgentId,
  startAgent,
  stopAgent,
  isLoading,
}) => {
  const { agents } = useSystemControl();

  return (
    <div>
      {/* Render a table/grid of agents with clickable rows */}
      <table style={{ width: '100%', background: 'rgba(30,34,45,0.9)', color: '#fff', borderRadius: 8 }}>
        <thead>
          <tr>
            <th style={{ padding: 8 }}>Name</th>
            <th>Status</th>
            <th>Type</th>
            <th>Last Active</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {agents.map(agent => (
            <tr
              key={agent.agent_id}
              onClick={() => onSelectAgent(agent.agent_id)}
              style={{
                background: agent.agent_id === selectedAgentId ? '#263047' : undefined,
                cursor: 'pointer',
                fontWeight: agent.agent_id === selectedAgentId ? 'bold' : undefined,
              }}
            >
              <td style={{ padding: 8 }}>{agent.name}</td>
              <td>{agent.status}</td>
              <td>{agent.type}</td>
              <td>{agent.last_active}</td>
              <td>
                {agent.status === 'running' ? (
                  <button onClick={e => { e.stopPropagation(); stopAgent(agent.agent_id); }}>Stop</button>
                ) : (
                  <button onClick={e => { e.stopPropagation(); startAgent(agent.agent_id); }}>Start</button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {isLoading && <div style={{ color: '#ff0', marginTop: 8 }}>Loading...</div>}
    </div>
  );
};

export default DashboardAgentStatusGrid;
