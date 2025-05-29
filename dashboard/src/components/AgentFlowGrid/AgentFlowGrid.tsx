import React from 'react';
import ReactFlow, { Background, MiniMap, Controls, Node, Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useSystemControl } from '../../context/SystemControlContext';
import AgentCard from './AgentCard';

// Helper to render AgentCard in a React Flow node
const AgentNode = ({ data }: any) => <AgentCard {...data} />;

const nodeTypes = { agent: AgentNode };

const AgentFlowGrid: React.FC = () => {
  const { agents, startAgent, stopAgent } = useSystemControl();

  // Layout nodes in a grid (future: dynamic/flow layout)
  const nodes: Node[] = agents.map((agent, idx) => ({
    id: agent.agent_id,
    type: 'agent',
    position: { x: 60 + (idx % 3) * 380, y: 60 + Math.floor(idx / 3) * 260 },
    data: { agent, onStart: startAgent, onStop: stopAgent },
  }));

  // For now, draw simple arrows from each node to the next (can be replaced with real dependencies later)
  const edges: Edge[] = agents.slice(1).map((agent, idx) => ({
    id: `e${agents[idx].agent_id}-${agent.agent_id}`,
    source: agents[idx].agent_id,
    target: agent.agent_id,
    animated: true,
    style: { stroke: '#6ff' },
  }));

  return (
    <div style={{ width: '100%', height: '70vh', background: '#151a24', borderRadius: 12 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        panOnScroll
        zoomOnScroll
        minZoom={0.5}
        maxZoom={1.5}
      >
        <MiniMap nodeColor={() => '#6ff'} maskColor="#232b3b" />
        <Controls />
        <Background color="#232b3b" gap={16} />
      </ReactFlow>
    </div>
  );
};

export default AgentFlowGrid;
