import {
  addEdge,
  Background,
  BackgroundVariant,
  Connection,
  Controls,
  Edge,
  MarkerType,
  Panel,
  useReactFlow,
  MiniMap,
  Node,
  Position,
  ReactFlow,
  useEdgesState,
  useNodesState
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
// Add custom styles for enhanced data flow visualization
import './flow-styles.css';
import React, { useCallback, useEffect, useMemo, useState } from 'react';

import { Agent as AgentType, useSystemControl } from '../../context/SystemControlContext';
import AgentCard from './AgentCard';

// Define a type for the custom node data
interface AgentNodeData {
  agent: AgentType;
  onStart: (agentId: string) => void;
  onStop: (agentId: string) => void;
  [key: string]: any; // Re-add index signature
}

const nodeWidth = 320;
const nodeHeight = 280; // Approximate height of AgentCard, adjust as needed

// Utility function to determine edge color based on agent status and type
const getEdgeColor = (status: string, sourceAgent?: AgentType, targetAgent?: AgentType): string => {
  // Special coloring for sentiment analysis data flows
  if (sourceAgent?.agent_role === 'specialized_sentiment' && status === 'running') {
    return '#4CAF50'; // Green for sentiment data flows when running
  }
  
  // Special coloring for data flowing into sentiment analysis agents
  if (targetAgent?.agent_role === 'specialized_sentiment' && status === 'running') {
    return '#8BC34A'; // Light green for data flowing into sentiment agents
  }
  
  switch (status) {
    case 'running':
      return '#FFFF00'; // Bright yellow for running
    case 'error':
      return '#FF4444'; // Red for error
    case 'initializing':
      return '#FFAA00'; // Orange for initializing
    default:
      return 'rgba(255, 255, 0, 0.5)'; // Dimmer yellow for inactive
  }
};

// Utility function to get edge width based on status
const getEdgeWidth = (status: string): number => {
  return status === 'running' ? 2.5 : 1.5;
};

// Utility function to get node border color based on status
const getNodeBorderColor = (status: string): string => {
  switch (status) {
    case 'running':
      return '#4caf50';
    case 'error':
      return '#f44336';
    case 'initializing':
      return '#ff9800';
    default:
      return '#2c3e50'; // Default border color
  }
};

const AgentFlowGrid: React.FC = () => {
  const { agents, startAgent, stopAgent } = useSystemControl();

  // Correctly type state hooks
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<AgentNodeData>>([]); // Reverted type
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]); // Correct to Edge
  const [isLayoutReady, setIsLayoutReady] = useState(false); // New state for layout readiness

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Define custom node types
  const nodeTypes = useMemo(() => ({ agentNode: AgentCard }), []);

  useEffect(() => {
    if (agents && agents.length > 0) {
      // Fixed dimensions for layout
      const columnGap = 400; // Increased horizontal gap for better spacing
      const rowGap = 50;     // Smaller row gap for more compact arrangement
      const startX = 100;    // Left margin
      const startY = 100;    // Top margin

      // Define column positions for a 3-column layout
      const columnPositions = [startX, startX + nodeWidth + columnGap, startX + 2 * (nodeWidth + columnGap)];
      
      // Clear categorization of agents by their primary role
      const columnAgents: AgentType[][] = [[], [], []]; // [left, middle, right]
      
      // Group agents by their role
      agents.forEach(agent => {
        const role = agent.agent_role || '';
        
        if (role === 'decision_aggregator') {
          columnAgents[1].push(agent);  // Middle column
        } else if (role === 'execution_broker') {
          columnAgents[2].push(agent);  // Right column
        } else {
          columnAgents[0].push(agent);  // Left column (specialized agents)
        }
      });
      
      // Sort agents in each column by type for predictable ordering
      columnAgents.forEach(column => {
        column.sort((a, b) => {
          // Sort by role for predictability
          const roleA = a.agent_role || '';
          const roleB = b.agent_role || '';
          return roleA.localeCompare(roleB);
        });
      });
      
      // Calculate heights for each column
      const columnHeights = columnAgents.map(column => column.length * (nodeHeight + rowGap));
      
      // Create nodes with proper positioning
      const initialNodes: Node<AgentNodeData>[] = [];
      
      // Generate nodes for each column with proper vertical positioning
      columnAgents.forEach((column, columnIndex) => {
        column.forEach((agent, rowIndex) => {
          // Position this node at its calculated position
          const position = {
            x: columnPositions[columnIndex],
            y: startY + rowIndex * (nodeHeight + rowGap)
          };
          
          // Adjust y-position to center columns with fewer nodes
          const maxHeight = Math.max(...columnHeights);
          const columnOffset = (maxHeight - columnHeights[columnIndex]) / 2;
          position.y += columnOffset;
          
          // Create the node and add to the array
          initialNodes.push({
            id: agent.agent_id,
            type: 'agentNode',
            position,
            data: {
              agent,
              onStart: startAgent,
              onStop: stopAgent
            },
            // We'll use class names for targeting in CSS
            className: `agent-node ${agent.agent_role === 'specialized_sentiment' ? 'sentiment-agent-node' : ''} ${agent.status === 'running' ? 'running-node' : ''} ${agent.status === 'error' ? 'error-node' : ''}`,
            dragHandle: '.agent-drag-handle',
            draggable: true,
            selectable: true,
            style: {
              width: nodeWidth,
              padding: 0,
              border: `2px solid ${getNodeBorderColor(agent.status)}`,
              background: 'transparent',
              filter: agent.status === 'error' ? 'drop-shadow(0 0 8px rgba(255, 0, 0, 0.6))' : 
                     agent.status === 'running' ? 'drop-shadow(0 0 8px rgba(0, 255, 128, 0.4))' : 'none',
              transition: 'filter 0.3s, border-color 0.3s, transform 0.2s'
            },
          });
        });
      });
      
      setNodes(initialNodes);

      // Create the edges based on agent connections
      const newEdges: Edge[] = [];
      agents.forEach(agent => {
        if (agent.outputs_to && Array.isArray(agent.outputs_to)) {
          agent.outputs_to.forEach((targetId: string) => {
            // Only create edge if target agent exists
            const targetAgent = agents.find(a => a.agent_id === targetId);
            if (targetAgent) {
              const sourceStatus = agent.status;
              const edgeColor = getEdgeColor(sourceStatus, agent, targetAgent);
              
              // Special styling for sentiment-related edges
              const isSentimentFlow = 
                agent.agent_role === 'specialized_sentiment' || 
                targetAgent.agent_role === 'specialized_sentiment';
                
              const edgeClassName = sourceStatus === 'running' ?
                (isSentimentFlow ? 'animated-edge sentiment-edge' : 'animated-edge') : '';
              
              // Enhanced edge width for sentiment flows
              const edgeWidth = isSentimentFlow && sourceStatus === 'running' ? 
                getEdgeWidth(sourceStatus) + 0.5 : getEdgeWidth(sourceStatus);
              
              newEdges.push({
                id: `e${agent.agent_id}-${targetId}`,
                source: agent.agent_id,
                target: targetId,
                type: 'smoothstep',
                animated: sourceStatus === 'running',
                className: edgeClassName,
                markerEnd: {
                  type: MarkerType.ArrowClosed,
                  width: 15,
                  height: 15,
                  color: edgeColor,
                },
                // Edge styling
                style: { 
                  stroke: edgeColor,
                  strokeWidth: edgeWidth,
                  opacity: sourceStatus === 'running' ? 1 : 0.5,
                  zIndex: 1000,
                  // Add enhanced glow effect for sentiment connections
                  filter: isSentimentFlow && sourceStatus === 'running' ? 
                    'drop-shadow(0 0 5px rgba(76, 175, 80, 0.7))' : 
                    (sourceStatus === 'running' ? 'drop-shadow(0 0 5px rgba(255, 255, 0, 0.5))' : 'none'),
                },
                // Labels for edges
                label: sourceStatus === 'running' ? 
                  (isSentimentFlow ? 'SENTIMENT' : 'ACTIVE') : '',
                labelBgStyle: { fill: 'rgba(0, 0, 0, 0.7)', fillOpacity: 0.8 },
                labelStyle: { 
                  fill: isSentimentFlow ? '#4CAF50' : '#FFFF00', 
                  fontSize: 11, 
                  fontWeight: 'bold', 
                  fontFamily: 'Arial' 
                },
              });
            }
          });
        }
      });
      setEdges(newEdges);

    } else {
      // Clear nodes and edges if no agents
      setNodes([]);
      setEdges([]);
      setIsLayoutReady(false); // Reset if agents are cleared
    }
    // Set layout ready only if there are agents to process
    if (agents && agents.length > 0) {
      setIsLayoutReady(true);
    } else if (nodes.length > 0) { // Also set ready if we have dummy nodes
      setIsLayoutReady(true);
    }
    else {
      // If no agents, ensure layout is not considered ready, or set to true if an empty grid is desired.
      // For now, let's assume an empty grid is fine if no agents.
      setIsLayoutReady(true);
    }
  }, [agents, startAgent, stopAgent, setNodes, setEdges]);

  return (
    <div style={{ width: '100%', height: '70vh', background: 'rgba(21,26,36,0.7)', borderRadius: '8px', position: 'relative', zIndex: 0 }}> {/* Added position relative and base zIndex, slightly transparent background for debugging */}
      {isLayoutReady ? (
        <>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            minZoom={0.5}
            maxZoom={1.5}
            attributionPosition="bottom-left"
            style={{ background: 'transparent', zIndex: 1 }}
            // Better default edge options for new connections
            defaultEdgeOptions={{
              type: 'smart',
              animated: false,
              style: { stroke: 'rgba(255, 255, 0, 0.5)', strokeWidth: 1.5 }
            }}
            // Smoother connection line when dragging
            connectionLineStyle={{ stroke: '#FFFF00', strokeWidth: 2, strokeDasharray: '5,5' }}
          >
            {/* Inject styles for the minimap */}
            <style>
              {`
            .react-flow__minimap {
              background-color: #2D2D2D !important; /* Dark background from theme */
              border: 1px solid #444444 !important; /* Dark border from theme */
            }
            .react-flow__minimap-mask {
              fill: rgba(100, 100, 100, 0.3) !important; /* Semi-transparent mask, slightly more opaque */
            }
            /* Style for nodes within the minimap if needed */
            .react-flow__minimap-node {
              fill: #555 !important; /* Default node color in minimap */
              stroke: #888 !important;
            }

            /* Style for the controls */
            .react-flow__controls {
              background-color: rgba(45, 45, 45, 0.8) !important; /* Dark semi-transparent background */
              border-radius: 6px !important;
              padding: 5px !important;
              box-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
            }
            .react-flow__controls button {
              background-color: rgba(60, 60, 60, 0.9) !important;
              color: #e0e0e0 !important;
              border: 1px solid #555 !important;
              border-bottom: none !important; /* Remove default bottom border */
              margin-bottom: 1px !important; /* Add margin to separate buttons */
            }
            .react-flow__controls button:hover {
              background-color: rgba(75, 75, 75, 0.9) !important;
            }
            .react-flow__controls button:last-of-type {
              border-bottom: 1px solid #555 !important; /* Add border to last button */
            }
            .react-flow__controls svg {
              fill: #e0e0e0 !important;
            }

            /* Attempt to force edge visibility - Commented out for now */
            /* .react-flow__edge-path {
              stroke: #FFFF00 !important;
              stroke-width: 3px !important;
            }
            .react-flow__marker-arrowclosed path {
                fill: #FFFF00 !important;
            } */
          `}
            </style>
            <Controls />
            <MiniMap
              nodeStrokeWidth={2}
              // Enhanced coloring system for minimap nodes
              nodeColor={(node: Node<AgentNodeData>) => {
                if (!node.data?.agent) return '#757575';
                
                const status = node.data.agent.status;
                const role = node.data.agent.agent_role || '';
                
                // Different colors by role and status
                if (status === 'running') {
                  if (role.includes('specialized')) return '#4caf50';     // Green for specialized
                  if (role.includes('decision')) return '#2196f3';        // Blue for decision 
                  if (role.includes('execution')) return '#ff9800';       // Orange for execution
                  return '#4caf50';                                      // Default green
                }
                
                if (status === 'error') return '#f44336';                 // Red for errors
                if (status === 'initializing') return '#ffeb3b';          // Yellow for initializing
                
                // Different gray shades for stopped agents by role
                if (role.includes('specialized')) return '#757575';       // Dark gray
                if (role.includes('decision')) return '#9e9e9e';          // Medium gray
                if (role.includes('execution')) return '#bdbdbd';         // Light gray
                
                return '#757575';                                        // Default gray
              }}
              maskColor="rgba(0, 0, 0, 0.6)"                           // Darker mask for better contrast
              zoomable
              pannable
            />
            <Background color="#232b3b" gap={20} variant={BackgroundVariant.Dots} />
            
            {/* Legend removed as requested */}
            
            {/* Add flow metrics panel */}
            <Panel position="top-right" style={{ background: 'rgba(0,0,0,0.7)', borderRadius: '4px', padding: '10px', margin: '10px', minWidth: '180px' }}>
              <div style={{ color: '#fff', marginBottom: '8px', fontSize: '13px', fontWeight: 'bold' }}>Flow Metrics</div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <div style={{ color: '#bbb', fontSize: '11px' }}>Active Nodes:</div>
                <div style={{ color: '#fff', fontSize: '11px', fontWeight: 'bold' }}>
                  {agents.filter(a => a.status === 'running').length}/{agents.length}
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <div style={{ color: '#bbb', fontSize: '11px' }}>Active Flows:</div>
                <div style={{ color: '#fff', fontSize: '11px', fontWeight: 'bold' }}>
                  {edges.filter(e => {
                    const sourceAgent = agents.find(a => a.agent_id === e.source);
                    return sourceAgent && sourceAgent.status === 'running';
                  }).length}/{edges.length}
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <div style={{ color: '#bbb', fontSize: '11px' }}>Last Update:</div>
                <div style={{ color: '#fff', fontSize: '11px', fontWeight: 'bold' }}>
                  {new Date().toLocaleTimeString()}
                </div>
              </div>
            </Panel>
          </ReactFlow>
        </>
      ) : (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'white' }}>
          Calculating layout...
        </div>
      )}
    </div>
  );
};

export default AgentFlowGrid;