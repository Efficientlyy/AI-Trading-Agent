import React, { useEffect, useRef, useState } from 'react';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import webSocketService, { WebSocketTopic } from '../../../services/WebSocketService';

interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'idle' | 'error';
  lastActivity: string;
  processingTime?: number;
  dataVolume?: number;
  signalStrength?: number;
  connections: string[];
}

interface DataSource {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'idle' | 'error';
  lastUpdated: string;
  symbols: string[];
}

interface AgentActivityData {
  agents: Agent[];
  dataSources: DataSource[];
  interactions: {
    timestamp: string;
    from: string;
    to: string;
    type: string;
    data?: any;
  }[];
}

const PaperTradingAgentVisualization: React.FC = () => {
  const { state } = usePaperTrading();
  const [agentData, setAgentData] = useState<AgentActivityData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [selectedDataSource, setSelectedDataSource] = useState<DataSource | null>(null);
  const [timelineView, setTimelineView] = useState<boolean>(false);

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    if (state.activeSessions.length > 0) {
      // Connect to WebSocket
      webSocketService.connect([WebSocketTopic.AGENT_STATUS])
        .then(() => {
          setIsConnected(true);
          console.log('Connected to WebSocket for agent visualization');
        })
        .catch(error => {
          console.error('Failed to connect to WebSocket for agent visualization:', error);
          setError('Failed to connect to WebSocket');
        });

      // Set up event handler
      const handleAgentStatusUpdate = (data: any) => {
        if (data.error) {
          setError(data.error);
          return;
        }

        if (data.agents) {
          setAgentData(data);
          setError(null);
        }
      };

      // Register event handler
      webSocketService.on(WebSocketTopic.AGENT_STATUS, handleAgentStatusUpdate);

      // Cleanup function
      return () => {
        webSocketService.off(WebSocketTopic.AGENT_STATUS, handleAgentStatusUpdate);
        
        // Disconnect from WebSocket
        webSocketService.disconnect();
        setIsConnected(false);
      };
    }
  }, [state.activeSessions.length]);

  // Draw the agent visualization diagram
  useEffect(() => {
    if (!canvasRef.current || !agentData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas dimensions
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    // Draw data flow diagram
    drawDataFlowDiagram(ctx, agentData);
  }, [agentData, canvasRef.current?.offsetWidth, canvasRef.current?.offsetHeight]);

  // Draw data flow diagram
  const drawDataFlowDiagram = (ctx: CanvasRenderingContext2D, data: AgentActivityData) => {
    const { agents, dataSources } = data;
    
    // Calculate positions
    const margin = 50;
    const dataSourceY = margin;
    const canvasElement = canvasRef.current;
    if (!canvasElement) return;
    
    const agentY = canvasElement.height - margin;
    const dataSourceWidth = (canvasElement.width - 2 * margin) / Math.max(dataSources.length, 1);
    const agentWidth = (canvasElement.width - 2 * margin) / Math.max(agents.length, 1);
    
    // Draw data sources
    dataSources.forEach((source, index) => {
      const x = margin + index * dataSourceWidth + dataSourceWidth / 2;
      drawNode(ctx, x, dataSourceY, source.name, getStatusColor(source.status));
    });
    
    // Draw agents
    agents.forEach((agent, index) => {
      const x = margin + index * agentWidth + agentWidth / 2;
      drawNode(ctx, x, agentY, agent.name, getStatusColor(agent.status));
    });
    
    // Draw connections
    agents.forEach((agent, agentIndex) => {
      const agentX = margin + agentIndex * agentWidth + agentWidth / 2;
      
      // Find connected data sources
      agent.connections.forEach(connectionId => {
        const sourceIndex = dataSources.findIndex(source => source.id === connectionId);
        if (sourceIndex >= 0) {
          const sourceX = margin + sourceIndex * dataSourceWidth + dataSourceWidth / 2;
          drawConnection(ctx, sourceX, dataSourceY, agentX, agentY, getStatusColor(agent.status));
        }
      });
    });
  };
  
  // Draw a node (agent or data source)
  const drawNode = (
    ctx: CanvasRenderingContext2D, 
    x: number, 
    y: number, 
    label: string, 
    color: string
  ) => {
    const radius = 30;
    
    // Draw circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw label
    ctx.fillStyle = '#fff';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y);
  };
  
  // Draw a connection between nodes
  const drawConnection = (
    ctx: CanvasRenderingContext2D, 
    x1: number, 
    y1: number, 
    x2: number, 
    y2: number, 
    color: string
  ) => {
    ctx.beginPath();
    ctx.moveTo(x1, y1 + 30); // Start below the data source node
    ctx.lineTo(x2, y2 - 30); // End above the agent node
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw arrow
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const arrowSize = 10;
    const arrowY = y2 - 30;
    const arrowX = x2;
    
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX - arrowSize * Math.cos(angle - Math.PI / 6),
      arrowY - arrowSize * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      arrowX - arrowSize * Math.cos(angle + Math.PI / 6),
      arrowY - arrowSize * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  };
  
  // Get color based on status
  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'active':
        return '#4caf50'; // Green
      case 'idle':
        return '#2196f3'; // Blue
      case 'error':
        return '#f44336'; // Red
      default:
        return '#9e9e9e'; // Grey
    }
  };

  // Handle agent click
  const handleAgentClick = (agent: Agent) => {
    setSelectedAgent(agent);
    setSelectedDataSource(null);
  };

  // Handle data source click
  const handleDataSourceClick = (source: DataSource) => {
    setSelectedDataSource(source);
    setSelectedAgent(null);
  };

  // Toggle timeline view
  const toggleTimelineView = () => {
    setTimelineView(!timelineView);
  };

  // Render agent details
  const renderAgentDetails = (agent: Agent) => {
    return (
      <div className="agent-details">
        <h4>{agent.name}</h4>
        <p><strong>Type:</strong> {agent.type}</p>
        <p><strong>Status:</strong> <span className={`status-${agent.status}`}>{agent.status}</span></p>
        <p><strong>Last Activity:</strong> {new Date(agent.lastActivity).toLocaleTimeString()}</p>
        {agent.processingTime !== undefined && (
          <p><strong>Processing Time:</strong> {agent.processingTime.toFixed(2)} ms</p>
        )}
        {agent.dataVolume !== undefined && (
          <p><strong>Data Volume:</strong> {agent.dataVolume} points</p>
        )}
        {agent.signalStrength !== undefined && (
          <p><strong>Signal Strength:</strong> {(agent.signalStrength * 100).toFixed(1)}%</p>
        )}
      </div>
    );
  };

  // Render data source details
  const renderDataSourceDetails = (source: DataSource) => {
    return (
      <div className="data-source-details">
        <h4>{source.name}</h4>
        <p><strong>Type:</strong> {source.type}</p>
        <p><strong>Status:</strong> <span className={`status-${source.status}`}>{source.status}</span></p>
        <p><strong>Last Updated:</strong> {new Date(source.lastUpdated).toLocaleTimeString()}</p>
        <p><strong>Symbols:</strong> {source.symbols.join(', ')}</p>
      </div>
    );
  };

  // Render timeline view
  const renderTimelineView = () => {
    if (!agentData || !agentData.interactions) return null;

    return (
      <div className="timeline-view">
        <h4>Agent Interactions Timeline</h4>
        <div className="timeline-container">
          {agentData.interactions.map((interaction, index) => (
            <div key={index} className="timeline-item">
              <div className="timeline-time">
                {new Date(interaction.timestamp).toLocaleTimeString()}
              </div>
              <div className="timeline-content">
                <div className="timeline-title">
                  {interaction.from} → {interaction.to}
                </div>
                <div className="timeline-type">{interaction.type}</div>
                {interaction.data && (
                  <div className="timeline-data">
                    {typeof interaction.data === 'object' 
                      ? JSON.stringify(interaction.data).substring(0, 50) + '...'
                      : interaction.data.toString().substring(0, 50) + '...'}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="paper-trading-agent-visualization">
      <div className="panel-header">
        <h3>Agent Activity Visualization</h3>
        <div className="panel-controls">
          <button 
            className={`view-toggle ${timelineView ? 'active' : ''}`} 
            onClick={toggleTimelineView}
          >
            {timelineView ? 'Show Diagram' : 'Show Timeline'}
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}

      {!isConnected && !error && (
        <div className="loading-message">
          Connecting to agent status feed...
        </div>
      )}

      {isConnected && !agentData && !error && (
        <div className="loading-message">
          Waiting for agent activity data...
        </div>
      )}

      {agentData && !timelineView && (
        <div className="visualization-container">
          <canvas ref={canvasRef} className="agent-canvas" />
          
          <div className="agent-list">
            <h4>Agents</h4>
            <div className="agent-items">
              {agentData.agents.map(agent => (
                <div 
                  key={agent.id}
                  className={`agent-item ${selectedAgent?.id === agent.id ? 'selected' : ''}`}
                  onClick={() => handleAgentClick(agent)}
                >
                  <div className={`status-indicator status-${agent.status}`} />
                  <div className="agent-name">{agent.name}</div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="data-source-list">
            <h4>Data Sources</h4>
            <div className="data-source-items">
              {agentData.dataSources.map(source => (
                <div 
                  key={source.id}
                  className={`data-source-item ${selectedDataSource?.id === source.id ? 'selected' : ''}`}
                  onClick={() => handleDataSourceClick(source)}
                >
                  <div className={`status-indicator status-${source.status}`} />
                  <div className="data-source-name">{source.name}</div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="details-panel">
            {selectedAgent && renderAgentDetails(selectedAgent)}
            {selectedDataSource && renderDataSourceDetails(selectedDataSource)}
            {!selectedAgent && !selectedDataSource && (
              <div className="no-selection">
                <p>Select an agent or data source to view details</p>
              </div>
            )}
          </div>
        </div>
      )}

      {agentData && timelineView && renderTimelineView()}
    </div>
  );
};

export default PaperTradingAgentVisualization;
