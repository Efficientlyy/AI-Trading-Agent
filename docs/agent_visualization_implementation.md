# Agent Visualization Implementation

This document outlines the implementation details for visualizing agent activities and data flow in the AI Trading Agent system.

## Overview

The agent visualization system will provide a real-time view of:
1. Data sources and their status
2. Agent activities and interactions
3. Decision-making processes
4. Data flow between components

## Implementation Components

### 1. Agent Activity Tracking

Each agent in the system needs to emit activity events that can be tracked and visualized:

```python
# In ai_trading_agent/common/event_system.py
import time
from typing import Dict, Any, List, Optional
from queue import Queue
import threading

class EventSystem:
    """Central event system for tracking agent activities."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = EventSystem()
        return cls._instance
    
    def __init__(self):
        """Initialize the event system."""
        self.event_queue = Queue()
        self.subscribers = []
        self.event_history = []
        self.max_history = 1000
        
        # Start event processing thread
        self.processing_thread = threading.Thread(target=self._process_events)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def emit_event(self, agent_id: str, event_type: str, data: Dict[str, Any]):
        """Emit an event from an agent."""
        event = {
            'timestamp': time.time(),
            'agent_id': agent_id,
            'event_type': event_type,
            'data': data
        }
        self.event_queue.put(event)
    
    def subscribe(self, callback):
        """Subscribe to events."""
        self.subscribers.append(callback)
        return len(self.subscribers) - 1
    
    def unsubscribe(self, subscription_id):
        """Unsubscribe from events."""
        if 0 <= subscription_id < len(self.subscribers):
            self.subscribers[subscription_id] = None
    
    def _process_events(self):
        """Process events from the queue."""
        while True:
            event = self.event_queue.get()
            
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
            
            # Notify subscribers
            for subscriber in self.subscribers:
                if subscriber is not None:
                    try:
                        subscriber(event)
                    except Exception as e:
                        print(f"Error in event subscriber: {e}")
            
            self.event_queue.task_done()
    
    def get_recent_events(self, limit: int = 100, 
                         agent_id: Optional[str] = None,
                         event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent events, optionally filtered."""
        events = self.event_history.copy()
        
        # Apply filters
        if agent_id:
            events = [e for e in events if e['agent_id'] == agent_id]
        if event_type:
            events = [e for e in events if e['event_type'] == event_type]
        
        # Return most recent events first
        return sorted(events, key=lambda e: e['timestamp'], reverse=True)[:limit]
```

### 2. Agent Instrumentation

Each agent class needs to be instrumented to emit events at key points:

```python
# Example for PublicApiDataProvider
from ai_trading_agent.common.event_system import EventSystem

class PublicApiDataProvider(BaseDataProvider):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.event_system = EventSystem.get_instance()
        # Rest of initialization...
    
    async def fetch_historical_data(self, symbols, timeframe, start_date, end_date, params=None):
        # Emit event at start
        self.event_system.emit_event(
            agent_id='public_api_provider',
            event_type='fetch_historical_data_start',
            data={
                'symbols': symbols,
                'timeframe': timeframe,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'primary_source': self.primary_source
            }
        )
        
        try:
            # Existing implementation...
            results = {}
            client = self.api_clients[self.primary_source]
            
            # Track which source was used for each symbol
            data_sources_used = {}
            
            for symbol in symbols:
                try:
                    data = await client.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if data is not None and not data.empty:
                        results[symbol] = data
                        data_sources_used[symbol] = self.primary_source
                    else:
                        # Try backup sources
                        for backup_source in self.backup_sources:
                            backup_client = self.api_clients[backup_source]
                            data = await backup_client.get_historical_data(
                                symbol=symbol,
                                timeframe=timeframe,
                                start_date=start_date,
                                end_date=end_date
                            )
                            if data is not None and not data.empty:
                                results[symbol] = data
                                data_sources_used[symbol] = backup_source
                                break
                except Exception as e:
                    self.event_system.emit_event(
                        agent_id='public_api_provider',
                        event_type='fetch_error',
                        data={
                            'symbol': symbol,
                            'source': self.primary_source,
                            'error': str(e)
                        }
                    )
            
            # Emit completion event
            self.event_system.emit_event(
                agent_id='public_api_provider',
                event_type='fetch_historical_data_complete',
                data={
                    'symbols_fetched': list(results.keys()),
                    'data_sources_used': data_sources_used,
                    'success_rate': len(results) / len(symbols) if symbols else 0
                }
            )
            
            return results
            
        except Exception as e:
            # Emit error event
            self.event_system.emit_event(
                agent_id='public_api_provider',
                event_type='fetch_historical_data_error',
                data={
                    'error': str(e)
                }
            )
            raise
```

### 3. API Endpoints for Visualization Data

Add API endpoints to expose agent activity data to the dashboard:

```python
# In api_server.py or a dedicated visualization_api.py
from flask import Blueprint, jsonify, request
from ai_trading_agent.common.event_system import EventSystem

visualization_api = Blueprint('visualization_api', __name__)

@visualization_api.route('/api/visualization/events', methods=['GET'])
def get_events():
    """Get recent events for visualization."""
    limit = int(request.args.get('limit', 100))
    agent_id = request.args.get('agent_id')
    event_type = request.args.get('event_type')
    
    event_system = EventSystem.get_instance()
    events = event_system.get_recent_events(
        limit=limit,
        agent_id=agent_id,
        event_type=event_type
    )
    
    return jsonify({'events': events})

@visualization_api.route('/api/visualization/agent-status', methods=['GET'])
def get_agent_status():
    """Get current status of all agents."""
    event_system = EventSystem.get_instance()
    
    # Get the most recent event for each agent
    all_events = event_system.get_recent_events(limit=1000)
    
    # Group by agent
    agents = {}
    for event in all_events:
        agent_id = event['agent_id']
        if agent_id not in agents or event['timestamp'] > agents[agent_id]['last_activity']:
            agents[agent_id] = {
                'last_activity': event['timestamp'],
                'last_event_type': event['event_type'],
                'last_event_data': event['data']
            }
    
    # Add status based on last activity
    current_time = time.time()
    for agent_id, data in agents.items():
        time_since_last = current_time - data['last_activity']
        if time_since_last < 60:  # Within the last minute
            data['status'] = 'active'
        elif time_since_last < 300:  # Within the last 5 minutes
            data['status'] = 'idle'
        else:
            data['status'] = 'inactive'
    
    return jsonify({'agents': agents})

@visualization_api.route('/api/visualization/data-flow', methods=['GET'])
def get_data_flow():
    """Get data flow information between agents."""
    # This would require analyzing events to determine data flow
    # For now, return a static representation of the system architecture
    data_flow = {
        'nodes': [
            {'id': 'public_api_provider', 'type': 'data_source', 'name': 'Public API Provider'},
            {'id': 'data_manager', 'type': 'manager', 'name': 'Data Manager'},
            {'id': 'strategy_manager', 'type': 'manager', 'name': 'Strategy Manager'},
            {'id': 'ma_crossover_strategy', 'type': 'strategy', 'name': 'MA Crossover Strategy'},
            {'id': 'sentiment_strategy', 'type': 'strategy', 'name': 'Sentiment Strategy'},
            {'id': 'risk_manager', 'type': 'manager', 'name': 'Risk Manager'},
            {'id': 'portfolio_manager', 'type': 'manager', 'name': 'Portfolio Manager'},
            {'id': 'execution_handler', 'type': 'execution', 'name': 'Execution Handler'}
        ],
        'edges': [
            {'source': 'public_api_provider', 'target': 'data_manager', 'type': 'data'},
            {'source': 'data_manager', 'target': 'strategy_manager', 'type': 'data'},
            {'source': 'strategy_manager', 'target': 'ma_crossover_strategy', 'type': 'control'},
            {'source': 'strategy_manager', 'target': 'sentiment_strategy', 'type': 'control'},
            {'source': 'ma_crossover_strategy', 'target': 'strategy_manager', 'type': 'signal'},
            {'source': 'sentiment_strategy', 'target': 'strategy_manager', 'type': 'signal'},
            {'source': 'strategy_manager', 'target': 'risk_manager', 'type': 'signal'},
            {'source': 'risk_manager', 'target': 'portfolio_manager', 'type': 'risk_adjusted_signal'},
            {'source': 'portfolio_manager', 'target': 'execution_handler', 'type': 'order'},
            {'source': 'execution_handler', 'target': 'portfolio_manager', 'type': 'execution_result'}
        ]
    }
    
    # Enhance with activity data
    event_system = EventSystem.get_instance()
    recent_events = event_system.get_recent_events(limit=100)
    
    # Add activity level to nodes
    for node in data_flow['nodes']:
        node_events = [e for e in recent_events if e['agent_id'] == node['id']]
        node['activity_level'] = len(node_events)
        if node_events:
            node['last_activity'] = max(e['timestamp'] for e in node_events)
            node['status'] = 'active' if time.time() - node['last_activity'] < 60 else 'idle'
        else:
            node['status'] = 'inactive'
    
    # Add activity to edges based on event flow
    for edge in data_flow['edges']:
        edge_events = [e for e in recent_events if 
                      e['agent_id'] == edge['source'] and 
                      e['event_type'].endswith('_complete') and
                      edge['target'] in str(e['data'])]
        edge['activity_level'] = len(edge_events)
    
    return jsonify(data_flow)
```

### 4. React Component for Data Flow Visualization

Create a React component to visualize the data flow using a library like `react-flow` or `d3.js`:

```jsx
// In dashboard/src/components/AgentVisualization.js
import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, { 
  Controls, 
  Background, 
  MiniMap,
  useNodesState,
  useEdgesState
} from 'reactflow';
import 'reactflow/dist/style.css';
import axios from 'axios';

// Custom node types
const nodeTypes = {
  dataSource: ({ data }) => (
    <div className={`node data-source-node ${data.status}`}>
      <div className="node-header">{data.name}</div>
      <div className="node-content">
        <div className="status-indicator"></div>
        <div className="node-stats">
          <div>Status: {data.status}</div>
          <div>Activity: {data.activity_level}</div>
          {data.last_event_type && (
            <div>Last: {data.last_event_type}</div>
          )}
        </div>
      </div>
    </div>
  ),
  strategy: ({ data }) => (
    <div className={`node strategy-node ${data.status}`}>
      <div className="node-header">{data.name}</div>
      <div className="node-content">
        <div className="status-indicator"></div>
        <div className="node-stats">
          <div>Status: {data.status}</div>
          <div>Activity: {data.activity_level}</div>
          {data.last_event_type && (
            <div>Last: {data.last_event_type}</div>
          )}
        </div>
      </div>
    </div>
  ),
  // Similar custom nodes for other types...
};

const AgentVisualization = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [events, setEvents] = useState([]);
  
  // Fetch data flow information
  const fetchDataFlow = useCallback(async () => {
    try {
      const response = await axios.get('/api/visualization/data-flow');
      const dataFlow = response.data;
      
      // Transform nodes for ReactFlow
      const flowNodes = dataFlow.nodes.map(node => ({
        id: node.id,
        type: node.type,
        data: { 
          ...node,
          label: node.name 
        },
        position: getNodePosition(node.id) // Function to position nodes
      }));
      
      // Transform edges for ReactFlow
      const flowEdges = dataFlow.edges.map((edge, i) => ({
        id: `e${i}`,
        source: edge.source,
        target: edge.target,
        animated: edge.activity_level > 0,
        style: { 
          strokeWidth: Math.min(1 + edge.activity_level, 5),
          stroke: getEdgeColor(edge.type)
        },
        label: edge.type
      }));
      
      setNodes(flowNodes);
      setEdges(flowEdges);
    } catch (error) {
      console.error('Error fetching data flow:', error);
    }
  }, [setNodes, setEdges]);
  
  // Fetch recent events
  const fetchEvents = useCallback(async () => {
    try {
      const response = await axios.get('/api/visualization/events', {
        params: { limit: 20 }
      });
      setEvents(response.data.events);
    } catch (error) {
      console.error('Error fetching events:', error);
    }
  }, []);
  
  // Position nodes in a logical layout
  const getNodePosition = (nodeId) => {
    // This would be a more sophisticated layout algorithm in production
    const positions = {
      'public_api_provider': { x: 100, y: 100 },
      'data_manager': { x: 300, y: 100 },
      'strategy_manager': { x: 500, y: 100 },
      'ma_crossover_strategy': { x: 400, y: 250 },
      'sentiment_strategy': { x: 600, y: 250 },
      'risk_manager': { x: 700, y: 100 },
      'portfolio_manager': { x: 900, y: 100 },
      'execution_handler': { x: 1100, y: 100 }
    };
    
    return positions[nodeId] || { x: 0, y: 0 };
  };
  
  // Get color based on edge type
  const getEdgeColor = (type) => {
    const colors = {
      'data': '#3498db',
      'control': '#2ecc71',
      'signal': '#e74c3c',
      'risk_adjusted_signal': '#9b59b6',
      'order': '#f39c12',
      'execution_result': '#1abc9c'
    };
    
    return colors[type] || '#95a5a6';
  };
  
  // Update data periodically
  useEffect(() => {
    fetchDataFlow();
    fetchEvents();
    
    const interval = setInterval(() => {
      fetchDataFlow();
      fetchEvents();
    }, 5000);
    
    return () => clearInterval(interval);
  }, [fetchDataFlow, fetchEvents]);
  
  return (
    <div className="agent-visualization">
      <h2>Agent Activity Visualization</h2>
      
      <div className="visualization-container">
        <div className="flow-diagram" style={{ height: 600 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
          >
            <Controls />
            <MiniMap />
            <Background />
          </ReactFlow>
        </div>
        
        <div className="event-log">
          <h3>Recent Events</h3>
          <div className="events-container">
            {events.map((event, index) => (
              <div key={index} className={`event-item ${event.event_type}`}>
                <div className="event-time">
                  {new Date(event.timestamp * 1000).toLocaleTimeString()}
                </div>
                <div className="event-agent">{event.agent_id}</div>
                <div className="event-type">{event.event_type}</div>
                <div className="event-data">
                  {Object.entries(event.data).map(([key, value]) => (
                    <div key={key} className="data-item">
                      <span className="data-key">{key}:</span>
                      <span className="data-value">
                        {typeof value === 'object' 
                          ? JSON.stringify(value).substring(0, 30) + '...'
                          : String(value).substring(0, 30)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="data-source-indicators">
        <h3>Data Sources</h3>
        <div className="source-indicators">
          {['coingecko', 'cryptocompare'].map(source => (
            <div key={source} className="source-indicator">
              <div className={`indicator ${isSourceActive(source, events) ? 'active' : 'inactive'}`}></div>
              <div className="source-name">{source}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Helper function to check if a data source is active
const isSourceActive = (source, events) => {
  const sourceEvents = events.filter(e => 
    e.agent_id === 'public_api_provider' && 
    e.data && 
    (e.data.primary_source === source || 
     (e.data.data_sources_used && Object.values(e.data.data_sources_used).includes(source)))
  );
  
  if (sourceEvents.length === 0) return false;
  
  const latestEvent = sourceEvents.reduce((latest, event) => 
    event.timestamp > latest.timestamp ? event : latest, sourceEvents[0]);
  
  return (Date.now() / 1000) - latestEvent.timestamp < 60; // Active in the last minute
};

export default AgentVisualization;
```

### 5. CSS Styling for Visualization

Add CSS to style the visualization components:

```css
/* In dashboard/src/styles/AgentVisualization.css */
.agent-visualization {
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.visualization-container {
  display: flex;
  margin-top: 20px;
}

.flow-diagram {
  flex: 3;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
}

.event-log {
  flex: 1;
  margin-left: 20px;
  max-height: 600px;
  overflow-y: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
  padding: 10px;
}

.events-container {
  display: flex;
  flex-direction: column;
}

.event-item {
  padding: 8px;
  margin-bottom: 8px;
  border-radius: 4px;
  background-color: #f1f1f1;
  font-size: 12px;
  border-left: 4px solid #ccc;
}

.event-item.fetch_historical_data_start { border-left-color: #3498db; }
.event-item.fetch_historical_data_complete { border-left-color: #2ecc71; }
.event-item.fetch_historical_data_error { border-left-color: #e74c3c; }
.event-item.fetch_error { border-left-color: #f39c12; }

.event-time {
  font-weight: bold;
  color: #555;
}

.event-agent {
  color: #3498db;
  font-weight: bold;
}

.event-type {
  color: #7f8c8d;
  font-style: italic;
}

.event-data {
  margin-top: 5px;
  display: flex;
  flex-direction: column;
}

.data-item {
  display: flex;
  margin-bottom: 2px;
}

.data-key {
  color: #555;
  margin-right: 5px;
  font-weight: bold;
}

.data-value {
  color: #333;
  word-break: break-word;
}

/* Node styling */
.node {
  padding: 10px;
  border-radius: 5px;
  width: 180px;
  color: white;
}

.node-header {
  font-weight: bold;
  text-align: center;
  margin-bottom: 5px;
  font-size: 14px;
}

.node-content {
  display: flex;
  align-items: center;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 10px;
}

.node.active .status-indicator {
  background-color: #2ecc71;
  box-shadow: 0 0 8px #2ecc71;
}

.node.idle .status-indicator {
  background-color: #f39c12;
}

.node.inactive .status-indicator {
  background-color: #e74c3c;
}

.node-stats {
  font-size: 12px;
}

/* Node type styling */
.data-source-node {
  background-color: #3498db;
}

.strategy-node {
  background-color: #9b59b6;
}

.manager-node {
  background-color: #2c3e50;
}

.execution-node {
  background-color: #e67e22;
}

/* Data source indicators */
.data-source-indicators {
  margin-top: 20px;
  padding: 10px;
  background-color: white;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.source-indicators {
  display: flex;
  gap: 20px;
  margin-top: 10px;
}

.source-indicator {
  display: flex;
  align-items: center;
}

.indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 8px;
}

.indicator.active {
  background-color: #2ecc71;
  box-shadow: 0 0 8px #2ecc71;
}

.indicator.inactive {
  background-color: #e74c3c;
}

.source-name {
  font-weight: bold;
}
```

## Integration with Dashboard

To integrate the agent visualization into the dashboard:

1. Add the visualization component to the main dashboard
2. Create a dedicated route for the visualization page
3. Add a navigation link to access the visualization

```jsx
// In dashboard/src/App.js
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import PaperTradingPanel from './components/PaperTradingPanel';
import AgentVisualization from './components/AgentVisualization';

function App() {
  return (
    <Router>
      <div className="App">
        <Switch>
          <Route exact path="/" component={Dashboard} />
          <Route path="/paper-trading" component={PaperTradingPanel} />
          <Route path="/agent-visualization" component={AgentVisualization} />
        </Switch>
      </div>
    </Router>
  );
}

export default App;
```

## Example Visualization

The final visualization will show:

1. A network diagram of all agents with:
   - Nodes representing different agents
   - Edges showing data flow between agents
   - Color coding for different agent types
   - Status indicators (active/idle/inactive)
   - Animation for active data flows

2. An event log showing recent activities:
   - Timestamped events
   - Source agent
   - Event type
   - Key event data

3. Data source indicators:
   - Which data sources are currently active
   - Which symbols are being provided by each source
   - Status of each data source (active/inactive)

This implementation provides a comprehensive view of the system's operation, allowing you to see exactly where data is coming from and what each agent is doing in real-time.
