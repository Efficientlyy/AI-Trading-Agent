import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Typography,
  Paper,
  Tabs,
  Tab,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import Timeline from '@mui/lab/Timeline';
import TimelineItem from '@mui/lab/TimelineItem';
import TimelineSeparator from '@mui/lab/TimelineSeparator';
import TimelineConnector from '@mui/lab/TimelineConnector';
import TimelineContent from '@mui/lab/TimelineContent';
import TimelineDot from '@mui/lab/TimelineDot';
import TimelineOppositeContent from '@mui/lab/TimelineOppositeContent';

// Import visualization libraries
import { ResponsiveLine } from '@nivo/line';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

// Component colors for visualization
const componentColors = {
  'data_manager': '#1f77b4',
  'strategy_manager': '#ff7f0e',
  'risk_manager': '#2ca02c',
  'portfolio_manager': '#d62728',
  'execution_handler': '#9467bd',
  'data_provider': '#8c564b',
  'api_client': '#e377c2',
  'orchestrator': '#7f7f7f',
  'default': '#bcbd22'
};

// Get color for a component
const getComponentColor = (component) => {
  return componentColors[component] || componentColors.default;
};

// Format timestamp
const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString();
};

const AgentVisualization = () => {
  // State
  const [activeTab, setActiveTab] = useState(0);
  const [timeWindow, setTimeWindow] = useState(30);
  const [loading, setLoading] = useState(false);
  const [events, setEvents] = useState([]);
  const [timelineData, setTimelineData] = useState(null);
  const [dataFlowData, setDataFlowData] = useState(null);
  const [componentStates, setComponentStates] = useState({});
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [refreshTimer, setRefreshTimer] = useState(null);
  
  // Refs for visualization
  const graphRef = useRef(null);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Fetch data on component mount and periodically
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Fetch different data based on active tab
        if (activeTab === 0) {
          // Timeline view
          const response = await axios.get(`${API_BASE_URL}/api/agent-visualization/activity-timeline?minutes=${timeWindow}`);
          setTimelineData(response.data);
        } else if (activeTab === 1) {
          // Data flow view
          const response = await axios.get(`${API_BASE_URL}/api/agent-visualization/data-flow`);
          setDataFlowData(response.data);
        } else if (activeTab === 2) {
          // Events view
          const response = await axios.get(`${API_BASE_URL}/api/agent-visualization/events?limit=50`);
          setEvents(response.data);
        }
        
        // Always fetch component states
        const statesResponse = await axios.get(`${API_BASE_URL}/api/agent-visualization/component-states`);
        setComponentStates(statesResponse.data);
      } catch (error) {
        console.error('Error fetching visualization data:', error);
      } finally {
        setLoading(false);
      }
    };
    
    // Initial fetch
    fetchData();
    
    // Set up periodic refresh
    const timer = setInterval(fetchData, refreshInterval);
    setRefreshTimer(timer);
    
    // Clean up on unmount
    return () => {
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
    };
  }, [activeTab, timeWindow, refreshInterval]);
  
  // Handle refresh button click
  const handleRefresh = async () => {
    setLoading(true);
    try {
      // Fetch different data based on active tab
      if (activeTab === 0) {
        // Timeline view
        const response = await axios.get(`${API_BASE_URL}/api/agent-visualization/activity-timeline?minutes=${timeWindow}`);
        setTimelineData(response.data);
      } else if (activeTab === 1) {
        // Data flow view
        const response = await axios.get(`${API_BASE_URL}/api/agent-visualization/data-flow`);
        setDataFlowData(response.data);
      } else if (activeTab === 2) {
        // Events view
        const response = await axios.get(`${API_BASE_URL}/api/agent-visualization/events?limit=50`);
        setEvents(response.data);
      }
      
      // Always fetch component states
      const statesResponse = await axios.get(`${API_BASE_URL}/api/agent-visualization/component-states`);
      setComponentStates(statesResponse.data);
    } catch (error) {
      console.error('Error refreshing visualization data:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // Prepare data for force graph
  const prepareGraphData = () => {
    if (!dataFlowData) return { nodes: [], links: [] };
    
    const nodes = [];
    const links = [];
    const nodeMap = {};
    
    // Add data sources as nodes
    dataFlowData.data_sources.forEach(source => {
      if (!nodeMap[source]) {
        const node = {
          id: source,
          name: source,
          group: 'source',
          val: 1
        };
        nodes.push(node);
        nodeMap[source] = true;
      }
    });
    
    // Add data sinks as nodes
    dataFlowData.data_sinks.forEach(sink => {
      if (!nodeMap[sink]) {
        const node = {
          id: sink,
          name: sink,
          group: 'sink',
          val: 1
        };
        nodes.push(node);
        nodeMap[sink] = true;
      }
    });
    
    // Add data flows as links
    dataFlowData.data_flows.forEach(flow => {
      links.push({
        source: flow.source,
        target: flow.destination,
        value: 1,
        label: flow.data_type
      });
    });
    
    return { nodes, links };
  };
  
  // Render timeline view
  const renderTimelineView = () => {
    if (!timelineData) return <Typography>No timeline data available</Typography>;
    
    return (
      <Box sx={{ height: 500, overflowY: 'auto' }}>
        <Timeline position="alternate">
          {timelineData.components.map((component, componentIndex) => (
            <React.Fragment key={component}>
              {timelineData.events_by_component[component].slice(0, 10).map((event, eventIndex) => (
                <TimelineItem key={`${component}-${eventIndex}`}>
                  <TimelineOppositeContent color="text.secondary">
                    {formatTimestamp(event.timestamp)}
                  </TimelineOppositeContent>
                  <TimelineSeparator>
                    <TimelineDot sx={{ bgcolor: getComponentColor(component) }} />
                    {eventIndex < timelineData.events_by_component[component].length - 1 && <TimelineConnector />}
                  </TimelineSeparator>
                  <TimelineContent>
                    <Paper elevation={3} sx={{ p: 2, bgcolor: 'background.paper' }}>
                      <Typography variant="h6" component="span">
                        {component}
                      </Typography>
                      <Typography>{event.action}</Typography>
                      {event.symbol && (
                        <Typography variant="body2" color="text.secondary">
                          Symbol: {event.symbol}
                        </Typography>
                      )}
                    </Paper>
                  </TimelineContent>
                </TimelineItem>
              ))}
            </React.Fragment>
          ))}
        </Timeline>
      </Box>
    );
  };
  
  // Render data flow view
  const renderDataFlowView = () => {
    if (!dataFlowData) return <Typography>No data flow information available</Typography>;
    
    const graphData = prepareGraphData();
    
    return (
      <Box sx={{ height: 500, p: 2, overflowY: 'auto' }}>
        <Typography variant="h6" gutterBottom>Data Flow Visualization</Typography>
        
        {/* Simple table-based visualization instead of force graph */}
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="subtitle1" gutterBottom>Components and Data Flow</Typography>
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle2">Data Sources:</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                {dataFlowData.data_sources.map((source, idx) => (
                  <Chip 
                    key={idx} 
                    label={source} 
                    sx={{ bgcolor: getComponentColor(source.split('_')[0]), color: 'white' }} 
                  />
                ))}
              </Box>
              
              <Typography variant="subtitle2">Data Sinks:</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                {dataFlowData.data_sinks.map((sink, idx) => (
                  <Chip 
                    key={idx} 
                    label={sink} 
                    sx={{ bgcolor: getComponentColor(sink.split('_')[0]), color: 'white' }} 
                  />
                ))}
              </Box>
              
              <Typography variant="subtitle2">Data Flows:</Typography>
              <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Source</TableCell>
                      <TableCell>Destination</TableCell>
                      <TableCell>Data Type</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {dataFlowData.data_flows.map((flow, idx) => (
                      <TableRow key={idx}>
                        <TableCell>{flow.source}</TableCell>
                        <TableCell>{flow.destination}</TableCell>
                        <TableCell>{flow.data_type}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    );
  };
  
  // Render events view
  const renderEventsView = () => {
    if (!events || events.length === 0) return <Typography>No events available</Typography>;
    
    return (
      <Box sx={{ height: 500, overflowY: 'auto' }}>
        {events.map((event, index) => (
          <Paper key={index} elevation={1} sx={{ p: 2, mb: 2, bgcolor: 'background.paper' }}>
            <Typography variant="subtitle1">
              {event.component} - {event.action}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {formatTimestamp(event.timestamp)}
              {event.symbol && ` | Symbol: ${event.symbol}`}
            </Typography>
            <Box sx={{ mt: 1 }}>
              <Typography variant="body2">
                {Object.entries(event.data).map(([key, value]) => (
                  <Box key={key} component="span" display="block">
                    <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : value}
                  </Box>
                ))}
              </Typography>
            </Box>
          </Paper>
        ))}
      </Box>
    );
  };
  
  // Render component states
  const renderComponentStates = () => {
    if (!componentStates || Object.keys(componentStates).length === 0) {
      return <Typography>No component state information available</Typography>;
    }
    
    return (
      <Grid container spacing={2}>
        {Object.entries(componentStates).map(([component, state]) => (
          <Grid item xs={12} md={6} lg={4} key={component}>
            <Paper 
              elevation={2} 
              sx={{ 
                p: 2, 
                borderLeft: `4px solid ${getComponentColor(component.split('_')[0])}`,
                height: '100%'
              }}
            >
              <Typography variant="subtitle1" gutterBottom>
                {component}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Last updated: {formatTimestamp(state.last_updated)}
              </Typography>
              <Divider sx={{ my: 1 }} />
              <Box sx={{ mt: 1 }}>
                {Object.entries(state.state).map(([key, value]) => (
                  <Typography key={key} variant="body2">
                    <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : value}
                  </Typography>
                ))}
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    );
  };
  
  return (
    <Card>
      <CardHeader 
        title="Agent Visualization" 
        subheader="Visualize agent activities and data flow"
        action={
          <Button
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Refresh'}
          </Button>
        }
      />
      <Divider />
      <CardContent>
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <Tabs value={activeTab} onChange={handleTabChange} aria-label="visualization tabs">
                <Tab label="Timeline" />
                <Tab label="Data Flow" />
                <Tab label="Events" />
              </Tabs>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl sx={{ minWidth: 200 }}>
                <InputLabel id="time-window-label">Time Window</InputLabel>
                <Select
                  labelId="time-window-label"
                  value={timeWindow}
                  onChange={(e) => setTimeWindow(e.target.value)}
                  label="Time Window"
                >
                  <MenuItem value={5}>Last 5 minutes</MenuItem>
                  <MenuItem value={15}>Last 15 minutes</MenuItem>
                  <MenuItem value={30}>Last 30 minutes</MenuItem>
                  <MenuItem value={60}>Last hour</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Box>
        
        <Box sx={{ mt: 3 }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              {/* Tab content */}
              <Box sx={{ mb: 4 }}>
                {activeTab === 0 && renderTimelineView()}
                {activeTab === 1 && renderDataFlowView()}
                {activeTab === 2 && renderEventsView()}
              </Box>
              
              {/* Component states (always shown) */}
              <Box sx={{ mt: 4 }}>
                <Typography variant="h6" gutterBottom>Component States</Typography>
                <Divider sx={{ mb: 2 }} />
                {renderComponentStates()}
              </Box>
            </>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default AgentVisualization;
