import axios from 'axios';
import React, { createContext, ReactNode, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { mockAgents, getMockAgents } from '../api/mockData/mockAgents';
import { useMockData } from './MockDataContext';

// API base URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Define types for our context
export interface Agent {
  agent_id: string;
  name: string;
  status: 'running' | 'stopped' | 'error' | 'initializing'; // Matches backend status enum values
  type: string; // Corresponds to backend agent_type
  last_updated: string; // Changed from last_active to match backend
  metrics?: { // Changed from performance_metrics to general metrics, AgentCard will need to adapt or backend metrics need specific keys
    win_rate?: number;
    profit_factor?: number;
    avg_profit_loss?: number;
    max_drawdown?: number;
    [key: string]: any; // Allow other metrics
  };
  symbols: string[];
  strategy?: string; // This can be derived from 'type' or 'config_details' if needed by UI
  agent_role?: string; // Matches backend agent_role
  inputs_from?: string[]; // Added to match backend and for graph rendering
  outputs_to?: string[]; // Matches backend
  config_details?: Record<string, any>; // Added to match backend
}

export interface Session {
  session_id: string;
  status: 'running' | 'stopped' | 'paused' | 'completed' | 'error';
  start_time: string;
  uptime_seconds: number;
  symbols: string[];
  current_portfolio?: any;
  performance_metrics?: any;
  last_updated?: string;
}

export type SystemStatus = {
  status: 'running' | 'stopped' | 'partial' | 'error' | 'unknown';
  total_agents: number;
  active_agents: number;
  error_agents: number;
  total_sessions: number;
  active_sessions: number;
  uptime_seconds?: number;
  start_time?: string;
  health_metrics?: {
    cpu_usage?: number;
    memory_usage?: number;
    disk_usage?: number;
    data_feed_connected?: boolean;
    data_feed_uptime?: number;
  };
  cpu_usage?: number;
  memory_usage?: number;
  disk_usage?: number;
};

interface SystemControlContextType {
  systemStatus: SystemStatus | null;
  agents: Agent[];
  sessions: Session[];
  isLoading: boolean;
  error: string | null;
  startSystem: () => Promise<void>;
  stopSystem: () => Promise<void>;
  startAgent: (agentId: string) => Promise<void>;
  stopAgent: (agentId: string) => Promise<void>;
  pauseSession: (sessionId: string) => Promise<void>;
  resumeSession: (sessionId: string) => Promise<void>;
  pauseAllSessions: () => Promise<void>;
  refreshSystemStatus: () => Promise<void>;
}

// Create the context with default values
const SystemControlContext = createContext<SystemControlContextType>({
  systemStatus: null,
  agents: [],
  sessions: [],
  isLoading: false,
  error: null,
  startSystem: async () => { },
  stopSystem: async () => { },
  startAgent: async () => { },
  stopAgent: async () => { },
  pauseSession: async () => { },
  resumeSession: async () => { },
  pauseAllSessions: async () => { },
  refreshSystemStatus: async () => { },
});

// Create a provider component
export const SystemControlProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { useMockData: shouldUseMockData } = useMockData(); // Get mock data preference from context
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      // Close any existing connection
      if (socket) {
        socket.close();
      }

      // Create a new WebSocket connection (Point to backend port 8000)
      const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//127.0.0.1:8000/ws/system`);

      ws.onopen = () => {
        console.log('WebSocket connection established');
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          // Handle different message types
          switch (message.type) {
            case 'system_status':
              setSystemStatus(message.data);
              break;
            case 'agent_update':
              // Update a single agent
              if (message.data && message.data.agent_id) {
                setAgents(prevAgents => {
                  const existingIndex = prevAgents.findIndex(a => a.agent_id === message.data.agent_id);
                  if (existingIndex >= 0) {
                    // Update existing agent
                    const updatedAgents = [...prevAgents];
                    updatedAgents[existingIndex] = message.data;
                    return updatedAgents;
                  } else {
                    // Add new agent
                    return [...prevAgents, message.data];
                  }
                });
              }
              break;
            case 'agents_update':
              // Update all agents
              if (Array.isArray(message.data)) {
                setAgents(message.data);
              }
              break;
            case 'session_update':
              // Update a single session
              if (message.data && message.data.session_id) {
                setSessions(prevSessions => {
                  const existingIndex = prevSessions.findIndex(s => s.session_id === message.data.session_id);
                  if (existingIndex >= 0) {
                    // Update existing session
                    const updatedSessions = [...prevSessions];
                    updatedSessions[existingIndex] = message.data;
                    return updatedSessions;
                  } else {
                    // Add new session
                    return [...prevSessions, message.data];
                  }
                });
              }
              break;
            case 'sessions_update':
              // Update all sessions
              if (Array.isArray(message.data)) {
                setSessions(message.data);
              }
              break;
            default:
              console.warn('Unknown message type:', message.type);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Consider a retry mechanism here
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
        // Consider a reconnect mechanism here
      };

      // Set the socket reference
      setSocket(ws);
    };

    // Initialize WebSocket
    try {
      connectWebSocket();
    } catch (wsError) {
      console.error('WebSocket setup error:', wsError);
    }

    // Cleanup on unmount
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, []);

  // Fetch initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        await fetchAgents();
        await fetchSessions();
        await refreshSystemStatus();
        setIsLoading(false);
      } catch (err) {
        console.error('Error fetching initial data:', err);
        setError('Failed to fetch system data');
        setIsLoading(false);
      }
    };

    fetchInitialData();
  }, [shouldUseMockData]); // Re-fetch data when mock data preference changes

  // The refreshSystemStatus function for periodic status updates
  const refreshSystemStatus = async () => {
    try {
      // Determine system status based on agents and sessions
      let status: 'running' | 'stopped' | 'partial' | 'error' | 'unknown' = 'unknown';

      // Count active and error agents
      const activeAgents = agents.filter(agent => agent.status === 'running').length;
      const errorAgents = agents.filter(agent => agent.status === 'error').length;

      // Count active sessions
      const activeSessions = sessions.filter(session => session.status === 'running').length;

      // Determine overall system status
      if (agents.length === 0 && sessions.length === 0) {
        status = 'unknown';
      } else if (activeAgents === 0 && activeSessions === 0) {
        status = 'stopped';
      } else if (activeAgents === agents.length && activeSessions === sessions.length) {
        status = 'running';
      } else if (errorAgents > 0) {
        status = 'error';
      } else {
        status = 'partial';
      }

      // Variables for metrics
      let uptime = 0;
      let startTime = '';
      let healthMetrics = null;

      // Check if we should use mock data
      if (shouldUseMockData) {
        console.log('Using mock system status data based on user preference');
        // Generate mock uptime and health metrics
        uptime = 7200; // 2 hours in seconds
        startTime = new Date(Date.now() - uptime * 1000).toISOString();
        healthMetrics = {
          cpu_usage: 25 + Math.random() * 15, // Random values between 25-40%
          memory_usage: 40 + Math.random() * 20, // Random values between 40-60%
          disk_usage: 30 + Math.random() * 10 // Random values between 30-40%
        };
      } else {
        // Try to get uptime from API
        try {
          const uptimeResponse = await axios.get(`${API_BASE_URL}/api/system/uptime`);
          if (uptimeResponse.data) {
            uptime = uptimeResponse.data.uptime_seconds || 0;
            startTime = uptimeResponse.data.start_time || '';
          }
        } catch (uptimeErr) {
          console.warn('Could not fetch uptime metrics:', uptimeErr);
          // In development mode, provide mock data as fallback
          if (process.env.NODE_ENV === 'development') {
            uptime = 3600; // 1 hour in seconds
            startTime = new Date(Date.now() - uptime * 1000).toISOString();
          }
        }

        // Try to get health metrics from API
        try {
          const healthResponse = await axios.get(`${API_BASE_URL}/api/system/health`);
          if (healthResponse.data) {
            healthMetrics = {
              cpu_usage: healthResponse.data.cpu_usage || 0,
              memory_usage: healthResponse.data.memory_usage || 0,
              disk_usage: healthResponse.data.disk_usage || 0
            };
          }
        } catch (healthErr) {
          console.warn('Could not fetch health metrics:', healthErr);
          // In development mode, provide mock health metrics as fallback
          if (process.env.NODE_ENV === 'development') {
            healthMetrics = {
              cpu_usage: 20 + Math.random() * 10, // Random values
              memory_usage: 35 + Math.random() * 15,
              disk_usage: 25 + Math.random() * 10
            };
          }
        }
      }

      setSystemStatus({
        status,
        total_agents: agents.length,
        active_agents: activeAgents,
        error_agents: errorAgents,
        total_sessions: sessions.length,
        active_sessions: activeSessions,
        uptime_seconds: uptime,
        start_time: startTime,
        ...(healthMetrics ? {
          health_metrics: healthMetrics,
          cpu_usage: healthMetrics.cpu_usage,
          memory_usage: healthMetrics.memory_usage,
          disk_usage: healthMetrics.disk_usage
        } : {})
      });
    } catch (err) {
      console.error('Error refreshing system status:', err);
      setError('Failed to refresh system status');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch agents from the backend
  const fetchAgents = async () => {
    try {
      setIsLoading(true);
      let agentsData: Agent[] = [];

      // Check if we should use mock data (based on user preference)
      if (shouldUseMockData) {
        console.log('Using mock agent data based on user preference');
        agentsData = await getMockAgents();
      } else {
        // Try to get real data from the API
        try {
          const response = await axios.get(`${API_BASE_URL}/api/system/agents`);
          agentsData = response.data;

          if (!Array.isArray(agentsData)) {
            throw new Error('Invalid response format');
          }

          // Transform backend data to match our frontend Agent interface if needed
          agentsData = agentsData.map(agent => {
            return {
              ...agent,
              // Ensure any required transformations between API and UI models
              status: agent.status.toLowerCase() as 'running' | 'stopped' | 'error' | 'initializing'
            };
          });
        } catch (apiError) {
          // If API fails and we're in development, fallback to mock data
          if (process.env.NODE_ENV === 'development') {
            console.warn('API call failed, falling back to mock agent data');
            agentsData = await getMockAgents();
          } else {
            console.error('Error fetching agents:', apiError);
            setError('Failed to fetch agents data');
            throw apiError;
          }
        }
      }

      // If we have no agents data but we're in development mode, use mock data
      if (agentsData.length === 0 && process.env.NODE_ENV === 'development') {
        console.warn('No agents found, using mock agents data');
        const mockAgentData = await getMockAgents();
        setAgents(mockAgentData);
        return;
      }

      // Set our agents state
      setAgents(agentsData);

      // Optionally get real-time status on each agent
      for (const agent of agentsData) {
        try {
          const statusResponse = await axios.get(`${API_BASE_URL}/api/system/agents/${agent.agent_id}/status`);
          const updatedAgent = {
            ...agent,
            status: statusResponse.data.status.toLowerCase() as 'running' | 'stopped' | 'error' | 'initializing',
            last_updated: statusResponse.data.last_updated || agent.last_updated
          };

          // Update this specific agent in our state
          setAgents(prevAgents =>
            prevAgents.map(a => a.agent_id === agent.agent_id ? updatedAgent : a)
          );
        } catch (statusError) {
          console.warn(`Could not get status for agent ${agent.agent_id}:`, statusError);
        }
      }

      // Now try to get performance metrics for each agent
      for (const agent of agentsData) {
        try {
          const metricsResponse = await axios.get(`${API_BASE_URL}/api/system/agents/${agent.agent_id}/metrics`);
          const updatedAgent = {
            ...agent,
            metrics: metricsResponse.data
          };

          // Update this specific agent in our state
          setAgents(prevAgents =>
            prevAgents.map(a => a.agent_id === agent.agent_id ? updatedAgent : a)
          );
        } catch (metricsError) {
          console.warn(`Could not get metrics for agent ${agent.agent_id}:`, metricsError);
        }
      }
    } catch (err) {
      console.error('Error in fetchAgents:', err);
      setError('Error fetching agents');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch sessions from the backend
  const fetchSessions = async () => {
    try {
      setIsLoading(true);

      // Check if we should use mock data (based on user preference)
      if (shouldUseMockData) {
        console.log('Using mock session data based on user preference');
        // You need to create or import mock sessions data
        // For now, using an empty array
        setSessions([]);
      } else {
        try {
          const response = await axios.get(`${API_BASE_URL}/api/paper-trading/sessions`);
          const sessionsData = response.data;

          if (!Array.isArray(sessionsData)) {
            throw new Error('Invalid sessions response format');
          }

          setSessions(sessionsData);
        } catch (apiError) {
          console.error('Error fetching sessions:', apiError);
          // For development - set empty sessions array if API fails
          setSessions([]);
          if (process.env.NODE_ENV !== 'development') {
            setError('Failed to fetch sessions data');
            throw apiError;
          }
        }
      }
    } catch (err) {
      console.error('Error in fetchSessions:', err);
      setError('Error fetching sessions');
    } finally {
      setIsLoading(false);
    }
  };

  const startSystem = async () => {
    try {
      console.log('Starting system at:', `${API_BASE_URL}/api/system/start`);
      const response = await axios.post(`${API_BASE_URL}/api/system/start`);
      console.log('Start system response:', response.data);
      await refreshSystemStatus();
      return response.data;
    } catch (err: any) {
      console.error('Error starting system:', err);
      let errorMessage = 'Failed to start system';
      if (err.response) {
        errorMessage = `Server error: ${err.response.status} - ${err.response.data?.detail || err.response.statusText}`;
      } else if (err.request) {
        errorMessage = 'No response received from server. Please check if the backend is running.';
      } else {
        errorMessage = `Error: ${err.message}`;
      }
      setError(errorMessage);
      throw err;
    }
  };

  const stopSystem = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/system/stop`);
      refreshSystemStatus();
    } catch (err) {
      console.error('Error stopping system:', err);
      setError('Failed to stop system');
    }
  };

  const startAgent = async (agentId: string) => {
    try {
      await axios.post(`${API_BASE_URL}/api/system/agents/${agentId}/start`);
      refreshSystemStatus();
    } catch (err) {
      console.error(`Error starting agent ${agentId}:`, err);
      setError(`Failed to start agent ${agentId}`);
    }
  };

  const stopAgent = async (agentId: string) => {
    try {
      await axios.post(`${API_BASE_URL}/api/system/agents/${agentId}/stop`);
      refreshSystemStatus();
    } catch (err) {
      console.error(`Error stopping agent ${agentId}:`, err);
      setError(`Failed to stop agent ${agentId}`);
    }
  };

  const pauseSession = async (sessionId: string) => {
    try {
      await axios.post(`${API_BASE_URL}/api/paper-trading/sessions/${sessionId}/pause`);
      refreshSystemStatus();
    } catch (err) {
      console.error(`Error pausing session ${sessionId}:`, err);
      setError(`Failed to pause session ${sessionId}`);
    }
  };

  const resumeSession = async (sessionId: string) => {
    try {
      await axios.post(`${API_BASE_URL}/api/paper-trading/sessions/${sessionId}/resume`);
      refreshSystemStatus();
    } catch (err) {
      console.error(`Error resuming session ${sessionId}:`, err);
      setError(`Failed to resume session ${sessionId}`);
    }
  };

  const pauseAllSessions = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/paper-trading/sessions/pause-all`);
      refreshSystemStatus();
    } catch (err) {
      console.error('Error pausing all sessions:', err);
      setError('Failed to pause all sessions');
    }
  };

  // Context value
  const value = {
    systemStatus,
    agents,
    sessions,
    isLoading,
    error,
    startSystem,
    stopSystem,
    startAgent,
    stopAgent,
    pauseSession,
    resumeSession,
    pauseAllSessions,
    refreshSystemStatus,
  };

  return (
    <SystemControlContext.Provider value={value}>
      {children}
    </SystemControlContext.Provider>
  );
};

// Custom hook for using the context
export const useSystemControl = () => useContext(SystemControlContext);

export default SystemControlContext;
