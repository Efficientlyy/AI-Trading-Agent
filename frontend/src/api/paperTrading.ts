import { createAuthenticatedClient } from './client';

interface PaperTradingConfig {
  config_path: string;
  duration_minutes: number;
  interval_minutes: number;
  autonomous_mode?: boolean;
}

interface PaperTradingStatus {
  status: 'idle' | 'starting' | 'running' | 'stopping' | 'completed' | 'error';
  uptime_seconds?: number;
  symbols?: string[];
  current_portfolio?: any;
  recent_trades?: any[];
  performance_metrics?: any;
}

interface PaperTradingSession {
  session_id: string;
  status: 'starting' | 'running' | 'stopping' | 'completed' | 'error';
  start_time?: string;
  uptime_seconds?: number;
  symbols?: string[];
  current_portfolio?: any;
}

interface PaperTradingResults {
  portfolio_history: any[];
  trades: any[];
  performance_metrics: any;
}

/**
 * Paper Trading API client
 */
export const paperTradingApi = {
  /**
   * Start a new paper trading session
   */
  startPaperTrading: async (config: PaperTradingConfig): Promise<{ status: string; session_id: string; message: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.post('/api/paper-trading/start', config);
    return response.data;
  },

  /**
   * Stop the current paper trading session
   */
  stopPaperTrading: async (sessionId: string): Promise<{ status: string; message: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.post(`/api/paper-trading/stop/${sessionId}`);
    return response.data;
  },

  /**
   * Get the current status of a paper trading session
   * @param sessionId The ID of the session to get status for
   */
  getStatus: async (sessionId: string): Promise<PaperTradingStatus> => {
    const client = createAuthenticatedClient();
    const response = await client.get(`/api/paper-trading/status?session_id=${sessionId}`);
    return response.data;
  },

  /**
   * Get the results of the last completed paper trading session
   */
  getResults: async (sessionId: string): Promise<PaperTradingResults> => {
    const client = createAuthenticatedClient();
    const response = await client.get(`/api/paper-trading/results/${sessionId}`);
    return response.data;
  },

  /**
   * Get all paper trading sessions
   */
  getSessions: async (): Promise<{ sessions: PaperTradingSession[] }> => {
    const client = createAuthenticatedClient();
    const response = await client.get('/api/paper-trading/sessions');
    return response.data;
  },

  /**
   * Get details for a specific paper trading session
   */
  getSessionDetails: async (sessionId: string): Promise<PaperTradingSession> => {
    const client = createAuthenticatedClient();
    const response = await client.get(`/api/paper-trading/sessions/${sessionId}`);
    return response.data;
  },

  /**
   * Get alerts for a specific paper trading session
   */
  getSessionAlerts: async (sessionId: string): Promise<{ alerts: any[] }> => {
    const client = createAuthenticatedClient();
    const response = await client.get(`/api/paper-trading/alerts/${sessionId}`);
    return response.data;
  },

  /**
   * Update alert settings for a paper trading session
   */
  updateAlertSettings: async (sessionId: string, settings: any): Promise<{ status: string; message: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.post(`/api/paper-trading/alerts/${sessionId}/settings`, settings);
    return response.data;
  },

  /**
   * Add an alert for a paper trading session
   */
  addAlert: async (sessionId: string, alert: any): Promise<{ status: string; message: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.post(`/api/paper-trading/alerts/${sessionId}`, alert);
    return response.data;
  }
};