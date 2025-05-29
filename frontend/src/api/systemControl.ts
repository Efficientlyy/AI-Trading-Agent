import { createAuthenticatedClient } from './client';

export const systemControlApi = {
  /**
   * Fetch logs for a specific agent (session)
   * @param agentId The agent/session ID
   * @param lines Number of lines to retrieve (default 100, max 1000)
   */
  getAgentLogs: async (agentId: string, lines: number = 100): Promise<{ agent_id: string; log_lines: string[]; line_count?: number }> => {
    const client = createAuthenticatedClient();
    try {
      const response = await client.get(`/api/system/agents/${agentId}/logs`, { params: { lines } });
      // Ensure response.data and response.data.log_lines exist and log_lines is an array
      if (response && response.data && Array.isArray(response.data.log_lines)) {
        // The backend might return line_count or total_lines, ensure we map it if present
        const dataToReturn: { agent_id: string; log_lines: string[]; line_count?: number } = {
          agent_id: response.data.agent_id,
          log_lines: response.data.log_lines,
        };
        if (response.data.line_count !== undefined) {
          dataToReturn.line_count = response.data.line_count;
        } else if (response.data.total_lines !== undefined) { // Fallback for older key
          dataToReturn.line_count = response.data.total_lines;
        }
        return dataToReturn;
      } else {
        // Log an error and return a default structure if data is not as expected
        console.error(`[API] getAgentLogs for ${agentId} received unexpected data:`, response?.data);
        return { agent_id: agentId, log_lines: [`Error: Unexpected log data format from server.`], line_count: 1 };
      }
    } catch (error: any) {
      console.error(`[API] Error fetching logs for ${agentId}:`, error);
      // Re-throw or return a structured error object that AgentLogViewer can handle
      // For now, return an error message within the lines array
      const errorMessage = error?.response?.data?.detail || error.message || 'Unknown error fetching logs.';
      return { agent_id: agentId, log_lines: [`Error fetching logs: ${errorMessage}`], line_count: 1 };
    }
  },
};
