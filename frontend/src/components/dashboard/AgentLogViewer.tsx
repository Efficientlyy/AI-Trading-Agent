import React, { useEffect, useState } from 'react';
import { systemControlApi } from '../../api/systemControl';

interface AgentLogViewerProps {
  agentId: string;
  lines?: number;
  autoRefresh?: boolean;
  refreshInterval?: number; // ms
}

const AgentLogViewer: React.FC<AgentLogViewerProps> = ({
  agentId,
  lines = 100,
  autoRefresh = false,
  refreshInterval = 5000,
}) => {
  const [logLines, setLogLines] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await systemControlApi.getAgentLogs(agentId, lines);
      if (result && Array.isArray(result.log_lines)) {
        setLogLines(result.log_lines);
        setError(null); // Clear previous errors if logs are fetched successfully
      } else {
        const warning = `Log data for agent ${agentId} is not in the expected format. Received: ${JSON.stringify(result)}`;
        console.warn(warning);
        setError(warning); // Display this warning in the UI via the error state
        setLogLines([]);
      }
    } catch (e: any) {
      const errorMsg = e?.response?.data?.detail || e.message || 'Failed to fetch logs';
      setError(errorMsg);
      setLogLines([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    if (autoRefresh) {
      const interval = setInterval(fetchLogs, refreshInterval);
      return () => clearInterval(interval);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId, lines, autoRefresh, refreshInterval]);

  return (
    <div className="agent-log-viewer bg-gray-900 text-green-200 rounded p-4 shadow-inner max-h-96 overflow-auto">
      <div className="flex items-center mb-2">
        <span className="font-bold mr-2">Agent Log:</span>
        <span className="text-xs text-gray-400">{agentId}</span>
        {loading && <span className="ml-4 text-xs text-yellow-400">Loading... (State: loading)</span>}
        {error && <span className="ml-4 text-xs text-red-400">Error: {error} (State: error)</span>}
      </div>
      {/* More aggressive diagnostic styling for the pre tag */}
      <pre
        className="whitespace-pre-wrap text-xs leading-tight"
        style={{
          border: '2px solid purple !important',
          minHeight: '50px !important',
          padding: '10px !important',
          color: 'white !important',
          backgroundColor: 'rgba(128, 0, 128, 0.3) !important', /* Light purple background */
          display: 'block !important', /* Ensure it's a block */
          boxSizing: 'border-box',
          width: '100%'
        }}
      >
        DIRECT PRE TEXT: Visible? loading: {loading.toString()}, error: {error || 'null'}, logLines.length: {logLines.length}
        <br /> {/* Line break for clarity */}
        {!loading && !error && logLines.length === 0 && <span style={{ color: 'orange', display: 'block' }}>No logs available (State: empty logLines)</span>}
        {!loading && !error && logLines.length > 0 && logLines.every(line => typeof line === 'string' && line.trim() === '') && <span style={{ color: 'yellow', display: 'block' }}>Logs fetched, but all lines are empty/whitespace. (Count: {logLines.length})</span>}
        {!loading && !error && logLines.length > 0 && !logLines.every(line => typeof line === 'string' && line.trim() === '') && <span style={{ color: 'cyan', display: 'block' }}>Rendering {logLines.length} log lines:</span>}
        {logLines.map((line, idx) => (
          <div key={idx} style={{ borderBottom: '1px dotted #aaa', margin: '2px 0', color: 'white' }}>{`[L${idx}]: "${line}"`}</div>
        ))}
      </pre>
    </div>
  );
};

export default AgentLogViewer;
