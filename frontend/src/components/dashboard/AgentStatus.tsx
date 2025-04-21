import React from 'react';

interface AgentStatusProps {
  status?: 'running' | 'stopped' | 'error';
  reasoning?: string;
  lastUpdated?: string;
}

const statusColors = {
  running: 'bg-green-500',
  stopped: 'bg-gray-400',
  error: 'bg-red-500',
};

const AgentStatus: React.FC<AgentStatusProps> = ({ status, reasoning, lastUpdated }) => {
  if (!status) {
    return (
      <div className="rounded-lg shadow p-4 bg-white border border-gray-200 flex flex-col gap-2 animate-pulse">
        <div className="flex items-center gap-2">
          <span className="inline-block w-3 h-3 rounded-full bg-gray-300"></span>
          <span className="font-semibold text-lg text-gray-400">Loading agent status...</span>
          <span className="ml-auto text-xs text-gray-300">Last updated: --</span>
        </div>
        <div className="mt-2">
          <span className="block text-xs text-gray-300 mb-1">Reasoning / Thoughts:</span>
          <div className="rounded bg-gray-50 p-2 text-sm text-gray-300 min-h-[48px]">Loading...</div>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-lg shadow p-4 bg-white border border-gray-200 flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className={`inline-block w-3 h-3 rounded-full ${statusColors[status]}`}></span>
        <span className="font-semibold text-lg">Agent Status: {status.charAt(0).toUpperCase() + status.slice(1)}</span>
        <span className="ml-auto text-xs text-gray-500">Last updated: {lastUpdated}</span>
      </div>
      <div className="mt-2">
        <span className="block text-xs text-gray-400 mb-1">Reasoning / Thoughts:</span>
        <div className="rounded bg-gray-50 p-2 text-sm text-gray-700 min-h-[48px]">
          {reasoning}
        </div>
      </div>
    </div>
  );
};

export default AgentStatus;
