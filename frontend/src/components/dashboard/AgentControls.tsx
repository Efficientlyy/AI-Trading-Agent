import React from 'react';

interface AgentControlsProps {
  status?: 'running' | 'stopped' | 'error';
  onStart: () => void;
  onStop: () => void;
  isLoading?: boolean;
}

const AgentControls: React.FC<AgentControlsProps> = ({ status, onStart, onStop, isLoading }) => {
  return (
    <div className="flex items-center gap-4 p-2 bg-gray-50 border rounded shadow-sm mb-2">
      <button
        className="px-4 py-1 rounded bg-green-600 text-white font-medium disabled:opacity-50"
        onClick={onStart}
        disabled={isLoading || status === 'running'}
      >
        Start
      </button>
      <button
        className="px-4 py-1 rounded bg-red-600 text-white font-medium disabled:opacity-50"
        onClick={onStop}
        disabled={isLoading || status === 'stopped'}
      >
        Stop
      </button>
      <span className="ml-4 text-xs text-gray-500">
        Status: <span className={`font-semibold ${status === 'running' ? 'text-green-600' : status === 'error' ? 'text-red-600' : 'text-gray-600'}`}>{status || 'unknown'}</span>
      </span>
    </div>
  );
};

export default AgentControls;
