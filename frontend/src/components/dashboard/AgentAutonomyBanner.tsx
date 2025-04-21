import React from 'react';

interface AgentAutonomyBannerProps {
  status?: 'running' | 'stopped' | 'error';
  lastUpdated?: string;
  lastTrade?: {
    symbol: string;
    side: string;
    price: number;
    timestamp: string;
  };
  currentStrategy?: string;
  reasoning?: string;
  activityFeed?: Array<{
    time: string;
    message: string;
  }>;
}

const statusMap = {
  running: {
    color: 'bg-green-600',
    label: 'Autonomous Agent: ACTIVE',
    sub: 'The trading agent is running autonomously.'
  },
  stopped: {
    color: 'bg-gray-500',
    label: 'Autonomous Agent: PAUSED',
    sub: 'The trading agent is currently stopped.'
  },
  error: {
    color: 'bg-red-600',
    label: 'Autonomous Agent: ERROR',
    sub: 'The trading agent encountered an error.'
  },
  unknown: {
    color: 'bg-yellow-500',
    label: 'Autonomous Agent: UNKNOWN',
    sub: 'Unable to determine agent status.'
  }
};

const AgentAutonomyBanner: React.FC<AgentAutonomyBannerProps> = ({ status, lastUpdated, lastTrade, currentStrategy, reasoning, activityFeed }) => {
  const state = statusMap[status || 'unknown'];
  // Glowing border for running
  const borderAnim = status === 'running' ? 'border-4 border-green-400 animate-glow' : 'border-2';
  return (
    <div className={`w-full flex flex-col items-center py-3 px-4 mb-4 rounded-lg shadow-lg text-white ${state.color} ${borderAnim} border-white animate-fade-in relative`}>
      <div className="flex items-center gap-2">
        <span className={`inline-block w-3 h-3 rounded-full bg-white ${status === 'running' ? 'animate-pulse' : ''}`}></span>
        <span className="font-bold text-lg tracking-wide">{state.label}</span>
        {currentStrategy && (
          <span className="ml-2 px-2 py-1 rounded bg-white/20 text-xs font-semibold">Strategy: {currentStrategy}</span>
        )}
      </div>
      <div className="text-sm mt-1 opacity-90">{state.sub}</div>
      {lastTrade && (
        <div className="text-xs mt-1 opacity-90">
          <span className="font-semibold">Last trade:</span> {lastTrade.side.toUpperCase()} {lastTrade.symbol} @ ${lastTrade.price.toFixed(2)}
          <span className="ml-2 text-gray-200/70">({new Date(lastTrade.timestamp).toLocaleTimeString()})</span>
        </div>
      )}
      {lastUpdated && (
        <div className="text-xs mt-1 opacity-70">Last updated: {lastUpdated}</div>
      )}
      {reasoning && (
        <div className="mt-2 w-full max-w-xl text-xs bg-white/10 rounded p-2 text-white/90">
          <span className="font-semibold text-white/80">Agent reasoning:</span> {reasoning}
        </div>
      )}
      {activityFeed && activityFeed.length > 0 && (
        <div className="mt-2 w-full max-w-xl bg-black/20 rounded p-2 overflow-y-auto max-h-24 text-xs animate-fade-in">
          <div className="font-semibold text-white/80 mb-1">Live Activity:</div>
          <ul className="space-y-1">
            {activityFeed.slice(-5).reverse().map((item, idx) => (
              <li key={idx} className="flex gap-2 items-center">
                <span className="text-green-200/80">{item.time}</span>
                <span className="text-white/90">{item.message}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {/* Glowing/animated border effect for running */}
      <style>{`
        @keyframes glow {
          0% { box-shadow: 0 0 8px 2px #22d3ee, 0 0 0px 0px #22d3ee; }
          50% { box-shadow: 0 0 24px 8px #22d3ee, 0 0 8px 2px #22d3ee; }
          100% { box-shadow: 0 0 8px 2px #22d3ee, 0 0 0px 0px #22d3ee; }
        }
        .animate-glow { animation: glow 1.6s infinite alternate; }
      `}</style>
    </div>
  );
};

export default AgentAutonomyBanner;
