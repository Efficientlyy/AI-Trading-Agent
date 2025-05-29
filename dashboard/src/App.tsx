import React, { useEffect, useState } from 'react';
import { getSessions, pauseAll, resumeAll, stopAll, pauseSession, resumeSession, stopSession } from './api/paperTradingApi';
import { SystemControlPanel } from './components/SystemControlPanel';
import { SessionTable } from './components/SessionTable';

function calculateSystemStatus(sessions) {
  const states = sessions.map(s => s.status);
  if (states.every(s => s === 'running')) return 'Running';
  if (states.every(s => s === 'paused')) return 'Paused';
  if (states.every(s => ['stopped', 'completed'].includes(s))) return 'Stopped';
  return 'Mixed';
}

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [systemStatus, setSystemStatus] = useState('Loading...');
  const [error, setError] = useState('');

  const fetchSessions = async () => {
    try {
      const data = await getSessions();
      setSessions(data.sessions);
      setSystemStatus(calculateSystemStatus(data.sessions));
    } catch (e) {
      setError('Failed to fetch sessions');
    }
  };

  useEffect(() => { fetchSessions(); }, []);

  useEffect(() => {
    const interval = setInterval(fetchSessions, 5000);
    return () => clearInterval(interval);
  }, []);

  const handlePauseAll = async () => { await pauseAll(); fetchSessions(); };
  const handleResumeAll = async () => { await resumeAll(); fetchSessions(); };
  const handleStopAll = async () => { await stopAll(); fetchSessions(); };
  const handlePause = async (id) => { await pauseSession(id); fetchSessions(); };
  const handleResume = async (id) => { await resumeSession(id); fetchSessions(); };
  const handleStop = async (id) => { await stopSession(id); fetchSessions(); };

  return (
    <div style={{ padding: 32 }}>
      <h1>Paper Trading System Control Dashboard</h1>
      <SystemControlPanel
        onPauseAll={handlePauseAll}
        onResumeAll={handleResumeAll}
        onStopAll={handleStopAll}
        systemStatus={systemStatus}
      />
      <SessionTable
        sessions={sessions}
        onPause={handlePause}
        onResume={handleResume}
        onStop={handleStop}
      />
      {error && <div style={{ color: 'red' }}>{error}</div>}
    </div>
  );
}
