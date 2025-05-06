import React from 'react';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { useNavigate } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';
import '../../../styles/paperTrading.css';

const PaperTradingSessionsList: React.FC = () => {
  const { state, startPaperTrading, stopPaperTrading, refreshSessions } = usePaperTrading();
  const navigate = useNavigate();

  // Handle session selection
  const handleSelectSession = (sessionId: string) => {
    navigate(`/paper-trading/session/${sessionId}`);
  };

  // Handle stop session
  const handleStopSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent session selection
    await stopPaperTrading(sessionId);
  };

  // Handle refresh sessions
  const handleRefresh = () => {
    refreshSessions();
  };

  // Handle start new session
  const handleStartNew = () => {
    navigate('/paper-trading/new');
  };

  // Format the session start time
  const formatSessionTime = (timeString?: string) => {
    if (!timeString) return 'N/A';
    
    try {
      const date = new Date(timeString);
      return formatDistanceToNow(date, { addSuffix: true });
    } catch (error) {
      return timeString;
    }
  };

  // Get status badge class
  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'running':
        return 'status-badge status-running';
      case 'completed':
        return 'status-badge status-completed';
      case 'error':
        return 'status-badge status-error';
      case 'stopping':
        return 'status-badge status-stopping';
      case 'starting':
        return 'status-badge status-starting';
      default:
        return 'status-badge';
    }
  };

  return (
    <div className="paper-trading-sessions-list">
      <div className="sessions-header">
        <h2>Paper Trading Sessions</h2>
        <div className="sessions-actions">
          <button onClick={handleRefresh} className="refresh-button">
            Refresh
          </button>
          <button onClick={handleStartNew} className="start-button">
            Start New Session
          </button>
        </div>
      </div>

      {state.isLoading && (
        <div className="sessions-loading">
          <p>Loading sessions...</p>
        </div>
      )}

      {!state.isLoading && state.activeSessions.length === 0 && (
        <div className="no-sessions">
          <p>No active paper trading sessions found.</p>
          <button onClick={handleStartNew} className="primary-button">
            Start New Session
          </button>
        </div>
      )}

      {!state.isLoading && state.activeSessions.length > 0 && (
        <div className="sessions-grid">
          {state.activeSessions.map((session) => (
            <div 
              key={session.session_id} 
              className="session-card"
              onClick={() => handleSelectSession(session.session_id)}
            >
              <div className="session-header">
                <div className="session-id">
                  {session.session_id.substring(0, 8)}...
                </div>
                <div className={getStatusBadgeClass(session.status)}>
                  {session.status}
                </div>
              </div>
              
              <div className="session-details">
                <div className="detail-row">
                  <span className="detail-label">Started:</span>
                  <span className="detail-value">{formatSessionTime(session.start_time)}</span>
                </div>
                
                <div className="detail-row">
                  <span className="detail-label">Uptime:</span>
                  <span className="detail-value">
                    {session.uptime_seconds 
                      ? `${Math.floor(session.uptime_seconds / 60)} min ${session.uptime_seconds % 60} sec` 
                      : 'N/A'}
                  </span>
                </div>
                
                <div className="detail-row">
                  <span className="detail-label">Symbols:</span>
                  <span className="detail-value">
                    {session.symbols ? session.symbols.join(', ') : 'N/A'}
                  </span>
                </div>
              </div>
              
              <div className="session-actions">
                <button 
                  onClick={(e) => handleSelectSession(session.session_id)} 
                  className="view-button"
                >
                  View Details
                </button>
                
                {session.status === 'running' && (
                  <button 
                    onClick={(e) => handleStopSession(session.session_id, e)} 
                    className="stop-button"
                  >
                    Stop Session
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default PaperTradingSessionsList;