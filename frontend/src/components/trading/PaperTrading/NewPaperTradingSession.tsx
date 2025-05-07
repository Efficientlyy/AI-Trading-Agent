import React, { useState } from 'react';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { useNavigate } from 'react-router-dom';
import '../../../styles/paperTrading.css';

const NewPaperTradingSession: React.FC = () => {
  const { startPaperTrading, state } = usePaperTrading();
  const navigate = useNavigate();
  
  const [configPath, setConfigPath] = useState('configs/default_config.yaml');
  const [duration, setDuration] = useState(60);
  const [interval, setInterval] = useState(1);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      await startPaperTrading({
        configPath,
        duration,
        interval
      });
      
      // If we have a selectedSessionId after starting, navigate to it
      if (state.selectedSessionId) {
        navigate(`/paper-trading/session/${state.selectedSessionId}`);
      } else {
        // Otherwise go back to sessions list
        navigate('/paper-trading');
      }
    } catch (error) {
      console.error('Error starting paper trading session:', error);
    }
  };
  
  const handleCancel = () => {
    navigate('/paper-trading');
  };
  
  return (
    <div className="new-paper-trading-session-container">
      <div className="new-paper-trading-session">
        <div className="form-header">
          <h2>Start New Paper Trading Session</h2>
          <button onClick={handleCancel} className="back-button">
            <i className="fas fa-arrow-left"></i> Back to Sessions
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="paper-trading-form">
          <div className="form-group">
            <label htmlFor="configPath">Configuration File</label>
            <input
              type="text"
              id="configPath"
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              className="form-control"
              required
            />
            <small className="form-text">Path to the configuration file</small>
          </div>
          
          <div className="form-group">
            <label htmlFor="duration">Duration (minutes)</label>
            <input
              type="number"
              id="duration"
              value={duration}
              onChange={(e) => setDuration(parseInt(e.target.value))}
              className="form-control"
              min="1"
              required
            />
            <small className="form-text">Duration of the paper trading session in minutes</small>
          </div>
          
          <div className="form-group">
            <label htmlFor="interval">Interval (minutes)</label>
            <input
              type="number"
              id="interval"
              value={interval}
              onChange={(e) => setInterval(parseInt(e.target.value))}
              className="form-control"
              min="1"
              required
            />
            <small className="form-text">Interval between trades in minutes</small>
          </div>
          
          <div className="form-actions">
            <button type="button" onClick={handleCancel} className="cancel-button">
              Cancel
            </button>
            <button type="submit" className="submit-button" disabled={state.isLoading}>
              {state.isLoading ? 'Starting...' : 'Start Session'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NewPaperTradingSession;
