import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { SentimentDashboard, EnhancedSentimentDashboard } from '../components/dashboard';

/**
 * Main application routes
 */
const AppRoutes: React.FC = () => {
  return (
    <Router>
      <Routes>
        {/* Redirect root to the enhanced sentiment dashboard */}
        <Route path="/" element={<Navigate to="/sentiment/enhanced" replace />} />
        
        {/* Sentiment Dashboard Routes */}
        <Route path="/sentiment">
          <Route index element={<SentimentDashboard />} />
          <Route path="basic" element={<SentimentDashboard />} />
          <Route path="enhanced" element={<EnhancedSentimentDashboard />} />
        </Route>
        
        {/* 404 - Not Found */}
        <Route path="*" element={<div>Page Not Found</div>} />
      </Routes>
    </Router>
  );
};

export default AppRoutes;
