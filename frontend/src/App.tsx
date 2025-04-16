import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import NotificationProvider from './components/common/NotificationSystem';
import { DataSourceProvider } from './context/DataSourceContext';
import ProtectedRoute from './components/common/ProtectedRoute';
import MainLayout from './components/layout/MainLayout';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import ErrorBoundary from './components/common/ErrorBoundary';
import { AnimatedTransition, AnimationType } from './components/common/AnimatedTransition';

// Import CSS for theme variables
import './styles/theme.css';

function App() {
  return (
    <ErrorBoundary>
      <DataSourceProvider>
        <ThemeProvider>
          <AuthProvider>
            <NotificationProvider>
              <Router>
                <AnimatedTransition
                  type={AnimationType.FADE}
                  className="min-h-screen bg-bg-primary text-text-primary transition-colors"
                >
                  <Routes>
                    {/* Public routes */}
                    <Route path="/login" element={<Login />} />
                    <Route path="/register" element={<Register />} />

                    {/* Protected routes */}
                    <Route path="/" element={
                      <ProtectedRoute>
                        <MainLayout />
                      </ProtectedRoute>
                    }>
                      <Route index element={<Dashboard />} />
                      {/* Add more routes for other dashboard pages */}
                      <Route path="portfolio" element={<div>Portfolio Page (Coming Soon)</div>} />
                      <Route path="trading" element={<div>Trading Page (Coming Soon)</div>} />
                      <Route path="backtest" element={<div>Backtest Page (Coming Soon)</div>} />
                      <Route path="strategies" element={<div>Strategies Page (Coming Soon)</div>} />
                      <Route path="sentiment" element={<div>Sentiment Page (Coming Soon)</div>} />
                      <Route path="settings" element={<div>Settings Page (Coming Soon)</div>} />
                    </Route>

                    {/* Fallback route */}
                    <Route path="*" element={<Navigate to="/" replace />} />
                  </Routes>
                </AnimatedTransition>
              </Router>
            </NotificationProvider>
          </AuthProvider>
        </ThemeProvider>
      </DataSourceProvider>
    </ErrorBoundary>
  );
}

export default App;
