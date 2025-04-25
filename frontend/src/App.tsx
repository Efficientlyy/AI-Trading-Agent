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
import Trade from './pages/Trade';
import Strategies from './pages/Strategies';
import ErrorBoundary from './components/common/ErrorBoundary';
import { AnimatedTransition, AnimationType } from './components/common/AnimatedTransition';
import Settings from './pages/Settings';
import Portfolio from './pages/Portfolio';
import ApiHealthPage from './pages/ApiHealthPage';
import ApiLogsPage from './pages/ApiLogsPage';
import AlertsPage from './pages/AlertsPage';
import PerformancePage from './pages/PerformancePage';
import PerformanceTestPage from './pages/PerformanceTestPage';
import Sentiment from './pages/Sentiment';
import { SelectedAssetProvider } from './context/SelectedAssetContext';
import { AlertsProvider } from './context/AlertsContext';

// Import CSS for theme variables
import './styles/theme.css';

function App() {
  return (
    <ErrorBoundary>
      <DataSourceProvider>
        <ThemeProvider>
          <NotificationProvider>
            <AuthProvider>
              <AlertsProvider>
                <SelectedAssetProvider>
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
                          <Route path="portfolio" element={<Portfolio />} />
                          <Route path="/trade" element={
                            <ProtectedRoute>
                              <Trade />
                            </ProtectedRoute>
                          } />
                          <Route path="backtest" element={<div>Backtest Page (Coming Soon)</div>} />
                          <Route path="strategies" element={<Strategies />} />
                          <Route path="sentiment" element={<Sentiment />} />
                          <Route path="api-health" element={<ApiHealthPage />} />
                          <Route path="api-logs" element={<ApiLogsPage />} />
                          <Route path="alerts" element={<AlertsPage />} />
                          <Route path="performance" element={<PerformancePage />} />
                          <Route path="performance-test" element={<PerformanceTestPage />} />
                          <Route path="settings" element={<Settings />} />
                        </Route>

                        {/* Fallback route */}
                        <Route path="*" element={<Navigate to="/" replace />} />
                      </Routes>
                    </AnimatedTransition>
                  </Router>
                </SelectedAssetProvider>
              </AlertsProvider>
            </AuthProvider>
          </NotificationProvider>
        </ThemeProvider>
      </DataSourceProvider>
    </ErrorBoundary>
  );
}

export default App;
