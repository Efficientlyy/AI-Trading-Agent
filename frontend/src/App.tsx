import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import NotificationProvider from './components/common/NotificationSystem';
import { DataSourceProvider } from './context/DataSourceContext';
import { PaperTradingProvider } from './context/PaperTradingContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
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
import SentimentAnalysis from './pages/SentimentAnalysis';
import TradingSignals from './pages/TradingSignals';
import AdvancedSignals from './pages/AdvancedSignals';
import BacktestResultPage from './pages/BacktestResultPage';
import BacktestHistoryPage from './pages/BacktestHistoryPage';
import PaperTradingPage from './pages/PaperTradingPage';
import NewPaperTradingPage from './pages/NewPaperTradingPage';
import PaperTradingDashboard from './components/trading/PaperTrading/PaperTradingDashboard';
import PaperTradingSessionPage from './pages/PaperTradingSessionPage';
import { SelectedAssetProvider } from './context/SelectedAssetContext';
import { AlertsProvider } from './context/AlertsContext';

// Import CSS for theme variables
import './styles/theme.css';

// Import dark theme overrides for better visibility
import './styles/darkThemeOverrides.css';

// Import Mantine core styles and provider
import '@mantine/core/styles.css';
import { MantineProvider, createTheme } from '@mantine/core';

// Create a client
const queryClient = new QueryClient();

// Mantine Theme (can be customized later)
const theme = createTheme({
  /** Put your mantine theme override here */
});

function App() {
  return (
    <ErrorBoundary>
      {/* Provide the client to your App - must be at the top level */}
      <QueryClientProvider client={queryClient}>
        <DataSourceProvider>
          <ThemeProvider>
            {/* Mantine Provider must wrap components using Mantine */}
            <MantineProvider theme={theme} defaultColorScheme="auto">
              <NotificationProvider>
                <AuthProvider>
                  <AlertsProvider>
                    <SelectedAssetProvider>
                      <PaperTradingProvider>
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
                                <Route path="sentiment-analysis" element={<SentimentAnalysis />} />
                                <Route path="trading-signals" element={<TradingSignals />} />
                                <Route path="advanced-signals" element={<AdvancedSignals />} />
                                <Route path="paper-trading" element={<PaperTradingPage />} />
                                <Route path="paper-trading/new" element={<NewPaperTradingPage />} />
                                <Route path="paper-trading/session/:sessionId" element={<PaperTradingSessionPage />} />
                                <Route path="api-health" element={<ApiHealthPage />} />
                                <Route path="api-logs" element={<ApiLogsPage />} />
                                <Route path="alerts" element={<AlertsPage />} />
                                <Route path="performance" element={<PerformancePage />} />
                                <Route path="performance-test" element={<PerformanceTestPage />} />
                                <Route path="backtests/:backtestId" element={<BacktestResultPage />} />
                                <Route path="backtests" element={<BacktestHistoryPage />} />
                                <Route path="settings" element={<Settings />} />
                              </Route>

                              {/* Fallback route */}
                              <Route path="*" element={<Navigate to="/" replace />} />
                            </Routes>
                          </AnimatedTransition>
                        </Router>
                      </PaperTradingProvider>
                    </SelectedAssetProvider>
                  </AlertsProvider>
                </AuthProvider>
              </NotificationProvider>
            </MantineProvider>
          </ThemeProvider>
        </DataSourceProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
