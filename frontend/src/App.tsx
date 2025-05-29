import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import NotificationProvider from './components/common/NotificationSystem';
import { DataSourceProvider } from './context/DataSourceContext';
import { PaperTradingProvider } from './context/PaperTradingContext';
import { LLMOversightProvider } from './context/LLMOversightContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ProtectedRoute from './components/common/ProtectedRoute';
import MainLayout from './components/layout/MainLayout';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Portfolio from './pages/Portfolio';
import Trade from './pages/Trade';
import MexcLiveChartDemo from './pages/MexcLiveChartDemo';
import SimpleMexcDashboardPage from './pages/SimpleMexcDashboardPage';
import RealMexcDashboardPage from './pages/RealMexcDashboardPage';
import MexcApiDebugPage from './pages/MexcApiDebugPage';
import Strategies from './pages/Strategies';
import ErrorBoundary from './components/common/ErrorBoundary';
import { AnimatedSwitch, AnimationType, AnimationDuration } from './components/common/AnimatedTransition';
import Settings from './pages/Settings';
import ApiHealthPage from './pages/ApiHealthPage';
import ApiLogsPage from './pages/ApiLogsPage';
import AlertsPage from './pages/AlertsPage';
import PerformancePage from './pages/PerformancePage';
import PerformanceTestPage from './pages/PerformanceTestPage';
import Sentiment from './pages/Sentiment';
import SentimentAnalysis from './pages/SentimentAnalysis';
import LLMOversight from './pages/LLMOversight';
import TradingSignals from './pages/TradingSignals';
import AdvancedSignals from './pages/AdvancedSignals';
import BacktestResultPage from './pages/BacktestResultPage';
import BacktestHistoryPage from './pages/BacktestHistoryPage';
import PaperTradingPage from './pages/PaperTradingPage';
import NewPaperTradingPage from './pages/NewPaperTradingPage';
import PaperTradingSessionPage from './pages/PaperTradingSessionPage';
import SystemControlPage from './pages/SystemControlPage';
import { SelectedAssetProvider } from './context/SelectedAssetContext';
import { AlertsProvider } from './context/AlertsContext';

// Import CSS for theme variables
import './styles/theme.css';

// Import clean dark theme
import './styles/darkTheme.css';

// Import Mantine provider
import { MantineProvider, createTheme } from '@mantine/core';

// Import warning suppression utility
import { suppressDefaultPropsWarning } from './utils/suppressWarnings';

// Create a client
const queryClient = new QueryClient();

// Mantine Theme (can be customized later)
const theme = createTheme({
  /** Put your mantine theme override here */
});

interface AppRoutesAnimatorProps {
  children: React.ReactNode;
  animationType?: AnimationType;
  animationDuration?: AnimationDuration;
}

const AppRoutesAnimator: React.FC<AppRoutesAnimatorProps> = ({ 
  children,
  animationType = AnimationType.FADE,
  animationDuration = AnimationDuration.NORMAL 
}) => {
  const location = useLocation();
  return (
    <AnimatedSwitch 
      location={location} 
      animationType={animationType} 
      animationDuration={animationDuration}
    >
      {/* Pass location to Routes as well for react-router-dom v6 transitions */}
      {React.cloneElement(children as React.ReactElement, { location })}
    </AnimatedSwitch>
  );
};

function App() {
  // Suppress React warnings about defaultProps in function components
  // This is especially useful for third-party libraries like @nivo/pie
  useEffect(() => {
    // Call the function to suppress warnings and store the cleanup function
    const restoreConsoleError = suppressDefaultPropsWarning();
    
    // Return cleanup function to restore original console.error when component unmounts
    return () => restoreConsoleError();
  }, []);
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
                      <LLMOversightProvider>
                        <PaperTradingProvider>
                          <Router>
                            <AppRoutesAnimator>
                              <Routes>
                                {/* Public routes */}
                                <Route path="/login" element={<Login />} />
                                <Route path="/register" element={<Register />} />

                                {/* Protected routes */}
                                <Route 
                                  path="/"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <Dashboard />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/analytics"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <Analytics />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/portfolio"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <Portfolio />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/trade"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <Trade />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/mexc-demo"
                                  element={
                                    <MexcLiveChartDemo />
                                  }
                                />
                                {/* Add our new SimpleMexcDashboard route */}
                                <Route 
                                  path="/mexc-dashboard"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <SimpleMexcDashboardPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/mexc-dashboard-real"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <RealMexcDashboardPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/mexc-api-debug"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <MexcApiDebugPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/backtest"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <div>Backtest Page (Coming Soon)</div>
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/strategies"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <Strategies />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/sentiment"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <Sentiment />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/sentiment-analysis"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <SentimentAnalysis />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/trading-signals"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <TradingSignals />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/advanced-signals"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <AdvancedSignals />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/paper-trading"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <PaperTradingPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/paper-trading/new"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <NewPaperTradingPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/paper-trading/session/:sessionId"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <PaperTradingSessionPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/system-control"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <SystemControlPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/api-health"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <ApiHealthPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/api-logs"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <ApiLogsPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/alerts"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <AlertsPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/performance"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <PerformancePage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/performance-test"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <PerformanceTestPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/backtests/:backtestId"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <BacktestResultPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/backtests"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <BacktestHistoryPage />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/llm-oversight"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <LLMOversight />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />
                                <Route 
                                  path="/settings"
                                  element={
                                    <ProtectedRoute>
                                      <MainLayout>
                                        <Settings />
                                      </MainLayout>
                                    </ProtectedRoute>
                                  }
                                />

                                {/* Fallback route */}
                                <Route path="*" element={<Navigate to="/" replace />} />
                              </Routes>
                            </AppRoutesAnimator>
                          </Router>
                        </PaperTradingProvider>
                      </LLMOversightProvider>
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