/**
 * Main Dashboard Component
 * 
 * This is the main layout component for the AI Trading Agent dashboard.
 * It integrates the header, sidebar, and main content area.
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  CssBaseline, 
  Drawer, 
  ThemeProvider, 
  createTheme,
  useMediaQuery,
  Snackbar,
  Alert,
  Tab,
  Tabs,
  Divider
} from '@mui/material';

import DashboardHeader from './DashboardHeader';
import Sidebar from './Sidebar';
import TradingView from './TradingView';
import AgentStatusGrid from './AgentStatusGrid';
import ActivityFeed from './ActivityFeed';
import PerformanceMetricsPanel from './PerformanceMetricsPanel';
import TechnicalAnalysisView from './TechnicalAnalysisView';
import axios from 'axios';

// Default drawer width
const drawerWidth = 240;

/**
 * Dashboard component with responsive layout
 */
const Dashboard = () => {
  // State
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [currentMarketRegime, setCurrentMarketRegime] = useState('normal');
  const [mockDataActive, setMockDataActive] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  // Create theme based on mode
  const theme = createTheme({
    palette: {
      mode: isDarkMode ? 'dark' : 'light',
      background: {
        default: isDarkMode ? '#121212' : '#f5f5f5',
        paper: isDarkMode ? '#1e1e1e' : '#ffffff',
      },
      primary: {
        main: '#3a86ff',
      },
      secondary: {
        main: '#ff006e',
      },
    },
    typography: {
      fontFamily: "'Inter', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
          },
        },
      },
    },
  });

  // Check if using a mobile device
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Toggle drawer for mobile view
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  // Toggle theme between light and dark mode
  const handleThemeToggle = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Handle settings click
  const handleSettingsClick = () => {
    setNotification({
      open: true,
      message: 'Settings panel is under development',
      severity: 'info'
    });
  };

  // Fetch current data source status on component mount
  useEffect(() => {
    const fetchDataSourceStatus = async () => {
      try {
        const response = await axios.get('/api/data-source/status');
        setMockDataActive(response.data.use_mock_data);
      } catch (error) {
        console.error('Failed to fetch data source status:', error);
      }
    };

    const fetchMarketRegime = async () => {
      try {
        // This would be the actual API endpoint in a real implementation
        // For now, we'll just simulate different regimes
        const regimes = ['bullish', 'bearish', 'volatile', 'range_bound', 'normal'];
        setCurrentMarketRegime(regimes[Math.floor(Math.random() * regimes.length)]);
      } catch (error) {
        console.error('Failed to fetch market regime:', error);
      }
    };

    fetchDataSourceStatus();
    fetchMarketRegime();
  }, []);

  // Handle data source toggle from header
  const handleDataSourceChange = (isRealData) => {
    setMockDataActive(!isRealData);
    setNotification({
      open: true,
      message: `Data source switched to ${isRealData ? 'real' : 'mock'} data`,
      severity: 'success'
    });
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Close notification
  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex', height: '100vh' }}>
        <CssBaseline />
        
        {/* Header */}
        <DashboardHeader 
          onToggleTheme={handleThemeToggle} 
          isDarkMode={isDarkMode}
          onToggleSidebar={handleDrawerToggle}
          onSettingsClick={handleSettingsClick}
          currentMarketRegime={currentMarketRegime}
        />
        
        {/* Sidebar - Responsive drawer */}
        <Box
          component="nav"
          sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        >
          {/* Mobile drawer */}
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            ModalProps={{ keepMounted: true }}
            sx={{
              display: { xs: 'block', sm: 'none' },
              '& .MuiDrawer-paper': { 
                boxSizing: 'border-box', 
                width: drawerWidth,
                borderRight: `1px solid ${theme.palette.divider}`
              },
            }}
          >
            <Sidebar />
          </Drawer>
          
          {/* Desktop drawer */}
          <Drawer
            variant="permanent"
            sx={{
              display: { xs: 'none', sm: 'block' },
              '& .MuiDrawer-paper': { 
                boxSizing: 'border-box', 
                width: drawerWidth,
                borderRight: `1px solid ${theme.palette.divider}`,
                mt: '64px'
              },
            }}
            open
          >
            <Sidebar />
          </Drawer>
        </Box>
        
        {/* Main content area */}
        <Box
          component="main"
          sx={{ 
            flexGrow: 1, 
            p: 3, 
            width: { sm: `calc(100% - ${drawerWidth}px)` },
            mt: '64px',
            overflowY: 'auto'
          }}
        >
          {/* Show a mock data indicator when mock data is active */}
          {mockDataActive && (
            <Box
              sx={{
                bgcolor: theme.palette.warning.light,
                color: theme.palette.warning.contrastText,
                py: 0.5,
                px: 2,
                borderRadius: 1,
                mb: 2,
                display: 'inline-block'
              }}
            >
              Mock Data Mode Active
            </Box>
          )}
          
          {/* Dashboard content */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Main Content Tabs */}
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs 
                value={activeTab} 
                onChange={handleTabChange} 
                aria-label="dashboard tabs"
                variant={isMobile ? 'scrollable' : 'standard'}
                scrollButtons={isMobile ? 'auto' : false}
              >
                <Tab label="Overview" />
                <Tab label="Technical Analysis" />
              </Tabs>
            </Box>
            
            {/* Tab Panels */}
            <Box sx={{ display: activeTab === 0 ? 'block' : 'none' }}>
              {/* Overview Tab Content */}
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Trading View */}
                <Box sx={{ height: '400px' }}>
                  <TradingView />
                </Box>
                
                {/* Grid layout for dashboard widgets */}
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
                  {/* Agent Status */}
                  <Box>
                    <AgentStatusGrid />
                  </Box>
                  
                  {/* Performance Metrics */}
                  <Box>
                    <PerformanceMetricsPanel />
                  </Box>
                  
                  {/* Activity Feed */}
                  <Box sx={{ gridColumn: { xs: '1', md: '1 / span 2' } }}>
                    <ActivityFeed />
                  </Box>
                </Box>
              </Box>
            </Box>
            
            <Box sx={{ display: activeTab === 1 ? 'block' : 'none' }}>
              {/* Technical Analysis Tab Content */}
              <TechnicalAnalysisView />
            </Box>
          </Box>
        </Box>
        
        {/* Notifications */}
        <Snackbar
          open={notification.open}
          autoHideDuration={4000}
          onClose={handleCloseNotification}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert 
            onClose={handleCloseNotification} 
            severity={notification.severity} 
            sx={{ width: '100%' }}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
};

export default Dashboard;
