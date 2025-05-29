/**
 * Sidebar Component
 * 
 * This component provides the navigation sidebar for the AI Trading Agent dashboard.
 */

import React from 'react';
import { 
  Box, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Divider,
  Typography,
  ListItemButton
} from '@mui/material';
import { useTheme } from '@mui/material/styles';

// Icons
import DashboardIcon from '@mui/icons-material/Dashboard';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import SettingsIcon from '@mui/icons-material/Settings';
import HistoryIcon from '@mui/icons-material/History';
import TuneIcon from '@mui/icons-material/Tune';
import AssessmentIcon from '@mui/icons-material/Assessment';
import BugReportIcon from '@mui/icons-material/BugReport';
import BarChartIcon from '@mui/icons-material/BarChart';

/**
 * Sidebar navigation component
 */
const Sidebar = ({ activeSection = 'dashboard' }) => {
  const theme = useTheme();
  
  // Navigation sections
  const sections = [
    { id: 'dashboard', name: 'Dashboard', icon: <DashboardIcon /> },
    { id: 'trading', name: 'Trading', icon: <ShowChartIcon /> },
    { id: 'analysis', name: 'Analysis', icon: <AutoGraphIcon /> },
    { id: 'backtesting', name: 'Backtesting', icon: <HistoryIcon /> },
    { id: 'performance', name: 'Performance', icon: <AssessmentIcon /> },
    { id: 'regimes', name: 'Market Regimes', icon: <BarChartIcon /> }
  ];
  
  // System sections
  const systemSections = [
    { id: 'settings', name: 'Settings', icon: <SettingsIcon /> },
    { id: 'parameters', name: 'Parameters', icon: <TuneIcon /> },
    { id: 'debug', name: 'Debug Tools', icon: <BugReportIcon /> }
  ];
  
  return (
    <Box sx={{ overflow: 'auto' }}>
      {/* Logo area */}
      <Box 
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          py: 2,
          borderBottom: `1px solid ${theme.palette.divider}`
        }}
      >
        <Typography 
          variant="h6" 
          component="div" 
          sx={{ 
            fontWeight: 'bold',
            display: { sm: 'none' }
          }}
        >
          AI Trading Agent
        </Typography>
      </Box>
      
      {/* Main navigation */}
      <List>
        {sections.map((section) => (
          <ListItem key={section.id} disablePadding>
            <ListItemButton
              selected={activeSection === section.id}
              sx={{
                '&.Mui-selected': {
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)'
                    : 'rgba(0, 0, 0, 0.05)',
                  borderLeft: `3px solid ${theme.palette.primary.main}`,
                  '&:hover': {
                    bgcolor: theme.palette.mode === 'dark' 
                      ? 'rgba(255, 255, 255, 0.12)'
                      : 'rgba(0, 0, 0, 0.07)'
                  }
                },
                '&:hover': {
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.05)'
                    : 'rgba(0, 0, 0, 0.03)'
                }
              }}
            >
              <ListItemIcon>
                {section.icon}
              </ListItemIcon>
              <ListItemText primary={section.name} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      <Divider sx={{ my: 1 }} />
      
      {/* System navigation */}
      <List>
        {systemSections.map((section) => (
          <ListItem key={section.id} disablePadding>
            <ListItemButton
              selected={activeSection === section.id}
              sx={{
                '&.Mui-selected': {
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)'
                    : 'rgba(0, 0, 0, 0.05)',
                  borderLeft: `3px solid ${theme.palette.primary.main}`,
                  '&:hover': {
                    bgcolor: theme.palette.mode === 'dark' 
                      ? 'rgba(255, 255, 255, 0.12)'
                      : 'rgba(0, 0, 0, 0.07)'
                  }
                },
                '&:hover': {
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.05)'
                    : 'rgba(0, 0, 0, 0.03)'
                }
              }}
            >
              <ListItemIcon>
                {section.icon}
              </ListItemIcon>
              <ListItemText primary={section.name} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      {/* Version info */}
      <Box sx={{ p: 2, mt: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Version 1.0.0
        </Typography>
      </Box>
    </Box>
  );
};

export default Sidebar;
