/**
 * Dashboard Header Component
 * 
 * This component provides the top navigation bar for the AI Trading Agent dashboard,
 * including the data source toggle and other controls.
 */

import React, { useState, useEffect } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Box, 
  IconButton, 
  useMediaQuery,
  Chip,
  Button,
  Tooltip
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import SettingsIcon from '@mui/icons-material/Settings';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';

import DataSourceToggle from './data_source_toggle';

/**
 * Dashboard header component with integrated data source toggle
 */
const DashboardHeader = ({ 
  onToggleTheme, 
  isDarkMode, 
  onToggleSidebar, 
  onSettingsClick,
  currentMarketRegime = 'normal'
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [dataSourceChanged, setDataSourceChanged] = useState(false);

  // Handle data source toggle change
  const handleDataSourceChange = (isRealData) => {
    setDataSourceChanged(true);
    setTimeout(() => setDataSourceChanged(false), 3000);
  };

  // Map market regimes to colors
  const regimeColors = {
    'bullish': theme.palette.success.main,
    'bearish': theme.palette.error.main,
    'volatile': theme.palette.warning.main,
    'range_bound': theme.palette.info.main,
    'normal': theme.palette.primary.main
  };

  return (
    <AppBar 
      position="fixed" 
      color="default" 
      elevation={1}
      sx={{
        zIndex: theme.zIndex.drawer + 1,
        bgcolor: theme.palette.background.paper,
        borderBottom: `1px solid ${theme.palette.divider}`
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        {/* Left side */}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={onToggleSidebar}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, display: { xs: 'none', sm: 'block' } }}>
            AI Trading Agent
          </Typography>
          
          {!isMobile && (
            <Chip 
              label={`Market Regime: ${currentMarketRegime.charAt(0).toUpperCase() + currentMarketRegime.slice(1).replace('_', ' ')}`}
              size="small"
              sx={{ 
                ml: 2, 
                bgcolor: regimeColors[currentMarketRegime] || theme.palette.primary.main,
                color: '#fff',
                fontWeight: 500
              }}
            />
          )}
        </Box>
        
        {/* Right side */}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {/* Data Source Toggle */}
          <Box sx={{ mr: 1 }}>
            <DataSourceToggle 
              onChange={handleDataSourceChange}
              disabled={false}
            />
          </Box>
          
          {/* Theme Toggle */}
          <Tooltip title={`Switch to ${isDarkMode ? 'light' : 'dark'} mode`}>
            <IconButton onClick={onToggleTheme} color="inherit" size="small">
              {isDarkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Tooltip>
          
          {/* Notifications */}
          <Tooltip title="Notifications">
            <IconButton color="inherit" size="small">
              <NotificationsIcon />
            </IconButton>
          </Tooltip>
          
          {/* Settings */}
          <Tooltip title="Settings">
            <IconButton 
              edge="end" 
              color="inherit" 
              onClick={onSettingsClick}
              size="small"
            >
              <SettingsIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default DashboardHeader;
