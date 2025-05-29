/**
 * Technical Analysis View
 * 
 * This component integrates the TechnicalChartViewer and PatternRecognitionView
 * into a unified dashboard section for technical analysis.
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Grid,
  Tab,
  Tabs,
  Divider,
  Button,
  Chip,
  useTheme,
  useMediaQuery
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import ShareIcon from '@mui/icons-material/Share';
import SaveIcon from '@mui/icons-material/Save';
import WarningIcon from '@mui/icons-material/Warning';
import BarChartIcon from '@mui/icons-material/BarChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import EnhancedTechnicalChartViewer from './EnhancedTechnicalChartViewer';
import EnhancedPatternRecognitionView from './EnhancedPatternRecognitionView';
import axios from 'axios';

/**
 * Tab panel component for displaying tab content
 */
const TabPanel = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      style={{ height: '100%' }}
      {...other}
    >
      {value === index && (
        <Box sx={{ height: '100%', pt: 2 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

/**
 * Technical Analysis View Component
 */
const TechnicalAnalysisView = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [activeTab, setActiveTab] = useState(0);
  const [isMockData, setIsMockData] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  
  // Fetch data source status on component mount
  useEffect(() => {
    const fetchDataSourceStatus = async () => {
      try {
        const response = await axios.get('/api/data-source/status');
        setIsMockData(response.data.use_mock_data);
      } catch (error) {
        console.error('Failed to fetch data source status:', error);
      }
    };

    fetchDataSourceStatus();
    
    // Set up listener for data source changes
    const checkDataSourceInterval = setInterval(fetchDataSourceStatus, 30000);
    
    return () => clearInterval(checkDataSourceInterval);
  }, []);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Handle fullscreen toggle
  const handleFullscreenToggle = () => {
    setFullscreen(!fullscreen);
  };
  
  return (
    <Paper 
      elevation={0} 
      sx={{ 
        p: 2, 
        height: fullscreen ? '100vh' : '100%',
        width: fullscreen ? '100vw' : '100%',
        position: fullscreen ? 'fixed' : 'relative',
        top: fullscreen ? 0 : 'auto',
        left: fullscreen ? 0 : 'auto',
        zIndex: fullscreen ? 1300 : 'auto',
        border: `1px solid ${theme.palette.divider}`
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <ShowChartIcon sx={{ mr: 1, color: theme.palette.primary.main }} />
          <Typography variant="h6">Technical Analysis</Typography>
          
          {isMockData && (
            <Chip 
              icon={<WarningIcon fontSize="small" />}
              label="Mock Data" 
              size="small"
              color="warning"
              sx={{ ml: 2 }}
            />
          )}
        </Box>
        
        <Box>
          <Button 
            size="small" 
            startIcon={<FullscreenIcon />}
            onClick={handleFullscreenToggle}
            sx={{ mr: 1 }}
          >
            {fullscreen ? 'Exit' : 'Fullscreen'}
          </Button>
          
          <Button 
            size="small" 
            startIcon={<SaveIcon />}
            sx={{ mr: 1 }}
          >
            Save
          </Button>
          
          <Button 
            size="small" 
            startIcon={<ShareIcon />}
          >
            Share
          </Button>
        </Box>
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: fullscreen ? 'calc(100vh - 120px)' : 'calc(100% - 40px)'
      }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange}
            aria-label="technical analysis tabs"
            variant={isMobile ? 'fullWidth' : 'standard'}
          >
            <Tab 
              icon={<TimelineIcon />} 
              iconPosition="start" 
              label="Chart Analysis" 
              id="analysis-tab-0" 
              aria-controls="analysis-tabpanel-0" 
            />
            <Tab 
              icon={<BarChartIcon />} 
              iconPosition="start" 
              label="Pattern Recognition" 
              id="analysis-tab-1" 
              aria-controls="analysis-tabpanel-1" 
            />
          </Tabs>
        </Box>
        
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          <TabPanel value={activeTab} index={0}>
            <EnhancedTechnicalChartViewer fullscreen={fullscreen} onToggleFullscreen={handleFullscreenToggle} />
          </TabPanel>
          
          <TabPanel value={activeTab} index={1}>
            <EnhancedPatternRecognitionView fullscreen={fullscreen} onToggleFullscreen={handleFullscreenToggle} />
          </TabPanel>
        </Box>
      </Box>
      
      {isMockData && (
        <Box sx={{ 
          mt: 2, 
          p: 1, 
          borderRadius: 1,
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 152, 0, 0.1)' : 'rgba(255, 152, 0, 0.05)',
          border: `1px solid ${theme.palette.warning.light}`
        }}>
          <Typography variant="caption" color="warning.main">
            <WarningIcon sx={{ fontSize: 16, verticalAlign: 'text-bottom', mr: 0.5 }} />
            Currently viewing mock data. Toggle to real data in the header for actual market analysis.
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default TechnicalAnalysisView;
