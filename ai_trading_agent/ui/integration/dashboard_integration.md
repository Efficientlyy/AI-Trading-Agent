# Dashboard Integration Guide

This guide explains how to integrate the `DataSourceToggle` component with the existing dashboard UI.

## Integration Steps

### 1. Import the Component

In your dashboard header or navbar component, import the `DataSourceToggle` component:

```jsx
import DataSourceToggle from '../components/data_source_toggle';
```

### 2. Add the Component to the Layout

Place the component in your dashboard header or navbar:

```jsx
// Inside your dashboard header/navbar component
<AppBar position="static">
  <Toolbar>
    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
      AI Trading Agent
    </Typography>
    
    {/* Add the DataSourceToggle component here */}
    <DataSourceToggle 
      onChange={(isRealData) => {
        console.log(`Data source changed to: ${isRealData ? 'real' : 'mock'}`);
        // Any other action needed when the data source changes
      }}
    />
    
    {/* Other navbar items */}
    <IconButton color="inherit">
      <AccountCircle />
    </IconButton>
  </Toolbar>
</AppBar>
```

### 3. Handle Data Source Changes

The `DataSourceToggle` component provides an `onChange` callback that you can use to respond to data source changes:

```jsx
const handleDataSourceChange = (isRealData) => {
  // Update application state
  dispatch({
    type: 'SET_DATA_SOURCE',
    payload: isRealData ? 'real' : 'mock'
  });
  
  // Refresh data if needed
  if (shouldRefreshData) {
    fetchLatestMarketData();
  }
  
  // Show notification to the user
  setNotification({
    open: true,
    message: `Switched to ${isRealData ? 'real' : 'mock'} data`,
    severity: 'info'
  });
};

// Pass the handler to the component
<DataSourceToggle onChange={handleDataSourceChange} />
```

### 4. Add Visual Indicators

When using mock data, it's important to provide clear visual indication throughout the application. Add a banner or indicator when mock data is in use:

```jsx
{!isRealData && (
  <Box sx={{ 
    bgcolor: 'warning.light', 
    color: 'warning.contrastText',
    p: 1,
    textAlign: 'center'
  }}>
    <Alert severity="warning" icon={<MockDataIcon />}>
      Using mock data - Not for real trading decisions
    </Alert>
  </Box>
)}
```

### 5. API Connection

The `DataSourceToggle` component automatically connects to the following API endpoints:

- `GET /api/data-source/status` - To retrieve the current data source status
- `POST /api/data-source/toggle` - To toggle between mock and real data

Ensure these endpoints are properly set up in your API server.

## Example: Complete Dashboard Integration

```jsx
import React, { useState, useEffect } from 'react';
import { AppBar, Toolbar, Typography, Box, Alert, IconButton } from '@mui/material';
import { AccountCircle } from '@mui/icons-material';
import DataSourceToggle from '../components/data_source_toggle';
import MockDataIcon from '../icons/MockDataIcon';

const Dashboard = () => {
  const [isRealData, setIsRealData] = useState(true);
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  const handleDataSourceChange = (isRealData) => {
    setIsRealData(isRealData);
    
    // Show notification
    setNotification({
      open: true,
      message: `Switched to ${isRealData ? 'real' : 'mock'} data`,
      severity: 'info'
    });
    
    // Refresh data
    fetchMarketData();
  };
  
  return (
    <div>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            AI Trading Agent
          </Typography>
          
          <DataSourceToggle onChange={handleDataSourceChange} />
          
          <IconButton color="inherit">
            <AccountCircle />
          </IconButton>
        </Toolbar>
      </AppBar>
      
      {!isRealData && (
        <Box sx={{ 
          bgcolor: 'warning.light', 
          color: 'warning.contrastText',
          p: 1,
          textAlign: 'center'
        }}>
          <Alert severity="warning" icon={<MockDataIcon />}>
            Using mock data - Not for real trading decisions
          </Alert>
        </Box>
      )}
      
      {/* Rest of your dashboard content */}
    </div>
  );
};

export default Dashboard;
```

## Testing the Integration

After integrating the component, test the following scenarios:

1. Toggle between mock and real data
2. Verify that the visual indicators appear when using mock data
3. Check that data refreshes appropriately when toggling
4. Confirm that the toggle state persists across page refreshes
