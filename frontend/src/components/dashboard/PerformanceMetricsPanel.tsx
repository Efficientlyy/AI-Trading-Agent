import {
  Box,
  Card,
  CardContent,
  CircularProgress,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  SelectChangeEvent,
  Tab,
  Tabs,
  Typography,
  useTheme
} from '@mui/material';
import React, { useState } from 'react';
import { useSystemControl } from '../../context/SystemControlContext';

// Import chart components from recharts
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`metrics-tabpanel-${index}`}
      aria-labelledby={`metrics-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 2, bgcolor: darkPaperBg, borderRadius: 1, mt: 1 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `metrics-tab-${index}`,
    'aria-controls': `metrics-tabpanel-${index}`,
  };
}

// Dark theme color constants
const darkBg = 'rgba(30, 34, 45, 0.9)';
const darkPaperBg = 'rgba(45, 50, 65, 0.8)';
const darkText = '#ffffff'; // White text for maximum visibility
const darkSecondaryText = 'rgba(255, 255, 255, 0.7)';
const darkBorder = 'rgba(255, 255, 255, 0.1)';

const PerformanceMetricsPanel: React.FC = () => {
  const theme = useTheme();
  const { agents, sessions, isLoading } = useSystemControl();
  const [timeRange, setTimeRange] = useState<string>('7d');
  const [selectedAgent, setSelectedAgent] = useState<string>('all');
  const [tabValue, setTabValue] = useState(0);

  // Handle time range change
  const handleTimeRangeChange = (event: SelectChangeEvent) => {
    setTimeRange(event.target.value);
  };

  // Handle agent selection change
  const handleAgentChange = (event: SelectChangeEvent) => {
    setSelectedAgent(event.target.value);
  };

  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Generate mock performance data
  const generatePerformanceData = () => {
    const data = [];
    const now = new Date();
    const daysToGenerate = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;

    let cumulativeReturn = 0;

    for (let i = daysToGenerate; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);

      const dailyReturn = (Math.random() * 2 - 0.5) * (selectedAgent === 'all' ? 1 : 1.5);
      cumulativeReturn += dailyReturn;

      data.push({
        date: date.toISOString().split('T')[0],
        dailyReturn: dailyReturn.toFixed(2),
        cumulativeReturn: cumulativeReturn.toFixed(2),
        winRate: (50 + Math.random() * 20).toFixed(1),
        trades: Math.floor(Math.random() * 15),
        drawdown: (Math.random() * 5).toFixed(2),
      });
    }

    return data;
  };

  // Generate mock trade distribution data
  const generateTradeDistributionData = () => {
    return [
      { name: '< -5%', value: Math.floor(Math.random() * 5) },
      { name: '-5% to -2%', value: Math.floor(Math.random() * 10) + 5 },
      { name: '-2% to 0%', value: Math.floor(Math.random() * 15) + 10 },
      { name: '0% to 2%', value: Math.floor(Math.random() * 20) + 15 },
      { name: '2% to 5%', value: Math.floor(Math.random() * 15) + 10 },
      { name: '> 5%', value: Math.floor(Math.random() * 10) + 5 },
    ];
  };

  // Generate mock drawdown data
  const generateDrawdownData = () => {
    const data = [];
    const now = new Date();
    const daysToGenerate = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;

    let drawdown = 0;
    let maxDrawdown = 0;

    for (let i = daysToGenerate; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);

      // Simulate drawdown fluctuations
      drawdown = Math.max(-10, Math.min(0, drawdown + (Math.random() * 2 - 1)));
      maxDrawdown = Math.min(maxDrawdown, drawdown);

      data.push({
        date: date.toISOString().split('T')[0],
        drawdown: drawdown.toFixed(2),
        maxDrawdown: maxDrawdown.toFixed(2),
      });
    }

    return data;
  };

  // Generate mock win rate data over time
  const generateWinRateData = () => {
    const data = [];
    const now = new Date();
    const daysToGenerate = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;

    let winRate = 50;

    for (let i = daysToGenerate; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);

      // Simulate win rate fluctuations
      winRate = Math.max(30, Math.min(80, winRate + (Math.random() * 6 - 3)));

      data.push({
        date: date.toISOString().split('T')[0],
        winRate: winRate.toFixed(1),
      });
    }

    return data;
  };

  // Generate summary metrics
  const generateSummaryMetrics = () => {
    return {
      totalReturn: (Math.random() * 30 - 5).toFixed(2),
      winRate: (50 + Math.random() * 20).toFixed(1),
      profitFactor: (1 + Math.random() * 1.5).toFixed(2),
      sharpeRatio: (1 + Math.random() * 2).toFixed(2),
      maxDrawdown: (Math.random() * 15).toFixed(2),
      avgWin: (1 + Math.random() * 3).toFixed(2),
      avgLoss: (1 + Math.random() * 2).toFixed(2),
      totalTrades: Math.floor(Math.random() * 100) + 50,
      winningTrades: Math.floor(Math.random() * 60) + 30,
      losingTrades: Math.floor(Math.random() * 40) + 20,
    };
  };

  const performanceData = generatePerformanceData();
  const tradeDistributionData = generateTradeDistributionData();
  const drawdownData = generateDrawdownData();
  const winRateData = generateWinRateData();
  const summaryMetrics = generateSummaryMetrics();

  return (
    <Card elevation={3} sx={{ mb: 3, bgcolor: '#101010', color: darkText, borderRadius: 2, border: '1px solid #555' }}> {/* Made bg even darker, added border */}
      <CardContent sx={{ bgcolor: 'transparent' }}> {/* Ensure CardContent is transparent */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="div" sx={{ fontWeight: 'bold', color: darkText }}>
            Performance Metrics
          </Typography>

          <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel id="agent-select-label" sx={{ color: darkText }}>Agent</InputLabel>
              <Select
                labelId="agent-select-label"
                id="agent-select"
                value={selectedAgent}
                label="Agent"
                onChange={handleAgentChange}
                sx={{
                  color: darkText,
                  '.MuiOutlinedInput-notchedOutline': { borderColor: darkBorder },
                  '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255, 255, 255, 0.3)' },
                  '.MuiSvgIcon-root': { color: darkText }
                }}
              >
                <MenuItem value="all">All Agents</MenuItem>
                {agents.map((agent) => (
                  <MenuItem key={agent.agent_id} value={agent.agent_id}>
                    {agent.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel id="time-range-select-label" sx={{ color: darkText }}>Time Range</InputLabel>
              <Select
                labelId="time-range-select-label"
                id="time-range-select"
                value={timeRange}
                label="Time Range"
                onChange={handleTimeRangeChange}
                sx={{
                  color: darkText,
                  '.MuiOutlinedInput-notchedOutline': { borderColor: darkBorder },
                  '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255, 255, 255, 0.3)' },
                  '.MuiSvgIcon-root': { color: darkText }
                }}
              >
                <MenuItem value="7d">7 Days</MenuItem>
                <MenuItem value="30d">30 Days</MenuItem>
                <MenuItem value="90d">90 Days</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Box>

        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {/* Summary Metrics */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={6} sm={3}>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                  <Typography variant="subtitle2" sx={{ color: darkSecondaryText }}>
                    Total Return
                  </Typography>
                  <Typography
                    variant="h5"
                    sx={{
                      color: Number(summaryMetrics.totalReturn) >= 0 ? '#4caf50' : '#f44336',
                      fontWeight: 'bold'
                    }}
                  >
                    {Number(summaryMetrics.totalReturn) >= 0 ? '+' : ''}
                    {summaryMetrics.totalReturn}%
                  </Typography>
                </Paper>
              </Grid>

              <Grid item xs={6} sm={3}>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                  <Typography variant="subtitle2" sx={{ color: darkSecondaryText }}>
                    Win Rate
                  </Typography>
                  <Typography variant="h5" sx={{ color: darkText, fontWeight: 'bold' }}>
                    {summaryMetrics.winRate}%
                  </Typography>
                </Paper>
              </Grid>

              <Grid item xs={6} sm={3}>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                  <Typography variant="subtitle2" sx={{ color: darkSecondaryText }}>
                    Profit Factor
                  </Typography>
                  <Typography
                    variant="h5"
                    sx={{
                      color: Number(summaryMetrics.profitFactor) >= 1 ? '#4caf50' : '#f44336',
                      fontWeight: 'bold'
                    }}
                  >
                    {summaryMetrics.profitFactor}
                  </Typography>
                </Paper>
              </Grid>

              <Grid item xs={6} sm={3}>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: darkPaperBg, borderColor: darkBorder }}>
                  <Typography variant="subtitle2" sx={{ color: darkSecondaryText }}>
                    Max Drawdown
                  </Typography>
                  <Typography variant="h5" sx={{ color: '#f44336', fontWeight: 'bold' }}>
                    -{summaryMetrics.maxDrawdown}%
                  </Typography>
                </Paper>
              </Grid>
            </Grid>

            {/* Tabs for different metrics */}
            <Box sx={{ borderBottom: 1, borderColor: darkBorder }}>
              <Tabs
                value={tabValue}
                onChange={handleTabChange}
                aria-label="performance metrics tabs"
                variant="scrollable"
                scrollButtons="auto"
                sx={{
                  '& .MuiTab-root': { color: darkSecondaryText },
                  '& .Mui-selected': { color: darkText },
                  '& .MuiTabs-indicator': { backgroundColor: '#4caf50' }
                }}
              >
                <Tab label="Returns" {...a11yProps(0)} />
                <Tab label="Drawdown" {...a11yProps(1)} />
                <Tab label="Win Rate" {...a11yProps(2)} />
                <Tab label="Trade Distribution" {...a11yProps(3)} />
                <Tab label="Detailed Stats" {...a11yProps(4)} />
              </Tabs>
            </Box>

            {/* Returns Chart */}
            <TabPanel value={tabValue} index={0}>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={performanceData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="cumulativeReturn"
                      name="Cumulative Return (%)"
                      stroke="#4caf50"
                      strokeWidth={2}
                      activeDot={{ r: 8, fill: '#4caf50' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="dailyReturn"
                      name="Daily Return (%)"
                      stroke="#9c27b0"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </TabPanel>

            {/* Drawdown Chart */}
            <TabPanel value={tabValue} index={1}>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={drawdownData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="drawdown"
                      name="Drawdown (%)"
                      stroke={theme.palette.error.main}
                      fill={theme.palette.error.light}
                      fillOpacity={0.3}
                    />
                    <Area
                      type="monotone"
                      dataKey="maxDrawdown"
                      name="Max Drawdown (%)"
                      stroke={theme.palette.error.dark}
                      fill={theme.palette.error.dark}
                      fillOpacity={0.1}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </TabPanel>

            {/* Win Rate Chart */}
            <TabPanel value={tabValue} index={2}>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={winRateData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="winRate"
                      name="Win Rate (%)"
                      stroke={theme.palette.success.main}
                      activeDot={{ r: 8 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </TabPanel>

            {/* Trade Distribution Chart */}
            <TabPanel value={tabValue} index={3}>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={tradeDistributionData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar
                      dataKey="value"
                      name="Number of Trades"
                      fill={theme.palette.primary.main}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </TabPanel>

            {/* Detailed Stats */}
            <TabPanel value={tabValue} index={4}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Trade Statistics
                    </Typography>
                    <Divider sx={{ mb: 2 }} />

                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Total Trades:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {summaryMetrics.totalTrades}
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Winning Trades:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="success.main">
                          {summaryMetrics.winningTrades}
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" sx={{ color: darkSecondaryText }}>
                          Losing Trades:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" sx={{ color: '#f44336', fontWeight: 'medium' }}>
                          {summaryMetrics.losingTrades}
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Win Rate:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {summaryMetrics.winRate}%
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Avg. Win:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="success.main">
                          +{summaryMetrics.avgWin}%
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Avg. Loss:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="error.main">
                          -{summaryMetrics.avgLoss}%
                        </Typography>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Performance Metrics
                    </Typography>
                    <Divider sx={{ mb: 2 }} />

                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Total Return:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography
                          variant="body2"
                          color={Number(summaryMetrics.totalReturn) >= 0 ? 'success.main' : 'error.main'}
                        >
                          {Number(summaryMetrics.totalReturn) >= 0 ? '+' : ''}
                          {summaryMetrics.totalReturn}%
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Profit Factor:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {summaryMetrics.profitFactor}
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Sharpe Ratio:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {summaryMetrics.sharpeRatio}
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Max Drawdown:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="error.main">
                          -{summaryMetrics.maxDrawdown}%
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Win/Loss Ratio:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {(Number(summaryMetrics.avgWin) / Number(summaryMetrics.avgLoss)).toFixed(2)}
                        </Typography>
                      </Grid>

                      <Grid item xs={6}>
                        <Typography variant="body2" color="textSecondary">
                          Expectancy:
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          {((Number(summaryMetrics.winRate) / 100 * Number(summaryMetrics.avgWin)) -
                            ((100 - Number(summaryMetrics.winRate)) / 100 * Number(summaryMetrics.avgLoss))).toFixed(2)}%
                        </Typography>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>
              </Grid>
            </TabPanel>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default PerformanceMetricsPanel;
