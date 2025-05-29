import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  useTheme,
  Chip,
  CircularProgress,
  Paper,
  Divider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent
} from '@mui/material';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, AreaChart, Area,
  ScatterChart, Scatter, ZAxis
} from 'recharts';
import apiClient from '../api/apiClient';

interface MarketRegimePoint {
  timestamp: string;
  regime: string;
  volatility: number;
  correlation: number;
  strength: number;
  duration: number;
}

interface RegimeSummary {
  regime: string;
  count: number;
  averageDuration: number;
  lastObserved: string;
  color: string;
}

interface MarketRegimeVisualizationProps {
  asset?: string;
  timeframe?: string;
}

/**
 * Market Regime Visualization Component
 * 
 * Visualizes market regime classifications and transitions over time,
 * including volatility clustering, correlation analysis, and regime strength.
 */
const MarketRegimeVisualization: React.FC<MarketRegimeVisualizationProps> = ({
  asset = 'global',
  timeframe = '6M'
}) => {
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [regimeData, setRegimeData] = useState<MarketRegimePoint[]>([]);
  const [regimeSummary, setRegimeSummary] = useState<RegimeSummary[]>([]);
  const [selectedAsset, setSelectedAsset] = useState<string>(asset);
  const [availableAssets, setAvailableAssets] = useState<string[]>(['global']);
  
  // Color mapping for different regimes
  const regimeColors = {
    'bull': theme.palette.success.main,
    'bear': theme.palette.error.main,
    'volatile': theme.palette.warning.main,
    'consolidation': theme.palette.info.main,
    'sideways': theme.palette.text.secondary,
    'recovery': theme.palette.secondary.main,
    'unknown': theme.palette.grey[500]
  };
  
  // Fetch regime data
  useEffect(() => {
    const fetchRegimeData = async () => {
      try {
        setLoading(true);
        
        // Fetch available assets for regime analysis
        const assetsResponse = await apiClient.get('/market-regime/assets');
        if (assetsResponse.data && assetsResponse.data.assets) {
          setAvailableAssets(['global', ...assetsResponse.data.assets]);
        }
        
        // Fetch regime data for selected asset and timeframe
        const params = new URLSearchParams({
          asset: selectedAsset,
          timeframe: timeframe
        });
        
        const response = await apiClient.get(`/market-regime/history?${params.toString()}`);
        
        if (response.data && response.data.regimes) {
          setRegimeData(response.data.regimes);
          setRegimeSummary(response.data.summary || []);
        } else {
          setRegimeData([]);
          setRegimeSummary([]);
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching regime data:', err);
        setError('Failed to load market regime data');
      } finally {
        setLoading(false);
      }
    };
    
    fetchRegimeData();
  }, [selectedAsset, timeframe]);
  
  // Handle asset change
  const handleAssetChange = (event: SelectChangeEvent<string>) => {
    setSelectedAsset(event.target.value);
  };

  // Show loading state
  if (loading && regimeData.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  // Show error state
  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }
  
  // Process regime transitions for visualization
  const regimeTransitions = regimeData.map((point, index, array) => {
    if (index === 0) return null;
    const from = array[index - 1].regime;
    const to = point.regime;
    return { from, to, timestamp: point.timestamp };
  }).filter(item => item !== null && item.from !== item.to);
  
  // Get current regime
  const currentRegime = regimeData.length > 0 ? regimeData[regimeData.length - 1].regime : 'unknown';
  const currentRegimeColor = regimeColors[currentRegime as keyof typeof regimeColors] || regimeColors.unknown;
  
  return (
    <Box sx={{ width: '100%' }}>
      <Grid container spacing={3}>
        {/* Regime selection and current regime */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth variant="outlined" size="small">
                    <InputLabel id="asset-select-label">Asset</InputLabel>
                    <Select
                      labelId="asset-select-label"
                      id="asset-select"
                      value={selectedAsset}
                      onChange={handleAssetChange}
                      label="Asset"
                    >
                      {availableAssets.map((assetOption) => (
                        <MenuItem key={assetOption} value={assetOption}>
                          {assetOption === 'global' ? 'Global Market' : assetOption}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={8}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                    <Typography variant="subtitle1" sx={{ mr: 2 }}>
                      Current Regime:
                    </Typography>
                    <Chip
                      label={currentRegime.toUpperCase()}
                      sx={{
                        bgcolor: currentRegimeColor,
                        color: theme.palette.getContrastText(currentRegimeColor),
                        fontWeight: 'bold'
                      }}
                    />
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Regime Timeline */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Market Regime Timeline
              </Typography>
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer>
                  <AreaChart
                    data={regimeData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="timestamp"
                      tickFormatter={(tick) => new Date(tick).toLocaleDateString()}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tickFormatter={(value) => {
                        const regimes = ['bear', 'volatile', 'sideways', 'consolidation', 'bull'];
                        const index = Math.round(value * (regimes.length - 1));
                        return regimes[index] || '';
                      }}
                    />
                    <Tooltip
                      labelFormatter={(label) => new Date(label).toLocaleString()}
                      formatter={(value: any, name: string, props: any) => {
                        if (name === 'strength') return [`${(Number(value) * 100).toFixed(1)}%`, 'Confidence'];
                        return [props.payload?.regime || value, 'Regime'];
                      }}
                    />
                    <defs>
                      {Object.entries(regimeColors).map(([regime, color]) => (
                        <linearGradient key={regime} id={`color-${regime}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={color} stopOpacity={0.8} />
                          <stop offset="95%" stopColor={color} stopOpacity={0.1} />
                        </linearGradient>
                      ))}
                    </defs>
                    {/* Custom areas for each regime */}
                    {regimeData.length > 0 && Object.keys(regimeColors).map((regime) => {
                      const filteredData = regimeData.map(item => ({
                        ...item,
                        value: item.regime === regime ? item.strength : 0
                      }));
                      
                      return (
                        <Area
                          key={regime}
                          type="monotone"
                          dataKey="value"
                          name={regime}
                          stroke={regimeColors[regime as keyof typeof regimeColors]}
                          fill={`url(#color-${regime})`}
                          fillOpacity={1}
                          stackId={1}
                        />
                      );
                    })}
                    {/* Overlay for regime strength */}
                    <Line
                      type="monotone"
                      dataKey="strength"
                      name="strength"
                      stroke={theme.palette.text.primary}
                      dot={false}
                      strokeDasharray="3 3"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Regime Summary */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Regime Distribution
              </Typography>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie
                      data={regimeSummary}
                      dataKey="count"
                      nameKey="regime"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    >
                      {regimeSummary.map((entry) => (
                        <Cell
                          key={entry.regime}
                          fill={entry.color || regimeColors[entry.regime as keyof typeof regimeColors] || regimeColors.unknown}
                        />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value} occurrences`, 'Count']} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Volatility vs Correlation */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Volatility vs. Correlation Clusters
              </Typography>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                  >
                    <CartesianGrid />
                    <XAxis
                      type="number"
                      dataKey="volatility"
                      name="Volatility"
                      label={{ value: 'Volatility', position: 'bottom' }}
                      domain={[0, 'dataMax']}
                    />
                    <YAxis
                      type="number"
                      dataKey="correlation"
                      name="Correlation"
                      label={{ value: 'Correlation', angle: -90, position: 'left' }}
                      domain={[-1, 1]}
                    />
                    <ZAxis type="number" dataKey="strength" range={[20, 100]} />
                    <Tooltip
                      formatter={(value, name) => {
                        if (name === 'Volatility') return [(Number(value) * 100).toFixed(1) + '%', name];
                        if (name === 'Correlation') return [Number(value).toFixed(2), name];
                        return [value, name];
                      }}
                      labelFormatter={(_, payload) => {
                        if (payload && payload.length) {
                          return `${payload[0].payload.regime.toUpperCase()} - ${new Date(payload[0].payload.timestamp).toLocaleString()}`;
                        }
                        return '';
                      }}
                    />
                    <Legend />
                    
                    {/* Cluster each regime separately for better visualization */}
                    {Object.keys(regimeColors).map((regime) => {
                      const filteredData = regimeData.filter(item => item.regime === regime);
                      if (filteredData.length === 0) return null;
                      
                      return (
                        <Scatter
                          key={regime}
                          name={regime.charAt(0).toUpperCase() + regime.slice(1)}
                          data={filteredData}
                          fill={regimeColors[regime as keyof typeof regimeColors]}
                        />
                      );
                    })}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Regime Transitions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Regime Transitions
              </Typography>
              <Box sx={{ mt: 2 }}>
                {regimeTransitions.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {regimeTransitions.slice(-6).reverse().map((transition, index) => {
                      if (!transition) return null;
                      const fromColor = regimeColors[transition.from as keyof typeof regimeColors] || regimeColors.unknown;
                      const toColor = regimeColors[transition.to as keyof typeof regimeColors] || regimeColors.unknown;
                      
                      return (
                        <Paper key={index} elevation={1} sx={{ p: 2 }}>
                          <Grid container alignItems="center" spacing={1}>
                            <Grid item>
                              <Chip
                                label={transition.from.toUpperCase()}
                                size="small"
                                sx={{
                                  bgcolor: fromColor,
                                  color: theme.palette.getContrastText(fromColor)
                                }}
                              />
                            </Grid>
                            <Grid item>
                              <Typography variant="body2">→</Typography>
                            </Grid>
                            <Grid item>
                              <Chip
                                label={transition.to.toUpperCase()}
                                size="small"
                                sx={{
                                  bgcolor: toColor,
                                  color: theme.palette.getContrastText(toColor)
                                }}
                              />
                            </Grid>
                            <Grid item xs>
                              <Typography variant="body2" align="right" color="text.secondary">
                                {new Date(transition.timestamp).toLocaleString()}
                              </Typography>
                            </Grid>
                          </Grid>
                        </Paper>
                      );
                    })}
                  </div>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No regime transitions in the selected timeframe
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MarketRegimeVisualization;
