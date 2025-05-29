import React, { useState, useEffect } from 'react';
import { Box, Container, Typography, Grid, Paper, Tabs, Tab, CircularProgress, Button, ButtonGroup } from '@mui/material';
import { styled } from '@mui/material/styles';
import { 
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import oversightClient from '../api/oversightClient';
import { useLLMOversight } from '../context/LLMOversightContext';
import OpenRouterTestComponent from '../components/oversight/OpenRouterTestComponent';

// Custom styled components
const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  boxShadow: '0px 3px 15px rgba(0, 0, 0, 0.1)',
  height: '100%',
  color: '#e0e0e0',
  backgroundColor: '#1e2030',
  borderRadius: '6px'
}));

const MetricCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  boxShadow: '0px 3px 15px rgba(0, 0, 0, 0.1)',
  color: '#e0e0e0',
  backgroundColor: '#2c3144',
  borderRadius: '6px',
  '& .MuiTypography-subtitle2': {
    color: '#b0b0b0',
    marginBottom: '8px'
  },
  '& .MuiTypography-h4': {
    color: '#ffffff',
    fontWeight: 500
  }
}));

const TabPanel = (props: { children: React.ReactNode; value: number; index: number }) => {
  const { children, value, index } = props;
  return (
    <div role="tabpanel" hidden={value !== index} style={{ padding: '16px 0' }}>
      {value === index && <>{children}</>}
    </div>
  );
};

interface MetricData {
  accuracy: number;
  precision: number;
  recall: number;
  consistency: number;
  pnl_impact: number;
  risk_reduction: number;
  false_positives: number;
  false_negatives: number;
}

interface OversightInsight {
  type: string;
  description: string;
  priority?: string;
  recommendation?: string;
}

interface DecisionHistoryItem {
  decision_id: string;
  timestamp: string;
  symbol: string;
  action: string;
  oversight_action: string;
  confidence: number;
  outcome: string;
  pnl_impact: number;
}

interface OversightDashboardData {
  metrics: MetricData;
  decision_history: DecisionHistoryItem[];
  insights: OversightInsight[];
  recommendations: OversightInsight[];
}

const COLORS = ['#4285F4', '#00C49F', '#FFB833', '#FF5252', '#9C27B0'];

const LLMOversightPage: React.FC = () => {
  const { status } = useLLMOversight();
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<OversightDashboardData | null>(null);
  const [metricsTrend, setMetricsTrend] = useState<any[]>([]);
  const [decisionDistribution, setDecisionDistribution] = useState<any[]>([]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleTimeRangeChange = (range: '7d' | '30d' | '90d') => {
    setTimeRange(range);
    fetchDashboardData(range);
  };

  const fetchDashboardData = async (range: string) => {
    setLoading(true);
    try {
      // Fetch all necessary data
      const [metricsData, decisionHistoryData, insightsData] = await Promise.all([
        oversightClient.getDetailedMetrics(range),
        oversightClient.getDecisionHistory({
          startDate: getStartDateFromRange(range),
          limit: 50
        }),
        oversightClient.getInsights()
      ]);
      
      // Construct dashboard data from API responses
      const dashboardData: OversightDashboardData = {
        metrics: metricsData.metrics,
        decision_history: decisionHistoryData.decisions,
        insights: insightsData.insights || [],
        recommendations: insightsData.recommendations || []
      };
      
      setData(dashboardData);
      
      // Process metrics trend data
      if (metricsData.trends) {
        setMetricsTrend(metricsData.trends);
      }
      
      // Process decision distribution
      if (decisionHistoryData.decisions) {
        const distribution = processDecisionDistribution(decisionHistoryData.decisions);
        setDecisionDistribution(distribution);
      }
      
    } catch (error) {
      console.error('Error fetching oversight dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // Helper function to convert time range to start date
  const getStartDateFromRange = (range: string): string => {
    const now = new Date();
    let startDate = new Date();
    
    if (range === '7d') {
      startDate.setDate(now.getDate() - 7);
    } else if (range === '30d') {
      startDate.setDate(now.getDate() - 30);
    } else if (range === '90d') {
      startDate.setDate(now.getDate() - 90);
    }
    
    return startDate.toISOString().split('T')[0]; // YYYY-MM-DD format
  };

  const processDecisionDistribution = (decisions: DecisionHistoryItem[]) => {
    const distribution = [
      { name: 'Approved', value: 0 },
      { name: 'Rejected', value: 0 },
      { name: 'Modified', value: 0 }
    ];
    
    decisions.forEach(decision => {
      if (decision.oversight_action === 'approve') {
        distribution[0].value++;
      } else if (decision.oversight_action === 'reject') {
        distribution[1].value++;
      } else if (decision.oversight_action === 'modify') {
        distribution[2].value++;
      }
    });
    
    return distribution;
  };

  useEffect(() => {
    fetchDashboardData(timeRange);
    // Refresh data every 5 minutes
    const interval = setInterval(() => {
      fetchDashboardData(timeRange);
    }, 5 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, [timeRange]);

  if (loading && !data) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" sx={{ mb: 2, color: '#e0e0e0' }}>
          LLM Oversight Dashboard
        </Typography>
        
        {/* OpenRouter Test Component */}
        <OpenRouterTestComponent />
        
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            variant="fullWidth"
            centered
            sx={{
              '& .MuiTabs-indicator': {
                backgroundColor: '#4285F4',
                height: 3
              },
              '& .MuiTab-root': {
                color: '#b0b0b0',
                '&.Mui-selected': {
                  color: '#ffffff',
                  fontWeight: 'bold'
                }
              }
            }}
          >
            <Tab label="Performance Overview" />
            <Tab label="Decision Analytics" />
            <Tab label="Insights & Recommendations" />
          </Tabs>
          
          <Box>
            <ButtonGroup variant="outlined" aria-label="time range button group">
              <Button
                onClick={() => handleTimeRangeChange('7d')}
                variant={timeRange === '7d' ? 'contained' : 'outlined'}
                sx={{
                  backgroundColor: timeRange === '7d' ? '#4285F4' : 'transparent',
                  color: timeRange === '7d' ? 'white' : '#b0b0b0',
                  borderColor: '#4285F4',
                  '&:hover': {
                    backgroundColor: timeRange === '7d' ? '#4285F4' : 'rgba(66, 133, 244, 0.1)',
                    borderColor: '#4285F4'
                  }
                }}
              >
                7 Days
              </Button>
              <Button
                onClick={() => handleTimeRangeChange('30d')}
                variant={timeRange === '30d' ? 'contained' : 'outlined'}
                sx={{
                  backgroundColor: timeRange === '30d' ? '#4285F4' : 'transparent',
                  color: timeRange === '30d' ? 'white' : '#b0b0b0',
                  borderColor: '#4285F4',
                  '&:hover': {
                    backgroundColor: timeRange === '30d' ? '#4285F4' : 'rgba(66, 133, 244, 0.1)',
                    borderColor: '#4285F4'
                  }
                }}
              >
                30 Days
              </Button>
              <Button
                onClick={() => handleTimeRangeChange('90d')}
                variant={timeRange === '90d' ? 'contained' : 'outlined'}
                sx={{
                  backgroundColor: timeRange === '90d' ? '#4285F4' : 'transparent',
                  color: timeRange === '90d' ? 'white' : '#b0b0b0',
                  borderColor: '#4285F4',
                  '&:hover': {
                    backgroundColor: timeRange === '90d' ? '#4285F4' : 'rgba(66, 133, 244, 0.1)',
                    borderColor: '#4285F4'
                  }
                }}
              >
                90 Days
              </Button>
            </ButtonGroup>
          </Box>
        </Box>
        
        {data && (
          <>
            <TabPanel value={tabValue} index={0}>
              <Grid container spacing={3}>
                {/* Key Metrics */}
                <Grid item xs={12} md={6} lg={3}>
                  <MetricCard>
                    <Typography variant="subtitle2" color="textSecondary">Accuracy</Typography>
                    <Typography variant="h4">{(data.metrics.accuracy * 100).toFixed(1)}%</Typography>
                  </MetricCard>
                </Grid>
                <Grid item xs={12} md={6} lg={3}>
                  <MetricCard>
                    <Typography variant="subtitle2" color="textSecondary">Precision</Typography>
                    <Typography variant="h4">{(data.metrics.precision * 100).toFixed(1)}%</Typography>
                  </MetricCard>
                </Grid>
                <Grid item xs={12} md={6} lg={3}>
                  <MetricCard>
                    <Typography variant="subtitle2" color="textSecondary">PnL Impact</Typography>
                    <Typography variant="h4" color={data.metrics.pnl_impact >= 0 ? 'success.main' : 'error.main'}>
                      ${data.metrics.pnl_impact.toFixed(2)}
                    </Typography>
                  </MetricCard>
                </Grid>
                <Grid item xs={12} md={6} lg={3}>
                  <MetricCard>
                    <Typography variant="subtitle2" color="textSecondary">Risk Reduction</Typography>
                    <Typography variant="h4">{data.metrics.risk_reduction.toFixed(2)}</Typography>
                  </MetricCard>
                </Grid>
                
                {/* Performance Metrics Chart */}
                <Grid item xs={12} md={8}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ color: '#e0e0e0' }}>
                      Performance Metrics Over Time
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart width={500} height={300} data={metricsTrend} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="date" tick={{ fill: '#e0e0e0' }} />
                        <YAxis domain={[0.5, 1]} tickCount={6} tick={{ fill: '#e0e0e0' }} />
                        <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                        <Legend />
                        <Line type="monotone" dataKey="accuracy" stroke="#4285F4" name="Accuracy" strokeWidth={2} dot={{ r: 4, fill: '#4285F4' }} />
                        <Line type="monotone" dataKey="precision" stroke="#00C49F" name="Precision" strokeWidth={2} dot={{ r: 4, fill: '#00C49F' }} />
                        <Line type="monotone" dataKey="recall" stroke="#FFB833" name="Recall" strokeWidth={2} dot={{ r: 4, fill: '#FFB833' }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </StyledPaper>
                </Grid>
                
                {/* Decision Distribution Pie Chart */}
                <Grid item xs={12} md={4}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ color: '#e0e0e0' }}>
                      Oversight Decision Distribution
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={decisionDistribution}
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        >
                          {decisionDistribution.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => value} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </StyledPaper>
                </Grid>
              </Grid>
            </TabPanel>
            
            <TabPanel value={tabValue} index={1}>
              <Grid container spacing={3}>
                {/* Decision Outcomes */}
                <Grid item xs={12}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ color: '#e0e0e0' }}>
                      Decision Outcomes by Action
                    </Typography>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart
                        data={[
                          { name: 'Approve', correct: 12, incorrect: 2 },
                          { name: 'Reject', correct: 8, incorrect: 1 },
                          { name: 'Modify', correct: 5, incorrect: 3 }
                        ]} // Mock data with better visibility
                        margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis 
                          dataKey="name"
                          tick={{ fill: '#e0e0e0' }}
                          axisLine={{ stroke: '#4c5067' }}
                        />
                        <YAxis 
                          tick={{ fill: '#e0e0e0' }}
                          axisLine={{ stroke: '#4c5067' }}
                          tickCount={6}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#2c3144', 
                            color: '#ffffff', 
                            border: 'none', 
                            borderRadius: '4px',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                          }} 
                          cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                        />
                        <Legend 
                          wrapperStyle={{ 
                            paddingTop: '10px', 
                            paddingBottom: '10px',
                            color: '#ffffff'
                          }}
                          formatter={(value) => <span style={{ color: '#e0e0e0' }}>{value}</span>}
                        />
                        <Bar 
                          dataKey="correct" 
                          name="Correct" 
                          stackId="a" 
                          fill="#00C49F"
                          radius={[4, 4, 0, 0]}
                          barSize={50}
                        />
                        <Bar 
                          dataKey="incorrect" 
                          name="Incorrect" 
                          stackId="a" 
                          fill="#FF5252"
                          radius={[0, 0, 4, 4]}
                          barSize={50}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </StyledPaper>
                </Grid>
                
                {/* Recent Decisions Table */}
                <Grid item xs={12}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ color: '#e0e0e0' }}>
                      Recent Decisions
                    </Typography>
                    <Box sx={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ borderBottom: '2px solid rgba(255, 255, 255, 0.2)' }}>
                            <th style={{ padding: '8px', textAlign: 'left', color: '#e0e0e0' }}>Time</th>
                            <th style={{ padding: '8px', textAlign: 'left', color: '#e0e0e0' }}>Symbol</th>
                            <th style={{ padding: '8px', textAlign: 'left', color: '#e0e0e0' }}>Action</th>
                            <th style={{ padding: '8px', textAlign: 'left', color: '#e0e0e0' }}>Oversight</th>
                            <th style={{ padding: '8px', textAlign: 'left', color: '#e0e0e0' }}>Confidence</th>
                            <th style={{ padding: '8px', textAlign: 'left', color: '#e0e0e0' }}>Outcome</th>
                            <th style={{ padding: '8px', textAlign: 'right', color: '#e0e0e0' }}>PnL Impact</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.decision_history && data.decision_history.slice(0, 10).map((decision) => (
                            <tr key={decision.decision_id}>
                              <td style={{ padding: '8px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#e0e0e0' }}>
                                {new Date(decision.timestamp).toLocaleString()}
                              </td>
                              <td style={{ padding: '8px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#e0e0e0' }}>{decision.symbol}</td>
                              <td style={{ padding: '8px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#e0e0e0' }}>{decision.action}</td>
                              <td style={{ padding: '8px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#e0e0e0' }}>
                                <span style={{
                                  padding: '4px 8px',
                                  borderRadius: '4px',
                                  backgroundColor: 
                                    decision.oversight_action === 'approve' ? 'rgba(0, 196, 159, 0.2)' :
                                    decision.oversight_action === 'reject' ? 'rgba(255, 82, 82, 0.2)' : 'rgba(255, 184, 51, 0.2)',
                                  color:
                                    decision.oversight_action === 'approve' ? '#00C49F' :
                                    decision.oversight_action === 'reject' ? '#FF5252' : '#FFB833'
                                }}>
                                  {decision.oversight_action}
                                </span>
                              </td>
                              <td style={{ padding: '8px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#e0e0e0' }}>
                                {(decision.confidence * 100).toFixed(0)}%
                              </td>
                              <td style={{ padding: '8px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#e0e0e0' }}>
                                <span style={{
                                  padding: '4px 8px',
                                  borderRadius: '4px',
                                  backgroundColor: 
                                    decision.outcome === 'profitable' ? 'rgba(0, 196, 159, 0.2)' :
                                    decision.outcome === 'loss' ? 'rgba(255, 82, 82, 0.2)' : 'rgba(117, 117, 117, 0.2)',
                                  color:
                                    decision.outcome === 'profitable' ? '#00C49F' :
                                    decision.outcome === 'loss' ? '#FF5252' : '#b0b0b0'
                                }}>
                                  {decision.outcome}
                                </span>
                              </td>
                              <td style={{ 
                                padding: '8px', 
                                textAlign: 'right', 
                                borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                                color: decision.pnl_impact >= 0 ? '#00C49F' : '#FF5252'
                              }}>
                                ${decision.pnl_impact?.toFixed(2) || 'N/A'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </Box>
                  </StyledPaper>
                </Grid>
              </Grid>
            </TabPanel>
            
            <TabPanel value={tabValue} index={2}>
              <Grid container spacing={3}>
                {/* Key Insights */}
                <Grid item xs={12} md={6}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ color: '#e0e0e0' }}>
                      Key Insights
                    </Typography>
                    <Box sx={{ mt: 2 }}>
                      {data.insights && data.insights.length > 0 ? (
                        data.insights.map((insight, index) => (
                          <Paper 
                            key={index} 
                            elevation={0} 
                            sx={{ 
                              p: 2, 
                              mb: 2, 
                              backgroundColor: '#2c3144', 
                              borderLeft: '4px solid #4285F4',
                              borderRadius: '4px'
                            }}
                          >
                            <Typography variant="body2" sx={{ mb: 1, fontWeight: 500, color: '#e0e0e0' }}>
                              {insight.type.replace(/_/g, ' ').toUpperCase()}
                            </Typography>
                            <Typography variant="body1" sx={{ color: '#b0b0b0' }}>
                              {insight.description}
                            </Typography>
                          </Paper>
                        ))
                      ) : (
                        <Typography color="textSecondary">
                          No insights available for the selected time period
                        </Typography>
                      )}
                    </Box>
                  </StyledPaper>
                </Grid>
                
                {/* Improvement Recommendations */}
                <Grid item xs={12} md={6}>
                  <StyledPaper>
                    <Typography variant="h6" gutterBottom sx={{ color: '#e0e0e0' }}>Improvement Recommendations</Typography>
                    <Box sx={{ mt: 2 }}>
                      {data.recommendations && data.recommendations.length > 0 ? (
                        data.recommendations.map((rec, index) => (
                          <Paper 
                            key={index} 
                            elevation={0} 
                            sx={{ 
                              p: 2, 
                              mb: 2, 
                              backgroundColor: '#2c3144', 
                              borderLeft: '4px solid',
                              borderLeftColor: rec.priority === 'critical' ? '#ff5252' : 
                                            rec.priority === 'high' ? '#ffb833' : 
                                            rec.priority === 'medium' ? '#4285F4' : '#00C49F',
                              borderRadius: '4px'
                            }}
                          >
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                              <Typography variant="body2" sx={{ fontWeight: 500, color: rec.priority === 'critical' ? '#ff5252' : rec.priority === 'high' ? '#ffb833' : rec.priority === 'medium' ? '#4285F4' : '#00C49F' }}>
                              {rec.priority === 'critical' ? '🚨 CRITICAL' : 
                               rec.priority === 'high' ? '⚠️ HIGH' : 
                               rec.priority === 'medium' ? '📊 MEDIUM' : '📝 LOW'}
                            </Typography>
                            </Box>
                            <Typography variant="body1" sx={{ mb: 1, color: '#b0b0b0' }}>
                              {rec.recommendation}
                            </Typography>
                          </Paper>
                        ))
                      ) : (
                        <Typography color="textSecondary">
                          No recommendations available for the selected time period
                        </Typography>
                      )}
                    </Box>
                  </StyledPaper>
                </Grid>
              </Grid>
            </TabPanel>
          </>
        )}
      </Box>
    </Container>
  );
};

export default LLMOversightPage;
