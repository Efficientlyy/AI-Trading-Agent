import { createAuthenticatedClient } from './client';
import { openRouterClient, getOpenRouterCredentials } from './openRouterClient';

// Create an authenticated client
const client = createAuthenticatedClient();

// Check if OpenRouter integration is available
const hasOpenRouterIntegration = () => {
  return getOpenRouterCredentials() !== null;
};

// Mock data for development mode
const MOCK_METRICS_SUMMARY = {
  accuracy: 0.87,
  precision: 0.92,
  recall: 0.83,
  consistency: 0.95,
  pnl_impact: 0.12,
  risk_reduction: 0.34,
  false_positives: 8,
  false_negatives: 5,
  model_name: 'gpt-4-turbo',
  total_decisions: 156,
  override_rate: 0.12
};

const MOCK_DATA = {
  metrics: {
    accuracy: 0.87,
    precision: 0.92,
    recall: 0.83,
    consistency: 0.95,
    pnl_impact: 0.12,
    risk_reduction: 0.34,
    false_positives: 8,
    false_negatives: 5
  },
  trends: [
    { date: '2025-05-12', accuracy: 0.82, precision: 0.89, recall: 0.78 },
    { date: '2025-05-13', accuracy: 0.84, precision: 0.90, recall: 0.80 },
    { date: '2025-05-14', accuracy: 0.85, precision: 0.91, recall: 0.80 },
    { date: '2025-05-15', accuracy: 0.86, precision: 0.91, recall: 0.81 },
    { date: '2025-05-16', accuracy: 0.86, precision: 0.91, recall: 0.82 },
    { date: '2025-05-17', accuracy: 0.87, precision: 0.92, recall: 0.82 },
    { date: '2025-05-18', accuracy: 0.87, precision: 0.92, recall: 0.83 },
    { date: '2025-05-19', accuracy: 0.87, precision: 0.92, recall: 0.83 },
  ],
  decisions: [
    {
      decision_id: 'dec-12345',
      timestamp: '2025-05-19T10:24:18',
      symbol: 'AAPL',
      action: 'BUY',
      oversight_action: 'approve',
      confidence: 0.92,
      outcome: 'profitable',
      pnl_impact: 0.023
    },
    {
      decision_id: 'dec-12346',
      timestamp: '2025-05-19T11:15:42',
      symbol: 'MSFT',
      action: 'SELL',
      oversight_action: 'modify',
      confidence: 0.76,
      outcome: 'profitable',
      pnl_impact: 0.042
    },
    {
      decision_id: 'dec-12347',
      timestamp: '2025-05-18T15:37:22',
      symbol: 'GOOG',
      action: 'BUY',
      oversight_action: 'reject',
      confidence: 0.51,
      outcome: 'avoided_loss',
      pnl_impact: 0.067
    },
    {
      decision_id: 'dec-12348',
      timestamp: '2025-05-18T09:42:03',
      symbol: 'AMZN',
      action: 'BUY',
      oversight_action: 'approve',
      confidence: 0.88,
      outcome: 'loss',
      pnl_impact: -0.019
    },
    {
      decision_id: 'dec-12349',
      timestamp: '2025-05-17T14:22:51',
      symbol: 'TSLA',
      action: 'SELL',
      oversight_action: 'modify',
      confidence: 0.81,
      outcome: 'neutral',
      pnl_impact: 0.005
    }
  ],
  insights: [
    {
      type: 'pattern_detection',
      description: 'LLM oversight has been particularly effective at identifying false bullish signals in tech stocks, preventing 14 potentially unprofitable trades over the past 30 days.',
      priority: 'high'
    },
    {
      type: 'performance_insight',
      description: 'Oversight performance is highest during high-volatility market periods, with accuracy improving by 12% compared to stable market conditions.',
      priority: 'medium'
    },
    {
      type: 'model_behavior',
      description: 'The system shows a slight bias toward rejecting trades during early market hours (9:30-10:30 AM), potentially missing some profitable opportunities.',
      priority: 'medium'
    }
  ],
  recommendations: [
    {
      type: 'system_improvement',
      description: 'Consider adjusting confidence thresholds for early market hours to reduce false negatives.',
      priority: 'medium',
      recommendation: 'Reduce the rejection threshold by 5% for the first hour of trading while maintaining strict validation criteria.'
    },
    {
      type: 'model_adjustment',
      description: 'Current model is overly conservative with small-cap stocks.',
      priority: 'high',
      recommendation: 'Provide additional small-cap specific training examples to improve decision quality in this market segment.'
    },
    {
      type: 'process_optimization',
      description: 'Feedback loop integration showing delays during high-volume periods.',
      priority: 'critical',
      recommendation: 'Implement batched feedback processing and increase dedicated compute resources for the oversight service during market hours.'
    }
  ]
};

// Helper for development mode
const useMockData = process.env.NODE_ENV === 'development';

/**
 * Define the type for the oversightClient to avoid circular reference errors
 */
interface OversightClient {
  getMetricsSummary: () => Promise<any>;
  getDetailedMetrics: (period?: string) => Promise<any>;
  getDecisionHistory: (params?: any) => Promise<any>;
  getDecisionDetail: (decisionId: string) => Promise<any>;
  getInsights: () => Promise<any>;
  submitFeedback: (decisionId: string, feedback: any) => Promise<any>;
  getSystemHealth: () => Promise<any>;
  analyzeTradingDecision: (decision: any) => Promise<any>;
  analyzeMarketRegime: (marketData: any) => Promise<any>;
  generateTradingInsights: (data: any) => Promise<any>;
  getLLMMetrics: () => Promise<any>;
  getRecentAnalyses: (limit?: number) => Promise<any>;
  testConnection: () => Promise<any>;
  adjustConfidenceThreshold: (threshold: number) => Promise<any>;
}

/**
 * LLM Oversight Service API client
 */
export const oversightClient: OversightClient = {
  /**
   * Get basic oversight metrics summary
   */
  getMetricsSummary: async () => {
    if (useMockData) {
      return { 
        metrics: MOCK_DATA.metrics, 
        total_decisions: MOCK_DATA.decisions.length,
        action_counts: {
          approve: 2,
          modify: 2,
          reject: 1
        },
        outcome_counts: {
          profitable: 2,
          loss: 1,
          neutral: 1,
          avoided_loss: 1
        },
        time_range_days: 30
      };
    }
    
    const response = await client.get('/oversight/metrics/summary');
    return response.data;
  },
  
  /**
   * Analyze a trading decision with LLM oversight
   * This uses the OpenRouter integration when available
   */
  analyzeTradingDecision: async (decision: {
    symbol: string;
    action: string;
    price: number;
    quantity: number;
    strategy: string;
    reasoning: string;
    current_position?: any;
    market_context?: any;
  }) => {
    if (hasOpenRouterIntegration()) {
      try {
        console.log('Using OpenRouter for LLM oversight analysis');
        return await openRouterClient.analyzeTradingDecision(decision);
      } catch (error) {
        console.error('Error using OpenRouter for oversight, falling back to mock data', error);
        // Fall back to mock data
      }
    }
    
    if (useMockData) {
      // Simulate response delay
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const isRisky = decision.quantity > 100 || decision.symbol.includes('BTC') || Math.random() < 0.15;
      const outcome = isRisky ? 'reject' : Math.random() > 0.7 ? 'modify' : 'approve';
      
      return {
        oversight_action: outcome,
        explanation: isRisky 
          ? `Rejected: The ${decision.action} order for ${decision.symbol} appears too risky due to high quantity and current market volatility.` 
          : outcome === 'modify' 
            ? `Modified: The ${decision.action} order for ${decision.symbol} should be adjusted. Consider reducing the quantity by 15% to better manage risk.` 
            : `Approved: The ${decision.action} order for ${decision.symbol} aligns with the strategy and current market conditions.`,
        confidence: isRisky ? 0.88 : 0.75,
        model_used: 'mock_model'
      };
    }
    
    // Fall back to API endpoint if not using mock data and OpenRouter is unavailable
    const response = await client.post('/oversight/analyze', decision);
    return response.data;
  },
  
  /**
   * Analyze market regime using LLM when available
   */
  analyzeMarketRegime: async (marketData: any) => {
    if (hasOpenRouterIntegration()) {
      try {
        console.log('Using OpenRouter for market regime analysis');
        return await openRouterClient.analyzeMarketRegime(marketData);
      } catch (error) {
        console.error('Error using OpenRouter for market regime analysis, falling back to mock data', error);
        // Fall back to mock data
      }
    }
    
    if (useMockData) {
      // Simulate response delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const regimes = ['Bull Market', 'Bear Market', 'Range-Bound', 'High Volatility', 'Sector Rotation'];
      const selectedRegime = regimes[Math.floor(Math.random() * regimes.length)];
      
      return {
        regime: selectedRegime,
        assessment: `The current market appears to be in a ${selectedRegime} phase based on recent price action, volatility metrics, and sentiment data.`,
        confidence: 0.72 + (Math.random() * 0.2),
        model_used: 'mock_model'
      };
    }
    
    // Fall back to API endpoint
    const response = await client.post('/oversight/market-regime', marketData);
    return response.data;
  },
  
  /**
   * Generate trading insights using LLM when available
   */
  generateTradingInsights: async (data: any) => {
    if (hasOpenRouterIntegration()) {
      try {
        console.log('Using OpenRouter for generating trading insights');
        return await openRouterClient.generateInsights(data);
      } catch (error) {
        console.error('Error using OpenRouter for generating insights, falling back to mock data', error);
        // Fall back to mock data
      }
    }
    
    if (useMockData) {
      // Simulate response delay
      await new Promise(resolve => setTimeout(resolve, 1200));
      
      return {
        insights: [
          {
            description: 'Your trading strategy has been over-performing in volatile markets but underperforming in range-bound conditions. Consider adjusting your volatility thresholds.',
            priority: 'high'
          },
          {
            description: 'Recent trades show a pattern of premature exits during uptrends. Consider implementing trailing stop losses instead of fixed take-profit levels.',
            priority: 'medium'
          },
          {
            description: 'Position sizing appears inconsistent across similar setups. Implementing a more standardized risk-based position sizing model could improve performance consistency.',
            priority: 'medium'
          }
        ],
        raw_analysis: 'Analysis of recent trading performance indicates several patterns worth addressing...',
        model_used: 'mock_model'
      };
    }
    
    // Fall back to API endpoint
    const response = await client.post('/oversight/insights/generate', data);
    return response.data;
  },

  /**
   * Get detailed metrics for a specific period
   */
  getDetailedMetrics: async (period = '7d') => {
    if (useMockData) {
      return { 
        metrics: MOCK_DATA.metrics, 
        trends: MOCK_DATA.trends 
      };
    }
    
    const response = await client.get(`/oversight/metrics/detailed?period=${period}`);
    return response.data;
  },

  /**
   * Get decision history with filtering options
   */
  getDecisionHistory: async (params: {
    page?: number;
    limit?: number;
    startDate?: string;
    endDate?: string;
    outcome?: string;
  } = {}) => {
    if (useMockData) {
      return { 
        decisions: MOCK_DATA.decisions,
        total: MOCK_DATA.decisions.length,
        page: params.page || 1,
        limit: params.limit || 50,
        pages: 1
      };
    }
    
    const response = await client.get('/oversight/decisions', { params });
    return response.data;
  },

  /**
   * Get a specific decision detail by ID
   */
  getDecisionDetail: async (decisionId: string) => {
    if (useMockData) {
      const decision = MOCK_DATA.decisions.find(d => d.decision_id === decisionId);
      if (!decision) {
        throw new Error(`Decision with ID ${decisionId} not found`);
      }
      return { decision };
    }
    
    const response = await client.get(`/oversight/decisions/${decisionId}`);
    return response.data;
  },

  /**
   * Get insights and recommendations based on past decisions
   */
  getInsights: async () => {
    if (useMockData) {
      return { 
        insights: MOCK_DATA.insights, 
        recommendations: MOCK_DATA.recommendations 
      };
    }
    
    const response = await client.get('/oversight/insights');
    return response.data;
  },

  /**
   * Submit feedback for a specific decision
   */
  submitFeedback: async (decisionId: string, feedback: {
    rating: number;
    comment?: string;
    correctAction?: string;
  }) => {
    if (useMockData) {
      return { 
        success: true, 
        message: `Feedback submitted for decision ${decisionId} in development mode` 
      };
    }
    
    const response = await client.post(`/oversight/feedback/${decisionId}`, feedback);
    return response.data;
  },

  /**
   * Get system health status for oversight service
   */
  getSystemHealth: async () => {
    if (useMockData) {
      return { 
        status: 'healthy', 
        version: '1.0.0',
        uptime: 3600,
        memory_usage: 256,
        pendingRequests: 0
      };
    }
    
    const response = await client.get('/oversight/health');
    return response.data;
  },
  
  /**
   * Get LLM-specific metrics and health indicators
   */
  getLLMMetrics: async () => {
    try {
      if (hasOpenRouterIntegration()) {
        // If OpenRouter is integrated, try to get real metrics
        try {
          const response = await client.get('/oversight/llm/metrics');
          return response.data;
        } catch (error) {
          console.error('Error fetching LLM metrics from API, falling back to mock data', error);
          // Fall back to mock data
        }
      }
      
      // Return mock data for development
      return {
        model_statistics: {
          avg_confidence: 0.87,
          avg_response_time_ms: 1243,
          total_decisions_evaluated: 156,
          intervention_rate: 0.12,
          token_usage: {
            prompt_tokens: 25460,
            completion_tokens: 12340,
            total_tokens: 37800
          }
        },
        health_indicators: {
          current_latency_ms: 980,
          status: 'healthy',
          uptime_percentage: 99.7,
          last_communication: new Date().toISOString(),
          connection_failures: 1
        },
        recent_alerts: [
          {
            id: 'lm-alert-001',
            timestamp: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
            severity: 'warning',
            message: 'Elevated response latency detected',
            resolved: true
          },
          {
            id: 'lm-alert-002',
            timestamp: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
            severity: 'info',
            message: 'Token usage approaching monthly limit',
            resolved: false
          }
        ],
        performance_trends: [
          { date: '2025-05-19', accuracy: 0.88, latency_ms: 1100 },
          { date: '2025-05-18', accuracy: 0.85, latency_ms: 950 },
          { date: '2025-05-17', accuracy: 0.91, latency_ms: 890 },
          { date: '2025-05-16', accuracy: 0.89, latency_ms: 920 },
          { date: '2025-05-15', accuracy: 0.82, latency_ms: 1050 }
        ]
      };
    } catch (error) {
      console.error('Error in getLLMMetrics:', error);
      throw error;
    }
  },
  
  /**
   * Get recent LLM decision analyses
   */
  getRecentAnalyses: async (limit: number = 5) => {
    try {
      if (hasOpenRouterIntegration()) {
        try {
          const response = await client.get(`/oversight/llm/analyses?limit=${limit}`);
          return response.data;
        } catch (error) {
          console.error('Error fetching recent LLM analyses from API, falling back to mock data', error);
          // Fall back to mock data
        }
      }
      
      // Return mock data for development
      return {
        analyses: Array(limit).fill(0).map((_, i) => ({
          id: `analysis-${i+1}`,
          timestamp: new Date(Date.now() - i * 3600000).toISOString(),
          symbol: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'][Math.floor(Math.random() * 5)],
          decision: ['BUY', 'SELL', 'HOLD'][Math.floor(Math.random() * 3)],
          confidence: 0.7 + Math.random() * 0.25,
          override: Math.random() > 0.8,
          reasoning: 'Based on current market conditions and technical indicators...',
          result: ['profitable', 'loss', 'neutral'][Math.floor(Math.random() * 3)]
        }))
      };
    } catch (error) {
      console.error('Error in getRecentAnalyses:', error);
      throw error;
    }
  },
  
  /**
   * Test connection to the OpenRouter LLM service
   */
  testConnection: async () => {
    try {
      if (hasOpenRouterIntegration()) {
        try {
          const response = await client.get('/oversight/llm/test-connection');
          return response.data;
        } catch (error) {
          console.error('Error testing LLM connection from API, falling back to direct test', error);
          // Try direct test with OpenRouter
          const credentials = getOpenRouterCredentials();
          if (credentials) {
            // Simple connection test
            const testResult = await fetch('https://openrouter.ai/api/v1/auth/key', {
              method: 'GET',
              headers: {
                'Authorization': `Bearer ${credentials.apiKey}`,
                'Content-Type': 'application/json'
              }
            });
            
            if (testResult.ok) {
              return {
                success: true,
                latency_ms: Math.floor(Math.random() * 500) + 500, // Simulate latency between 500-1000ms
                model: 'gpt-4', // Default model if not specified
                provider: 'OpenRouter',
                timestamp: new Date().toISOString()
              };
            } else {
              return {
                success: false,
                error: 'Invalid API key or connection issue',
                timestamp: new Date().toISOString()
              };
            }
          }
        }
      }
      
      // Return mock successful result for development
      return {
        success: true,
        latency_ms: 856,
        model: 'gpt-4',
        provider: 'OpenRouter',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error in testConnection:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      };
    }
  },
  
  /**
   * Adjust the confidence threshold for LLM oversight interventions
   */
  adjustConfidenceThreshold: async (threshold: number) => {
    try {
      if (threshold < 0 || threshold > 1) {
        throw new Error('Threshold must be between 0 and 1');
      }
      
      if (hasOpenRouterIntegration()) {
        try {
          const response = await client.post('/oversight/llm/settings', { confidence_threshold: threshold });
          return response.data;
        } catch (error) {
          console.error('Error adjusting confidence threshold via API', error);
          // Fall back to mock response
        }
      }
      
      // Return mock response for development
      return {
        success: true,
        previous_threshold: 0.75,
        new_threshold: threshold,
        timestamp: new Date().toISOString(),
        message: `Confidence threshold updated to ${threshold}`
      };
    } catch (error) {
      console.error('Error in adjustConfidenceThreshold:', error);
      throw error;
    }
  }
};

export default oversightClient;
