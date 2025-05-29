import axios, { AxiosInstance } from 'axios';

/**
 * Interface for the OpenRouter API credentials
 */
export interface OpenRouterCredentials {
  apiKey: string;
  model: string;
}

/**
 * Get OpenRouter credentials from localStorage if available
 */
export const getOpenRouterCredentials = (): OpenRouterCredentials | null => {
  try {
    // Retrieve saved integrations from localStorage
    const savedIntegrationsString = localStorage.getItem('tradingAgentIntegrations');
    if (!savedIntegrationsString) return null;
    
    const savedIntegrations = JSON.parse(savedIntegrationsString);
    
    // Find LLM oversight integration
    const llmIntegration = savedIntegrations.find((integration: any) => 
      integration.id === 'llm_oversight' && 
      integration.status === 'connected'
    );
    
    if (!llmIntegration) return null;
    
    // Check if provider is OpenRouter
    const providerField = llmIntegration.additionalFields?.find(
      (field: any) => field.name === 'provider'
    );
    
    if (!providerField || providerField.value !== 'OpenRouter') return null;
    
    // Get model name
    const modelField = llmIntegration.additionalFields?.find(
      (field: any) => field.name === 'model'
    );
    
    // Return credentials if available
    if (llmIntegration.apiKeyMasked) {
      // Get the real API key from secure storage
      // For this implementation, we need to retrieve the actual API key from a more secure place
      // For now, we'll use localStorage, but in production this should be more secure
      const apiKeys = JSON.parse(localStorage.getItem('apiKeys') || '{}');
      const apiKey = apiKeys['llm_oversight'] || '';
      
      if (!apiKey) {
        console.error('API key for OpenRouter not found in secure storage');
        return null;
      }
      
      return {
        apiKey,
        model: modelField?.value || 'gpt-4'
      };
    }
    
    return null;
  } catch (error) {
    console.error('Error retrieving OpenRouter credentials:', error);
    return null;
  }
};

/**
 * Create an OpenRouter API client
 */
export const createOpenRouterClient = (credentials: OpenRouterCredentials): AxiosInstance => {
  const client = axios.create({
    baseURL: 'https://openrouter.ai/api/v1',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${credentials.apiKey}`,
      'HTTP-Referer': window.location.origin,
      'X-Title': 'AI Trading Agent'
    }
  });
  
  return client;
};

/**
 * OpenRouter client for making LLM API calls
 */
/**
 * Verify if the OpenRouter credentials are valid
 */
export const verifyOpenRouterCredentials = async (): Promise<{ valid: boolean; modelAvailable: boolean }> => {
  try {
    const credentials = getOpenRouterCredentials();
    if (!credentials) {
      return { valid: false, modelAvailable: false };
    }
    
    const client = createOpenRouterClient(credentials);
    
    // Check if we can access the models endpoint
    const response = await client.get('/models');
    
    // Check if the specified model exists in the available models
    const modelAvailable = response.data.data.some(
      (model: any) => model.id === credentials.model || 
                      model.id.includes(credentials.model)
    );
    
    return { 
      valid: true, 
      modelAvailable 
    };
  } catch (error) {
    console.error('Error verifying OpenRouter credentials:', error);
    return { valid: false, modelAvailable: false };
  }
};

export const openRouterClient = {
  /**
   * Get available models from OpenRouter
   */
  getModels: async () => {
    const credentials = getOpenRouterCredentials();
    if (!credentials) {
      throw new Error('OpenRouter credentials not available');
    }
    
    const client = createOpenRouterClient(credentials);
    const response = await client.get('/models');
    return response.data;
  },
  
  /**
   * Make a chat completion request to OpenRouter
   */
  createChatCompletion: async (messages: any[], options: any = {}) => {
    const credentials = getOpenRouterCredentials();
    if (!credentials) {
      throw new Error('OpenRouter credentials not available');
    }
    
    const client = createOpenRouterClient(credentials);
    
    const response = await client.post('/chat/completions', {
      model: credentials.model,
      messages,
      ...options
    });
    
    return response.data;
  },
  
  /**
   * Analyze a trading decision with LLM oversight
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
    const messages = [
      {
        role: 'system',
        content: `You are an expert trading oversight system for an AI-powered trading agent. 
        Your job is to analyze trading decisions before they are executed and determine if they should be:
        1. APPROVE - The trade appears reasonable and can be executed as is
        2. MODIFY - The trade idea has merit but some parameters should be adjusted
        3. REJECT - The trade should not be executed due to risks or flaws in reasoning
        
        Provide a brief explanation of your decision and any recommendations.`
      },
      {
        role: 'user',
        content: `Please analyze the following trading decision:
        
        Symbol: ${decision.symbol}
        Action: ${decision.action}
        Price: ${decision.price}
        Quantity: ${decision.quantity}
        Strategy: ${decision.strategy}
        Reasoning: ${decision.reasoning}
        ${decision.current_position ? `Current Position: ${JSON.stringify(decision.current_position)}` : ''}
        ${decision.market_context ? `Market Context: ${JSON.stringify(decision.market_context)}` : ''}
        
        Provide your oversight decision (APPROVE, MODIFY, or REJECT) and explanation.`
      }
    ];
    
    try {
      const response = await openRouterClient.createChatCompletion(messages, {
        temperature: 0.2,
        max_tokens: 300
      });
      
      const responseText = response.choices[0]?.message?.content || '';
      
      // Parse the decision from the response
      let decision = 'APPROVE'; // Default
      if (responseText.includes('REJECT')) {
        decision = 'REJECT';
      } else if (responseText.includes('MODIFY')) {
        decision = 'MODIFY';
      }
      
      const creds = getOpenRouterCredentials();
      return {
        oversight_action: decision.toLowerCase(),
        explanation: responseText,
        confidence: response.choices[0]?.message?.tool_calls?.[0]?.function?.arguments?.confidence || 0.85,
        model_used: creds?.model || 'unknown'
      };
    } catch (error) {
      console.error('Error analyzing trading decision with OpenRouter:', error);
      // Fallback to approving with a warning
      return {
        oversight_action: 'approve',
        explanation: 'ERROR: Unable to complete oversight analysis. Trade approved by default, but review is recommended.',
        confidence: 0.5,
        model_used: 'fallback',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  },
  
  /**
   * Analyze market conditions to detect regime changes
   */
  analyzeMarketRegime: async (marketData: {
    recent_price_action: any[];
    volatility_metrics: any;
    technical_indicators: any;
    news_sentiment: any;
    macro_indicators?: any;
  }) => {
    const messages = [
      {
        role: 'system',
        content: `You are an expert market analyst for an AI-powered trading system.
        Your job is to analyze current market conditions and identify the market regime.
        Market regimes include: Bull Market, Bear Market, Range-Bound, High Volatility, Low Volatility, 
        Crisis/Crash, Recovery, Sector Rotation, etc.
        
        Provide your assessment of the current market regime, confidence level, and key factors influencing your decision.`
      },
      {
        role: 'user',
        content: `Please analyze the following market data and identify the current market regime:
        
        ${JSON.stringify(marketData, null, 2)}
        
        Provide your assessment of the current market regime, confidence level (0-1), and key factors influencing your decision.`
      }
    ];
    
    try {
      const response = await openRouterClient.createChatCompletion(messages, {
        temperature: 0.3,
        max_tokens: 500
      });
      
      const responseText = response.choices[0]?.message?.content || '';
      
      // Extract regime from response (simple extraction - could be enhanced)
      const regimeMatch = responseText.match(/regime:?\s*([A-Za-z\s\-/]+)/i);
      const confidenceMatch = responseText.match(/confidence:?\s*(0\.\d+|1\.0|1)/i);
      
      const creds = getOpenRouterCredentials();
      return {
        regime: regimeMatch ? regimeMatch[1].trim() : 'Unknown',
        assessment: responseText,
        confidence: confidenceMatch ? parseFloat(confidenceMatch[1]) : 0.7,
        model_used: creds?.model || 'unknown'
      };
    } catch (error) {
      console.error('Error analyzing market regime with OpenRouter:', error);
      return {
        regime: 'Unknown',
        assessment: 'ERROR: Unable to complete market regime analysis.',
        confidence: 0,
        model_used: 'fallback',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  },
  
  /**
   * Generate trading insights based on recent performance and market conditions
   */
  generateInsights: async (data: {
    recent_trades: any[];
    performance_metrics: any;
    market_conditions: any;
  }) => {
    const messages = [
      {
        role: 'system',
        content: `You are an expert trading insights analyst for an AI-powered trading system.
        Your job is to analyze recent trading performance and market conditions to generate actionable insights.
        Provide 3-5 specific insights that could help improve trading performance, along with recommendations.`
      },
      {
        role: 'user',
        content: `Please analyze the following trading data and generate strategic insights:
        
        ${JSON.stringify(data, null, 2)}
        
        Provide 3-5 specific insights about the trading performance and market conditions, along with actionable recommendations.`
      }
    ];
    
    try {
      const response = await openRouterClient.createChatCompletion(messages, {
        temperature: 0.4,
        max_tokens: 800
      });
      
      const responseText = response.choices[0]?.message?.content || '';
      
      // Parse insights from the response (basic parsing)
      const insightsArray = responseText.split(/\n\n|\r\n\r\n/)
        .filter((section: string) => section.trim().length > 0)
        .map((section: string) => {
          return {
            description: section.trim(),
            priority: section.toLowerCase().includes('urgent') || section.toLowerCase().includes('critical')
              ? 'high'
              : section.toLowerCase().includes('important')
                ? 'medium'
                : 'low'
          };
        });
      
      const creds = getOpenRouterCredentials();
      return {
        insights: insightsArray,
        raw_analysis: responseText,
        model_used: creds?.model || 'unknown'
      };
    } catch (error) {
      console.error('Error generating insights with OpenRouter:', error);
      return {
        insights: [{
          description: 'ERROR: Unable to generate trading insights due to service disruption.',
          priority: 'low'
        }],
        raw_analysis: 'Service error occurred',
        model_used: 'fallback',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
};

export default openRouterClient;
