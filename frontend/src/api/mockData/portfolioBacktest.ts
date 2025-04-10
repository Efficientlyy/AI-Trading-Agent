import { BacktestResult, BacktestMetrics } from '../../components/dashboard/BacktestingInterface';

export interface PortfolioBacktestParams {
  assets: string[];
  weights: number[];
  startDate: string;
  endDate: string;
  initialCapital: number;
  rebalancingPeriod: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly' | 'none';
  strategies: Record<string, string>;
  riskManagement: {
    maxDrawdown: number;
    stopLoss: number;
    trailingStop: boolean;
    correlationThreshold: number;
  };
}

export interface AssetPerformance {
  symbol: string;
  weight: number;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  contribution: number;
}

export interface PortfolioBacktestResult {
  equityCurve: BacktestResult[];
  metrics: BacktestMetrics;
  assetPerformance: AssetPerformance[];
  correlationMatrix: Record<string, Record<string, number>>;
  drawdownEvents: {
    startDate: string;
    endDate: string;
    depth: number;
    duration: number;
    recovery: number;
  }[];
}

// Generate a correlation matrix for the assets
const generateCorrelationMatrix = (assets: string[]): Record<string, Record<string, number>> => {
  const matrix: Record<string, Record<string, number>> = {};
  
  assets.forEach(asset1 => {
    matrix[asset1] = {};
    
    assets.forEach(asset2 => {
      if (asset1 === asset2) {
        matrix[asset1][asset2] = 1; // Perfect correlation with self
      } else {
        // Generate random correlation between -0.8 and 0.9
        // Biased towards positive correlations as is common in markets
        const correlation = Math.round((Math.random() * 1.7 - 0.8) * 100) / 100;
        matrix[asset1][asset2] = correlation;
        
        // Ensure symmetry in the correlation matrix
        if (matrix[asset2] && matrix[asset2][asset1] === undefined) {
          matrix[asset2][asset1] = correlation;
        }
      }
    });
  });
  
  return matrix;
};

// Generate mock portfolio backtest results
export const runPortfolioBacktest = (params: PortfolioBacktestParams): PortfolioBacktestResult => {
  const { assets, weights, startDate, endDate, initialCapital, rebalancingPeriod } = params;
  
  // Generate equity curve
  const startDateObj = new Date(startDate);
  const endDateObj = new Date(endDate);
  const equityCurve: BacktestResult[] = [];
  
  let currentDate = new Date(startDateObj);
  let equity = initialCapital;
  let benchmark = initialCapital;
  let highWatermark = equity;
  
  // Generate daily returns
  while (currentDate <= endDateObj) {
    if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) { // Skip weekends
      // Portfolio return is weighted sum of asset returns
      let portfolioReturn = 0;
      
      assets.forEach((asset, index) => {
        // Random daily return between -2% and 2%, with some assets more volatile than others
        const volatilityFactor = asset.includes('BTC') || asset.includes('ETH') ? 2 : 1;
        const assetReturn = (Math.random() * 0.04 - 0.02) * volatilityFactor;
        portfolioReturn += assetReturn * weights[index];
      });
      
      // Apply risk management - limit drawdowns
      if (params.riskManagement.maxDrawdown > 0) {
        const currentDrawdown = ((highWatermark - equity) / highWatermark) * 100;
        if (currentDrawdown >= params.riskManagement.maxDrawdown) {
          portfolioReturn = Math.max(portfolioReturn, 0); // Prevent further losses
        }
      }
      
      // Benchmark return (S&P 500 like)
      const benchmarkReturn = (Math.random() * 0.02 - 0.008); // Slightly positive bias
      
      equity = equity * (1 + portfolioReturn);
      benchmark = benchmark * (1 + benchmarkReturn);
      
      // Update high watermark
      highWatermark = Math.max(highWatermark, equity);
      
      // Calculate drawdown
      const drawdown = ((highWatermark - equity) / highWatermark) * 100;
      
      equityCurve.push({
        date: currentDate.toISOString().split('T')[0],
        equity: Math.round(equity * 100) / 100,
        benchmark: Math.round(benchmark * 100) / 100,
        drawdown: Math.round(drawdown * 100) / 100
      });
      
      // Simulate rebalancing
      if (rebalancingPeriod !== 'none') {
        let shouldRebalance = false;
        
        switch (rebalancingPeriod) {
          case 'daily':
            shouldRebalance = true;
            break;
          case 'weekly':
            shouldRebalance = currentDate.getDay() === 5; // Friday
            break;
          case 'monthly':
            shouldRebalance = currentDate.getDate() === 28; // End of month (simplified)
            break;
          case 'quarterly':
            shouldRebalance = 
              currentDate.getDate() === 28 && 
              [2, 5, 8, 11].includes(currentDate.getMonth()); // Mar, Jun, Sep, Dec
            break;
          case 'yearly':
            shouldRebalance = 
              currentDate.getDate() === 28 && 
              currentDate.getMonth() === 11; // December
            break;
        }
        
        if (shouldRebalance) {
          // Rebalancing would happen here in a real implementation
          // For the mock, we'll just add a small performance boost
          equity *= 1.001; // 0.1% rebalancing bonus
        }
      }
    }
    
    // Move to next day
    currentDate.setDate(currentDate.getDate() + 1);
  }
  
  // Calculate metrics
  const initialValue = initialCapital;
  const finalValue = equityCurve[equityCurve.length - 1].equity;
  const totalReturn = ((finalValue - initialValue) / initialValue) * 100;
  
  const daysDiff = Math.floor((endDateObj.getTime() - startDateObj.getTime()) / (1000 * 60 * 60 * 24));
  const years = daysDiff / 365;
  
  const annualizedReturn = (Math.pow((finalValue / initialValue), (1 / years)) - 1) * 100;
  
  // Calculate max drawdown from the equity curve
  const maxDrawdown = Math.max(...equityCurve.map(point => point.drawdown || 0));
  
  // Generate drawdown events
  const drawdownEvents = [];
  let inDrawdown = false;
  let drawdownStart = '';
  let drawdownPeak = 0;
  let drawdownTrough = 0;
  
  for (let i = 1; i < equityCurve.length; i++) {
    const currentEquity = equityCurve[i].equity;
    const prevEquity = equityCurve[i-1].equity;
    
    if (!inDrawdown && currentEquity < prevEquity) {
      // Start of drawdown
      inDrawdown = true;
      drawdownStart = equityCurve[i].date;
      drawdownPeak = prevEquity;
      drawdownTrough = currentEquity;
    } else if (inDrawdown) {
      if (currentEquity < drawdownTrough) {
        // Deeper drawdown
        drawdownTrough = currentEquity;
      } else if (currentEquity >= drawdownPeak) {
        // End of drawdown
        const depth = ((drawdownPeak - drawdownTrough) / drawdownPeak) * 100;
        const startDate = new Date(drawdownStart);
        const endDate = new Date(equityCurve[i].date);
        const duration = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
        
        if (depth > 5) { // Only record significant drawdowns
          drawdownEvents.push({
            startDate: drawdownStart,
            endDate: equityCurve[i].date,
            depth: Math.round(depth * 100) / 100,
            duration,
            recovery: Math.floor(duration / 2) // Simplified recovery calculation
          });
        }
        
        inDrawdown = false;
      }
    }
  }
  
  // Generate asset performance data
  const assetPerformance: AssetPerformance[] = assets.map((asset, index) => {
    // Generate random performance metrics for each asset
    const assetReturn = totalReturn * (0.7 + Math.random() * 0.6); // Between 70% and 130% of portfolio return
    const assetSharpe = (Math.random() * 1 + 1.5); // Between 1.5 and 2.5
    const assetDrawdown = maxDrawdown * (0.8 + Math.random() * 0.4); // Between 80% and 120% of portfolio drawdown
    
    return {
      symbol: asset,
      weight: weights[index],
      totalReturn: Math.round(assetReturn * 100) / 100,
      sharpeRatio: Math.round(assetSharpe * 100) / 100,
      maxDrawdown: Math.round(assetDrawdown * 100) / 100,
      contribution: Math.round((assetReturn * weights[index]) * 100) / 100
    };
  });
  
  // Generate correlation matrix
  const correlationMatrix = generateCorrelationMatrix(assets);
  
  // Return the complete result
  return {
    equityCurve,
    metrics: {
      totalReturn: Math.round(totalReturn * 100) / 100,
      annualizedReturn: Math.round(annualizedReturn * 100) / 100,
      sharpeRatio: Math.round((annualizedReturn / 12) * 100) / 100, // Assuming 12% volatility
      maxDrawdown: Math.round(maxDrawdown * 100) / 100,
      winRate: Math.round(Math.random() * 15 + 55), // Random win rate between 55% and 70%
      profitFactor: Math.round((Math.random() * 0.8 + 1.5) * 100) / 100, // Random profit factor between 1.5 and 2.3
      averageWin: Math.round(Math.random() * 200 + 100) / 100, // Random average win between $1 and $3
      averageLoss: Math.round(Math.random() * 100 + 50) / 100, // Random average loss between $0.5 and $1.5
      tradesPerMonth: Math.round(Math.random() * 50 + 30), // Random trades per month between 30 and 80
      totalTrades: Math.round(Math.random() * 300 + 200) // Random total trades between 200 and 500
    },
    assetPerformance,
    correlationMatrix,
    drawdownEvents
  };
};
