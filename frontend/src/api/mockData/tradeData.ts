import { Trade } from '../../components/dashboard/TradeStatistics';
import { BacktestParams } from '../../components/dashboard/BacktestingInterface';

// Generate mock trade data for backtesting
export const generateMockTrades = (params: BacktestParams): Trade[] => {
  const { symbol, strategy, startDate, endDate, initialCapital } = params;
  
  // Parse dates
  const startDateObj = new Date(startDate);
  const endDateObj = new Date(endDate);
  
  // Calculate number of days in the backtest period
  const daysDiff = Math.floor((endDateObj.getTime() - startDateObj.getTime()) / (1000 * 60 * 60 * 24));
  
  // Generate a random number of trades based on the backtest period
  // Assume 1-3 trades per week on average
  const numTrades = Math.floor(daysDiff / 7 * (1 + Math.random() * 2));
  
  // Generate trades
  const trades: Trade[] = [];
  
  // Set win rate based on strategy
  let winRate = 0.55; // Default win rate
  
  switch (strategy) {
    case 'MACD Crossover':
      winRate = 0.58;
      break;
    case 'RSI Oscillator':
      winRate = 0.62;
      break;
    case 'Bollinger Breakout':
      winRate = 0.53;
      break;
    case 'Moving Average Crossover':
      winRate = 0.56;
      break;
    case 'Sentiment-Based':
      winRate = 0.60;
      break;
  }
  
  // Generate random trade dates within the backtest period
  const tradeDates: Date[] = [];
  for (let i = 0; i < numTrades * 2; i++) { // *2 because we need entry and exit dates
    const randomDayOffset = Math.floor(Math.random() * daysDiff);
    const tradeDate = new Date(startDateObj);
    tradeDate.setDate(startDateObj.getDate() + randomDayOffset);
    
    // Skip weekends
    if (tradeDate.getDay() === 0 || tradeDate.getDay() === 6) {
      i--;
      continue;
    }
    
    tradeDates.push(tradeDate);
  }
  
  // Sort dates
  tradeDates.sort((a, b) => a.getTime() - b.getTime());
  
  // Generate trades
  for (let i = 0; i < numTrades; i++) {
    const entryDate = tradeDates[i * 2];
    const exitDate = tradeDates[i * 2 + 1];
    
    // If entry date is after exit date, swap them
    if (entryDate > exitDate) {
      [tradeDates[i * 2], tradeDates[i * 2 + 1]] = [tradeDates[i * 2 + 1], tradeDates[i * 2]];
    }
    
    // Calculate trade duration in days
    const duration = Math.floor((exitDate.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24));
    
    // Generate random prices based on the symbol
    let basePrice = 0;
    let volatility = 0;
    
    switch (symbol) {
      case 'BTC':
        basePrice = 40000;
        volatility = 0.05;
        break;
      case 'ETH':
        basePrice = 2500;
        volatility = 0.06;
        break;
      case 'AAPL':
        basePrice = 180;
        volatility = 0.02;
        break;
      case 'MSFT':
        basePrice = 350;
        volatility = 0.02;
        break;
      case 'AMZN':
        basePrice = 3500;
        volatility = 0.03;
        break;
      case 'GOOGL':
        basePrice = 2800;
        volatility = 0.025;
        break;
      case 'TSLA':
        basePrice = 800;
        volatility = 0.04;
        break;
      default:
        basePrice = 100;
        volatility = 0.03;
    }
    
    // Generate entry price with some randomness
    const entryPrice = basePrice * (1 + (Math.random() * 0.2 - 0.1));
    
    // Determine if this is a winning trade based on win rate
    const isWinningTrade = Math.random() < winRate;
    
    // Generate exit price based on whether it's a winning trade
    let exitPrice;
    let type: 'buy' | 'sell';
    
    if (Math.random() < 0.5) {
      // Long trade
      type = 'buy';
      if (isWinningTrade) {
        // Winning long trade: exit price > entry price
        exitPrice = entryPrice * (1 + Math.random() * volatility * 2);
      } else {
        // Losing long trade: exit price < entry price
        exitPrice = entryPrice * (1 - Math.random() * volatility);
      }
    } else {
      // Short trade
      type = 'sell';
      if (isWinningTrade) {
        // Winning short trade: exit price < entry price
        exitPrice = entryPrice * (1 - Math.random() * volatility * 2);
      } else {
        // Losing short trade: exit price > entry price
        exitPrice = entryPrice * (1 + Math.random() * volatility);
      }
    }
    
    // Calculate quantity based on initial capital and position sizing
    // Assume using 5-10% of capital per trade
    const positionSize = (0.05 + Math.random() * 0.05) * initialCapital;
    const quantity = Math.floor(positionSize / entryPrice);
    
    // Calculate P&L
    let pnl = 0;
    if (type === 'buy') {
      pnl = (exitPrice - entryPrice) * quantity;
    } else {
      pnl = (entryPrice - exitPrice) * quantity;
    }
    
    // Calculate P&L percent
    const pnlPercent = (pnl / (entryPrice * quantity)) * 100;
    
    trades.push({
      id: `trade-${i + 1}`,
      symbol,
      type,
      entryDate: entryDate.toISOString().split('T')[0],
      exitDate: exitDate.toISOString().split('T')[0],
      entryPrice,
      exitPrice,
      quantity,
      pnl,
      pnlPercent,
      duration,
      strategy
    });
  }
  
  return trades;
};
