/*
 * Backtesting engine for the Rust trading engine
 */

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

use crate::market_data::{CandleData, OrderBookData, TimeFrame, TradeData, TradeSide};

/// Errors that can occur during backtesting
#[derive(Error, Debug)]
pub enum BacktestError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Simulation error: {0}")]
    SimulationError(String),
}

/// The result of a backtest
pub type BacktestResult = Result<BacktestStats, BacktestError>;

/// Stats from a completed backtest
#[derive(Debug, Clone)]
pub struct BacktestStats {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub initial_balance: Decimal,
    pub final_balance: Decimal,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_profit: Decimal,
    pub total_loss: Decimal,
    pub max_drawdown: Decimal,
    pub max_drawdown_pct: Decimal,
    pub sharpe_ratio: Option<Decimal>,
    pub profit_factor: Option<Decimal>,
    pub win_rate: Decimal,
    pub avg_win: Option<Decimal>,
    pub avg_loss: Option<Decimal>,
    pub largest_win: Option<Decimal>,
    pub largest_loss: Option<Decimal>,
}

impl BacktestStats {
    /// Create a new empty stats object
    pub fn new(initial_balance: Decimal) -> Self {
        Self {
            start_time: Utc::now(),
            end_time: Utc::now(),
            initial_balance,
            final_balance: initial_balance,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            total_profit: Decimal::ZERO,
            total_loss: Decimal::ZERO,
            max_drawdown: Decimal::ZERO,
            max_drawdown_pct: Decimal::ZERO,
            sharpe_ratio: None,
            profit_factor: None,
            win_rate: Decimal::ZERO,
            avg_win: None,
            avg_loss: None,
            largest_win: None,
            largest_loss: None,
        }
    }
    
    /// Update stats with a completed trade
    pub fn update_with_trade(&mut self, profit_loss: Decimal) {
        self.total_trades += 1;
        
        if profit_loss > Decimal::ZERO {
            self.winning_trades += 1;
            self.total_profit += profit_loss;
            
            if let Some(largest_win) = self.largest_win {
                if profit_loss > largest_win {
                    self.largest_win = Some(profit_loss);
                }
            } else {
                self.largest_win = Some(profit_loss);
            }
        } else if profit_loss < Decimal::ZERO {
            self.losing_trades += 1;
            self.total_loss += profit_loss.abs();
            
            if let Some(largest_loss) = self.largest_loss {
                if profit_loss.abs() > largest_loss {
                    self.largest_loss = Some(profit_loss.abs());
                }
            } else {
                self.largest_loss = Some(profit_loss.abs());
            }
        }
        
        // Update average win/loss
        if self.winning_trades > 0 {
            self.avg_win = Some(self.total_profit / Decimal::from(self.winning_trades));
        }
        
        if self.losing_trades > 0 {
            self.avg_loss = Some(self.total_loss / Decimal::from(self.losing_trades));
        }
        
        // Update win rate
        if self.total_trades > 0 {
            self.win_rate = Decimal::from(self.winning_trades) / Decimal::from(self.total_trades) * dec!(100);
        }
        
        // Update profit factor
        if self.total_loss > Decimal::ZERO {
            self.profit_factor = Some(self.total_profit / self.total_loss);
        }
    }
    
    /// Finalize the statistics after a completed backtest
    pub fn finalize(&mut self, end_time: DateTime<Utc>, final_balance: Decimal, max_drawdown: Decimal, sharpe_ratio: Option<Decimal>) {
        self.end_time = end_time;
        self.final_balance = final_balance;
        self.max_drawdown = max_drawdown;
        
        if self.initial_balance > Decimal::ZERO {
            self.max_drawdown_pct = (max_drawdown / self.initial_balance) * dec!(100);
        }
        
        self.sharpe_ratio = sharpe_ratio;
    }
}

/// Order information for backtesting
#[derive(Debug, Clone)]
pub struct BacktestOrder {
    pub id: String,
    pub symbol: String,
    pub side: TradeSide,
    pub order_type: BacktestOrderType,
    pub price: Option<Decimal>,
    pub amount: Decimal,
    pub status: BacktestOrderStatus,
    pub created_at: DateTime<Utc>,
    pub executed_at: Option<DateTime<Utc>>,
    pub executed_price: Option<Decimal>,
    pub stop_price: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub stop_loss: Option<Decimal>,
}

/// Types of orders supported in backtesting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacktestOrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
}

/// Order status for backtesting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacktestOrderStatus {
    Created,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
}

/// Mode for the backtesting engine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacktestMode {
    Candles,    // Use OHLCV candles
    Trades,     // Use individual trades
    OrderBook,  // Use order book snapshots
}

/// Position information for backtesting
#[derive(Debug, Clone)]
pub struct BacktestPosition {
    pub symbol: String,
    pub amount: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl BacktestPosition {
    /// Create a new position
    pub fn new(symbol: String, amount: Decimal, price: Decimal, time: DateTime<Utc>) -> Self {
        Self {
            symbol,
            amount,
            entry_price: price,
            current_price: price,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            opened_at: time,
            updated_at: time,
        }
    }
    
    /// Update the position with current price
    pub fn update_price(&mut self, price: Decimal, time: DateTime<Utc>) {
        self.current_price = price;
        self.updated_at = time;
        
        // Calculate unrealized PnL
        let direction = if self.amount > Decimal::ZERO { Decimal::ONE } else { Decimal::NEGATIVE_ONE };
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.amount.abs() * direction;
    }
    
    /// Close part or all of the position
    pub fn close(&mut self, amount: Decimal, price: Decimal, time: DateTime<Utc>) -> Decimal {
        if amount.abs() > self.amount.abs() {
            return Decimal::ZERO; // Cannot close more than we have
        }
        
        let direction = if self.amount > Decimal::ZERO { Decimal::ONE } else { Decimal::NEGATIVE_ONE };
        let pnl = (price - self.entry_price) * amount.abs() * direction;
        
        // Update the position
        self.amount -= amount;
        self.realized_pnl += pnl;
        self.updated_at = time;
        
        // If position is closed, reset unrealized PnL
        if self.amount.is_zero() {
            self.unrealized_pnl = Decimal::ZERO;
        } else {
            // Recalculate unrealized PnL for remaining position
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.amount.abs() * direction;
        }
        
        pnl
    }
    
    /// Get the total PnL (realized + unrealized)
    pub fn total_pnl(&self) -> Decimal {
        self.realized_pnl + self.unrealized_pnl
    }
}

/// Configuration for the backtest engine
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_balance: Decimal,
    pub symbols: Vec<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub mode: BacktestMode,
    pub commission_rate: Decimal,
    pub slippage: Decimal,
    pub enable_fractional_sizing: bool,
}

/// The backtesting engine
pub struct BacktestEngine {
    config: BacktestConfig,
    current_time: DateTime<Utc>,
    balance: Decimal,
    equity: Decimal,
    peak_equity: Decimal,
    positions: HashMap<String, BacktestPosition>,
    open_orders: VecDeque<BacktestOrder>,
    filled_orders: Vec<BacktestOrder>,
    equity_curve: Vec<(DateTime<Utc>, Decimal)>,
    stats: BacktestStats,
}

impl BacktestEngine {
    /// Create a new backtesting engine
    pub fn new(config: BacktestConfig) -> Self {
        let initial_balance = config.initial_balance;
        Self {
            current_time: config.start_time,
            balance: initial_balance,
            equity: initial_balance,
            peak_equity: initial_balance,
            positions: HashMap::new(),
            open_orders: VecDeque::new(),
            filled_orders: Vec::new(),
            equity_curve: vec![(config.start_time, initial_balance)],
            stats: BacktestStats::new(initial_balance),
            config,
        }
    }
    
    /// Process a candle update
    pub fn process_candle(&mut self, candle: &CandleData) -> Result<(), BacktestError> {
        if self.config.mode != BacktestMode::Candles {
            return Err(BacktestError::SimulationError(
                "Cannot process candles in non-candle mode".to_string()
            ));
        }
        
        // Update current time
        self.current_time = candle.timestamp;
        
        // Process orders that could be triggered by this candle
        self.process_orders_with_candle(candle)?;
        
        // Update positions with the closing price
        if let Some(position) = self.positions.get_mut(&candle.symbol) {
            position.update_price(candle.close, self.current_time);
        }
        
        // Update equity and check for new equity peak
        self.update_equity();
        
        // Add to equity curve
        self.equity_curve.push((self.current_time, self.equity));
        
        Ok(())
    }
    
    /// Process orders that could be triggered by a candle
    fn process_orders_with_candle(&mut self, candle: &CandleData) -> Result<(), BacktestError> {
        let mut orders_to_process = Vec::new();
        
        // Select orders for this symbol that could be triggered
        while let Some(order) = self.open_orders.front() {
            if order.symbol != candle.symbol {
                break;
            }
            
            // Remove from open orders for processing
            orders_to_process.push(self.open_orders.pop_front().unwrap());
        }
        
        // Process each order
        for order in orders_to_process {
            let executed = match order.order_type {
                BacktestOrderType::Market => {
                    // Market orders execute at current price with slippage
                    let execution_price = match order.side {
                        TradeSide::Buy => candle.close * (Decimal::ONE + self.config.slippage),
                        TradeSide::Sell => candle.close * (Decimal::ONE - self.config.slippage),
                    };
                    Some(execution_price)
                },
                BacktestOrderType::Limit => {
                    // Limit orders execute if price crosses the limit price
                    match (order.side, order.price) {
                        (TradeSide::Buy, Some(price)) if candle.low <= price => Some(price),
                        (TradeSide::Sell, Some(price)) if candle.high >= price => Some(price),
                        _ => None,
                    }
                },
                BacktestOrderType::StopMarket => {
                    // Stop orders execute if price crosses the stop price
                    match (order.side, order.stop_price) {
                        (TradeSide::Buy, Some(price)) if candle.high >= price => {
                            // Buy stop triggers, execute at stop price with slippage
                            Some(price * (Decimal::ONE + self.config.slippage))
                        },
                        (TradeSide::Sell, Some(price)) if candle.low <= price => {
                            // Sell stop triggers, execute at stop price with slippage
                            Some(price * (Decimal::ONE - self.config.slippage))
                        },
                        _ => None,
                    }
                },
                BacktestOrderType::StopLimit => {
                    // Stop limit orders: first the stop price must be reached,
                    // then the limit order is placed
                    match (order.side, order.stop_price, order.price) {
                        (TradeSide::Buy, Some(stop), Some(limit)) if candle.high >= stop && candle.low <= limit => {
                            Some(limit)
                        },
                        (TradeSide::Sell, Some(stop), Some(limit)) if candle.low <= stop && candle.high >= limit => {
                            Some(limit)
                        },
                        _ => None,
                    }
                },
            };
            
            if let Some(execution_price) = executed {
                self.execute_order(order, execution_price)?;
            } else {
                // Put the order back if not executed
                self.open_orders.push_back(order);
            }
        }
        
        Ok(())
    }
    
    /// Execute an order at the given price
    fn execute_order(&mut self, mut order: BacktestOrder, price: Decimal) -> Result<(), BacktestError> {
        let commission = price * order.amount.abs() * self.config.commission_rate;
        
        // Update order info
        order.executed_price = Some(price);
        order.executed_at = Some(self.current_time);
        order.status = BacktestOrderStatus::Filled;
        
        // Adjust balance and handle position
        match order.side {
            TradeSide::Buy => {
                // Subtract the cost plus commission from balance
                let cost = price * order.amount + commission;
                if cost > self.balance {
                    return Err(BacktestError::InsufficientData(
                        "Insufficient balance for buy order".to_string()
                    ));
                }
                self.balance -= cost;
                
                // Update or create position
                if let Some(position) = self.positions.get_mut(&order.symbol) {
                    // If position exists, update it
                    if position.amount < Decimal::ZERO {
                        // If it's a short position, reduce or close it
                        let realized_pnl = position.close(order.amount, price, self.current_time);
                        self.stats.update_with_trade(realized_pnl);
                        
                        if position.amount.is_zero() {
                            // Position fully closed
                            self.positions.remove(&order.symbol);
                        }
                    } else {
                        // If it's a long position, increase it (dollar-cost averaging)
                        let total_amount = position.amount + order.amount;
                        let total_cost = (position.entry_price * position.amount) + (price * order.amount);
                        position.entry_price = total_cost / total_amount;
                        position.amount = total_amount;
                        position.updated_at = self.current_time;
                    }
                } else {
                    // Create new long position
                    let position = BacktestPosition::new(
                        order.symbol.clone(),
                        order.amount,
                        price,
                        self.current_time
                    );
                    self.positions.insert(order.symbol.clone(), position);
                }
            },
            TradeSide::Sell => {
                // We first need to check if we have a position to sell
                let sell_amount = order.amount;
                
                if let Some(position) = self.positions.get_mut(&order.symbol) {
                    if position.amount > Decimal::ZERO {
                        // Close long position (partially or fully)
                        let realized_pnl = position.close(-sell_amount, price, self.current_time);
                        self.stats.update_with_trade(realized_pnl);
                        
                        // Add the proceeds minus commission to balance
                        let proceeds = price * sell_amount - commission;
                        self.balance += proceeds;
                        
                        if position.amount.is_zero() {
                            // Position fully closed
                            self.positions.remove(&order.symbol);
                        }
                    } else {
                        // Increase short position
                        let total_amount = position.amount - sell_amount;
                        let total_cost = (position.entry_price * position.amount.abs()) + (price * sell_amount);
                        position.entry_price = total_cost / total_amount.abs();
                        position.amount = total_amount;
                        position.updated_at = self.current_time;
                        
                        // Add the proceeds minus commission to balance
                        let proceeds = price * sell_amount - commission;
                        self.balance += proceeds;
                    }
                } else {
                    // Create new short position
                    let position = BacktestPosition::new(
                        order.symbol.clone(),
                        -sell_amount,
                        price,
                        self.current_time
                    );
                    self.positions.insert(order.symbol.clone(), position);
                    
                    // Add the proceeds minus commission to balance
                    let proceeds = price * sell_amount - commission;
                    self.balance += proceeds;
                }
            }
        }
        
        // Add to filled orders
        self.filled_orders.push(order);
        
        // Update equity
        self.update_equity();
        
        Ok(())
    }
    
    /// Submit a new order to the backtest engine
    pub fn submit_order(&mut self, order: BacktestOrder) -> Result<String, BacktestError> {
        // Validate the order
        match order.order_type {
            BacktestOrderType::Limit | BacktestOrderType::StopLimit if order.price.is_none() => {
                return Err(BacktestError::InvalidParameter(
                    "Limit orders require a price".to_string()
                ));
            },
            BacktestOrderType::StopMarket | BacktestOrderType::StopLimit if order.stop_price.is_none() => {
                return Err(BacktestError::InvalidParameter(
                    "Stop orders require a stop price".to_string()
                ));
            },
            _ => {}
        }
        
        // Add to open orders
        let order_id = order.id.clone();
        self.open_orders.push_back(order);
        
        Ok(order_id)
    }
    
    /// Cancel an open order
    pub fn cancel_order(&mut self, order_id: &str) -> Result<(), BacktestError> {
        let index = self.open_orders.iter().position(|o| o.id == order_id);
        
        if let Some(idx) = index {
            let mut order = self.open_orders.remove(idx).unwrap();
            order.status = BacktestOrderStatus::Canceled;
            self.filled_orders.push(order);
            Ok(())
        } else {
            Err(BacktestError::InvalidParameter(
                format!("Order with ID {} not found", order_id)
            ))
        }
    }
    
    /// Update equity calculation based on current positions
    fn update_equity(&mut self) {
        let mut total_equity = self.balance;
        
        // Add value of all positions
        for (_, position) in &self.positions {
            total_equity += position.unrealized_pnl;
        }
        
        self.equity = total_equity;
        
        // Update peak equity for drawdown calculations
        if self.equity > self.peak_equity {
            self.peak_equity = self.equity;
        }
    }
    
    /// Calculate the current drawdown
    fn calculate_drawdown(&self) -> Decimal {
        if self.peak_equity > self.equity {
            self.peak_equity - self.equity
        } else {
            Decimal::ZERO
        }
    }
    
    /// Calculate the Sharpe ratio based on equity curve
    fn calculate_sharpe_ratio(&self) -> Option<Decimal> {
        if self.equity_curve.len() < 2 {
            return None;
        }
        
        // Calculate daily returns
        let mut returns = Vec::with_capacity(self.equity_curve.len() - 1);
        for i in 1..self.equity_curve.len() {
            let prev = self.equity_curve[i - 1].1;
            let current = self.equity_curve[i].1;
            
            if prev > Decimal::ZERO {
                returns.push((current - prev) / prev);
            }
        }
        
        if returns.is_empty() {
            return None;
        }
        
        // Calculate average return
        let sum = returns.iter().fold(Decimal::ZERO, |acc, r| acc + r);
        let avg_return = sum / Decimal::from(returns.len());
        
        // Calculate standard deviation of returns
        let sum_sq_diff = returns.iter()
            .fold(Decimal::ZERO, |acc, r| acc + (*r - avg_return) * (*r - avg_return));
        
        let std_dev = if returns.len() > 1 {
            (sum_sq_diff / Decimal::from(returns.len() - 1)).sqrt().unwrap_or(Decimal::ONE)
        } else {
            return None;
        };
        
        if std_dev == Decimal::ZERO {
            return None;
        }
        
        // Sharpe ratio = (average return - risk free rate) / standard deviation
        // We use 0 as the risk-free rate for simplicity
        Some(avg_return / std_dev * Decimal::from(252_u32).sqrt().unwrap())
    }
    
    /// Run the backtest
    pub fn run(&mut self) -> BacktestResult {
        // This would be the main entry point for running a backtest
        // In a real implementation, you would feed in data and process it
        
        // For now, we just calculate the final stats
        let max_drawdown = self.calculate_drawdown();
        let sharpe_ratio = self.calculate_sharpe_ratio();
        
        // Finalize stats
        self.stats.finalize(
            self.current_time,
            self.equity,
            max_drawdown,
            sharpe_ratio
        );
        
        Ok(self.stats.clone())
    }
    
    /// Get the current statistics
    pub fn get_stats(&self) -> BacktestStats {
        self.stats.clone()
    }
    
    /// Get the equity curve
    pub fn get_equity_curve(&self) -> &[(DateTime<Utc>, Decimal)] {
        &self.equity_curve
    }
    
    /// Get the current balance
    pub fn get_balance(&self) -> Decimal {
        self.balance
    }
    
    /// Get the current equity
    pub fn get_equity(&self) -> Decimal {
        self.equity
    }
    
    /// Get all positions
    pub fn get_positions(&self) -> &HashMap<String, BacktestPosition> {
        &self.positions
    }
    
    /// Get all open orders
    pub fn get_open_orders(&self) -> &VecDeque<BacktestOrder> {
        &self.open_orders
    }
    
    /// Get all filled orders
    pub fn get_filled_orders(&self) -> &[BacktestOrder] {
        &self.filled_orders
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use uuid::Uuid;
    
    fn create_test_config() -> BacktestConfig {
        BacktestConfig {
            initial_balance: dec!(10000),
            symbols: vec!["BTCUSDT".to_string()],
            start_time: Utc::now(),
            end_time: Utc::now() + Duration::days(30),
            mode: BacktestMode::Candles,
            commission_rate: dec!(0.001),
            slippage: dec!(0.0005),
            enable_fractional_sizing: true,
        }
    }
    
    fn create_test_candle(symbol: &str, close_price: Decimal) -> CandleData {
        CandleData::new(
            symbol.to_string(),
            "test_exchange".to_string(),
            Utc::now(),
            close_price,
            close_price * dec!(1.01),
            close_price * dec!(0.99),
            close_price,
            dec!(100),
            TimeFrame::Hour1,
        )
    }
    
    fn create_test_order(symbol: &str, side: TradeSide, amount: Decimal) -> BacktestOrder {
        BacktestOrder {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            side,
            order_type: BacktestOrderType::Market,
            price: None,
            amount,
            status: BacktestOrderStatus::Created,
            created_at: Utc::now(),
            executed_at: None,
            executed_price: None,
            stop_price: None,
            take_profit: None,
            stop_loss: None,
        }
    }
    
    #[test]
    fn test_order_execution() {
        let config = create_test_config();
        let mut engine = BacktestEngine::new(config);
        
        // Create a buy order
        let order = create_test_order("BTCUSDT", TradeSide::Buy, dec!(1));
        let order_id = engine.submit_order(order).unwrap();
        
        // Process a candle to trigger the order
        let candle = create_test_candle("BTCUSDT", dec!(10000));
        engine.process_candle(&candle).unwrap();
        
        // Check that the order was filled
        assert_eq!(engine.open_orders.len(), 0);
        assert_eq!(engine.filled_orders.len(), 1);
        assert_eq!(engine.filled_orders[0].status, BacktestOrderStatus::Filled);
        
        // Check that we have a position
        assert_eq!(engine.positions.len(), 1);
        let position = engine.positions.get("BTCUSDT").unwrap();
        assert_eq!(position.amount, dec!(1));
        assert_eq!(position.entry_price, dec!(10000));
        
        // Check that balance was updated (purchase + commission)
        let expected_balance = dec!(10000) - dec!(10000) * dec!(1) - dec!(10000) * dec!(1) * dec!(0.001);
        assert_eq!(engine.balance, expected_balance);
    }
    
    #[test]
    fn test_position_closure() {
        let config = create_test_config();
        let mut engine = BacktestEngine::new(config);
        
        // Create a buy order
        let buy_order = create_test_order("BTCUSDT", TradeSide::Buy, dec!(1));
        engine.submit_order(buy_order).unwrap();
        
        // Process a candle to trigger the buy order
        let buy_candle = create_test_candle("BTCUSDT", dec!(10000));
        engine.process_candle(&buy_candle).unwrap();
        
        // Create a sell order to close the position
        let sell_order = create_test_order("BTCUSDT", TradeSide::Sell, dec!(1));
        engine.submit_order(sell_order).unwrap();
        
        // Process a candle with higher price to trigger the sell order
        let sell_candle = create_test_candle("BTCUSDT", dec!(11000));
        engine.process_candle(&sell_candle).unwrap();
        
        // Check that both orders were filled
        assert_eq!(engine.filled_orders.len(), 2);
        
        // Check that position is closed
        assert_eq!(engine.positions.len(), 0);
        
        // Check that we made a profit (minus commissions)
        let buy_cost = dec!(10000) * dec!(1) + dec!(10000) * dec!(1) * dec!(0.001);
        let sell_proceeds = dec!(11000) * dec!(1) - dec!(11000) * dec!(1) * dec!(0.001);
        let expected_balance = dec!(10000) - buy_cost + sell_proceeds;
        
        assert!(engine.balance > dec!(10000)); // We should have made a profit
        assert_eq!(engine.balance, expected_balance);
        
        // Check that stats were updated
        assert_eq!(engine.stats.total_trades, 1);
        assert_eq!(engine.stats.winning_trades, 1);
    }
} 