use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use numpy::{PyArray1, PyArray2, IntoPyArray};
use ndarray::{Array1, Array2};

/// Order side enum (Buy or Sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Order status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Created,
    Submitted,
    Partial,
    Filled,
    Canceled,
    Rejected,
}

/// Order struct representing a trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub status: OrderStatus,
    pub fills: Vec<Fill>,
    pub created_at: i64,  // Unix timestamp
}

/// Fill struct representing a partial or complete order execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub quantity: f64,
    pub price: f64,
    pub timestamp: i64,  // Unix timestamp
}

/// Trade struct representing a completed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub trade_id: String,
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: i64,  // Unix timestamp
}

/// Position struct representing a trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub market_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

/// Portfolio struct representing the full trading portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub cash: f64,
    pub total_value: f64,
    pub positions: HashMap<String, Position>,
}

/// PortfolioSnapshot struct for recording portfolio state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSnapshot {
    pub timestamp: i64,  // Unix timestamp
    pub cash: f64,
    pub total_value: f64,
    pub positions: HashMap<String, Position>,
}

/// BacktestResult struct containing all backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub portfolio_history: Vec<PortfolioSnapshot>,
    pub trade_history: Vec<Trade>,
    pub order_history: Vec<Order>,
    pub metrics: PerformanceMetrics,
}

/// PerformanceMetrics struct containing calculated performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: i64,
    pub total_trades: i64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_profit_per_trade: f64,
    pub avg_loss_per_trade: f64,
    pub avg_profit_loss_ratio: f64,
}

/// OHLCV bar struct for price data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCVBar {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// BacktestConfig struct for configuring the backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub slippage: f64,
    pub enable_fractional: bool,
}

/// Calculate execution price based on order type and current bar
fn calculate_execution_price(order: &Order, bar: &OHLCVBar) -> Option<f64> {
    match order.order_type {
        OrderType::Market => Some(bar.open),
        
        OrderType::Limit => {
            match order.side {
                OrderSide::Buy => {
                    if let Some(limit_price) = order.limit_price {
                        if bar.low <= limit_price {
                            Some(limit_price.min(bar.open))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                OrderSide::Sell => {
                    if let Some(limit_price) = order.limit_price {
                        if bar.high >= limit_price {
                            Some(limit_price.max(bar.open))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            }
        },
        
        OrderType::Stop => {
            match order.side {
                OrderSide::Buy => {
                    if let Some(stop_price) = order.stop_price {
                        if bar.high >= stop_price {
                            Some(stop_price.max(bar.open))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                OrderSide::Sell => {
                    if let Some(stop_price) = order.stop_price {
                        if bar.low <= stop_price {
                            Some(stop_price.min(bar.open))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            }
        },
        
        OrderType::StopLimit => {
            // For simplicity, we'll implement this as a stop order that becomes a limit order
            // when triggered
            match order.side {
                OrderSide::Buy => {
                    if let (Some(stop_price), Some(limit_price)) = (order.stop_price, order.limit_price) {
                        if bar.high >= stop_price && bar.low <= limit_price {
                            Some(limit_price.min(bar.open.max(stop_price)))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                OrderSide::Sell => {
                    if let (Some(stop_price), Some(limit_price)) = (order.stop_price, order.limit_price) {
                        if bar.low <= stop_price && bar.high >= limit_price {
                            Some(limit_price.max(bar.open.min(stop_price)))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            }
        }
    }
}

/// Apply transaction costs (commission and slippage) to execution price
fn apply_transaction_costs(
    order: &Order, 
    executed_price: f64, 
    commission_rate: f64, 
    slippage: f64
) -> f64 {
    let slippage_factor = match order.side {
        OrderSide::Buy => 1.0 + slippage,
        OrderSide::Sell => 1.0 - slippage,
    };
    
    // Apply slippage
    let price_with_slippage = executed_price * slippage_factor;
    
    // Note: Commission is typically applied to the total value, not the price
    // We'll handle that in the portfolio update logic
    
    price_with_slippage
}

/// Update portfolio based on a trade
fn update_portfolio_from_trade(portfolio: &mut Portfolio, trade: &Trade) {
    // Update cash
    let trade_value = trade.quantity * trade.price;
    match trade.side {
        OrderSide::Buy => {
            portfolio.cash -= trade_value;
        },
        OrderSide::Sell => {
            portfolio.cash += trade_value;
        }
    }
    
    // Update position
    let position = portfolio.positions.entry(trade.symbol.clone()).or_insert(Position {
        symbol: trade.symbol.clone(),
        quantity: 0.0,
        entry_price: 0.0,
        market_price: trade.price,
        unrealized_pnl: 0.0,
        realized_pnl: 0.0,
    });
    
    // Calculate P&L if reducing position
    if (position.quantity > 0.0 && trade.side == OrderSide::Sell) || 
       (position.quantity < 0.0 && trade.side == OrderSide::Buy) {
        let trade_quantity = trade.quantity.min(position.quantity.abs());
        let trade_pnl = match trade.side {
            OrderSide::Sell => trade_quantity * (trade.price - position.entry_price),
            OrderSide::Buy => trade_quantity * (position.entry_price - trade.price),
        };
        position.realized_pnl += trade_pnl;
    }
    
    // Update position quantity
    match trade.side {
        OrderSide::Buy => {
            position.quantity += trade.quantity;
        },
        OrderSide::Sell => {
            position.quantity -= trade.quantity;
        }
    }
    
    // Update entry price (weighted average)
    if position.quantity != 0.0 {
        // Only update entry price if adding to position
        if (position.quantity > 0.0 && trade.side == OrderSide::Buy) || 
           (position.quantity < 0.0 && trade.side == OrderSide::Sell) {
            let old_value = position.entry_price * (position.quantity - trade.quantity).abs();
            let new_value = trade.price * trade.quantity;
            position.entry_price = (old_value + new_value) / position.quantity.abs();
        }
    } else {
        // If position is flat, reset entry price
        position.entry_price = 0.0;
    }
    
    // Update market price
    position.market_price = trade.price;
    
    // Update unrealized P&L
    position.unrealized_pnl = position.quantity * (position.market_price - position.entry_price);
    
    // Update total portfolio value
    update_portfolio_value(portfolio);
}

/// Update portfolio total value
fn update_portfolio_value(portfolio: &mut Portfolio) {
    let mut total_value = portfolio.cash;
    
    // Add value of all positions
    for (_, position) in portfolio.positions.iter() {
        total_value += position.quantity * position.market_price;
    }
    
    portfolio.total_value = total_value;
}

/// Update position market price and unrealized P&L
fn update_position_market_price(position: &mut Position, market_price: f64) {
    position.market_price = market_price;
    position.unrealized_pnl = position.quantity * (position.market_price - position.entry_price);
}

/// Run a backtest with the given data and configuration
#[pyfunction]
pub fn run_backtest_rs(
    py: Python<'_>,
    data: HashMap<String, Vec<OHLCVBar>>,
    orders: Vec<Order>,
    config: BacktestConfig,
) -> PyResult<Py<PyDict>> {
    // Initialize portfolio
    let mut portfolio = Portfolio {
        cash: config.initial_capital,
        total_value: config.initial_capital,
        positions: HashMap::new(),
    };
    
    // Initialize result containers
    let mut portfolio_history: Vec<PortfolioSnapshot> = Vec::new();
    let mut trade_history: Vec<Trade> = Vec::new();
    let mut order_history: Vec<Order> = orders.clone();
    
    // Get reference symbol for iteration (assuming all data has same timestamps)
    let reference_symbol = data.keys().next().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("No data provided for backtesting")
    })?;
    
    let reference_data = &data[reference_symbol];
    
    // Main backtesting loop
    for (i, reference_bar) in reference_data.iter().enumerate() {
        let current_timestamp = reference_bar.timestamp;
        
        // Process orders for this timestamp
        for order in order_history.iter_mut() {
            // Skip already filled or canceled orders
            if order.status == OrderStatus::Filled || order.status == OrderStatus::Canceled {
                continue;
            }
            
            // Get current bar for the order's symbol
            if let Some(bars) = data.get(&order.symbol) {
                if i < bars.len() {
                    let current_bar = &bars[i];
                    
                    // Calculate execution price
                    if let Some(executed_price) = calculate_execution_price(order, current_bar) {
                        // Apply commission and slippage
                        let effective_price = apply_transaction_costs(
                            order, 
                            executed_price, 
                            config.commission_rate, 
                            config.slippage
                        );
                        
                        // Create fill
                        let mut fill_qty = order.quantity;
                        if !config.enable_fractional {
                            fill_qty = fill_qty.floor(); // Round down to nearest integer
                        }
                        
                        // Add fill to order
                        order.fills.push(Fill {
                            quantity: fill_qty,
                            price: effective_price,
                            timestamp: current_timestamp,
                        });
                        
                        // Create trade record
                        let trade = Trade {
                            trade_id: format!("t-{}-{}", order.order_id, order.fills.len()),
                            order_id: order.order_id.clone(),
                            symbol: order.symbol.clone(),
                            side: order.side,
                            quantity: fill_qty,
                            price: effective_price,
                            timestamp: current_timestamp,
                        };
                        
                        // Update portfolio with the trade
                        update_portfolio_from_trade(&mut portfolio, &trade);
                        
                        // Record trade
                        trade_history.push(trade);
                        
                        // Update order status
                        order.status = OrderStatus::Filled;
                    }
                }
            }
        }
        
        // Update market prices for all positions
        for (symbol, position) in portfolio.positions.iter_mut() {
            if let Some(bars) = data.get(symbol) {
                if i < bars.len() {
                    let current_price = bars[i].close;
                    update_position_market_price(position, current_price);
                }
            }
        }
        
        // Update portfolio total value
        update_portfolio_value(&mut portfolio);
        
        // Record portfolio snapshot
        let snapshot = PortfolioSnapshot {
            timestamp: current_timestamp,
            cash: portfolio.cash,
            total_value: portfolio.total_value,
            positions: portfolio.positions.clone(),
        };
        
        portfolio_history.push(snapshot);
    }
    
    // Calculate performance metrics
    let metrics = calculate_performance_metrics(&portfolio_history, &trade_history, config.initial_capital);
    
    // Create Python dictionary for results
    let result_dict = PyDict::new(py);
    
    // Convert portfolio history to Python list
    let py_portfolio_history = PyList::empty(py);
    for snapshot in portfolio_history {
        let py_snapshot = PyDict::new(py);
        py_snapshot.set_item("timestamp", snapshot.timestamp)?;
        py_snapshot.set_item("cash", snapshot.cash)?;
        py_snapshot.set_item("total_value", snapshot.total_value)?;
        
        let py_positions = PyDict::new(py);
        for (symbol, position) in snapshot.positions {
            let py_position = PyDict::new(py);
            py_position.set_item("quantity", position.quantity)?;
            py_position.set_item("entry_price", position.entry_price)?;
            py_position.set_item("market_price", position.market_price)?;
            py_position.set_item("unrealized_pnl", position.unrealized_pnl)?;
            py_position.set_item("realized_pnl", position.realized_pnl)?;
            
            py_positions.set_item(symbol, py_position)?;
        }
        
        py_snapshot.set_item("positions", py_positions)?;
        py_portfolio_history.append(py_snapshot)?;
    }
    
    // Convert trade history to Python list
    let py_trade_history = PyList::empty(py);
    for trade in trade_history {
        let py_trade = PyDict::new(py);
        py_trade.set_item("trade_id", trade.trade_id)?;
        py_trade.set_item("order_id", trade.order_id)?;
        py_trade.set_item("symbol", trade.symbol)?;
        py_trade.set_item("side", format!("{:?}", trade.side))?;
        py_trade.set_item("quantity", trade.quantity)?;
        py_trade.set_item("price", trade.price)?;
        py_trade.set_item("timestamp", trade.timestamp)?;
        
        py_trade_history.append(py_trade)?;
    }
    
    // Convert order history to Python list
    let py_order_history = PyList::empty(py);
    for order in order_history {
        let py_order = PyDict::new(py);
        py_order.set_item("order_id", order.order_id)?;
        py_order.set_item("symbol", order.symbol)?;
        py_order.set_item("side", format!("{:?}", order.side))?;
        py_order.set_item("order_type", format!("{:?}", order.order_type))?;
        py_order.set_item("quantity", order.quantity)?;
        py_order.set_item("status", format!("{:?}", order.status))?;
        py_order.set_item("created_at", order.created_at)?;
        
        if let Some(limit_price) = order.limit_price {
            py_order.set_item("limit_price", limit_price)?;
        }
        
        if let Some(stop_price) = order.stop_price {
            py_order.set_item("stop_price", stop_price)?;
        }
        
        let py_fills = PyList::empty(py);
        for fill in order.fills {
            let py_fill = PyDict::new(py);
            py_fill.set_item("quantity", fill.quantity)?;
            py_fill.set_item("price", fill.price)?;
            py_fill.set_item("timestamp", fill.timestamp)?;
            
            py_fills.append(py_fill)?;
        }
        
        py_order.set_item("fills", py_fills)?;
        py_order_history.append(py_order)?;
    }
    
    // Convert metrics to Python dict
    let py_metrics = PyDict::new(py);
    py_metrics.set_item("total_return", metrics.total_return)?;
    py_metrics.set_item("annualized_return", metrics.annualized_return)?;
    py_metrics.set_item("volatility", metrics.volatility)?;
    py_metrics.set_item("sharpe_ratio", metrics.sharpe_ratio)?;
    py_metrics.set_item("sortino_ratio", metrics.sortino_ratio)?;
    py_metrics.set_item("max_drawdown", metrics.max_drawdown)?;
    py_metrics.set_item("max_drawdown_duration", metrics.max_drawdown_duration)?;
    py_metrics.set_item("total_trades", metrics.total_trades)?;
    py_metrics.set_item("win_rate", metrics.win_rate)?;
    py_metrics.set_item("profit_factor", metrics.profit_factor)?;
    py_metrics.set_item("avg_profit_per_trade", metrics.avg_profit_per_trade)?;
    py_metrics.set_item("avg_loss_per_trade", metrics.avg_loss_per_trade)?;
    py_metrics.set_item("avg_profit_loss_ratio", metrics.avg_profit_loss_ratio)?;
    
    // Add everything to result dict
    result_dict.set_item("portfolio_history", py_portfolio_history)?;
    result_dict.set_item("trade_history", py_trade_history)?;
    result_dict.set_item("order_history", py_order_history)?;
    result_dict.set_item("metrics", py_metrics)?;
    
    Ok(result_dict.into())
}

/// Calculate performance metrics from backtest results
fn calculate_performance_metrics(
    portfolio_history: &[PortfolioSnapshot],
    trade_history: &[Trade],
    initial_capital: f64,
) -> PerformanceMetrics {
    // Default metrics
    let mut metrics = PerformanceMetrics {
        total_return: 0.0,
        annualized_return: 0.0,
        volatility: 0.0,
        sharpe_ratio: 0.0,
        sortino_ratio: 0.0,
        max_drawdown: 0.0,
        max_drawdown_duration: 0,
        total_trades: trade_history.len() as i64,
        win_rate: 0.0,
        profit_factor: 0.0,
        avg_profit_per_trade: 0.0,
        avg_loss_per_trade: 0.0,
        avg_profit_loss_ratio: 0.0,
    };
    
    // Need at least 2 portfolio snapshots to calculate returns
    if portfolio_history.len() < 2 {
        return metrics;
    }
    
    // Calculate total return
    let final_value = portfolio_history.last().unwrap().total_value;
    metrics.total_return = (final_value / initial_capital) - 1.0;
    
    // Calculate daily returns
    let mut returns = Vec::with_capacity(portfolio_history.len() - 1);
    for i in 1..portfolio_history.len() {
        let prev_value = portfolio_history[i-1].total_value;
        let curr_value = portfolio_history[i].total_value;
        let daily_return = (curr_value / prev_value) - 1.0;
        returns.push(daily_return);
    }
    
    // Calculate volatility (annualized)
    if !returns.is_empty() {
        let sum_squared_deviation: f64 = returns.iter()
            .map(|&r| {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                (r - mean).powi(2)
            })
            .sum();
        
        let variance = sum_squared_deviation / returns.len() as f64;
        metrics.volatility = (variance.sqrt()) * (252.0_f64.sqrt()); // Annualize
    }
    
    // Calculate Sharpe ratio (assuming risk-free rate of 0)
    if metrics.volatility > 0.0 {
        metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility;
    }
    
    // Calculate drawdowns
    let mut max_value = initial_capital;
    let mut max_drawdown = 0.0;
    let mut drawdown_start = 0;
    let mut max_drawdown_duration = 0;
    let mut current_drawdown_duration = 0;
    
    for (i, snapshot) in portfolio_history.iter().enumerate() {
        let current_value = snapshot.total_value;
        
        if current_value > max_value {
            max_value = current_value;
            // Reset drawdown tracking if we're at a new high
            drawdown_start = i;
            current_drawdown_duration = 0;
        } else {
            // Calculate drawdown
            let drawdown = (max_value - current_value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
            
            // Track drawdown duration
            current_drawdown_duration = i - drawdown_start;
            if current_drawdown_duration > max_drawdown_duration {
                max_drawdown_duration = current_drawdown_duration;
            }
        }
    }
    
    metrics.max_drawdown = max_drawdown;
    metrics.max_drawdown_duration = max_drawdown_duration as i64;
    
    // Calculate win rate and profit factor
    if !trade_history.is_empty() {
        let mut winning_trades = 0;
        let mut total_profit = 0.0;
        let mut total_loss = 0.0;
        let mut profit_trades = Vec::new();
        let mut loss_trades = Vec::new();
        
        for trade in trade_history {
            // Simplified P&L calculation
            let trade_pnl = match trade.side {
                OrderSide::Buy => -trade.quantity * trade.price,
                OrderSide::Sell => trade.quantity * trade.price,
            };
            
            if trade_pnl > 0.0 {
                winning_trades += 1;
                total_profit += trade_pnl;
                profit_trades.push(trade_pnl);
            } else {
                total_loss += trade_pnl.abs();
                loss_trades.push(trade_pnl);
            }
        }
        
        metrics.win_rate = winning_trades as f64 / trade_history.len() as f64;
        
        if total_loss > 0.0 {
            metrics.profit_factor = total_profit / total_loss;
        } else {
            metrics.profit_factor = if total_profit > 0.0 { f64::INFINITY } else { 0.0 };
        }
        
        // Calculate average profit/loss per trade
        if !profit_trades.is_empty() {
            metrics.avg_profit_per_trade = profit_trades.iter().sum::<f64>() / profit_trades.len() as f64;
        }
        
        if !loss_trades.is_empty() {
            metrics.avg_loss_per_trade = loss_trades.iter().sum::<f64>() / loss_trades.len() as f64;
        }
        
        // Calculate profit/loss ratio
        if metrics.avg_loss_per_trade != 0.0 {
            metrics.avg_profit_loss_ratio = metrics.avg_profit_per_trade.abs() / metrics.avg_loss_per_trade.abs();
        } else {
            metrics.avg_profit_loss_ratio = if metrics.avg_profit_per_trade > 0.0 { f64::INFINITY } else { 0.0 };
        }
    }
    
    // Calculate annualized return
    let first_timestamp = portfolio_history.first().unwrap().timestamp;
    let last_timestamp = portfolio_history.last().unwrap().timestamp;
    let days = (last_timestamp - first_timestamp) as f64 / (24.0 * 60.0 * 60.0);
    
    if days > 0.0 {
        metrics.annualized_return = ((1.0 + metrics.total_return).powf(365.0 / days)) - 1.0;
    }
    
    // Calculate Sortino ratio (using downside deviation)
    let downside_returns: Vec<f64> = returns.iter()
        .filter(|&&r| r < 0.0)
        .cloned()
        .collect();
    
    if !downside_returns.is_empty() {
        let sum_squared_downside: f64 = downside_returns.iter()
            .map(|&r| r.powi(2))
            .sum();
        
        let downside_deviation = (sum_squared_downside / downside_returns.len() as f64).sqrt() * (252.0_f64.sqrt());
        
        if downside_deviation > 0.0 {
            metrics.sortino_ratio = metrics.annualized_return / downside_deviation;
        }
    }
    
    metrics
}
