/*
 * Python bindings for the Rust backtesting engine
 */

use chrono::{DateTime, Duration, Utc};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::wrap_pyfunction;
use rust_decimal::Decimal;
use uuid::Uuid;
use std::collections::HashMap;

use crate::backtesting::{
    BacktestConfig, BacktestEngine, BacktestError, BacktestMode,
    BacktestOrder, BacktestOrderStatus, BacktestOrderType,
    BacktestPosition, BacktestResult, BacktestStats
};
use crate::market_data::{CandleData, TimeFrame, TradeSide};

/// Initialize the backtesting module for Python
pub fn init_module(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_backtest_engine, m)?)?;
    m.add_function(wrap_pyfunction!(create_candle, m)?)?;
    m.add_function(wrap_pyfunction!(create_market_order, m)?)?;
    m.add_function(wrap_pyfunction!(create_limit_order, m)?)?;
    m.add_function(wrap_pyfunction!(create_stop_market_order, m)?)?;
    m.add_function(wrap_pyfunction!(create_stop_limit_order, m)?)?;
    
    // Register classes
    m.add_class::<PyBacktestEngine>()?;
    m.add_class::<PyBacktestStats>()?;
    
    Ok(())
}

/// Convert a BacktestError to a Python exception
fn backtest_error_to_pyerr(err: BacktestError) -> PyErr {
    match err {
        BacktestError::InsufficientData(msg) => PyValueError::new_err(msg),
        BacktestError::InvalidParameter(msg) => PyValueError::new_err(msg),
        BacktestError::SimulationError(msg) => PyRuntimeError::new_err(msg),
    }
}

/// Convert a string to a TimeFrame
fn str_to_timeframe(tf_str: &str) -> PyResult<TimeFrame> {
    match tf_str.to_uppercase().as_str() {
        "1M" => Ok(TimeFrame::Minute1),
        "5M" => Ok(TimeFrame::Minute5),
        "15M" => Ok(TimeFrame::Minute15),
        "30M" => Ok(TimeFrame::Minute30),
        "1H" => Ok(TimeFrame::Hour1),
        "4H" => Ok(TimeFrame::Hour4),
        "1D" => Ok(TimeFrame::Day1),
        "1W" => Ok(TimeFrame::Week1),
        _ => Err(PyValueError::new_err(format!("Invalid timeframe: {}", tf_str))),
    }
}

/// Convert a string to an order type
fn str_to_order_type(order_type_str: &str) -> PyResult<BacktestOrderType> {
    match order_type_str.to_uppercase().as_str() {
        "MARKET" => Ok(BacktestOrderType::Market),
        "LIMIT" => Ok(BacktestOrderType::Limit),
        "STOP_MARKET" => Ok(BacktestOrderType::StopMarket),
        "STOP_LIMIT" => Ok(BacktestOrderType::StopLimit),
        _ => Err(PyValueError::new_err(format!("Invalid order type: {}", order_type_str))),
    }
}

/// Convert a string to a trade side
fn str_to_trade_side(side_str: &str) -> PyResult<TradeSide> {
    match side_str.to_uppercase().as_str() {
        "BUY" => Ok(TradeSide::Buy),
        "SELL" => Ok(TradeSide::Sell),
        _ => Err(PyValueError::new_err(format!("Invalid trade side: {}", side_str))),
    }
}

/// Convert a string to a backtest mode
fn str_to_backtest_mode(mode_str: &str) -> PyResult<BacktestMode> {
    match mode_str.to_uppercase().as_str() {
        "CANDLES" => Ok(BacktestMode::Candles),
        "TRADES" => Ok(BacktestMode::Trades),
        "ORDERBOOK" => Ok(BacktestMode::OrderBook),
        _ => Err(PyValueError::new_err(format!("Invalid backtest mode: {}", mode_str))),
    }
}

/// Python wrapper for BacktestStats
#[pyclass(name = "BacktestStats")]
struct PyBacktestStats {
    stats: BacktestStats,
}

#[pymethods]
impl PyBacktestStats {
    #[getter]
    fn start_time(&self) -> PyResult<i64> {
        Ok(self.stats.start_time.timestamp())
    }
    
    #[getter]
    fn end_time(&self) -> PyResult<i64> {
        Ok(self.stats.end_time.timestamp())
    }
    
    #[getter]
    fn initial_balance(&self) -> PyResult<f64> {
        Ok(self.stats.initial_balance.to_f64().unwrap_or(0.0))
    }
    
    #[getter]
    fn final_balance(&self) -> PyResult<f64> {
        Ok(self.stats.final_balance.to_f64().unwrap_or(0.0))
    }
    
    #[getter]
    fn total_trades(&self) -> PyResult<usize> {
        Ok(self.stats.total_trades)
    }
    
    #[getter]
    fn winning_trades(&self) -> PyResult<usize> {
        Ok(self.stats.winning_trades)
    }
    
    #[getter]
    fn losing_trades(&self) -> PyResult<usize> {
        Ok(self.stats.losing_trades)
    }
    
    #[getter]
    fn total_profit(&self) -> PyResult<f64> {
        Ok(self.stats.total_profit.to_f64().unwrap_or(0.0))
    }
    
    #[getter]
    fn total_loss(&self) -> PyResult<f64> {
        Ok(self.stats.total_loss.to_f64().unwrap_or(0.0))
    }
    
    #[getter]
    fn max_drawdown(&self) -> PyResult<f64> {
        Ok(self.stats.max_drawdown.to_f64().unwrap_or(0.0))
    }
    
    #[getter]
    fn max_drawdown_pct(&self) -> PyResult<f64> {
        Ok(self.stats.max_drawdown_pct.to_f64().unwrap_or(0.0))
    }
    
    #[getter]
    fn sharpe_ratio(&self) -> PyResult<Option<f64>> {
        Ok(self.stats.sharpe_ratio.and_then(|d| d.to_f64()))
    }
    
    #[getter]
    fn profit_factor(&self) -> PyResult<Option<f64>> {
        Ok(self.stats.profit_factor.and_then(|d| d.to_f64()))
    }
    
    #[getter]
    fn win_rate(&self) -> PyResult<f64> {
        Ok(self.stats.win_rate.to_f64().unwrap_or(0.0))
    }
    
    #[getter]
    fn avg_win(&self) -> PyResult<Option<f64>> {
        Ok(self.stats.avg_win.and_then(|d| d.to_f64()))
    }
    
    #[getter]
    fn avg_loss(&self) -> PyResult<Option<f64>> {
        Ok(self.stats.avg_loss.and_then(|d| d.to_f64()))
    }
    
    #[getter]
    fn largest_win(&self) -> PyResult<Option<f64>> {
        Ok(self.stats.largest_win.and_then(|d| d.to_f64()))
    }
    
    #[getter]
    fn largest_loss(&self) -> PyResult<Option<f64>> {
        Ok(self.stats.largest_loss.and_then(|d| d.to_f64()))
    }
    
    /// Convert to a Python dictionary
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        dict.set_item("start_time", self.stats.start_time.timestamp())?;
        dict.set_item("end_time", self.stats.end_time.timestamp())?;
        dict.set_item("initial_balance", self.stats.initial_balance.to_f64().unwrap_or(0.0))?;
        dict.set_item("final_balance", self.stats.final_balance.to_f64().unwrap_or(0.0))?;
        dict.set_item("total_trades", self.stats.total_trades)?;
        dict.set_item("winning_trades", self.stats.winning_trades)?;
        dict.set_item("losing_trades", self.stats.losing_trades)?;
        dict.set_item("total_profit", self.stats.total_profit.to_f64().unwrap_or(0.0))?;
        dict.set_item("total_loss", self.stats.total_loss.to_f64().unwrap_or(0.0))?;
        dict.set_item("max_drawdown", self.stats.max_drawdown.to_f64().unwrap_or(0.0))?;
        dict.set_item("max_drawdown_pct", self.stats.max_drawdown_pct.to_f64().unwrap_or(0.0))?;
        
        if let Some(sr) = self.stats.sharpe_ratio {
            dict.set_item("sharpe_ratio", sr.to_f64().unwrap_or(0.0))?;
        } else {
            dict.set_item("sharpe_ratio", py.None())?;
        }
        
        if let Some(pf) = self.stats.profit_factor {
            dict.set_item("profit_factor", pf.to_f64().unwrap_or(0.0))?;
        } else {
            dict.set_item("profit_factor", py.None())?;
        }
        
        dict.set_item("win_rate", self.stats.win_rate.to_f64().unwrap_or(0.0))?;
        
        if let Some(aw) = self.stats.avg_win {
            dict.set_item("avg_win", aw.to_f64().unwrap_or(0.0))?;
        } else {
            dict.set_item("avg_win", py.None())?;
        }
        
        if let Some(al) = self.stats.avg_loss {
            dict.set_item("avg_loss", al.to_f64().unwrap_or(0.0))?;
        } else {
            dict.set_item("avg_loss", py.None())?;
        }
        
        if let Some(lw) = self.stats.largest_win {
            dict.set_item("largest_win", lw.to_f64().unwrap_or(0.0))?;
        } else {
            dict.set_item("largest_win", py.None())?;
        }
        
        if let Some(ll) = self.stats.largest_loss {
            dict.set_item("largest_loss", ll.to_f64().unwrap_or(0.0))?;
        } else {
            dict.set_item("largest_loss", py.None())?;
        }
        
        Ok(dict.into())
    }
    
    fn __str__(&self) -> PyResult<String> {
        let profit_or_loss = self.stats.final_balance - self.stats.initial_balance;
        let pnl_percent = if self.stats.initial_balance > Decimal::ZERO {
            (profit_or_loss / self.stats.initial_balance) * Decimal::from(100)
        } else {
            Decimal::ZERO
        };
        
        Ok(format!(
            "BacktestStats(profit: {:.2}%, trades: {}, win_rate: {:.2}%)",
            pnl_percent,
            self.stats.total_trades,
            self.stats.win_rate
        ))
    }
    
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

/// Python wrapper for BacktestEngine
#[pyclass(name = "BacktestEngine")]
struct PyBacktestEngine {
    engine: BacktestEngine,
}

#[pymethods]
impl PyBacktestEngine {
    #[staticmethod]
    fn from_config(
        initial_balance: f64,
        symbols: Vec<String>,
        start_time: i64,
        end_time: i64,
        mode: &str,
        commission_rate: f64,
        slippage: f64,
        enable_fractional_sizing: bool,
    ) -> PyResult<Self> {
        let backtest_mode = str_to_backtest_mode(mode)?;
        
        let config = BacktestConfig {
            initial_balance: Decimal::try_from(initial_balance)
                .map_err(|_| PyValueError::new_err("Invalid initial balance"))?,
            symbols,
            start_time: DateTime::from_timestamp(start_time, 0)
                .ok_or_else(|| PyValueError::new_err("Invalid start time"))?,
            end_time: DateTime::from_timestamp(end_time, 0)
                .ok_or_else(|| PyValueError::new_err("Invalid end time"))?,
            mode: backtest_mode,
            commission_rate: Decimal::try_from(commission_rate)
                .map_err(|_| PyValueError::new_err("Invalid commission rate"))?,
            slippage: Decimal::try_from(slippage)
                .map_err(|_| PyValueError::new_err("Invalid slippage"))?,
            enable_fractional_sizing,
        };
        
        Ok(Self {
            engine: BacktestEngine::new(config),
        })
    }
    
    /// Process a candle in the backtest
    fn process_candle(
        &mut self,
        symbol: &str,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        timeframe: &str,
    ) -> PyResult<()> {
        let tf = str_to_timeframe(timeframe)?;
        
        let candle = CandleData::new(
            symbol.to_string(),
            "".to_string(), // Exchange is not used in backtesting
            DateTime::from_timestamp(timestamp, 0)
                .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?,
            Decimal::try_from(close)
                .map_err(|_| PyValueError::new_err("Invalid close price"))?,
            Decimal::try_from(high)
                .map_err(|_| PyValueError::new_err("Invalid high price"))?,
            Decimal::try_from(low)
                .map_err(|_| PyValueError::new_err("Invalid low price"))?,
            Decimal::try_from(open)
                .map_err(|_| PyValueError::new_err("Invalid open price"))?,
            Decimal::try_from(volume)
                .map_err(|_| PyValueError::new_err("Invalid volume"))?,
            tf,
        );
        
        self.engine.process_candle(&candle)
            .map_err(backtest_error_to_pyerr)
    }
    
    /// Submit a market order to the backtesting engine
    fn submit_market_order(
        &mut self,
        symbol: &str,
        side: &str,
        amount: f64,
    ) -> PyResult<String> {
        let trade_side = str_to_trade_side(side)?;
        let order_id = Uuid::new_v4().to_string();
        
        let order = BacktestOrder {
            id: order_id.clone(),
            symbol: symbol.to_string(),
            side: trade_side,
            order_type: BacktestOrderType::Market,
            price: None,
            amount: Decimal::try_from(amount)
                .map_err(|_| PyValueError::new_err("Invalid amount"))?,
            status: BacktestOrderStatus::Created,
            created_at: Utc::now(),
            executed_at: None,
            executed_price: None,
            stop_price: None,
            take_profit: None,
            stop_loss: None,
        };
        
        self.engine.submit_order(order)
            .map_err(backtest_error_to_pyerr)
    }
    
    /// Submit a limit order to the backtesting engine
    fn submit_limit_order(
        &mut self,
        symbol: &str,
        side: &str,
        price: f64,
        amount: f64,
    ) -> PyResult<String> {
        let trade_side = str_to_trade_side(side)?;
        let order_id = Uuid::new_v4().to_string();
        
        let order = BacktestOrder {
            id: order_id.clone(),
            symbol: symbol.to_string(),
            side: trade_side,
            order_type: BacktestOrderType::Limit,
            price: Some(Decimal::try_from(price)
                .map_err(|_| PyValueError::new_err("Invalid price"))?),
            amount: Decimal::try_from(amount)
                .map_err(|_| PyValueError::new_err("Invalid amount"))?,
            status: BacktestOrderStatus::Created,
            created_at: Utc::now(),
            executed_at: None,
            executed_price: None,
            stop_price: None,
            take_profit: None,
            stop_loss: None,
        };
        
        self.engine.submit_order(order)
            .map_err(backtest_error_to_pyerr)
    }
    
    /// Submit a stop market order to the backtesting engine
    fn submit_stop_market_order(
        &mut self,
        symbol: &str,
        side: &str,
        stop_price: f64,
        amount: f64,
    ) -> PyResult<String> {
        let trade_side = str_to_trade_side(side)?;
        let order_id = Uuid::new_v4().to_string();
        
        let order = BacktestOrder {
            id: order_id.clone(),
            symbol: symbol.to_string(),
            side: trade_side,
            order_type: BacktestOrderType::StopMarket,
            price: None,
            amount: Decimal::try_from(amount)
                .map_err(|_| PyValueError::new_err("Invalid amount"))?,
            status: BacktestOrderStatus::Created,
            created_at: Utc::now(),
            executed_at: None,
            executed_price: None,
            stop_price: Some(Decimal::try_from(stop_price)
                .map_err(|_| PyValueError::new_err("Invalid stop price"))?),
            take_profit: None,
            stop_loss: None,
        };
        
        self.engine.submit_order(order)
            .map_err(backtest_error_to_pyerr)
    }
    
    /// Submit a stop limit order to the backtesting engine
    fn submit_stop_limit_order(
        &mut self,
        symbol: &str,
        side: &str,
        stop_price: f64,
        limit_price: f64,
        amount: f64,
    ) -> PyResult<String> {
        let trade_side = str_to_trade_side(side)?;
        let order_id = Uuid::new_v4().to_string();
        
        let order = BacktestOrder {
            id: order_id.clone(),
            symbol: symbol.to_string(),
            side: trade_side,
            order_type: BacktestOrderType::StopLimit,
            price: Some(Decimal::try_from(limit_price)
                .map_err(|_| PyValueError::new_err("Invalid limit price"))?),
            amount: Decimal::try_from(amount)
                .map_err(|_| PyValueError::new_err("Invalid amount"))?,
            status: BacktestOrderStatus::Created,
            created_at: Utc::now(),
            executed_at: None,
            executed_price: None,
            stop_price: Some(Decimal::try_from(stop_price)
                .map_err(|_| PyValueError::new_err("Invalid stop price"))?),
            take_profit: None,
            stop_loss: None,
        };
        
        self.engine.submit_order(order)
            .map_err(backtest_error_to_pyerr)
    }
    
    /// Cancel an open order
    fn cancel_order(&mut self, order_id: &str) -> PyResult<()> {
        self.engine.cancel_order(order_id)
            .map_err(backtest_error_to_pyerr)
    }
    
    /// Get the current balance
    fn get_balance(&self) -> PyResult<f64> {
        Ok(self.engine.get_balance().to_f64().unwrap_or(0.0))
    }
    
    /// Get the current equity
    fn get_equity(&self) -> PyResult<f64> {
        Ok(self.engine.get_equity().to_f64().unwrap_or(0.0))
    }
    
    /// Get the current open positions
    fn get_positions(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        for (symbol, position) in self.engine.get_positions() {
            let pos_dict = PyDict::new(py);
            pos_dict.set_item("symbol", &position.symbol)?;
            pos_dict.set_item("amount", position.amount.to_f64().unwrap_or(0.0))?;
            pos_dict.set_item("entry_price", position.entry_price.to_f64().unwrap_or(0.0))?;
            pos_dict.set_item("current_price", position.current_price.to_f64().unwrap_or(0.0))?;
            pos_dict.set_item("unrealized_pnl", position.unrealized_pnl.to_f64().unwrap_or(0.0))?;
            pos_dict.set_item("realized_pnl", position.realized_pnl.to_f64().unwrap_or(0.0))?;
            pos_dict.set_item("opened_at", position.opened_at.timestamp())?;
            pos_dict.set_item("updated_at", position.updated_at.timestamp())?;
            
            dict.set_item(symbol, pos_dict)?;
        }
        
        Ok(dict.into())
    }
    
    /// Get the equity curve
    fn get_equity_curve(&self, py: Python) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        
        for (timestamp, equity) in self.engine.get_equity_curve() {
            let point = PyDict::new(py);
            point.set_item("timestamp", timestamp.timestamp())?;
            point.set_item("equity", equity.to_f64().unwrap_or(0.0))?;
            list.append(point)?;
        }
        
        Ok(list.into())
    }
    
    /// Run the backtest and get the final statistics
    fn run(&mut self) -> PyResult<PyBacktestStats> {
        match self.engine.run() {
            Ok(stats) => Ok(PyBacktestStats { stats }),
            Err(err) => Err(backtest_error_to_pyerr(err)),
        }
    }
}

/// Create a new backtest engine from config
#[pyfunction]
fn create_backtest_engine(
    initial_balance: f64,
    symbols: Vec<String>,
    start_time: i64,
    end_time: i64,
    mode: &str,
    commission_rate: f64,
    slippage: f64,
    enable_fractional_sizing: bool,
) -> PyResult<PyBacktestEngine> {
    PyBacktestEngine::from_config(
        initial_balance,
        symbols,
        start_time,
        end_time,
        mode,
        commission_rate,
        slippage,
        enable_fractional_sizing,
    )
}

/// Helper function to create a candle for backtesting
#[pyfunction]
fn create_candle(
    symbol: &str,
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    timeframe: &str,
) -> PyResult<CandleData> {
    let tf = str_to_timeframe(timeframe)?;
    
    Ok(CandleData::new(
        symbol.to_string(),
        "".to_string(), // Exchange is not used in backtesting
        DateTime::from_timestamp(timestamp, 0)
            .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?,
        Decimal::try_from(close)
            .map_err(|_| PyValueError::new_err("Invalid close price"))?,
        Decimal::try_from(high)
            .map_err(|_| PyValueError::new_err("Invalid high price"))?,
        Decimal::try_from(low)
            .map_err(|_| PyValueError::new_err("Invalid low price"))?,
        Decimal::try_from(open)
            .map_err(|_| PyValueError::new_err("Invalid open price"))?,
        Decimal::try_from(volume)
            .map_err(|_| PyValueError::new_err("Invalid volume"))?,
        tf,
    ))
}

/// Helper function to create a market order for backtesting
#[pyfunction]
fn create_market_order(
    symbol: &str,
    side: &str,
    amount: f64,
) -> PyResult<BacktestOrder> {
    let trade_side = str_to_trade_side(side)?;
    let order_id = Uuid::new_v4().to_string();
    
    Ok(BacktestOrder {
        id: order_id,
        symbol: symbol.to_string(),
        side: trade_side,
        order_type: BacktestOrderType::Market,
        price: None,
        amount: Decimal::try_from(amount)
            .map_err(|_| PyValueError::new_err("Invalid amount"))?,
        status: BacktestOrderStatus::Created,
        created_at: Utc::now(),
        executed_at: None,
        executed_price: None,
        stop_price: None,
        take_profit: None,
        stop_loss: None,
    })
}

/// Helper function to create a limit order for backtesting
#[pyfunction]
fn create_limit_order(
    symbol: &str,
    side: &str,
    price: f64,
    amount: f64,
) -> PyResult<BacktestOrder> {
    let trade_side = str_to_trade_side(side)?;
    let order_id = Uuid::new_v4().to_string();
    
    Ok(BacktestOrder {
        id: order_id,
        symbol: symbol.to_string(),
        side: trade_side,
        order_type: BacktestOrderType::Limit,
        price: Some(Decimal::try_from(price)
            .map_err(|_| PyValueError::new_err("Invalid price"))?),
        amount: Decimal::try_from(amount)
            .map_err(|_| PyValueError::new_err("Invalid amount"))?,
        status: BacktestOrderStatus::Created,
        created_at: Utc::now(),
        executed_at: None,
        executed_price: None,
        stop_price: None,
        take_profit: None,
        stop_loss: None,
    })
}

/// Helper function to create a stop market order for backtesting
#[pyfunction]
fn create_stop_market_order(
    symbol: &str,
    side: &str,
    stop_price: f64,
    amount: f64,
) -> PyResult<BacktestOrder> {
    let trade_side = str_to_trade_side(side)?;
    let order_id = Uuid::new_v4().to_string();
    
    Ok(BacktestOrder {
        id: order_id,
        symbol: symbol.to_string(),
        side: trade_side,
        order_type: BacktestOrderType::StopMarket,
        price: None,
        amount: Decimal::try_from(amount)
            .map_err(|_| PyValueError::new_err("Invalid amount"))?,
        status: BacktestOrderStatus::Created,
        created_at: Utc::now(),
        executed_at: None,
        executed_price: None,
        stop_price: Some(Decimal::try_from(stop_price)
            .map_err(|_| PyValueError::new_err("Invalid stop price"))?),
        take_profit: None,
        stop_loss: None,
    })
}

/// Helper function to create a stop limit order for backtesting
#[pyfunction]
fn create_stop_limit_order(
    symbol: &str,
    side: &str,
    stop_price: f64,
    limit_price: f64,
    amount: f64,
) -> PyResult<BacktestOrder> {
    let trade_side = str_to_trade_side(side)?;
    let order_id = Uuid::new_v4().to_string();
    
    Ok(BacktestOrder {
        id: order_id,
        symbol: symbol.to_string(),
        side: trade_side,
        order_type: BacktestOrderType::StopLimit,
        price: Some(Decimal::try_from(limit_price)
            .map_err(|_| PyValueError::new_err("Invalid limit price"))?),
        amount: Decimal::try_from(amount)
            .map_err(|_| PyValueError::new_err("Invalid amount"))?,
        status: BacktestOrderStatus::Created,
        created_at: Utc::now(),
        executed_at: None,
        executed_price: None,
        stop_price: Some(Decimal::try_from(stop_price)
            .map_err(|_| PyValueError::new_err("Invalid stop price"))?),
        take_profit: None,
        stop_loss: None,
    })
} 