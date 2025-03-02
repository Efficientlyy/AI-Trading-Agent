/*
 * Python bindings for the market data models
 */

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::market_data::{
    CandleData, OrderBookData, OrderBookProcessor, OrderBookUpdate,
    ProcessorError, TimeFrame, TradeData, TradeSide, MarketImpact
};

/// Initialize the market_data module
pub fn init_module(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_candle, m)?)?;
    m.add_function(wrap_pyfunction!(create_order_book, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_mid_price, m)?)?;
    m.add_function(wrap_pyfunction!(create_order_book_processor, m)?)?;
    
    // Register TimeFrame enum as Python class
    m.add_class::<PyTimeFrame>()?;
    
    // Register data classes
    m.add_class::<PyCandleData>()?;
    m.add_class::<PyOrderBookData>()?;
    m.add_class::<PyOrderBookProcessor>()?;
    
    Ok(())
}

/// Python wrapper for TimeFrame enum
#[pyclass]
#[derive(Clone)]
pub struct PyTimeFrame {
    inner: TimeFrame,
}

#[pymethods]
impl PyTimeFrame {
    #[classattr]
    const MINUTE_1: &'static str = "1m";
    #[classattr]
    const MINUTE_5: &'static str = "5m";
    #[classattr]
    const MINUTE_15: &'static str = "15m";
    #[classattr]
    const HOUR_1: &'static str = "1h";
    #[classattr]
    const HOUR_4: &'static str = "4h";
    #[classattr]
    const DAY_1: &'static str = "1d";
    #[classattr]
    const WEEK_1: &'static str = "1w";
    
    #[new]
    fn new(timeframe: &str) -> PyResult<Self> {
        let inner = match timeframe {
            "1m" => TimeFrame::Minute1,
            "3m" => TimeFrame::Minute3,
            "5m" => TimeFrame::Minute5,
            "15m" => TimeFrame::Minute15,
            "30m" => TimeFrame::Minute30,
            "1h" => TimeFrame::Hour1,
            "2h" => TimeFrame::Hour2,
            "4h" => TimeFrame::Hour4,
            "6h" => TimeFrame::Hour6,
            "12h" => TimeFrame::Hour12,
            "1d" => TimeFrame::Day1,
            "3d" => TimeFrame::Day3,
            "1w" => TimeFrame::Week1,
            "1M" => TimeFrame::Month1,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid timeframe: {}", timeframe)
            )),
        };
        
        Ok(Self { inner })
    }
    
    #[getter]
    fn duration_seconds(&self) -> i64 {
        self.inner.duration_seconds()
    }
    
    fn __str__(&self) -> String {
        match self.inner {
            TimeFrame::Minute1 => "1m".to_string(),
            TimeFrame::Minute3 => "3m".to_string(),
            TimeFrame::Minute5 => "5m".to_string(),
            TimeFrame::Minute15 => "15m".to_string(),
            TimeFrame::Minute30 => "30m".to_string(),
            TimeFrame::Hour1 => "1h".to_string(),
            TimeFrame::Hour2 => "2h".to_string(),
            TimeFrame::Hour4 => "4h".to_string(),
            TimeFrame::Hour6 => "6h".to_string(),
            TimeFrame::Hour12 => "12h".to_string(),
            TimeFrame::Day1 => "1d".to_string(),
            TimeFrame::Day3 => "3d".to_string(),
            TimeFrame::Week1 => "1w".to_string(),
            TimeFrame::Month1 => "1M".to_string(),
        }
    }
}

/// Python wrapper for CandleData
#[pyclass]
#[derive(Clone)]
pub struct PyCandleData {
    inner: CandleData,
}

#[pymethods]
impl PyCandleData {
    #[new]
    fn new(
        symbol: String,
        exchange: String,
        timestamp: f64,  // Unix timestamp in seconds
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        timeframe: String,
    ) -> PyResult<Self> {
        // Convert timestamp to DateTime<Utc>
        let secs = timestamp.trunc() as i64;
        let nsecs = ((timestamp - secs as f64) * 1_000_000_000.0) as u32;
        let timestamp = DateTime::from_timestamp(secs, nsecs)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid timestamp"))?;
        
        // Convert timeframe string to TimeFrame enum
        let timeframe = match timeframe.as_str() {
            "1m" => TimeFrame::Minute1,
            "3m" => TimeFrame::Minute3,
            "5m" => TimeFrame::Minute5,
            "15m" => TimeFrame::Minute15,
            "30m" => TimeFrame::Minute30,
            "1h" => TimeFrame::Hour1,
            "2h" => TimeFrame::Hour2,
            "4h" => TimeFrame::Hour4,
            "6h" => TimeFrame::Hour6,
            "12h" => TimeFrame::Hour12,
            "1d" => TimeFrame::Day1,
            "3d" => TimeFrame::Day3,
            "1w" => TimeFrame::Week1,
            "1M" => TimeFrame::Month1,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid timeframe: {}", timeframe)
            )),
        };
        
        let inner = CandleData {
            symbol,
            exchange,
            timestamp,
            open: Decimal::try_from(open).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid decimal value for open: {}", e)
            ))?,
            high: Decimal::try_from(high).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid decimal value for high: {}", e)
            ))?,
            low: Decimal::try_from(low).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid decimal value for low: {}", e)
            ))?,
            close: Decimal::try_from(close).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid decimal value for close: {}", e)
            ))?,
            volume: Decimal::try_from(volume).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid decimal value for volume: {}", e)
            ))?,
            timeframe,
            trade_count: None,
            vwap: None,
            quote_volume: None,
        };
        
        Ok(Self { inner })
    }
    
    #[getter]
    fn symbol(&self) -> String {
        self.inner.symbol.clone()
    }
    
    #[getter]
    fn exchange(&self) -> String {
        self.inner.exchange.clone()
    }
    
    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp.timestamp() as f64 + 
        self.inner.timestamp.timestamp_subsec_nanos() as f64 / 1_000_000_000.0
    }
    
    #[getter]
    fn open(&self) -> f64 {
        self.inner.open.to_f64().unwrap_or(0.0)
    }
    
    #[getter]
    fn high(&self) -> f64 {
        self.inner.high.to_f64().unwrap_or(0.0)
    }
    
    #[getter]
    fn low(&self) -> f64 {
        self.inner.low.to_f64().unwrap_or(0.0)
    }
    
    #[getter]
    fn close(&self) -> f64 {
        self.inner.close.to_f64().unwrap_or(0.0)
    }
    
    #[getter]
    fn volume(&self) -> f64 {
        self.inner.volume.to_f64().unwrap_or(0.0)
    }
    
    #[getter]
    fn timeframe(&self) -> String {
        match self.inner.timeframe {
            TimeFrame::Minute1 => "1m".to_string(),
            TimeFrame::Minute3 => "3m".to_string(),
            TimeFrame::Minute5 => "5m".to_string(),
            TimeFrame::Minute15 => "15m".to_string(),
            TimeFrame::Minute30 => "30m".to_string(),
            TimeFrame::Hour1 => "1h".to_string(),
            TimeFrame::Hour2 => "2h".to_string(),
            TimeFrame::Hour4 => "4h".to_string(),
            TimeFrame::Hour6 => "6h".to_string(),
            TimeFrame::Hour12 => "12h".to_string(),
            TimeFrame::Day1 => "1d".to_string(),
            TimeFrame::Day3 => "3d".to_string(),
            TimeFrame::Week1 => "1w".to_string(),
            TimeFrame::Month1 => "1M".to_string(),
        }
    }
    
    fn range(&self) -> f64 {
        self.inner.range().to_f64().unwrap_or(0.0)
    }
    
    fn body(&self) -> f64 {
        self.inner.body().to_f64().unwrap_or(0.0)
    }
    
    fn is_bullish(&self) -> bool {
        self.inner.is_bullish()
    }
    
    fn is_bearish(&self) -> bool {
        self.inner.is_bearish()
    }
    
    fn body_percent(&self) -> f64 {
        self.inner.body_percent().to_f64().unwrap_or(0.0)
    }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("symbol", self.symbol())?;
        dict.set_item("exchange", self.exchange())?;
        dict.set_item("timestamp", self.timestamp())?;
        dict.set_item("open", self.open())?;
        dict.set_item("high", self.high())?;
        dict.set_item("low", self.low())?;
        dict.set_item("close", self.close())?;
        dict.set_item("volume", self.volume())?;
        dict.set_item("timeframe", self.timeframe())?;
        
        Ok(dict.into())
    }
}

/// Python wrapper for OrderBookData
#[pyclass]
#[derive(Clone)]
pub struct PyOrderBookData {
    inner: OrderBookData,
}

#[pymethods]
impl PyOrderBookData {
    #[new]
    fn new(
        symbol: String,
        exchange: String,
        timestamp: f64,  // Unix timestamp in seconds
        bids: Vec<(f64, f64)>,  // (price, amount) pairs
        asks: Vec<(f64, f64)>,  // (price, amount) pairs
    ) -> PyResult<Self> {
        // Convert timestamp to DateTime<Utc>
        let secs = timestamp.trunc() as i64;
        let nsecs = ((timestamp - secs as f64) * 1_000_000_000.0) as u32;
        let timestamp = DateTime::from_timestamp(secs, nsecs)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid timestamp"))?;
        
        // Convert bids and asks to BTreeMap
        let mut bids_map = BTreeMap::new();
        for (price, amount) in bids {
            let price_dec = Decimal::try_from(price).map_err(|e| 
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid price: {}", e)))?;
            let amount_dec = Decimal::try_from(amount).map_err(|e| 
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid amount: {}", e)))?;
            bids_map.insert(price_dec, amount_dec);
        }
        
        let mut asks_map = BTreeMap::new();
        for (price, amount) in asks {
            let price_dec = Decimal::try_from(price).map_err(|e| 
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid price: {}", e)))?;
            let amount_dec = Decimal::try_from(amount).map_err(|e| 
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid amount: {}", e)))?;
            asks_map.insert(price_dec, amount_dec);
        }
        
        let inner = OrderBookData {
            symbol,
            exchange,
            timestamp,
            bids: bids_map,
            asks: asks_map,
            sequence: None,
        };
        
        Ok(Self { inner })
    }
    
    #[getter]
    fn symbol(&self) -> String {
        self.inner.symbol.clone()
    }
    
    #[getter]
    fn exchange(&self) -> String {
        self.inner.exchange.clone()
    }
    
    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp.timestamp() as f64 + 
        self.inner.timestamp.timestamp_subsec_nanos() as f64 / 1_000_000_000.0
    }
    
    #[getter]
    fn bids(&self, py: Python) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        
        for (price, amount) in self.inner.bids.iter().rev() {  // Reverse to get highest bids first
            let tuple = PyTuple::new(py, &[
                price.to_f64().unwrap_or(0.0).into_py(py),
                amount.to_f64().unwrap_or(0.0).into_py(py),
            ]);
            list.append(tuple)?;
        }
        
        Ok(list.into())
    }
    
    #[getter]
    fn asks(&self, py: Python) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        
        for (price, amount) in self.inner.asks.iter() {
            let tuple = PyTuple::new(py, &[
                price.to_f64().unwrap_or(0.0).into_py(py),
                amount.to_f64().unwrap_or(0.0).into_py(py),
            ]);
            list.append(tuple)?;
        }
        
        Ok(list.into())
    }
    
    fn best_bid(&self) -> Option<(f64, f64)> {
        self.inner.best_bid().map(|price| {
            let amount = self.inner.bids.get(&price).unwrap();
            (price.to_f64().unwrap_or(0.0), amount.to_f64().unwrap_or(0.0))
        })
    }
    
    fn best_ask(&self) -> Option<(f64, f64)> {
        self.inner.best_ask().map(|price| {
            let amount = self.inner.asks.get(&price).unwrap();
            (price.to_f64().unwrap_or(0.0), amount.to_f64().unwrap_or(0.0))
        })
    }
    
    fn spread(&self) -> Option<f64> {
        self.inner.spread().map(|spread| spread.to_f64().unwrap_or(0.0))
    }
    
    fn mid_price(&self) -> Option<f64> {
        self.inner.mid_price().map(|price| price.to_f64().unwrap_or(0.0))
    }
    
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("symbol", self.symbol())?;
        dict.set_item("exchange", self.exchange())?;
        dict.set_item("timestamp", self.timestamp())?;
        dict.set_item("bids", self.bids(py)?)?;
        dict.set_item("asks", self.asks(py)?)?;
        
        if let Some(mid) = self.mid_price() {
            dict.set_item("mid_price", mid)?;
        }
        
        if let Some(spread) = self.spread() {
            dict.set_item("spread", spread)?;
        }
        
        Ok(dict.into())
    }
}

/// Create a new candle from Python values
#[pyfunction]
fn create_candle(
    symbol: String,
    exchange: String,
    timestamp: f64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    timeframe: String,
) -> PyResult<PyCandleData> {
    PyCandleData::new(
        symbol,
        exchange,
        timestamp,
        open,
        high,
        low,
        close,
        volume,
        timeframe,
    )
}

/// Create a new order book from Python values
#[pyfunction]
fn create_order_book(
    symbol: String,
    exchange: String,
    timestamp: f64,
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
) -> PyResult<PyOrderBookData> {
    PyOrderBookData::new(
        symbol,
        exchange,
        timestamp,
        bids,
        asks,
    )
}

/// Calculate the mid price from a list of bids and asks
#[pyfunction]
fn calculate_mid_price(
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
) -> Option<f64> {
    if bids.is_empty() || asks.is_empty() {
        return None;
    }
    
    // Find the highest bid
    let best_bid = bids.iter().map(|(price, _)| *price).fold(0.0, f64::max);
    
    // Find the lowest ask
    let best_ask = asks.iter().map(|(price, _)| *price).fold(f64::INFINITY, f64::min);
    
    // Calculate mid price
    Some((best_bid + best_ask) / 2.0)
}

/// Python wrapper for OrderBookProcessor
#[pyclass(name = "OrderBookProcessor")]
pub struct PyOrderBookProcessor {
    processor: OrderBookProcessor,
}

#[pymethods]
impl PyOrderBookProcessor {
    #[new]
    fn new(symbol: String, exchange: String, max_depth: usize) -> Self {
        Self {
            processor: OrderBookProcessor::new(symbol, exchange, max_depth),
        }
    }
    
    /// Process a batch of order book updates
    #[pyo3(text_signature = "(self, updates)")]
    fn process_updates(&mut self, py: Python, updates: &PyList) -> PyResult<f64> {
        let mut rust_updates = Vec::with_capacity(updates.len());
        
        for item in updates.iter() {
            let update = item.downcast::<PyDict>()?;
            
            let price: f64 = update.get_item("price")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'price' field"))?
                .extract()?;
            
            let side: String = update.get_item("side")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'side' field"))?
                .extract()?;
            
            let quantity: f64 = update.get_item("quantity")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'quantity' field"))?
                .extract()?;
            
            let timestamp = if let Some(ts) = update.get_item("timestamp") {
                let ts_float: f64 = ts.extract()?;
                let seconds = ts_float.trunc() as i64;
                let nanos = ((ts_float.fract() * 1_000_000_000.0).round()) as u32;
                DateTime::from_timestamp(seconds, nanos).unwrap_or_else(|| Utc::now())
            } else {
                Utc::now()
            };
            
            let sequence: u64 = update.get_item("sequence")
                .map(|seq| seq.extract())
                .transpose()?
                .unwrap_or(0);
            
            let side = match side.to_lowercase().as_str() {
                "buy" | "bid" => TradeSide::Buy,
                "sell" | "ask" => TradeSide::Sell,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid side, must be 'buy' or 'sell'")),
            };
            
            rust_updates.push(OrderBookUpdate {
                price: Decimal::from_f64(price).unwrap_or(Decimal::ZERO),
                side,
                quantity: Decimal::from_f64(quantity).unwrap_or(Decimal::ZERO),
                timestamp,
                sequence,
            });
        }
        
        match self.processor.process_updates(rust_updates) {
            Ok(duration) => Ok(duration.as_secs_f64() * 1000.0), // Return milliseconds
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Error processing updates: {}", e))),
        }
    }
    
    /// Calculate market impact for a given order size
    #[pyo3(text_signature = "(self, side, size)")]
    fn calculate_market_impact(&self, py: Python, side: &str, size: f64) -> PyResult<Py<PyDict>> {
        let side = match side.to_lowercase().as_str() {
            "buy" | "bid" => TradeSide::Buy,
            "sell" | "ask" => TradeSide::Sell,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid side, must be 'buy' or 'sell'")),
        };
        
        let size_decimal = Decimal::from_f64(size).unwrap_or(Decimal::ZERO);
        let impact = self.processor.calculate_market_impact(side, size_decimal);
        
        let result = PyDict::new(py);
        result.set_item("avg_price", impact.avg_price.to_f64().unwrap_or(0.0))?;
        result.set_item("slippage_pct", impact.slippage_pct.to_f64().unwrap_or(0.0))?;
        result.set_item("total_value", impact.total_value.to_f64().unwrap_or(0.0))?;
        result.set_item("fillable_quantity", impact.fillable_quantity.to_f64().unwrap_or(0.0))?;
        result.set_item("levels_consumed", impact.levels_consumed)?;
        
        Ok(result.into())
    }
    
    /// Get the best bid price
    #[pyo3(text_signature = "(self)")]
    fn best_bid_price(&self) -> f64 {
        self.processor.best_bid_price().to_f64().unwrap_or(0.0)
    }
    
    /// Get the best ask price
    #[pyo3(text_signature = "(self)")]
    fn best_ask_price(&self) -> f64 {
        self.processor.best_ask_price().to_f64().unwrap_or(0.0)
    }
    
    /// Get the mid price
    #[pyo3(text_signature = "(self)")]
    fn mid_price(&self) -> f64 {
        self.processor.mid_price().to_f64().unwrap_or(0.0)
    }
    
    /// Get the current bid-ask spread
    #[pyo3(text_signature = "(self)")]
    fn spread(&self) -> f64 {
        self.processor.spread().to_f64().unwrap_or(0.0)
    }
    
    /// Get the current bid-ask spread as a percentage of the mid price
    #[pyo3(text_signature = "(self)")]
    fn spread_pct(&self) -> f64 {
        self.processor.spread_pct().to_f64().unwrap_or(0.0)
    }
    
    /// Calculate the volume-weighted average price (VWAP) for a given side and depth
    #[pyo3(text_signature = "(self, side, depth)")]
    fn vwap(&self, side: &str, depth: usize) -> PyResult<f64> {
        let side = match side.to_lowercase().as_str() {
            "buy" | "bid" => TradeSide::Buy,
            "sell" | "ask" => TradeSide::Sell,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid side, must be 'buy' or 'sell'")),
        };
        
        Ok(self.processor.vwap(side, depth).to_f64().unwrap_or(0.0))
    }
    
    /// Calculate the total liquidity available up to a given price depth
    #[pyo3(text_signature = "(self, side, price_depth)")]
    fn liquidity_up_to(&self, side: &str, price_depth: f64) -> PyResult<f64> {
        let side = match side.to_lowercase().as_str() {
            "buy" | "bid" => TradeSide::Buy,
            "sell" | "ask" => TradeSide::Sell,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid side, must be 'buy' or 'sell'")),
        };
        
        let price_depth_decimal = Decimal::from_f64(price_depth).unwrap_or(Decimal::ZERO);
        Ok(self.processor.liquidity_up_to(side, price_depth_decimal).to_f64().unwrap_or(0.0))
    }
    
    /// Detect order book imbalance (ratio of buy to sell liquidity)
    #[pyo3(text_signature = "(self, depth)")]
    fn book_imbalance(&self, depth: usize) -> f64 {
        self.processor.book_imbalance(depth).to_f64().unwrap_or(1.0)
    }
    
    /// Get a snapshot of the current order book
    #[pyo3(text_signature = "(self)")]
    fn snapshot(&self, py: Python) -> PyResult<Py<PyDict>> {
        let book = self.processor.snapshot();
        
        let result = PyDict::new(py);
        
        let bids = PyList::new(py, book.bids().iter().map(|level| {
            let level_dict = PyDict::new(py);
            level_dict.set_item("price", level.price.to_f64().unwrap_or(0.0)).unwrap();
            level_dict.set_item("quantity", level.quantity.to_f64().unwrap_or(0.0)).unwrap();
            level_dict
        }));
        
        let asks = PyList::new(py, book.asks().iter().map(|level| {
            let level_dict = PyDict::new(py);
            level_dict.set_item("price", level.price.to_f64().unwrap_or(0.0)).unwrap();
            level_dict.set_item("quantity", level.quantity.to_f64().unwrap_or(0.0)).unwrap();
            level_dict
        }));
        
        result.set_item("symbol", &book.symbol)?;
        result.set_item("exchange", &book.exchange)?;
        result.set_item("timestamp", book.timestamp.timestamp_millis() as f64 / 1000.0)?;
        result.set_item("bids", bids)?;
        result.set_item("asks", asks)?;
        
        Ok(result.into())
    }
    
    /// Get processing statistics
    #[pyo3(text_signature = "(self)")]
    fn processing_stats(&self, py: Python) -> PyResult<Py<PyDict>> {
        let stats = self.processor.processing_stats();
        
        let result = PyDict::new(py);
        result.set_item("updates_processed", stats.updates_processed)?;
        result.set_item("levels_added", stats.levels_added)?;
        result.set_item("levels_removed", stats.levels_removed)?;
        result.set_item("levels_modified", stats.levels_modified)?;
        result.set_item("avg_processing_time_us", stats.avg_processing_time_us)?;
        result.set_item("max_processing_time_us", stats.max_processing_time_us)?;
        result.set_item("min_processing_time_us", stats.min_processing_time_us)?;
        
        Ok(result.into())
    }
    
    /// Reset the order book processor
    #[pyo3(text_signature = "(self)")]
    fn reset(&mut self) {
        self.processor.reset();
    }
}

/// Create a new order book processor
#[pyfunction]
pub fn create_order_book_processor(symbol: String, exchange: String, max_depth: usize) -> PyOrderBookProcessor {
    PyOrderBookProcessor::new(symbol, exchange, max_depth)
} 