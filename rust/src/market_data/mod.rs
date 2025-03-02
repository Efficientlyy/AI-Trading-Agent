/*
 * Market data module for the Rust trading engine
 */

pub mod models;
pub mod processor;

// Re-export the most commonly used items
pub use models::{
    CandleData, OrderBookData, PriceConversion, TickerData, 
    TimeFrame, TradeData, TradeSide
}; 

pub use processor::{
    OrderBookProcessor, OrderBookUpdate, MarketImpact,
    ProcessingStats, ProcessorError, ProcessorResult
}; 