/*
 * Technical analysis module for the Rust trading engine
 */

pub mod indicators;

// Re-export the most commonly used items
pub use indicators::{
    calculate_ema, calculate_sma, CrossoverSignal, EMA, MAType, 
    MACrossover, MovingAverage, SMA, WMA
}; 