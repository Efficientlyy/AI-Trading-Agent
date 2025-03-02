/*
 * Order book processor for high-performance market data processing
 * Provides real-time order book management with efficient updates and analytics
 */

use std::collections::{BTreeMap, VecDeque};
use std::time::{Instant, Duration};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use thiserror::Error;

use super::models::{OrderBookData, TradeSide, PriceLevel};

/// Errors that can occur during order book processing
#[derive(Error, Debug)]
pub enum ProcessorError {
    #[error("Invalid update sequence: {0}")]
    InvalidSequence(String),
    
    #[error("Invalid price level: {0}")]
    InvalidPriceLevel(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Result type for order book processing operations
pub type ProcessorResult<T> = Result<T, ProcessorError>;

/// Update to an order book price level
#[derive(Debug, Clone)]
pub struct OrderBookUpdate {
    /// The price level being updated
    pub price: Decimal,
    /// The side of the book (bid or ask)
    pub side: TradeSide,
    /// The new quantity (0 for removal)
    pub quantity: Decimal,
    /// Timestamp of the update
    pub timestamp: DateTime<Utc>,
    /// Sequence number for ordering
    pub sequence: u64,
}

/// Statistics about order book updates and processing
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    /// Total number of updates processed
    pub updates_processed: u64,
    /// Total number of price levels added
    pub levels_added: u64,
    /// Total number of price levels removed
    pub levels_removed: u64,
    /// Total number of price levels modified
    pub levels_modified: u64,
    /// Average processing time per update in microseconds
    pub avg_processing_time_us: f64,
    /// Maximum processing time in microseconds
    pub max_processing_time_us: u64,
    /// Minimum processing time in microseconds
    pub min_processing_time_us: u64,
}

/// Market impact calculation for a potential order
#[derive(Debug, Clone)]
pub struct MarketImpact {
    /// Average execution price
    pub avg_price: Decimal,
    /// Price slippage as a percentage
    pub slippage_pct: Decimal,
    /// Total value of the order
    pub total_value: Decimal,
    /// Total quantity that can be filled
    pub fillable_quantity: Decimal,
    /// Number of price levels needed to fill
    pub levels_consumed: usize,
}

/// High-performance order book processor for real-time market data
pub struct OrderBookProcessor {
    /// Symbol identifier
    symbol: String,
    /// Exchange identifier
    exchange: String,
    /// The current order book state
    book: OrderBookData,
    /// Queue of updates to be processed
    update_queue: VecDeque<OrderBookUpdate>,
    /// Maximum depth to maintain per side
    max_depth: usize,
    /// Last update time
    last_update_time: DateTime<Utc>,
    /// Last sequence number processed
    last_sequence: u64,
    /// Statistics about processing performance
    processing_stats: ProcessingStats,
}

impl OrderBookProcessor {
    /// Create a new order book processor
    pub fn new(symbol: String, exchange: String, max_depth: usize) -> Self {
        let now = Utc::now();
        Self {
            symbol: symbol.clone(),
            exchange: exchange.clone(),
            book: OrderBookData::new(symbol, exchange, now),
            update_queue: VecDeque::new(),
            max_depth,
            last_update_time: now,
            last_sequence: 0,
            processing_stats: ProcessingStats::default(),
        }
    }
    
    /// Process a batch of updates to the order book
    pub fn process_updates(&mut self, updates: Vec<OrderBookUpdate>) -> ProcessorResult<Duration> {
        if updates.is_empty() {
            return Ok(Duration::from_micros(0));
        }
        
        // Start timing the processing
        let start_time = Instant::now();
        
        // Add all updates to the queue
        for update in updates {
            self.update_queue.push_back(update);
        }
        
        // Process the queue
        let mut updates_processed = 0;
        let mut levels_added = 0;
        let mut levels_modified = 0;
        let mut levels_removed = 0;
        
        while let Some(update) = self.update_queue.pop_front() {
            // Check sequence number (if we care about order)
            if update.sequence < self.last_sequence {
                // This is an old update, skip it
                continue;
            }
            
            // Update the book
            match update.side {
                TradeSide::Buy => {
                    if update.quantity == dec!(0) {
                        // Remove price level
                        if self.book.remove_bid(update.price) {
                            levels_removed += 1;
                        }
                    } else {
                        // Add or update price level
                        if self.book.has_bid(update.price) {
                            self.book.update_bid(update.price, update.quantity);
                            levels_modified += 1;
                        } else {
                            self.book.add_bid(update.price, update.quantity);
                            levels_added += 1;
                        }
                    }
                },
                TradeSide::Sell => {
                    if update.quantity == dec!(0) {
                        // Remove price level
                        if self.book.remove_ask(update.price) {
                            levels_removed += 1;
                        }
                    } else {
                        // Add or update price level
                        if self.book.has_ask(update.price) {
                            self.book.update_ask(update.price, update.quantity);
                            levels_modified += 1;
                        } else {
                            self.book.add_ask(update.price, update.quantity);
                            levels_added += 1;
                        }
                    }
                }
            }
            
            // Update metadata
            self.last_update_time = update.timestamp;
            self.last_sequence = update.sequence;
            updates_processed += 1;
        }
        
        // Enforce max depth
        self.book.truncate_bids(self.max_depth);
        self.book.truncate_asks(self.max_depth);
        
        // Calculate processing time
        let processing_time = start_time.elapsed();
        let processing_time_us = processing_time.as_micros() as u64;
        
        // Update statistics
        self.update_processing_stats(
            updates_processed,
            levels_added,
            levels_modified,
            levels_removed,
            processing_time_us
        );
        
        Ok(processing_time)
    }
    
    /// Calculate market impact for a given order size
    pub fn calculate_market_impact(&self, side: TradeSide, size: Decimal) -> MarketImpact {
        let mut remaining_size = size;
        let mut total_value = dec!(0);
        let mut levels_consumed = 0;
        
        // Price levels to iterate over
        let levels = match side {
            // For a buy order, we consume ask levels (starting from lowest ask)
            TradeSide::Buy => self.book.asks(),
            // For a sell order, we consume bid levels (starting from highest bid)
            TradeSide::Sell => self.book.bids(),
        };
        
        // For sells, we iterate in reverse (highest to lowest)
        let mut price_levels: Vec<&PriceLevel> = levels.iter().collect();
        if side == TradeSide::Sell {
            price_levels.reverse();
        }
        
        for level in price_levels {
            levels_consumed += 1;
            
            let level_quantity = level.quantity;
            let level_price = level.price;
            
            if level_quantity >= remaining_size {
                // This level can fully fill the remaining size
                total_value += level_price * remaining_size;
                remaining_size = dec!(0);
                break;
            } else {
                // Partial fill from this level
                total_value += level_price * level_quantity;
                remaining_size -= level_quantity;
            }
        }
        
        // Calculate fillable quantity
        let fillable_quantity = size - remaining_size;
        
        // Calculate average price
        let avg_price = if fillable_quantity > dec!(0) {
            total_value / fillable_quantity
        } else {
            dec!(0)
        };
        
        // Calculate slippage as a percentage of the best price
        let slippage_pct = if fillable_quantity > dec!(0) {
            let best_price = match side {
                TradeSide::Buy => self.best_ask_price(),
                TradeSide::Sell => self.best_bid_price(),
            };
            
            if best_price == dec!(0) {
                dec!(0)
            } else {
                let price_diff = match side {
                    TradeSide::Buy => avg_price - best_price,
                    TradeSide::Sell => best_price - avg_price,
                };
                
                (price_diff * dec!(100)) / best_price
            }
        } else {
            dec!(0)
        };
        
        MarketImpact {
            avg_price,
            slippage_pct,
            total_value,
            fillable_quantity,
            levels_consumed,
        }
    }
    
    /// Get the best bid price
    pub fn best_bid_price(&self) -> Decimal {
        if let Some(best_bid) = self.book.bids().first() {
            best_bid.price
        } else {
            dec!(0)
        }
    }
    
    /// Get the best ask price
    pub fn best_ask_price(&self) -> Decimal {
        if let Some(best_ask) = self.book.asks().first() {
            best_ask.price
        } else {
            dec!(0)
        }
    }
    
    /// Get the mid price
    pub fn mid_price(&self) -> Decimal {
        let best_bid = self.best_bid_price();
        let best_ask = self.best_ask_price();
        
        if best_bid > dec!(0) && best_ask > dec!(0) {
            (best_bid + best_ask) / dec!(2)
        } else if best_bid > dec!(0) {
            best_bid
        } else if best_ask > dec!(0) {
            best_ask
        } else {
            dec!(0)
        }
    }
    
    /// Get the current bid-ask spread
    pub fn spread(&self) -> Decimal {
        let best_bid = self.best_bid_price();
        let best_ask = self.best_ask_price();
        
        if best_bid > dec!(0) && best_ask > dec!(0) {
            best_ask - best_bid
        } else {
            dec!(0)
        }
    }
    
    /// Get the current bid-ask spread as a percentage of the mid price
    pub fn spread_pct(&self) -> Decimal {
        let spread = self.spread();
        let mid = self.mid_price();
        
        if mid > dec!(0) {
            (spread * dec!(100)) / mid
        } else {
            dec!(0)
        }
    }
    
    /// Calculate the volume-weighted average price (VWAP) for a given side and depth
    pub fn vwap(&self, side: TradeSide, depth: usize) -> Decimal {
        let levels = match side {
            TradeSide::Buy => self.book.bids(),
            TradeSide::Sell => self.book.asks(),
        };
        
        let mut total_value = dec!(0);
        let mut total_volume = dec!(0);
        
        for (i, level) in levels.iter().enumerate() {
            if i >= depth {
                break;
            }
            
            total_value += level.price * level.quantity;
            total_volume += level.quantity;
        }
        
        if total_volume > dec!(0) {
            total_value / total_volume
        } else {
            dec!(0)
        }
    }
    
    /// Calculate the total liquidity available up to a given price depth
    pub fn liquidity_up_to(&self, side: TradeSide, price_depth: Decimal) -> Decimal {
        let (from_price, levels) = match side {
            TradeSide::Buy => {
                let best_bid = self.best_bid_price();
                (best_bid, self.book.bids())
            },
            TradeSide::Sell => {
                let best_ask = self.best_ask_price();
                (best_ask, self.book.asks())
            }
        };
        
        let mut total_liquidity = dec!(0);
        
        for level in levels {
            let price_diff = match side {
                TradeSide::Buy => from_price - level.price,
                TradeSide::Sell => level.price - from_price,
            };
            
            if price_diff <= price_depth {
                total_liquidity += level.quantity;
            } else {
                break;
            }
        }
        
        total_liquidity
    }
    
    /// Detect order book imbalance (ratio of buy to sell liquidity)
    pub fn book_imbalance(&self, depth: usize) -> Decimal {
        let mut bid_volume = dec!(0);
        let mut ask_volume = dec!(0);
        
        for (i, level) in self.book.bids().iter().enumerate() {
            if i >= depth {
                break;
            }
            bid_volume += level.quantity;
        }
        
        for (i, level) in self.book.asks().iter().enumerate() {
            if i >= depth {
                break;
            }
            ask_volume += level.quantity;
        }
        
        if ask_volume > dec!(0) {
            bid_volume / ask_volume
        } else if bid_volume > dec!(0) {
            dec!(10) // Arbitrary large number indicating strong bid imbalance
        } else {
            dec!(1) // No imbalance if both are zero
        }
    }
    
    /// Get a snapshot of the current order book
    pub fn snapshot(&self) -> &OrderBookData {
        &self.book
    }
    
    /// Get processing statistics
    pub fn processing_stats(&self) -> &ProcessingStats {
        &self.processing_stats
    }
    
    /// Reset the order book
    pub fn reset(&mut self) {
        let now = Utc::now();
        self.book = OrderBookData::new(self.symbol.clone(), self.exchange.clone(), now);
        self.update_queue.clear();
        self.last_update_time = now;
        self.last_sequence = 0;
        self.processing_stats = ProcessingStats::default();
    }
    
    // Private method to update processing statistics
    fn update_processing_stats(
        &mut self,
        updates_processed: u64,
        levels_added: u64,
        levels_modified: u64,
        levels_removed: u64,
        processing_time_us: u64
    ) {
        // Update total counters
        self.processing_stats.updates_processed += updates_processed;
        self.processing_stats.levels_added += levels_added;
        self.processing_stats.levels_modified += levels_modified;
        self.processing_stats.levels_removed += levels_removed;
        
        // Update timing statistics
        if self.processing_stats.min_processing_time_us == 0 || processing_time_us < self.processing_stats.min_processing_time_us {
            self.processing_stats.min_processing_time_us = processing_time_us;
        }
        
        if processing_time_us > self.processing_stats.max_processing_time_us {
            self.processing_stats.max_processing_time_us = processing_time_us;
        }
        
        // Exponential moving average for processing time
        if self.processing_stats.avg_processing_time_us == 0.0 {
            self.processing_stats.avg_processing_time_us = processing_time_us as f64;
        } else {
            const ALPHA: f64 = 0.05; // Smoothing factor
            self.processing_stats.avg_processing_time_us = 
                ALPHA * (processing_time_us as f64) + 
                (1.0 - ALPHA) * self.processing_stats.avg_processing_time_us;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper to create a test order book update
    fn create_update(price: Decimal, side: TradeSide, quantity: Decimal, sequence: u64) -> OrderBookUpdate {
        OrderBookUpdate {
            price,
            side,
            quantity,
            timestamp: Utc::now(),
            sequence,
        }
    }
    
    #[test]
    fn test_process_updates() {
        let mut processor = OrderBookProcessor::new("BTC/USD".to_string(), "test_exchange".to_string(), 10);
        
        // Create some test updates
        let updates = vec![
            create_update(dec!(10000), TradeSide::Buy, dec!(1.5), 1),
            create_update(dec!(10001), TradeSide::Buy, dec!(2.5), 2),
            create_update(dec!(10002), TradeSide::Sell, dec!(1.0), 3),
            create_update(dec!(10003), TradeSide::Sell, dec!(2.0), 4),
        ];
        
        // Process the updates
        let result = processor.process_updates(updates);
        assert!(result.is_ok());
        
        // Check the book state
        let snapshot = processor.snapshot();
        assert_eq!(snapshot.bids().len(), 2);
        assert_eq!(snapshot.asks().len(), 2);
        
        // Check best bid and ask
        assert_eq!(processor.best_bid_price(), dec!(10001));
        assert_eq!(processor.best_ask_price(), dec!(10002));
        
        // Check mid price
        assert_eq!(processor.mid_price(), dec!(10001.5));
        
        // Check spread
        assert_eq!(processor.spread(), dec!(1));
    }
    
    #[test]
    fn test_market_impact() {
        let mut processor = OrderBookProcessor::new("BTC/USD".to_string(), "test_exchange".to_string(), 10);
        
        // Create a test book with multiple levels
        let updates = vec![
            // Bids
            create_update(dec!(9995), TradeSide::Buy, dec!(5.0), 1),
            create_update(dec!(9996), TradeSide::Buy, dec!(4.0), 2),
            create_update(dec!(9997), TradeSide::Buy, dec!(3.0), 3),
            create_update(dec!(9998), TradeSide::Buy, dec!(2.0), 4),
            create_update(dec!(9999), TradeSide::Buy, dec!(1.0), 5),
            
            // Asks
            create_update(dec!(10000), TradeSide::Sell, dec!(1.0), 6),
            create_update(dec!(10001), TradeSide::Sell, dec!(2.0), 7),
            create_update(dec!(10002), TradeSide::Sell, dec!(3.0), 8),
            create_update(dec!(10003), TradeSide::Sell, dec!(4.0), 9),
            create_update(dec!(10004), TradeSide::Sell, dec!(5.0), 10),
        ];
        
        // Process the updates
        processor.process_updates(updates).unwrap();
        
        // Test market buy that consumes multiple levels
        let impact = processor.calculate_market_impact(TradeSide::Buy, dec!(3.5));
        assert_eq!(impact.levels_consumed, 3);
        assert_eq!(impact.fillable_quantity, dec!(3.5));
        
        // Expected average price = (1.0*10000 + 2.0*10001 + 0.5*10002) / 3.5
        let expected_avg_price = dec!(10000.857);
        assert!(
            (impact.avg_price - expected_avg_price).abs() < dec!(0.1),
            "Expected avg price around {}, got {}",
            expected_avg_price,
            impact.avg_price
        );
        
        // Test market sell that consumes multiple levels
        let impact = processor.calculate_market_impact(TradeSide::Sell, dec!(4.5));
        assert_eq!(impact.levels_consumed, 3);
        assert_eq!(impact.fillable_quantity, dec!(4.5));
    }
} 