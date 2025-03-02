/*
 * Market data models for the Rust trading engine
 */

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Time frame for candlestick data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeFrame {
    #[serde(rename = "1m")]
    Minute1,
    #[serde(rename = "3m")]
    Minute3,
    #[serde(rename = "5m")]
    Minute5,
    #[serde(rename = "15m")]
    Minute15,
    #[serde(rename = "30m")]
    Minute30,
    #[serde(rename = "1h")]
    Hour1,
    #[serde(rename = "2h")]
    Hour2,
    #[serde(rename = "4h")]
    Hour4,
    #[serde(rename = "6h")]
    Hour6,
    #[serde(rename = "12h")]
    Hour12,
    #[serde(rename = "1d")]
    Day1,
    #[serde(rename = "3d")]
    Day3,
    #[serde(rename = "1w")]
    Week1,
    #[serde(rename = "1M")]
    Month1,
}

impl TimeFrame {
    /// Get the duration in seconds
    pub fn duration_seconds(&self) -> i64 {
        match self {
            TimeFrame::Minute1 => 60,
            TimeFrame::Minute3 => 3 * 60,
            TimeFrame::Minute5 => 5 * 60,
            TimeFrame::Minute15 => 15 * 60,
            TimeFrame::Minute30 => 30 * 60,
            TimeFrame::Hour1 => 60 * 60,
            TimeFrame::Hour2 => 2 * 60 * 60,
            TimeFrame::Hour4 => 4 * 60 * 60,
            TimeFrame::Hour6 => 6 * 60 * 60,
            TimeFrame::Hour12 => 12 * 60 * 60,
            TimeFrame::Day1 => 24 * 60 * 60,
            TimeFrame::Day3 => 3 * 24 * 60 * 60,
            TimeFrame::Week1 => 7 * 24 * 60 * 60,
            TimeFrame::Month1 => 30 * 24 * 60 * 60, // Approximation
        }
    }
}

/// Candlestick data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleData {
    pub symbol: String,
    pub exchange: String,
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub timeframe: TimeFrame,
    pub trade_count: Option<u32>,
    pub vwap: Option<Decimal>,    // Volume Weighted Average Price
    pub quote_volume: Option<Decimal>,
}

impl CandleData {
    /// Create a new candlestick
    pub fn new(
        symbol: String,
        exchange: String,
        timestamp: DateTime<Utc>,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        timeframe: TimeFrame,
    ) -> Self {
        Self {
            symbol,
            exchange,
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            timeframe,
            trade_count: None,
            vwap: None,
            quote_volume: None,
        }
    }

    /// Calculate the range (high - low)
    pub fn range(&self) -> Decimal {
        self.high - self.low
    }

    /// Calculate the body size (abs(close - open))
    pub fn body(&self) -> Decimal {
        if self.close > self.open {
            self.close - self.open
        } else {
            self.open - self.close
        }
    }

    /// Is this a bullish candle? (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Is this a bearish candle? (close < open)
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    /// Calculate the body percentage of the total range
    pub fn body_percent(&self) -> Decimal {
        let range = self.range();
        if range.is_zero() {
            return Decimal::ZERO;
        }
        (self.body() / range) * Decimal::from(100)
    }
}

/// Trade data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub id: String,
    pub symbol: String,
    pub exchange: String,
    pub timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub amount: Decimal,
    pub side: TradeSide,
    pub is_liquidation: Option<bool>,
}

/// Trade side (buy or sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradeSide {
    #[serde(rename = "buy")]
    Buy,
    #[serde(rename = "sell")]
    Sell,
}

/// Order book data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookData {
    pub symbol: String,
    pub exchange: String,
    pub timestamp: DateTime<Utc>,
    pub bids: BTreeMap<Decimal, Decimal>, // Price -> Amount (sorted)
    pub asks: BTreeMap<Decimal, Decimal>, // Price -> Amount (sorted)
    pub sequence: Option<u64>,            // Sequence number for updates
}

impl OrderBookData {
    /// Create a new order book
    pub fn new(
        symbol: String,
        exchange: String,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol,
            exchange,
            timestamp,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            sequence: None,
        }
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.keys().next_back().cloned()
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.keys().next().cloned()
    }

    /// Get spread (best ask - best bid)
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get mid price ((best bid + best ask) / 2)
    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / Decimal::from(2)),
            _ => None,
        }
    }

    /// Update the order book with a price level
    pub fn update_level(&mut self, price: Decimal, amount: Decimal, is_bid: bool) {
        let book = if is_bid { &mut self.bids } else { &mut self.asks };
        
        if amount.is_zero() {
            book.remove(&price);
        } else {
            book.insert(price, amount);
        }
    }

    /// Get the total volume up to a certain price
    pub fn volume_up_to(&self, price: Decimal, is_bid: bool) -> Decimal {
        let book = if is_bid { &self.bids } else { &self.asks };
        
        if is_bid {
            book.iter()
                .filter(|(&p, _)| p >= price)
                .map(|(_, &amount)| amount)
                .sum()
        } else {
            book.iter()
                .filter(|(&p, _)| p <= price)
                .map(|(_, &amount)| amount)
                .sum()
        }
    }
}

/// Market ticker data (summary of current market state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    pub exchange: String,
    pub timestamp: DateTime<Utc>,
    pub last_price: Decimal,
    pub bid_price: Option<Decimal>,
    pub ask_price: Option<Decimal>,
    pub volume_24h: Option<Decimal>,
    pub price_change_24h: Option<Decimal>,
    pub price_change_pct_24h: Option<Decimal>,
    pub high_24h: Option<Decimal>,
    pub low_24h: Option<Decimal>,
}

/// Price conversion between currencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceConversion {
    pub from_currency: String,
    pub to_currency: String,
    pub conversion_rate: Decimal,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_candle_calculations() {
        let candle = CandleData::new(
            "BTCUSDT".to_string(),
            "binance".to_string(),
            Utc::now(),
            Decimal::from(10000),
            Decimal::from(10500),
            Decimal::from(9900),
            Decimal::from(10200),
            Decimal::from(100),
            TimeFrame::Minute15,
        );
        
        assert_eq!(candle.range(), Decimal::from(600)); // 10500 - 9900
        assert_eq!(candle.body(), Decimal::from(200));  // 10200 - 10000
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
        
        // Body percent should be 200/600 * 100 = 33.33%
        let expected = Decimal::from_str_exact("33.33333333333333333333333333").unwrap();
        assert!(candle.body_percent() - expected < Decimal::from_str_exact("0.01").unwrap());
    }
    
    #[test]
    fn test_orderbook_functions() {
        let mut orderbook = OrderBookData::new(
            "ETHUSDT".to_string(),
            "binance".to_string(),
            Utc::now(),
        );
        
        // Add some bids and asks
        orderbook.update_level(Decimal::from(1900), Decimal::from(1), true);
        orderbook.update_level(Decimal::from(1890), Decimal::from(2), true);
        orderbook.update_level(Decimal::from(1910), Decimal::from(0.5), true);
        
        orderbook.update_level(Decimal::from(1920), Decimal::from(1), false);
        orderbook.update_level(Decimal::from(1930), Decimal::from(1.5), false);
        orderbook.update_level(Decimal::from(1940), Decimal::from(0.8), false);
        
        assert_eq!(orderbook.best_bid().unwrap(), Decimal::from(1910));
        assert_eq!(orderbook.best_ask().unwrap(), Decimal::from(1920));
        assert_eq!(orderbook.spread().unwrap(), Decimal::from(10));
        assert_eq!(orderbook.mid_price().unwrap(), Decimal::from(1915));
        
        // Test removing levels
        orderbook.update_level(Decimal::from(1910), Decimal::ZERO, true);
        assert_eq!(orderbook.best_bid().unwrap(), Decimal::from(1900));
        
        // Test volume calculation
        assert_eq!(
            orderbook.volume_up_to(Decimal::from(1895), true),
            Decimal::from(3) // 1 + 2
        );
    }
} 