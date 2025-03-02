/*
 * Backtesting module for the Rust trading engine
 * This module provides high-performance backtesting capabilities
 */

pub mod engine;

// Re-export commonly used items
pub use engine::{
    BacktestConfig,
    BacktestEngine,
    BacktestError,
    BacktestMode,
    BacktestOrder,
    BacktestOrderStatus,
    BacktestOrderType,
    BacktestPosition,
    BacktestResult,
    BacktestStats,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_data::{CandleData, TimeFrame, TradeSide};
    use chrono::{Duration, Utc};
    use rust_decimal_macros::dec;
    use uuid::Uuid;
    
    #[test]
    fn test_simple_backtest() {
        // Create a simple backtest configuration
        let config = BacktestConfig {
            initial_balance: dec!(10000),
            symbols: vec!["BTCUSDT".to_string()],
            start_time: Utc::now(),
            end_time: Utc::now() + Duration::days(30),
            mode: BacktestMode::Candles,
            commission_rate: dec!(0.001),
            slippage: dec!(0.0005),
            enable_fractional_sizing: true,
        };
        
        // Create the backtest engine
        let mut engine = BacktestEngine::new(config);
        
        // Create a simple test strategy: buy and hold for 7 days
        let buy_order = BacktestOrder {
            id: Uuid::new_v4().to_string(),
            symbol: "BTCUSDT".to_string(),
            side: TradeSide::Buy,
            order_type: BacktestOrderType::Market,
            price: None,
            amount: dec!(1),
            status: BacktestOrderStatus::Created,
            created_at: Utc::now(),
            executed_at: None,
            executed_price: None,
            stop_price: None,
            take_profit: None,
            stop_loss: None,
        };
        
        // Submit the buy order
        let order_id = engine.submit_order(buy_order).expect("Order submission failed");
        
        // Create a series of candles with increasing prices
        let mut current_time = Utc::now();
        let mut price = dec!(10000);
        
        // Process candles for the first day
        for _ in 0..24 {
            let candle = CandleData::new(
                "BTCUSDT".to_string(),
                "test_exchange".to_string(),
                current_time,
                price,
                price * dec!(1.01),
                price * dec!(0.99),
                price,
                dec!(100),
                TimeFrame::Hour1,
            );
            
            engine.process_candle(&candle).expect("Failed to process candle");
            
            // Increment time and price
            current_time = current_time + Duration::hours(1);
            price = price * dec!(1.001); // Small increase each hour
        }
        
        // Check that the buy order was executed
        assert_eq!(engine.get_open_orders().len(), 0);
        assert_eq!(engine.get_filled_orders().len(), 1);
        
        // Check that we have a position
        assert_eq!(engine.get_positions().len(), 1);
        assert!(engine.get_positions().contains_key("BTCUSDT"));
        
        // Process candles for 6 more days (sell on day 7)
        for day in 1..7 {
            for _ in 0..24 {
                let candle = CandleData::new(
                    "BTCUSDT".to_string(),
                    "test_exchange".to_string(),
                    current_time,
                    price,
                    price * dec!(1.01),
                    price * dec!(0.99),
                    price,
                    dec!(100),
                    TimeFrame::Hour1,
                );
                
                engine.process_candle(&candle).expect("Failed to process candle");
                
                // Increment time and price
                current_time = current_time + Duration::hours(1);
                price = price * dec!(1.001); // Small increase each hour
            }
        }
        
        // Create a sell order on day 7
        let sell_order = BacktestOrder {
            id: Uuid::new_v4().to_string(),
            symbol: "BTCUSDT".to_string(),
            side: TradeSide::Sell,
            order_type: BacktestOrderType::Market,
            price: None,
            amount: dec!(1),
            status: BacktestOrderStatus::Created,
            created_at: current_time,
            executed_at: None,
            executed_price: None,
            stop_price: None,
            take_profit: None,
            stop_loss: None,
        };
        
        // Submit the sell order
        engine.submit_order(sell_order).expect("Order submission failed");
        
        // Process final candle to execute the sell order
        let final_candle = CandleData::new(
            "BTCUSDT".to_string(),
            "test_exchange".to_string(),
            current_time,
            price,
            price * dec!(1.01),
            price * dec!(0.99),
            price,
            dec!(100),
            TimeFrame::Hour1,
        );
        
        engine.process_candle(&final_candle).expect("Failed to process candle");
        
        // Check that the sell order was executed
        assert_eq!(engine.get_open_orders().len(), 0);
        assert_eq!(engine.get_filled_orders().len(), 2);
        
        // Check that position is closed
        assert_eq!(engine.get_positions().len(), 0);
        
        // Check final balance and profit
        let final_balance = engine.get_balance();
        assert!(final_balance > dec!(10000), "No profit was made");
        
        // Get final stats
        let result = engine.run().expect("Failed to get backtest results");
        
        // Check some key metrics
        assert_eq!(result.total_trades, 1);
        assert_eq!(result.winning_trades, 1);
        assert_eq!(result.losing_trades, 0);
        assert!(result.win_rate == dec!(100));
        assert!(result.total_profit > dec!(0));
    }
} 