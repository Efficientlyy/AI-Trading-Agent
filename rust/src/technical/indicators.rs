/*
 * Technical indicators for the Rust trading engine
 */

use rust_decimal::Decimal;
use std::collections::VecDeque;

/// Moving Average types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MAType {
    Simple,
    Exponential,
    Weighted,
}

/// A structure to calculate Simple Moving Average efficiently
pub struct SMA {
    period: usize,
    values: VecDeque<Decimal>,
    sum: Decimal,
}

impl SMA {
    /// Create a new Simple Moving Average calculator
    pub fn new(period: usize) -> Self {
        Self {
            period,
            values: VecDeque::with_capacity(period + 1),
            sum: Decimal::ZERO,
        }
    }

    /// Update the SMA with a new value
    pub fn update(&mut self, value: Decimal) -> Option<Decimal> {
        // Add the new value
        self.values.push_back(value);
        self.sum += value;

        // Remove oldest value if we exceed the period
        if self.values.len() > self.period {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
            }
        }

        // Calculate the average if we have enough values
        if self.values.len() == self.period {
            Some(self.sum / Decimal::from(self.period))
        } else {
            None
        }
    }

    /// Get the current SMA value
    pub fn current(&self) -> Option<Decimal> {
        if self.values.len() == self.period {
            Some(self.sum / Decimal::from(self.period))
        } else {
            None
        }
    }

    /// Get the period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Reset the SMA
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = Decimal::ZERO;
    }
}

/// A structure to calculate Exponential Moving Average efficiently
pub struct EMA {
    period: usize,
    k: Decimal,           // Smoothing factor: 2 / (period + 1)
    current_ema: Option<Decimal>,
    initialized: bool,
    sma: SMA,             // Used for initialization
}

impl EMA {
    /// Create a new Exponential Moving Average calculator
    pub fn new(period: usize) -> Self {
        // Calculate smoothing factor k = 2 / (period + 1)
        let k = Decimal::from(2) / Decimal::from(period + 1);
        
        Self {
            period,
            k,
            current_ema: None,
            initialized: false,
            sma: SMA::new(period),
        }
    }

    /// Update the EMA with a new value
    pub fn update(&mut self, value: Decimal) -> Option<Decimal> {
        if !self.initialized {
            // During initialization, use SMA
            if let Some(sma_value) = self.sma.update(value) {
                self.current_ema = Some(sma_value);
                self.initialized = true;
                return Some(sma_value);
            }
            return None;
        }

        // EMA = Current Price * k + Previous EMA * (1 - k)
        if let Some(prev_ema) = self.current_ema {
            let new_ema = value * self.k + prev_ema * (Decimal::ONE - self.k);
            self.current_ema = Some(new_ema);
            return Some(new_ema);
        }

        None
    }

    /// Get the current EMA value
    pub fn current(&self) -> Option<Decimal> {
        self.current_ema
    }

    /// Get the period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Reset the EMA
    pub fn reset(&mut self) {
        self.current_ema = None;
        self.initialized = false;
        self.sma.reset();
    }
}

/// A structure to calculate Weighted Moving Average efficiently
pub struct WMA {
    period: usize,
    values: VecDeque<Decimal>,
    total_weight: usize,
}

impl WMA {
    /// Create a new Weighted Moving Average calculator
    pub fn new(period: usize) -> Self {
        // Total weight is sum of 1 to period
        let total_weight = (period * (period + 1)) / 2;
        
        Self {
            period,
            values: VecDeque::with_capacity(period + 1),
            total_weight,
        }
    }

    /// Update the WMA with a new value
    pub fn update(&mut self, value: Decimal) -> Option<Decimal> {
        // Add the new value
        self.values.push_back(value);

        // Remove oldest value if we exceed the period
        if self.values.len() > self.period {
            self.values.pop_front();
        }

        // Calculate the weighted average if we have enough values
        if self.values.len() == self.period {
            let mut sum = Decimal::ZERO;
            let mut weight = 1;

            for i in 0..self.period {
                let idx = i;  // Oldest values have lowest weight
                sum += self.values[idx] * Decimal::from(weight);
                weight += 1;
            }

            return Some(sum / Decimal::from(self.total_weight));
        }

        None
    }

    /// Get the current WMA value
    pub fn current(&self) -> Option<Decimal> {
        if self.values.len() == self.period {
            let mut sum = Decimal::ZERO;
            let mut weight = 1;

            for i in 0..self.period {
                let idx = i;  // Oldest values have lowest weight
                sum += self.values[idx] * Decimal::from(weight);
                weight += 1;
            }

            Some(sum / Decimal::from(self.total_weight))
        } else {
            None
        }
    }

    /// Get the period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Reset the WMA
    pub fn reset(&mut self) {
        self.values.clear();
    }
}

/// Calculate moving average crossovers
pub struct MACrossover {
    fast_ma: Box<dyn MovingAverage>,
    slow_ma: Box<dyn MovingAverage>,
    last_signal: Option<CrossoverSignal>,
}

/// Crossover signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverSignal {
    BullishCrossover,  // Fast MA crosses above slow MA
    BearishCrossover,  // Fast MA crosses below slow MA
    NoSignal,          // No crossover
}

// Trait for common moving average interface
pub trait MovingAverage {
    fn update(&mut self, value: Decimal) -> Option<Decimal>;
    fn current(&self) -> Option<Decimal>;
    fn reset(&mut self);
    fn period(&self) -> usize;
}

impl MovingAverage for SMA {
    fn update(&mut self, value: Decimal) -> Option<Decimal> {
        SMA::update(self, value)
    }
    
    fn current(&self) -> Option<Decimal> {
        SMA::current(self)
    }
    
    fn reset(&mut self) {
        SMA::reset(self)
    }
    
    fn period(&self) -> usize {
        SMA::period(self)
    }
}

impl MovingAverage for EMA {
    fn update(&mut self, value: Decimal) -> Option<Decimal> {
        EMA::update(self, value)
    }
    
    fn current(&self) -> Option<Decimal> {
        EMA::current(self)
    }
    
    fn reset(&mut self) {
        EMA::reset(self)
    }
    
    fn period(&self) -> usize {
        EMA::period(self)
    }
}

impl MovingAverage for WMA {
    fn update(&mut self, value: Decimal) -> Option<Decimal> {
        WMA::update(self, value)
    }
    
    fn current(&self) -> Option<Decimal> {
        WMA::current(self)
    }
    
    fn reset(&mut self) {
        WMA::reset(self)
    }
    
    fn period(&self) -> usize {
        WMA::period(self)
    }
}

impl MACrossover {
    /// Create a new Moving Average crossover detector
    pub fn new(
        fast_period: usize,
        slow_period: usize,
        fast_type: MAType,
        slow_type: MAType,
    ) -> Self {
        let fast_ma: Box<dyn MovingAverage> = match fast_type {
            MAType::Simple => Box::new(SMA::new(fast_period)),
            MAType::Exponential => Box::new(EMA::new(fast_period)),
            MAType::Weighted => Box::new(WMA::new(fast_period)),
        };
        
        let slow_ma: Box<dyn MovingAverage> = match slow_type {
            MAType::Simple => Box::new(SMA::new(slow_period)),
            MAType::Exponential => Box::new(EMA::new(slow_period)),
            MAType::Weighted => Box::new(WMA::new(slow_period)),
        };
        
        Self {
            fast_ma,
            slow_ma,
            last_signal: None,
        }
    }
    
    /// Update with a new price and check for crossover
    pub fn update(&mut self, price: Decimal) -> CrossoverSignal {
        // Update both MAs
        let fast_value = self.fast_ma.update(price);
        let slow_value = self.slow_ma.update(price);
        
        // Need both values to check crossover
        if let (Some(fast), Some(slow)) = (fast_value, slow_value) {
            let prev_fast = self.fast_ma.current().unwrap();
            let prev_slow = self.slow_ma.current().unwrap();
            
            // Check for bullish crossover (fast crosses above slow)
            if fast > slow && prev_fast <= prev_slow {
                self.last_signal = Some(CrossoverSignal::BullishCrossover);
                return CrossoverSignal::BullishCrossover;
            }
            
            // Check for bearish crossover (fast crosses below slow)
            if fast < slow && prev_fast >= prev_slow {
                self.last_signal = Some(CrossoverSignal::BearishCrossover);
                return CrossoverSignal::BearishCrossover;
            }
        }
        
        self.last_signal = Some(CrossoverSignal::NoSignal);
        CrossoverSignal::NoSignal
    }
    
    /// Get the last crossover signal
    pub fn last_signal(&self) -> Option<CrossoverSignal> {
        self.last_signal
    }
    
    /// Reset the crossover detector
    pub fn reset(&mut self) {
        self.fast_ma.reset();
        self.slow_ma.reset();
        self.last_signal = None;
    }
}

/// Calculate simple moving average for a slice of values
pub fn calculate_sma(values: &[Decimal], period: usize) -> Vec<Decimal> {
    if values.len() < period {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(values.len() - period + 1);
    let mut sum = values.iter().take(period).sum::<Decimal>();
    
    // First SMA
    result.push(sum / Decimal::from(period));
    
    // Remaining SMAs using sliding window
    for i in period..values.len() {
        sum = sum + values[i] - values[i - period];
        result.push(sum / Decimal::from(period));
    }
    
    result
}

/// Calculate exponential moving average for a slice of values
pub fn calculate_ema(values: &[Decimal], period: usize) -> Vec<Decimal> {
    if values.len() < period {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(values.len() - period + 1);
    
    // Start with SMA for the first value
    let first_sma = values.iter().take(period).sum::<Decimal>() / Decimal::from(period);
    result.push(first_sma);
    
    // Calculate the multiplier
    let multiplier = Decimal::from(2) / Decimal::from(period + 1);
    
    // Calculate EMA for the rest of the values
    for i in period..values.len() {
        let previous_ema = result[result.len() - 1];
        let new_ema = values[i] * multiplier + previous_ema * (Decimal::ONE - multiplier);
        result.push(new_ema);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_sma_calculation() {
        let mut sma = SMA::new(3);
        
        // Not enough values yet
        assert_eq!(sma.update(dec!(10)), None);
        assert_eq!(sma.update(dec!(20)), None);
        
        // Now we have enough
        assert_eq!(sma.update(dec!(30)), Some(dec!(20))); // (10 + 20 + 30) / 3 = 20
        assert_eq!(sma.update(dec!(40)), Some(dec!(30))); // (20 + 30 + 40) / 3 = 30
        assert_eq!(sma.update(dec!(50)), Some(dec!(40))); // (30 + 40 + 50) / 3 = 40
        
        // Reset and test again
        sma.reset();
        assert_eq!(sma.update(dec!(10)), None);
    }
    
    #[test]
    fn test_ema_calculation() {
        let mut ema = EMA::new(3);
        
        // Should match SMA for first value
        assert_eq!(ema.update(dec!(10)), None);
        assert_eq!(ema.update(dec!(20)), None);
        assert_eq!(ema.update(dec!(30)), Some(dec!(20))); // First value is SMA = 20
        
        // EMA formula: EMA = Price * (2 / (period + 1)) + EMA(previous) * (1 - (2 / (period + 1)))
        // k = 2 / (3 + 1) = 0.5
        let expected = dec!(40) * dec!(0.5) + dec!(20) * dec!(0.5); // = 30
        assert_eq!(ema.update(dec!(40)), Some(expected));
    }
    
    #[test]
    fn test_crossover_detection() {
        let mut crossover = MACrossover::new(2, 4, MAType::Simple, MAType::Simple);
        
        // Initialize with some values
        assert_eq!(crossover.update(dec!(10)), CrossoverSignal::NoSignal);
        assert_eq!(crossover.update(dec!(20)), CrossoverSignal::NoSignal);
        assert_eq!(crossover.update(dec!(30)), CrossoverSignal::NoSignal);
        
        // Fast MA = (20 + 30) / 2 = 25, Slow MA = (10 + 20 + 30 + 40) / 4 = 25
        // No crossover yet
        assert_eq!(crossover.update(dec!(40)), CrossoverSignal::NoSignal);
        
        // Fast MA = (30 + 40) / 2 = 35, Slow MA = (20 + 30 + 40 + 10) / 4 = 25
        // Fast crosses above slow
        assert_eq!(crossover.update(dec!(10)), CrossoverSignal::BullishCrossover);
        
        // Fast MA = (40 + 10) / 2 = 25, Slow MA = (30 + 40 + 10 + 5) / 4 = 21.25
        // No crossover (fast still above slow)
        assert_eq!(crossover.update(dec!(5)), CrossoverSignal::NoSignal);
        
        // Fast MA = (10 + 5) / 2 = 7.5, Slow MA = (40 + 10 + 5 + 20) / 4 = 18.75
        // Fast crosses below slow
        assert_eq!(crossover.update(dec!(20)), CrossoverSignal::BearishCrossover);
    }
} 