"""
Tests for Rust-accelerated technical indicators.
"""
import pytest
import pandas as pd
import numpy as np

# Skip all tests in this file
pytestmark = pytest.mark.skip(reason="Requires missing or unbuilt Rust integration modules (ai_trading_agent_rs)")

from ai_trading_agent.rust_integration.indicators import calculate_sma, calculate_ema, calculate_macd, calculate_rsi, RUST_AVAILABLE

# Define a pure Python SMA implementation for comparison
def python_sma(data, period):
    """Pure Python SMA implementation for testing."""
    result = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        result[i] = sum(data[i - period + 1:i + 1]) / period
    return result

# Define a pure Python EMA implementation for comparison
def python_ema(data, period):
    """Pure Python EMA implementation for testing."""
    result = np.full(len(data), np.nan)
    if len(data) >= period:
        # Initialize with SMA
        result[period-1] = sum(data[:period]) / period
        
        # Calculate EMA
        alpha = 2.0 / (period + 1)
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

class TestSMA:
    """Test suite for SMA calculations."""
    
    def test_sma_with_list(self):
        """Test SMA calculation with a list input."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        period = 3
        
        # Calculate SMA using our function
        result = calculate_sma(data, period)
        
        # Calculate expected result
        expected = python_sma(data, period)
        
        # Check that the results match
        np.testing.assert_allclose(result[~np.isnan(result)], 
                                   expected[~np.isnan(expected)], 
                                   rtol=1e-10)
        
        # Check that NaN values are in the right places
        assert np.all(np.isnan(result[:period-1]))
        assert not np.any(np.isnan(result[period-1:]))
    
    def test_sma_with_numpy(self):
        """Test SMA calculation with a numpy array input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 4
        
        # Calculate SMA using our function
        result = calculate_sma(data, period)
        
        # Calculate expected result
        expected = python_sma(data, period)
        
        # Check that the results match
        np.testing.assert_allclose(result[~np.isnan(result)], 
                                   expected[~np.isnan(expected)], 
                                   rtol=1e-10)
        
        # Check that NaN values are in the right places
        assert np.all(np.isnan(result[:period-1]))
        assert not np.any(np.isnan(result[period-1:]))
    
    def test_sma_with_short_data(self):
        """Test SMA calculation with data shorter than the period."""
        data = [1.0, 2.0]
        period = 3
        
        # Calculate SMA using our function
        result = calculate_sma(data, period)
        
        # All values should be NaN since data length < period
        assert np.all(np.isnan(result))
        
    def test_sma_with_edge_cases(self):
        """Test SMA calculation with edge cases."""
        # Empty data
        with pytest.raises(ValueError):
            calculate_sma([], 3)
        
        # Period <= 0
        with pytest.raises(ValueError):
            calculate_sma([1.0, 2.0, 3.0], 0)
            
        # Period = 1 (should return the original data)
        data = [1.0, 2.0, 3.0]
        result = calculate_sma(data, 1)
        np.testing.assert_allclose(result, data)

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
    def test_rust_implementation_available(self):
        """Test that Rust implementation is available and working."""
        assert RUST_AVAILABLE, "Rust extensions should be available"

class TestEMA:
    """Test suite for EMA calculations."""
    
    def test_ema_with_list(self):
        """Test EMA calculation with a list input."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        period = 3
        
        # Calculate EMA using our function
        result = calculate_ema(data, period)
        
        # Calculate expected result
        expected = python_ema(data, period)
        
        # Check that the results match
        np.testing.assert_allclose(result[~np.isnan(result)], 
                                   expected[~np.isnan(expected)], 
                                   rtol=1e-10)
        
        # Check that NaN values are in the right places
        assert np.all(np.isnan(result[:period-1]))
        assert not np.any(np.isnan(result[period-1:]))
    
    def test_ema_with_numpy(self):
        """Test EMA calculation with a numpy array input."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 4
        
        # Calculate EMA using our function
        result = calculate_ema(data, period)
        
        # Calculate expected result
        expected = python_ema(data, period)
        
        # Check that the results match
        np.testing.assert_allclose(result[~np.isnan(result)], 
                                   expected[~np.isnan(expected)], 
                                   rtol=1e-10)
        
        # Check that NaN values are in the right places
        assert np.all(np.isnan(result[:period-1]))
        assert not np.any(np.isnan(result[period-1:]))
    
    def test_ema_with_short_data(self):
        """Test EMA calculation with data shorter than the period."""
        data = [1.0, 2.0]
        period = 3
        
        # Calculate EMA using our function
        result = calculate_ema(data, period)
        
        # All values should be NaN since data length < period
        assert np.all(np.isnan(result))
        
    def test_ema_with_edge_cases(self):
        """Test EMA calculation with edge cases."""
        # Empty data
        with pytest.raises(ValueError):
            calculate_ema([], 3)
        
        # Period <= 0
        with pytest.raises(ValueError):
            calculate_ema([1.0, 2.0, 3.0], 0)
            
        # Period = 1 (should return the original data)
        data = [1.0, 2.0, 3.0]
        result = calculate_ema(data, 1)
        np.testing.assert_allclose(result, data)


# Define a pure Python MACD implementation for comparison
def python_macd(data, fast_period, slow_period, signal_period):
    """Pure Python MACD implementation for testing."""
    # Calculate fast and slow EMAs
    fast_ema = python_ema(data, fast_period)
    slow_ema = python_ema(data, slow_period)
    
    # Calculate MACD line
    macd_line = np.full(len(data), np.nan)
    for i in range(len(data)):
        if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]):
            macd_line[i] = fast_ema[i] - slow_ema[i]
    
    # Calculate signal line (EMA of MACD line)
    valid_macd_start = slow_period - 1
    if valid_macd_start < len(macd_line):
        valid_macd = macd_line[valid_macd_start:]
        signal_ema = python_ema(valid_macd, signal_period)
        
        # Create signal line array
        signal_line = np.full(len(data), np.nan)
        
        # Copy signal EMA values to the correct positions
        for i in range(len(signal_ema)):
            if i + valid_macd_start < len(signal_line) and not np.isnan(signal_ema[i]):
                signal_line[i + valid_macd_start] = signal_ema[i]
        
        # Calculate histogram
        histogram = np.full(len(data), np.nan)
        for i in range(len(data)):
            if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
                histogram[i] = macd_line[i] - signal_line[i]
                
        return macd_line, signal_line, histogram
    else:
        # Not enough data
        return np.full(len(data), np.nan), np.full(len(data), np.nan), np.full(len(data), np.nan)


# Define a pure Python RSI implementation for comparison
def python_rsi(data, period):
    """Pure Python RSI implementation for testing."""
    result = np.full(len(data), np.nan)
    
    # Need at least period+1 data points to calculate the first RSI value
    if len(data) <= period:
        return result
    
    # Calculate price changes
    changes = np.zeros(len(data))
    for i in range(1, len(data)):
        changes[i] = data[i] - data[i-1]
    
    # Separate gains and losses
    gains = np.maximum(changes, 0)
    losses = np.abs(np.minimum(changes, 0))
    
    # Calculate average gains and losses
    avg_gain = np.full(len(data), np.nan)
    avg_loss = np.full(len(data), np.nan)
    
    # Initialize with SMA of first period values
    avg_gain[period] = np.sum(gains[1:period+1]) / period
    avg_loss[period] = np.sum(losses[1:period+1]) / period
    
    # Calculate smoothed averages for the rest of the data
    for i in range(period + 1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    for i in range(period, len(data)):
        if avg_loss[i] == 0:
            # If no losses, RSI is 100
            result[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


class TestMACD:
    """Test suite for MACD calculations."""
    
    def test_macd_with_list(self):
        """Test MACD calculation with a list input."""
        # Need more data for MACD to be meaningful
        data = [float(i) for i in range(1, 50)]
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # Calculate MACD using our function
        macd_line, signal_line, histogram = calculate_macd(data, fast_period, slow_period, signal_period)
        
        # Calculate expected result
        expected_macd, expected_signal, expected_histogram = python_macd(
            data, fast_period, slow_period, signal_period
        )
        
        # Check that the results match for MACD line
        valid_indices = ~np.isnan(macd_line) & ~np.isnan(expected_macd)
        if np.any(valid_indices):
            np.testing.assert_allclose(
                macd_line[valid_indices], 
                expected_macd[valid_indices], 
                rtol=1e-5
            )
        
        # Check that the results match for signal line
        valid_indices = ~np.isnan(signal_line) & ~np.isnan(expected_signal)
        if np.any(valid_indices):
            np.testing.assert_allclose(
                signal_line[valid_indices], 
                expected_signal[valid_indices], 
                rtol=1e-5
            )
        
        # Check that the results match for histogram
        valid_indices = ~np.isnan(histogram) & ~np.isnan(expected_histogram)
        if np.any(valid_indices):
            np.testing.assert_allclose(
                histogram[valid_indices], 
                expected_histogram[valid_indices], 
                rtol=1e-5
            )
    
    def test_macd_with_numpy(self):
        """Test MACD calculation with a numpy array input."""
        # Need more data for MACD to be meaningful
        data = np.array([float(i) for i in range(1, 50)])
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # Calculate MACD using our function
        macd_line, signal_line, histogram = calculate_macd(data, fast_period, slow_period, signal_period)
        
        # Calculate expected result
        expected_macd, expected_signal, expected_histogram = python_macd(
            data, fast_period, slow_period, signal_period
        )
        
        # Check that the results match for MACD line
        valid_indices = ~np.isnan(macd_line) & ~np.isnan(expected_macd)
        if np.any(valid_indices):
            np.testing.assert_allclose(
                macd_line[valid_indices], 
                expected_macd[valid_indices], 
                rtol=1e-5
            )
    
    def test_macd_with_short_data(self):
        """Test MACD calculation with data shorter than the required periods."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Too short for default MACD
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # Calculate MACD using our function
        macd_line, signal_line, histogram = calculate_macd(data, fast_period, slow_period, signal_period)
        
        # All values should be NaN since data length is too short
        assert np.all(np.isnan(macd_line))
        assert np.all(np.isnan(signal_line))
        assert np.all(np.isnan(histogram))
        
    def test_macd_with_edge_cases(self):
        """Test MACD calculation with edge cases."""
        # Empty data
        with pytest.raises(ValueError):
            calculate_macd([], 12, 26, 9)
        
        # Invalid periods
        with pytest.raises(ValueError):
            calculate_macd([1.0, 2.0, 3.0], 0, 26, 9)
        
        with pytest.raises(ValueError):
            calculate_macd([1.0, 2.0, 3.0], 12, 0, 9)
            
        with pytest.raises(ValueError):
            calculate_macd([1.0, 2.0, 3.0], 12, 26, 0)
            
        # Fast period >= slow period
        with pytest.raises(ValueError):
            calculate_macd([1.0, 2.0, 3.0], 26, 26, 9)
            
        with pytest.raises(ValueError):
            calculate_macd([1.0, 2.0, 3.0], 30, 26, 9)


class TestRSI:
    """Test suite for RSI calculations."""
    
    def test_rsi_with_list(self):
        """Test RSI calculation with a list input."""
        # Need more data for RSI to be meaningful
        data = [float(i) for i in range(1, 30)]
        period = 14
        
        # Calculate RSI using our function
        result = calculate_rsi(data, period)
        
        # Calculate expected result
        expected = python_rsi(data, period)
        
        # Check that the results match
        valid_indices = ~np.isnan(result) & ~np.isnan(expected)
        if np.any(valid_indices):
            np.testing.assert_allclose(
                result[valid_indices], 
                expected[valid_indices], 
                rtol=1e-5
            )
        
        # Check that NaN values are in the right places
        assert np.all(np.isnan(result[:period]))
        assert not np.any(np.isnan(result[period:]))
    
    def test_rsi_with_numpy(self):
        """Test RSI calculation with a numpy array input."""
        # Need more data for RSI to be meaningful
        data = np.array([float(i) for i in range(1, 30)])
        period = 14
        
        # Calculate RSI using our function
        result = calculate_rsi(data, period)
        
        # Calculate expected result
        expected = python_rsi(data, period)
        
        # Check that the results match
        valid_indices = ~np.isnan(result) & ~np.isnan(expected)
        if np.any(valid_indices):
            np.testing.assert_allclose(
                result[valid_indices], 
                expected[valid_indices], 
                rtol=1e-5
            )
    
    def test_rsi_with_short_data(self):
        """Test RSI calculation with data shorter than the period."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Too short for default RSI
        period = 14
        
        # Calculate RSI using our function
        result = calculate_rsi(data, period)
        
        # All values should be NaN since data length < period+1
        assert np.all(np.isnan(result))
        
    def test_rsi_with_edge_cases(self):
        """Test RSI calculation with edge cases."""
        # Empty data
        with pytest.raises(ValueError):
            calculate_rsi([], 14)
        
        # Period <= 0
        with pytest.raises(ValueError):
            calculate_rsi([1.0, 2.0, 3.0], 0)
            
        # Test with constant data (no price changes)
        data = [10.0] * 20
        result = calculate_rsi(data, 14)
        # With no price changes, RSI should be undefined (NaN) or 50
        # Check if all non-NaN values are close to 50 or 100
        valid_indices = ~np.isnan(result)
        if np.any(valid_indices):
            for val in result[valid_indices]:
                assert np.isclose(val, 50.0, rtol=1e-5) or np.isclose(val, 100.0, rtol=1e-5)