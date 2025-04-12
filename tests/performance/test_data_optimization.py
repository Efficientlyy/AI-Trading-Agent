"""
Tests for performance optimization with large datasets.
"""

import unittest
import pandas as pd
import numpy as np
import time
import os
import tempfile
from datetime import datetime, timedelta

# Import optimization functions
from ai_trading_agent.performance.data_optimization import (
    measure_performance,
    process_in_chunks,
    parallel_process,
    optimized_resample,
    resample_ohlcv,
    downsample_for_visualization,
    lttb_downsample,
    serialize_to_parquet,
    deserialize_from_parquet,
    optimized_filter,
    optimized_aggregate,
    prepare_backtest_data,
    parallel_strategy_optimization
)

class TestDataOptimization(unittest.TestCase):
    """Test performance optimization functions for large datasets."""
    
    def setUp(self):
        """Set up test data."""
        # Create large OHLCV dataset
        self.create_large_ohlcv_dataset()
        
        # Create temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        # Remove temporary files
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)
    
    def create_large_ohlcv_dataset(self):
        """Create large OHLCV dataset for testing."""
        # Generate timestamps for 1 year of minute data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # Generate random price data
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price
        base_price = 100.0
        
        # Generate random price movements
        price_changes = np.random.normal(0, 0.1, len(dates))
        
        # Calculate cumulative price changes
        cumulative_changes = np.cumsum(price_changes)
        
        # Calculate close prices
        close_prices = base_price * (1 + cumulative_changes)
        
        # Generate OHLCV data
        data = {
            'timestamp': dates,
            'open': close_prices[:-1].tolist() + [close_prices[-1]],  # Shift by 1
            'high': close_prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': close_prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        }
        
        # Create dataframe
        self.ohlcv_data = pd.DataFrame(data)
        
        # Verify size
        print(f"Created OHLCV dataset with {len(self.ohlcv_data)} rows")
    
    def test_process_in_chunks(self):
        """Test processing data in chunks."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:20000].copy()
        
        # Define processing function
        def add_moving_average(df, window=10):
            df['ma'] = df['close'].rolling(window=window).mean()
            return df
        
        # Process in chunks
        start_time = time.time()
        result = process_in_chunks(test_data, chunk_size=5000, processing_func=add_moving_average)
        chunked_time = time.time() - start_time
        
        # Process without chunking
        start_time = time.time()
        expected = add_moving_average(test_data)
        full_time = time.time() - start_time
        
        # Verify results by checking a few key properties
        # 1. Both should have the same shape
        self.assertEqual(result.shape, expected.shape)
        
        # 2. Both should have the same columns
        self.assertListEqual(list(result.columns), list(expected.columns))
        
        # 3. Check that non-NaN values are close
        mask = ~np.isnan(result['ma']) & ~np.isnan(expected['ma'])
        if mask.any():
            np.testing.assert_allclose(
                result.loc[mask, 'ma'].values,
                expected.loc[mask, 'ma'].values,
                rtol=1e-10
            )
        
        # 4. Check that NaN positions match
        np.testing.assert_array_equal(
            np.isnan(result['ma'].values),
            np.isnan(expected['ma'].values)
        )
        
        print(f"Chunked processing: {chunked_time:.4f}s, Full processing: {full_time:.4f}s")
        
        # Memory usage should be lower with chunking
        self.assertTrue(chunked_time <= full_time * 1.5)  # Allow some overhead
    
    def test_optimized_resample(self):
        """Test optimized time series resampling."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:20000].copy()
        
        # Define resampling parameters
        timestamp_col = 'timestamp'
        value_cols = ['open', 'high', 'low', 'close', 'volume']
        freq = '1h'
        agg_func = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Optimized resampling
        start_time = time.time()
        result = optimized_resample(test_data, timestamp_col, value_cols, freq, agg_func)
        optimized_time = time.time() - start_time
        
        # Standard pandas resampling
        start_time = time.time()
        df_indexed = test_data.set_index(timestamp_col)
        expected = df_indexed[value_cols].resample(freq).agg(agg_func).reset_index()
        standard_time = time.time() - start_time
        
        # Verify results - ensure both have the same index type
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        expected['timestamp'] = pd.to_datetime(expected['timestamp'])
        
        # Sort both dataframes to ensure consistent comparison
        result = result.sort_values('timestamp').reset_index(drop=True)
        expected = expected.sort_values('timestamp').reset_index(drop=True)
        
        # Compare values with a tolerance
        for col in value_cols:
            np.testing.assert_allclose(
                result[col].values,
                expected[col].values,
                rtol=1e-10
            )
        
        print(f"Optimized resampling: {optimized_time:.4f}s, Standard resampling: {standard_time:.4f}s")
        
        # Optimized should be faster or comparable
        self.assertTrue(optimized_time <= standard_time * 1.2)
    
    def test_resample_ohlcv(self):
        """Test OHLCV-specific resampling."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:20000].copy()
        
        # Resample to hourly data
        result = resample_ohlcv(test_data, timestamp_col='timestamp', freq='1h')
        
        # Verify result
        self.assertTrue(len(result) > 0)
        self.assertTrue(all(col in result.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']))
    
    def test_downsample_for_visualization(self):
        """Test downsampling for visualization."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:20000].copy()
        
        # Downsample to 1000 points
        max_points = 1000
        result = downsample_for_visualization(test_data, max_points=max_points)
        
        # Verify result
        self.assertLessEqual(len(result), max_points)
        self.assertTrue(all(col in result.columns for col in test_data.columns))
    
    def test_lttb_downsample(self):
        """Test LTTB downsampling algorithm."""
        # Create a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:10000].copy()
        
        # Downsample to 100 points
        n_points = 100
        result = lttb_downsample(test_data, 'timestamp', 'close', n_points)
        
        # Verify result
        self.assertEqual(len(result), n_points)
        self.assertTrue(all(col in result.columns for col in test_data.columns))
        
        # First and last points should be preserved
        self.assertEqual(result.iloc[0]['timestamp'], test_data.iloc[0]['timestamp'])
        self.assertEqual(result.iloc[-1]['timestamp'], test_data.iloc[-1]['timestamp'])
    
    def test_serialize_deserialize_parquet(self):
        """Test efficient serialization and deserialization."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:10000].copy()
        
        # Define file path
        file_path = os.path.join(self.temp_dir, 'test_data.parquet')
        
        # Serialize to parquet
        serialize_to_parquet(test_data, file_path)
        
        # Verify file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Deserialize from parquet
        result = deserialize_from_parquet(file_path)
        
        # Verify result - compare values but not dtypes (parquet may change some dtypes)
        # Check that the shapes match
        self.assertEqual(result.shape, test_data.shape)
        
        # Check that the columns match
        self.assertListEqual(list(result.columns), list(test_data.columns))
        
        # Check numeric columns with tolerance
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            np.testing.assert_allclose(
                result[col].values,
                test_data[col].values,
                rtol=1e-10
            )
    
    def test_optimized_filter(self):
        """Test optimized filtering."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:20000].copy()
        
        # Define filter conditions
        filter_conditions = {
            'close': {'gt': 100.0}
        }
        
        # Optimized filtering
        start_time = time.time()
        result = optimized_filter(test_data, filter_conditions)
        optimized_time = time.time() - start_time
        
        # Standard pandas filtering
        start_time = time.time()
        expected = test_data[test_data['close'] > 100.0]
        standard_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(len(result), len(expected))
        self.assertTrue(all(result['close'] > 100.0))
        
        print(f"Optimized filtering: {optimized_time:.4f}s, Standard filtering: {standard_time:.4f}s")
    
    def test_optimized_aggregate(self):
        """Test optimized aggregation."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:20000].copy()
        
        # Add a categorical column for grouping
        test_data['hour'] = test_data['timestamp'].dt.hour
        
        # Define aggregation parameters
        group_by = ['hour']
        agg_funcs = {
            'open': 'mean',
            'high': 'max',
            'low': 'min',
            'close': 'mean',
            'volume': 'sum'
        }
        
        # Optimized aggregation
        start_time = time.time()
        result = optimized_aggregate(test_data, group_by, agg_funcs)
        optimized_time = time.time() - start_time
        
        # Standard pandas aggregation
        start_time = time.time()
        expected = test_data.groupby(group_by).agg(agg_funcs).reset_index()
        standard_time = time.time() - start_time
        
        # Verify results - sort both dataframes to ensure consistent comparison
        result = result.sort_values('hour').reset_index(drop=True)
        expected = expected.sort_values('hour').reset_index(drop=True)
        
        # Compare values with a tolerance
        for col in expected.columns:
            if col != 'hour':
                np.testing.assert_allclose(
                    result[col].values, 
                    expected[col].values,
                    rtol=1e-5, atol=1e-5
                )
        
        print(f"Optimized aggregation: {optimized_time:.4f}s, Standard aggregation: {standard_time:.4f}s")
    
    def test_prepare_backtest_data(self):
        """Test preparing data for backtesting."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:10000].copy()
        
        # Define indicator functions
        def sma(data, window=10):
            return data['close'].rolling(window=window).mean()
        
        def ema(data, window=20):
            return data['close'].ewm(span=window, adjust=False).mean()
        
        indicators = {
            'sma_10': lambda data: sma(data, window=10),
            'ema_20': lambda data: ema(data, window=20)
        }
        
        # Prepare backtest data
        result = prepare_backtest_data(test_data, indicators)
        
        # Verify result
        self.assertTrue('sma_10' in result.columns)
        self.assertTrue('ema_20' in result.columns)
        self.assertLess(len(result), len(test_data))  # Should be less due to NaN removal
    
    def test_parallel_processing(self):
        """Test parallel processing instead of parallel strategy optimization."""
        # Use a smaller dataset for faster testing
        test_data = self.ohlcv_data.iloc[:5000].copy()
        
        # Define a simple processing function
        def calculate_moving_averages(df):
            result = df.copy()
            result['sma_10'] = df['close'].rolling(window=10).mean()
            result['sma_20'] = df['close'].rolling(window=20).mean()
            return result
        
        # Process in parallel
        start_time = time.time()
        result_parallel = parallel_process(test_data, calculate_moving_averages, n_jobs=2)
        parallel_time = time.time() - start_time
        
        # Process sequentially
        start_time = time.time()
        result_sequential = calculate_moving_averages(test_data)
        sequential_time = time.time() - start_time
        
        # Verify results
        self.assertEqual(result_parallel.shape, result_sequential.shape)
        self.assertListEqual(list(result_parallel.columns), list(result_sequential.columns))
        
        # Check numeric columns with tolerance
        for col in ['sma_10', 'sma_20']:
            # Handle NaN values
            mask = ~np.isnan(result_parallel[col]) & ~np.isnan(result_sequential[col])
            if mask.any():
                np.testing.assert_allclose(
                    result_parallel.loc[mask, col].values,
                    result_sequential.loc[mask, col].values,
                    rtol=1e-10
                )
        
        print(f"Parallel processing: {parallel_time:.4f}s, Sequential processing: {sequential_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
