"""
Data Optimization Module for AI Trading Agent.

This module provides optimized data structures and algorithms for efficient data processing,
particularly for large datasets used in backtesting and strategy optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union, AsyncGenerator, Callable
from datetime import datetime, timedelta
import logging
import time
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

@contextmanager
def measure_performance(task_name: str):
    """
    Context manager to measure execution time of a code block.
    
    Args:
        task_name: Name of the task being measured
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{task_name} completed in {execution_time:.4f} seconds")

def process_in_chunks(
    data: pd.DataFrame,
    chunk_size: int,
    processing_func: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """
    Process a large DataFrame in chunks to reduce memory usage.
    
    Args:
        data: Input DataFrame
        chunk_size: Number of rows per chunk
        processing_func: Function to apply to each chunk
        
    Returns:
        Processed DataFrame
    """
    if len(data) <= chunk_size:
        return processing_func(data)
    
    result_chunks = []
    
    # Process data in chunks
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size].copy()
        processed_chunk = processing_func(chunk)
        result_chunks.append(processed_chunk)
    
    # Combine results
    if result_chunks:
        return pd.concat(result_chunks, ignore_index=False)
    else:
        return pd.DataFrame()

def parallel_process(
    data: pd.DataFrame,
    processing_func: Callable[[pd.DataFrame], pd.DataFrame],
    n_jobs: int = -1,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Process a large DataFrame in parallel.
    
    Args:
        data: Input DataFrame
        processing_func: Function to apply to each chunk
        n_jobs: Number of parallel jobs (-1 for all available cores)
        chunk_size: Number of rows per chunk (None for automatic sizing)
        
    Returns:
        Processed DataFrame
    """
    # Determine number of workers
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    # Determine chunk size
    if chunk_size is None:
        chunk_size = max(1, len(data) // n_jobs)
    
    # Create chunks
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data.iloc[i:i+chunk_size].copy())
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(processing_func, chunks))
    
    # Combine results
    if results:
        return pd.concat(results, ignore_index=False)
    else:
        return pd.DataFrame()

def optimized_resample(
    data: pd.DataFrame,
    timestamp_col: str,
    value_cols: List[str],
    freq: str,
    agg_func: Dict[str, str]
) -> pd.DataFrame:
    """
    Optimized time series resampling.
    
    Args:
        data: Input DataFrame
        timestamp_col: Column containing timestamps
        value_cols: Columns to aggregate
        freq: Resampling frequency (e.g., '1h', '1d')
        agg_func: Aggregation functions for each column
        
    Returns:
        Resampled DataFrame
    """
    # Create a copy of the input data with only the required columns
    df = data[[timestamp_col] + value_cols].copy()
    
    # Set timestamp as index
    df = df.set_index(timestamp_col)
    
    # Perform resampling
    resampled = df.resample(freq).agg(agg_func)
    
    # Reset index
    resampled = resampled.reset_index()
    
    return resampled

def resample_ohlcv(
    data: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    freq: str = '1d'
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different frequency.
    
    Args:
        data: Input OHLCV DataFrame
        timestamp_col: Column containing timestamps
        freq: Resampling frequency (e.g., '1h', '1d')
        
    Returns:
        Resampled OHLCV DataFrame
    """
    # Define aggregation functions for OHLCV data
    agg_func = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Get value columns
    value_cols = [col for col in data.columns if col in agg_func]
    
    # Perform resampling
    return optimized_resample(data, timestamp_col, value_cols, freq, agg_func)

def downsample_for_visualization(
    data: pd.DataFrame,
    max_points: int = 1000
) -> pd.DataFrame:
    """
    Downsample data for visualization to reduce browser load.
    
    Args:
        data: Input DataFrame
        max_points: Maximum number of points to return
        
    Returns:
        Downsampled DataFrame
    """
    if len(data) <= max_points:
        return data
    
    # Calculate step size
    step = len(data) // max_points
    
    # Sample data
    return data.iloc[::step].copy()

def lttb_downsample(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_points: int
) -> pd.DataFrame:
    """
    Largest-Triangle-Three-Buckets downsampling algorithm for time series visualization.
    
    Args:
        data: Input DataFrame
        x_col: Column for x-axis (typically timestamp)
        y_col: Column for y-axis (typically price)
        n_points: Number of points to return
        
    Returns:
        Downsampled DataFrame
    """
    if len(data) <= n_points:
        return data
    
    # Make sure data is sorted by x
    data = data.sort_values(by=x_col)
    
    # Convert to numpy arrays for faster processing
    x = data[x_col].astype(np.float64).values
    y = data[y_col].astype(np.float64).values
    
    # Calculate effective x values if x is datetime
    if isinstance(data[x_col].iloc[0], (pd.Timestamp, datetime)):
        x = np.array([(d - data[x_col].iloc[0]).total_seconds() for d in data[x_col]])
    
    # Bucket size
    bucket_size = (len(data) - 2) / (n_points - 2)
    
    # Always include first and last points
    result_indices = [0, len(data) - 1]
    
    # Process buckets
    for i in range(n_points - 2):
        # Bucket range
        start_idx = int((i) * bucket_size) + 1
        end_idx = int((i + 1) * bucket_size) + 1
        
        # Last bucket might be smaller
        if end_idx >= len(data):
            end_idx = len(data) - 1
        
        # Points in current bucket
        bucket_points = range(start_idx, end_idx + 1)
        
        # Get the point from the previous bucket
        prev_x = x[result_indices[-1]]
        prev_y = y[result_indices[-1]]
        
        # Get the point from the next bucket
        next_x = x[end_idx + 1] if end_idx < len(data) - 1 else x[end_idx]
        next_y = y[end_idx + 1] if end_idx < len(data) - 1 else y[end_idx]
        
        # Calculate areas for each point in the bucket
        areas = []
        for idx in bucket_points:
            # Calculate triangle area
            area = 0.5 * abs((prev_x - next_x) * (y[idx] - prev_y) - 
                             (prev_x - x[idx]) * (next_y - prev_y))
            areas.append((idx, area))
        
        # Select the point with the largest area
        if areas:
            max_area_idx = max(areas, key=lambda x: x[1])[0]
            result_indices.append(max_area_idx)
    
    # Sort indices
    result_indices.sort()
    
    # Return downsampled data
    return data.iloc[result_indices].copy()

def serialize_to_parquet(
    data: pd.DataFrame,
    file_path: str,
    compression: str = 'snappy'
) -> None:
    """
    Serialize DataFrame to Parquet format for efficient storage.
    
    Args:
        data: Input DataFrame
        file_path: Path to save the Parquet file
        compression: Compression algorithm (snappy, gzip, brotli, etc.)
    """
    # Convert to PyArrow Table
    table = pa.Table.from_pandas(data)
    
    # Write to Parquet
    pq.write_table(table, file_path, compression=compression)

def deserialize_from_parquet(
    file_path: str
) -> pd.DataFrame:
    """
    Deserialize DataFrame from Parquet format.
    
    Args:
        file_path: Path to the Parquet file
        
    Returns:
        Deserialized DataFrame
    """
    # Read Parquet file
    table = pq.read_table(file_path)
    
    # Convert to Pandas DataFrame
    return table.to_pandas()

def optimized_filter(
    data: pd.DataFrame,
    filter_conditions: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Optimized filtering of DataFrame.
    
    Args:
        data: Input DataFrame
        filter_conditions: Dictionary of filter conditions
            Format: {
                'column1': {'gt': value1, 'lt': value2},
                'column2': {'eq': value3}
            }
        
    Returns:
        Filtered DataFrame
    """
    # Start with all rows
    mask = pd.Series(True, index=data.index)
    
    # Apply each filter condition
    for column, conditions in filter_conditions.items():
        if column not in data.columns:
            continue
        
        for op, value in conditions.items():
            if op == 'eq':
                mask &= (data[column] == value)
            elif op == 'ne':
                mask &= (data[column] != value)
            elif op == 'gt':
                mask &= (data[column] > value)
            elif op == 'lt':
                mask &= (data[column] < value)
            elif op == 'ge':
                mask &= (data[column] >= value)
            elif op == 'le':
                mask &= (data[column] <= value)
            elif op == 'in':
                mask &= data[column].isin(value)
            elif op == 'not_in':
                mask &= ~data[column].isin(value)
            elif op == 'contains':
                mask &= data[column].str.contains(value, na=False)
            elif op == 'between':
                mask &= data[column].between(value[0], value[1])
    
    # Apply mask
    return data[mask]

def optimized_aggregate(
    data: pd.DataFrame,
    group_by: List[str],
    agg_funcs: Dict[str, str]
) -> pd.DataFrame:
    """
    Optimized aggregation of DataFrame.
    
    Args:
        data: Input DataFrame
        group_by: Columns to group by
        agg_funcs: Aggregation functions for each column
        
    Returns:
        Aggregated DataFrame
    """
    # Perform aggregation
    result = data.groupby(group_by).agg(agg_funcs)
    
    # Reset index
    result = result.reset_index()
    
    return result

def prepare_backtest_data(
    data: pd.DataFrame,
    indicators: Dict[str, Callable[[pd.DataFrame], pd.Series]]
) -> pd.DataFrame:
    """
    Prepare data for backtesting by calculating indicators.
    
    Args:
        data: Input OHLCV DataFrame
        indicators: Dictionary of indicator functions
        
    Returns:
        DataFrame with calculated indicators
    """
    # Create a copy of the input data
    result = data.copy()
    
    # Calculate indicators
    for name, func in indicators.items():
        result[name] = func(data)
    
    # Drop rows with NaN values
    result = result.dropna()
    
    return result

def parallel_strategy_optimization(
    backtest_func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
    parameter_grid: List[Dict[str, Any]],
    data: pd.DataFrame,
    n_jobs: int = -1
) -> List[Dict[str, Any]]:
    """
    Run parallel optimization of strategy parameters.
    
    Args:
        backtest_func: Function to run backtest with parameters
        parameter_grid: List of parameter combinations to test
        data: Input data for backtesting
        n_jobs: Number of parallel jobs (-1 for all available cores)
        
    Returns:
        List of backtest results for each parameter combination
    """
    # Determine number of workers
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    # Define worker function
    def worker(params):
        try:
            # Run backtest with parameters
            result = backtest_func(data, params)
            
            # Add parameters to result
            result['parameters'] = params
            
            return result
        except Exception as e:
            logger.error(f"Error in optimization task: {e}")
            return {'parameters': params, 'error': str(e)}
    
    # Run optimization in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(worker, parameter_grid))
    
    return results

async def stream_data_chunks(
    data_generator: Callable[[], AsyncGenerator[pd.DataFrame, None]],
    processing_func: Callable[[pd.DataFrame], Dict[str, Any]],
    chunk_size: int = 1000
) -> AsyncGenerator[List[Dict], None]:
    """
    Stream and process data in chunks asynchronously.
    
    Args:
        data_generator: Async generator function that yields DataFrames
        processing_func: Function to process each chunk
        chunk_size: Number of rows per chunk
        
    Yields:
        Processed data chunks
    """
    current_chunk = []
    
    async for data_frame in data_generator():
        # Process data in chunks
        for i in range(0, len(data_frame), chunk_size):
            chunk = data_frame.iloc[i:i+chunk_size].copy()
            
            # Process chunk
            processed_items = processing_func(chunk)
            
            # Add to current chunk
            current_chunk.extend(processed_items)
            
            # Yield if current chunk is full
            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []
    
    # Yield remaining items
    if current_chunk:
        yield current_chunk

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of a DataFrame by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    result = df.copy()
    
    # Optimize integers
    int_columns = result.select_dtypes(include=['int']).columns
    for col in int_columns:
        result[col] = pd.to_numeric(result[col], downcast='integer')
    
    # Optimize floats
    float_columns = result.select_dtypes(include=['float']).columns
    for col in float_columns:
        result[col] = pd.to_numeric(result[col], downcast='float')
    
    # Optimize objects (strings)
    object_columns = result.select_dtypes(include=['object']).columns
    for col in object_columns:
        if result[col].nunique() / len(result) < 0.5:  # If column has low cardinality
            result[col] = result[col].astype('category')
    
    return result

def batch_process_database_query(
    query_func: Callable[[int, int], pd.DataFrame],
    processing_func: Callable[[pd.DataFrame], Any],
    total_rows: int,
    batch_size: int = 10000
) -> List[Any]:
    """
    Process database query results in batches to reduce memory usage.
    
    Args:
        query_func: Function to query database with offset and limit
        processing_func: Function to process each batch
        total_rows: Total number of rows to process
        batch_size: Number of rows per batch
        
    Returns:
        List of processed results
    """
    results = []
    
    # Process in batches
    for offset in range(0, total_rows, batch_size):
        # Query database
        batch = query_func(offset, batch_size)
        
        # Process batch
        result = processing_func(batch)
        
        # Add to results
        results.append(result)
    
    return results
