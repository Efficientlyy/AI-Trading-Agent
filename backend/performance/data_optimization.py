"""
Performance optimization module for handling large datasets.
Provides optimized data structures and algorithms for efficient data processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union, AsyncGenerator
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
from sqlalchemy.orm import Session
from sqlalchemy import text

# Configure logging
logger = logging.getLogger(__name__)

# Performance monitoring decorator
def measure_performance(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Chunked data processing for large datasets
def process_in_chunks(data: pd.DataFrame, chunk_size: int, processing_func, *args, **kwargs) -> pd.DataFrame:
    """
    Process large dataframes in chunks to reduce memory usage.
    
    Args:
        data: Input dataframe to process
        chunk_size: Number of rows per chunk
        processing_func: Function to apply to each chunk
        *args, **kwargs: Additional arguments for processing_func
        
    Returns:
        Processed dataframe
    """
    result_chunks = []
    
    # Process data in chunks
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size].copy()
        processed_chunk = processing_func(chunk, *args, **kwargs)
        result_chunks.append(processed_chunk)
    
    # Combine processed chunks
    if result_chunks:
        return pd.concat(result_chunks, ignore_index=True)
    return pd.DataFrame()

# Parallel data processing
def parallel_process(data: pd.DataFrame, processing_func, n_jobs: int = -1, *args, **kwargs) -> pd.DataFrame:
    """
    Process dataframe using parallel execution.
    
    Args:
        data: Input dataframe to process
        processing_func: Function to apply to each partition
        n_jobs: Number of parallel jobs (-1 for all available cores)
        *args, **kwargs: Additional arguments for processing_func
        
    Returns:
        Processed dataframe
    """
    # Split dataframe into partitions
    if n_jobs <= 0:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    
    partitions = np.array_split(data, n_jobs)
    
    # Process partitions in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(lambda df: processing_func(df, *args, **kwargs), partitions))
    
    # Combine results
    return pd.concat(results, ignore_index=True)

# Optimized time series resampling
@measure_performance
def optimized_resample(data: pd.DataFrame, 
                      timestamp_col: str, 
                      value_cols: List[str], 
                      freq: str, 
                      agg_func: Dict[str, str]) -> pd.DataFrame:
    """
    Optimized time series resampling for large datasets.
    
    Args:
        data: Input dataframe with time series data
        timestamp_col: Column name containing timestamps
        value_cols: List of columns to aggregate
        freq: Resampling frequency (e.g. '1min', '1h', '1d')
        agg_func: Dictionary mapping columns to aggregation functions
        
    Returns:
        Resampled dataframe
    """
    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    
    # Set timestamp as index for resampling
    data_indexed = data.set_index(timestamp_col)
    
    # Select only required columns for resampling
    data_to_resample = data_indexed[value_cols]
    
    # Perform resampling
    resampled = data_to_resample.resample(freq).agg(agg_func)
    
    # Reset index to convert timestamp back to column
    return resampled.reset_index()

# Optimized OHLCV resampling for market data
@measure_performance
def resample_ohlcv(data: pd.DataFrame, 
                  timestamp_col: str = 'timestamp', 
                  freq: str = '1h') -> pd.DataFrame:
    """
    Optimized OHLCV (Open, High, Low, Close, Volume) resampling for market data.
    
    Args:
        data: Input dataframe with OHLCV data
        timestamp_col: Column name containing timestamps
        freq: Resampling frequency (e.g. '1min', '1h', '1d')
        
    Returns:
        Resampled OHLCV dataframe
    """
    # Define aggregation functions for OHLCV data
    agg_func = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Ensure all required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    return optimized_resample(data, timestamp_col, required_cols, freq, agg_func)

# Efficient data downsampling for visualization
@measure_performance
def downsample_for_visualization(data: pd.DataFrame, 
                                max_points: int = 1000) -> pd.DataFrame:
    """
    Efficiently downsample data for visualization while preserving important features.
    
    Args:
        data: Input dataframe
        max_points: Maximum number of data points to return
        
    Returns:
        Downsampled dataframe
    """
    if len(data) <= max_points:
        return data
    
    # Calculate sampling interval
    interval = max(1, len(data) // max_points)
    
    # Use LTTB (Largest-Triangle-Three-Buckets) algorithm for downsampling
    # This preserves visual features better than simple sampling
    if 'timestamp' in data.columns and any(col in data.columns for col in ['close', 'value']):
        # For time series data, use specialized downsampling
        value_col = 'close' if 'close' in data.columns else 'value'
        return lttb_downsample(data, 'timestamp', value_col, max_points)
    else:
        # For other data, use simple interval sampling
        return data.iloc[::interval].reset_index(drop=True)

def lttb_downsample(data: pd.DataFrame, 
                   x_col: str, 
                   y_col: str, 
                   n_points: int) -> pd.DataFrame:
    """
    Largest-Triangle-Three-Buckets downsampling algorithm.
    Preserves visual features of the data better than uniform sampling.
    
    Args:
        data: Input dataframe
        x_col: Column name for x-axis values (usually timestamp)
        y_col: Column name for y-axis values
        n_points: Number of points in output
        
    Returns:
        Downsampled dataframe
    """
    data = data.copy()
    
    # Convert x column to numeric for calculation
    if pd.api.types.is_datetime64_any_dtype(data[x_col]):
        data['x_numeric'] = data[x_col].astype(np.int64)
        x_col_calc = 'x_numeric'
    else:
        x_col_calc = x_col
    
    # If data has fewer points than requested, return original
    if len(data) <= n_points:
        return data
    
    # Always include first and last points
    sampled_indices = [0, len(data) - 1]
    
    # Number of buckets
    bucket_size = (len(data) - 2) / (n_points - 2)
    
    # Create buckets and select highest-area triangle point from each
    for i in range(n_points - 2):
        # Bucket start and end indices
        bucket_start = int((i * bucket_size) + 1)
        bucket_end = int(((i + 1) * bucket_size) + 1)
        bucket_end = min(bucket_end, len(data) - 1)
        
        # Get points in current bucket
        bucket_data = data.iloc[bucket_start:bucket_end]
        
        # Calculate area of triangles formed with the last selected point
        # and the next point to be selected
        areas = []
        last_selected_idx = sampled_indices[-1]
        last_selected_x = data.iloc[last_selected_idx][x_col_calc]
        last_selected_y = data.iloc[last_selected_idx][y_col]
        
        for idx, row in bucket_data.iterrows():
            # Calculate triangle area
            x = row[x_col_calc]
            y = row[y_col]
            area = 0.5 * abs((last_selected_x - x) * (data.iloc[-1][y_col] - y) - 
                             (last_selected_x - data.iloc[-1][x_col_calc]) * (y - last_selected_y))
            areas.append((idx, area))
        
        # Select point with maximum area
        if areas:
            max_area_idx = max(areas, key=lambda x: x[1])[0]
            sampled_indices.append(max_area_idx)
    
    # Sort indices and return sampled data
    sampled_indices = sorted(set(sampled_indices))
    result = data.iloc[sampled_indices].copy()
    
    # Drop temporary column if created
    if 'x_numeric' in result.columns:
        result = result.drop(columns=['x_numeric'])
    
    return result.reset_index(drop=True)

# Optimized database queries for large datasets
@measure_performance
def optimized_db_query(db: Session, 
                      query_str: str, 
                      params: Dict[str, Any] = None,
                      chunk_size: int = 10000) -> pd.DataFrame:
    """
    Execute optimized database queries for large datasets with chunking.
    
    Args:
        db: SQLAlchemy database session
        query_str: SQL query string
        params: Query parameters
        chunk_size: Number of rows to fetch per chunk
        
    Returns:
        Query results as dataframe
    """
    params = params or {}
    
    # Create SQLAlchemy text query
    query = text(query_str)
    
    # Execute query with chunked fetching
    result_chunks = []
    with db.connection() as conn:
        result_proxy = conn.execute(query, params)
        
        # Fetch results in chunks
        while True:
            chunk = result_proxy.fetchmany(chunk_size)
            if not chunk:
                break
            
            # Convert chunk to dataframe
            columns = result_proxy.keys()
            chunk_df = pd.DataFrame(chunk, columns=columns)
            result_chunks.append(chunk_df)
    
    # Combine chunks
    if result_chunks:
        return pd.concat(result_chunks, ignore_index=True)
    return pd.DataFrame()

# Efficient data serialization/deserialization
@measure_performance
def serialize_to_parquet(data: pd.DataFrame, file_path: str) -> None:
    """
    Efficiently serialize dataframe to parquet format.
    
    Args:
        data: Input dataframe
        file_path: Output file path
    """
    # Convert to PyArrow table for better performance
    table = pa.Table.from_pandas(data)
    
    # Write to parquet with compression
    pq.write_table(table, file_path, compression='snappy')

@measure_performance
def deserialize_from_parquet(file_path: str) -> pd.DataFrame:
    """
    Efficiently deserialize dataframe from parquet format.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded dataframe
    """
    # Read parquet file
    table = pq.read_table(file_path)
    
    # Convert to pandas
    return table.to_pandas()

# Optimized data filtering for large datasets
@measure_performance
def optimized_filter(data: pd.DataFrame, 
                    filter_conditions: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply optimized filtering to large datasets.
    
    Args:
        data: Input dataframe
        filter_conditions: Dictionary of column-value pairs for filtering
        
    Returns:
        Filtered dataframe
    """
    # Convert to PyArrow table for better performance
    table = pa.Table.from_pandas(data)
    
    # Apply filters using PyArrow compute
    for col, value in filter_conditions.items():
        if isinstance(value, (list, tuple)):
            # Handle IN condition
            filter_expr = pc.is_in(table[col], value_set=pa.array(value))
        elif isinstance(value, dict) and len(value) == 1:
            # Handle range conditions
            op, val = next(iter(value.items()))
            if op == 'gt':
                filter_expr = pc.greater(table[col], val)
            elif op == 'gte':
                filter_expr = pc.greater_equal(table[col], val)
            elif op == 'lt':
                filter_expr = pc.less(table[col], val)
            elif op == 'lte':
                filter_expr = pc.less_equal(table[col], val)
            else:
                raise ValueError(f"Unsupported operation: {op}")
        else:
            # Handle equality condition
            filter_expr = pc.equal(table[col], value)
        
        # Apply filter
        table = table.filter(filter_expr)
    
    # Convert back to pandas
    return table.to_pandas()

# Optimized data aggregation for large datasets
@measure_performance
def optimized_aggregate(data: pd.DataFrame, 
                       group_by: List[str], 
                       agg_funcs: Dict[str, str]) -> pd.DataFrame:
    """
    Apply optimized aggregation to large datasets.
    
    Args:
        data: Input dataframe
        group_by: Columns to group by
        agg_funcs: Dictionary mapping columns to aggregation functions
        
    Returns:
        Aggregated dataframe
    """
    # For large datasets, use chunked processing
    if len(data) > 1000000:
        chunk_size = 100000
        return process_in_chunks(data, chunk_size, lambda df: df.groupby(group_by).agg(agg_funcs).reset_index())
    
    # For smaller datasets, use standard pandas aggregation
    return data.groupby(group_by).agg(agg_funcs).reset_index()

# Optimized backtesting data preparation
@measure_performance
def prepare_backtest_data(ohlcv_data: pd.DataFrame, 
                         indicators: Dict[str, Any],
                         chunk_size: int = 100000) -> pd.DataFrame:
    """
    Efficiently prepare data for backtesting with indicators.
    
    Args:
        ohlcv_data: OHLCV market data
        indicators: Dictionary of indicator functions and parameters
        chunk_size: Size of chunks for processing
        
    Returns:
        Prepared dataframe with indicators
    """
    # Process in chunks for large datasets
    if len(ohlcv_data) > chunk_size:
        def process_chunk(chunk):
            result = chunk.copy()
            for name, indicator_func in indicators.items():
                result[name] = indicator_func(chunk)
            return result
        
        return process_in_chunks(ohlcv_data, chunk_size, process_chunk)
    
    # For smaller datasets, process directly
    result = ohlcv_data.copy()
    for name, indicator_func in indicators.items():
        result[name] = indicator_func(ohlcv_data)
    
    return result

# Optimized strategy optimization
@measure_performance
def parallel_strategy_optimization(backtest_func, 
                                 parameter_grid: List[Dict[str, Any]],
                                 market_data: pd.DataFrame,
                                 n_jobs: int = -1) -> List[Dict[str, Any]]:
    """
    Run strategy optimization in parallel for multiple parameter combinations.
    
    Args:
        backtest_func: Function to run backtest with parameters
        parameter_grid: List of parameter combinations to test
        market_data: Market data for backtesting
        n_jobs: Number of parallel jobs
        
    Returns:
        List of optimization results with parameters and metrics
    """
    # Define worker function for each parameter set
    def worker(params):
        try:
            # Run backtest with parameters
            backtest_result = backtest_func(market_data, params)
            
            # Return parameters and metrics
            return {
                'parameters': params,
                'metrics': backtest_result['metrics'],
                'success': True
            }
        except Exception as e:
            logger.error(f"Error in optimization for parameters {params}: {str(e)}")
            return {
                'parameters': params,
                'error': str(e),
                'success': False
            }
    
    # Run optimization in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(worker, parameter_grid))
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    # Sort by sharpe ratio (or other primary metric)
    if successful_results and 'metrics' in successful_results[0] and 'sharpe_ratio' in successful_results[0]['metrics']:
        successful_results.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
    
    return successful_results

# Efficient caching for expensive computations
def get_cached_data(cache_key: str, 
                   data_loader_func, 
                   ttl_seconds: int = 3600,
                   *args, **kwargs) -> Any:
    """
    Get data with caching for expensive computations.
    
    Args:
        cache_key: Unique key for the cached data
        data_loader_func: Function to load data if not in cache
        ttl_seconds: Time-to-live for cached data in seconds
        *args, **kwargs: Arguments for data_loader_func
        
    Returns:
        Cached or freshly loaded data
    """
    from backend.database.cache import get_cache, set_cache
    
    # Try to get from cache
    cached_data = get_cache(cache_key)
    
    if cached_data is not None:
        logger.info(f"Cache hit for key: {cache_key}")
        return cached_data
    
    # Cache miss, load data
    logger.info(f"Cache miss for key: {cache_key}")
    data = data_loader_func(*args, **kwargs)
    
    # Store in cache
    set_cache(cache_key, data, ttl_seconds)
    
    return data

# Optimized JSON serialization for large objects
@measure_performance
def optimized_json_serialize(data: Any) -> str:
    """
    Efficiently serialize large objects to JSON.
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON string
    """
    # Convert pandas dataframes to records for more efficient serialization
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='records')
    
    # Use orjson for faster serialization if available
    try:
        import orjson
        return orjson.dumps(data).decode('utf-8')
    except ImportError:
        # Fall back to standard json
        return json.dumps(data)

@measure_performance
def optimized_json_deserialize(json_str: str) -> Any:
    """
    Efficiently deserialize JSON string.
    
    Args:
        json_str: JSON string to deserialize
        
    Returns:
        Deserialized data
    """
    # Use orjson for faster deserialization if available
    try:
        import orjson
        return orjson.loads(json_str)
    except ImportError:
        # Fall back to standard json
        return json.loads(json_str)

# Memory-efficient data streaming
async def stream_large_dataset(data_generator, 
                              chunk_size: int = 1000) -> AsyncGenerator[List[Dict], None]:
    """
    Stream large datasets in chunks for memory-efficient processing.
    
    Args:
        data_generator: Generator function that yields data items
        chunk_size: Number of items per chunk
        
    Yields:
        Chunks of data
    """
    chunk = []
    
    async for item in data_generator:
        chunk.append(item)
        
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    
    # Yield remaining items
    if chunk:
        yield chunk
