"""File-based storage implementation for the AI Crypto Trading System.

This module implements a file-based storage backend for storing market data.
Data is stored in CSV or Parquet files, organized by data type, exchange, and symbol.
"""

import asyncio
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.common.config import config
from src.common.logging import get_logger
from src.data_collection.persistence.storage import Storage
from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData


class FileStorage(Storage):
    """File-based storage backend for market data."""
    
    def __init__(self, name: str = "file_storage"):
        """Initialize the file storage backend.
        
        Args:
            name: The name of the storage backend
        """
        super().__init__(name)
        self.logger = get_logger("data_collection.storage", "file")
        self.base_dir = Path(config.get("data_collection.persistence.base_dir", "data"))
        self.file_format = config.get("data_collection.persistence.file_format", "csv").lower()
        self.max_chunk_size = config.get("data_collection.persistence.max_chunk_size", 10000)
        
        # Ensure the base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize subdirectories for different data types
        self.candles_dir = self.base_dir / "candles"
        self.orderbooks_dir = self.base_dir / "orderbooks"
        self.trades_dir = self.base_dir / "trades"
        
        # Create file locks for concurrent access
        self.file_locks = {}
    
    async def initialize(self) -> bool:
        """Initialize the file storage backend.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Create subdirectories
            self.candles_dir.mkdir(parents=True, exist_ok=True)
            self.orderbooks_dir.mkdir(parents=True, exist_ok=True)
            self.trades_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("File storage initialized", 
                           base_dir=str(self.base_dir), 
                           file_format=self.file_format)
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize file storage", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close the file storage backend."""
        self.logger.info("Closing file storage")
    
    async def store_candle(self, candle: CandleData) -> bool:
        """Store a single candle.
        
        Args:
            candle: The candle data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        return await self.store_candles([candle])
    
    async def store_candles(self, candles: List[CandleData]) -> bool:
        """Store multiple candles in batch.
        
        Args:
            candles: The list of candle data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not candles:
            return True
        
        # Group candles by exchange, symbol, and timeframe
        grouped_candles = {}
        for candle in candles:
            key = (candle.exchange, candle.symbol, candle.timeframe.value)
            if key not in grouped_candles:
                grouped_candles[key] = []
            grouped_candles[key].append(candle)
        
        # Store each group separately
        success = True
        for (exchange, symbol, timeframe), group in grouped_candles.items():
            try:
                # Convert candles to a dataframe
                candle_dicts = [
                    {
                        "timestamp": candle.timestamp.isoformat(),
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume
                    }
                    for candle in group
                ]
                
                df = pd.DataFrame(candle_dicts)
                
                # Determine file path
                exchange_dir = self.candles_dir / exchange
                symbol_dir = exchange_dir / symbol.replace("/", "_")
                symbol_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = symbol_dir / f"{timeframe}.{self.file_format}"
                
                # Acquire lock for this file
                file_lock = self._get_file_lock(str(file_path))
                async with file_lock:
                    # Load existing data if file exists
                    if file_path.exists():
                        existing_df = self._read_dataframe(file_path)
                        
                        # Combine and deduplicate
                        df = pd.concat([existing_df, df])
                        df = df.drop_duplicates(subset=["timestamp"])
                        
                        # Sort by timestamp
                        df = df.sort_values("timestamp")
                    
                    # Write the dataframe back to file
                    self._write_dataframe(df, file_path)
                
                self.logger.debug("Stored candles", 
                                exchange=exchange, 
                                symbol=symbol, 
                                timeframe=timeframe, 
                                count=len(group))
                
            except Exception as e:
                self.logger.error("Failed to store candles", 
                                exchange=exchange, 
                                symbol=symbol, 
                                timeframe=timeframe, 
                                error=str(e))
                success = False
        
        return success
    
    async def get_candles(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None, 
        limit: Optional[int] = None
    ) -> List[CandleData]:
        """Retrieve candles for a symbol, exchange, and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            start_time: Optional start time for the query
            end_time: Optional end time for the query
            limit: Optional limit for the number of candles to retrieve
            
        Returns:
            List[CandleData]: List of candle data objects
        """
        try:
            # Determine file path
            exchange_dir = self.candles_dir / exchange
            symbol_dir = exchange_dir / symbol.replace("/", "_")
            file_path = symbol_dir / f"{timeframe.value}.{self.file_format}"
            
            # Check if file exists
            if not file_path.exists():
                self.logger.debug("No candle data file found", 
                                exchange=exchange, 
                                symbol=symbol, 
                                timeframe=timeframe.value)
                return []
            
            # Acquire lock for this file
            file_lock = self._get_file_lock(str(file_path))
            async with file_lock:
                # Read the dataframe
                df = self._read_dataframe(file_path)
                
                # Convert timestamps to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Apply filters
                if start_time:
                    df = df[df["timestamp"] >= start_time]
                
                if end_time:
                    df = df[df["timestamp"] <= end_time]
                
                # Sort by timestamp
                df = df.sort_values("timestamp")
                
                # Apply limit
                if limit:
                    df = df.tail(limit)
                
                # Convert back to candle objects
                candles = []
                for _, row in df.iterrows():
                    candle = CandleData(
                        symbol=symbol,
                        exchange=exchange,
                        timestamp=row["timestamp"].to_pydatetime(),
                        timeframe=timeframe,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"])
                    )
                    candles.append(candle)
                
                self.logger.debug("Retrieved candles", 
                                exchange=exchange, 
                                symbol=symbol, 
                                timeframe=timeframe.value, 
                                count=len(candles))
                return candles
                
        except Exception as e:
            self.logger.error("Failed to retrieve candles", 
                            exchange=exchange, 
                            symbol=symbol, 
                            timeframe=timeframe.value, 
                            error=str(e))
            return []
    
    async def store_orderbook(self, orderbook: OrderBookData) -> bool:
        """Store an order book snapshot.
        
        Args:
            orderbook: The order book data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        try:
            # Convert orderbook to a flattened structure
            timestamp_str = orderbook.timestamp.isoformat()
            
            # Create separate entries for bids and asks
            rows = []
            for bid in orderbook.bids:
                rows.append({
                    "timestamp": timestamp_str,
                    "type": "bid",
                    "price": bid["price"],
                    "size": bid["size"]
                })
            
            for ask in orderbook.asks:
                rows.append({
                    "timestamp": timestamp_str,
                    "type": "ask",
                    "price": ask["price"],
                    "size": ask["size"]
                })
            
            df = pd.DataFrame(rows)
            
            # Determine file path
            exchange_dir = self.orderbooks_dir / orderbook.exchange
            symbol_dir = exchange_dir / orderbook.symbol.replace("/", "_")
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Use a timestamp-based filename to avoid large files
            date_str = orderbook.timestamp.strftime("%Y%m%d")
            file_path = symbol_dir / f"{date_str}.{self.file_format}"
            
            # Acquire lock for this file
            file_lock = self._get_file_lock(str(file_path))
            async with file_lock:
                # Load existing data if file exists
                if file_path.exists():
                    existing_df = self._read_dataframe(file_path)
                    
                    # Keep only entries from different timestamps
                    existing_df = existing_df[existing_df["timestamp"] != timestamp_str]
                    
                    # Combine
                    df = pd.concat([existing_df, df])
                
                # Write the dataframe back to file
                self._write_dataframe(df, file_path)
            
            self.logger.debug("Stored orderbook", 
                            exchange=orderbook.exchange, 
                            symbol=orderbook.symbol, 
                            timestamp=timestamp_str)
            return True
            
        except Exception as e:
            self.logger.error("Failed to store orderbook", 
                            exchange=orderbook.exchange, 
                            symbol=orderbook.symbol, 
                            error=str(e))
            return False
    
    async def get_orderbook(
        self, 
        symbol: str, 
        exchange: str, 
        timestamp: Optional[datetime] = None
    ) -> Optional[OrderBookData]:
        """Retrieve the latest (or specific) order book for a symbol and exchange.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timestamp: Optional specific timestamp to retrieve
            
        Returns:
            Optional[OrderBookData]: Order book data object, or None if not found
        """
        try:
            # Determine directory path
            exchange_dir = self.orderbooks_dir / exchange
            symbol_dir = exchange_dir / symbol.replace("/", "_")
            
            # Check if directory exists
            if not symbol_dir.exists():
                self.logger.debug("No orderbook data directory found", 
                                exchange=exchange, 
                                symbol=symbol)
                return None
            
            # List all files in the directory
            files = list(symbol_dir.glob(f"*.{self.file_format}"))
            if not files:
                self.logger.debug("No orderbook data files found", 
                                exchange=exchange, 
                                symbol=symbol)
                return None
            
            # For a specific timestamp, find the file for that date
            if timestamp:
                date_str = timestamp.strftime("%Y%m%d")
                file_path = symbol_dir / f"{date_str}.{self.file_format}"
                
                if not file_path.exists():
                    self.logger.debug("No orderbook data file found for date", 
                                    exchange=exchange, 
                                    symbol=symbol, 
                                    date=date_str)
                    return None
                
                # Acquire lock for this file
                file_lock = self._get_file_lock(str(file_path))
                async with file_lock:
                    # Read the dataframe
                    df = self._read_dataframe(file_path)
                    
                    # Filter for the specific timestamp
                    timestamp_str = timestamp.isoformat()
                    df = df["df["timestamp""] = = timestamp_str]
                    
                    if df.empty:
                        self.logger.debug("No orderbook data found for timestamp", 
                                        exchange=exchange, 
                                        symbol=symbol, 
                                        timestamp=timestamp_str)
                        return None
            else:
                # For latest, use the most recent file
                files.sort(reverse=True)
                file_path = files[0]
                
                # Acquire lock for this file
                file_lock = self._get_file_lock(str(file_path))
                async with file_lock:
                    # Read the dataframe
                    df = self._read_dataframe(file_path)
                    
                    if df.empty:
                        self.logger.debug("Empty orderbook data file", 
                                        exchange=exchange, 
                                        symbol=symbol, 
                                        file=file_path.name)
                        return None
                    
                    # Get the latest timestamp
                    latest_timestamp = df["timestamp"].max()
                    df = df["df["timestamp""] = = latest_timestamp]
            
            # Convert back to orderbook object
            bids = []
            asks = []
            
            for _, row in df.iterrows():
                entry = {
                    "price": float(row["price"]),
                    "size": float(row["size"])
                }
                
                if row["type"] = = "bid":
                    bids.append(entry)
                else:
                    asks.append(entry)
            
            # Sort bids (descending) and asks (ascending)
            bids.sort(key=lambda x: x["price"], reverse=True)
            asks.sort(key=lambda x: x["price"])
            
            # Create the orderbook object
            orderbook = OrderBookData(
                symbol=symbol,
                exchange=exchange,
                timestamp=datetime.fromisoformat(df["timestamp"].iloc[0]),
                bids=bids,
                asks=asks
            )
            
            self.logger.debug("Retrieved orderbook", 
                            exchange=exchange, 
                            symbol=symbol, 
                            timestamp=orderbook.timestamp.isoformat())
            return orderbook
            
        except Exception as e:
            self.logger.error("Failed to retrieve orderbook", 
                            exchange=exchange, 
                            symbol=symbol, 
                            error=str(e))
            return None
    
    async def store_trade(self, trade: TradeData) -> bool:
        """Store a single trade.
        
        Args:
            trade: The trade data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        return await self.store_trades([trade])
    
    async def store_trades(self, trades: List[TradeData]) -> bool:
        """Store multiple trades in batch.
        
        Args:
            trades: The list of trade data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not trades:
            return True
        
        # Group trades by exchange and symbol
        grouped_trades = {}
        for trade in trades:
            key = (trade.exchange, trade.symbol)
            if key not in grouped_trades:
                grouped_trades[key] = []
            grouped_trades[key].append(trade)
        
        # Store each group separately
        success = True
        for (exchange, symbol), group in grouped_trades.items():
            try:
                # Convert trades to a dataframe
                trade_dicts = [
                    {
                        "timestamp": trade.timestamp.isoformat(),
                        "price": trade.price,
                        "size": trade.size,
                        "side": trade.side
                    }
                    for trade in group
                ]
                
                df = pd.DataFrame(trade_dicts)
                
                # Determine file path
                exchange_dir = self.trades_dir / exchange
                symbol_dir = exchange_dir / symbol.replace("/", "_")
                symbol_dir.mkdir(parents=True, exist_ok=True)
                
                # Use a date-based filename to avoid large files
                date_str = group[0].timestamp.strftime("%Y%m%d")
                file_path = symbol_dir / f"{date_str}.{self.file_format}"
                
                # Acquire lock for this file
                file_lock = self._get_file_lock(str(file_path))
                async with file_lock:
                    # Load existing data if file exists
                    if file_path.exists():
                        existing_df = self._read_dataframe(file_path)
                        
                        # Combine
                        df = pd.concat([existing_df, df])
                        
                        # Sort by timestamp
                        df = df.sort_values("timestamp")
                    
                    # Write the dataframe back to file
                    self._write_dataframe(df, file_path)
                
                self.logger.debug("Stored trades", 
                                exchange=exchange, 
                                symbol=symbol, 
                                count=len(group))
                
            except Exception as e:
                self.logger.error("Failed to store trades", 
                                exchange=exchange, 
                                symbol=symbol, 
                                error=str(e))
                success = False
        
        return success
    
    async def get_trades(
        self, 
        symbol: str, 
        exchange: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None, 
        limit: Optional[int] = None
    ) -> List[TradeData]:
        """Retrieve trades for a symbol and exchange.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            start_time: Optional start time for the query
            end_time: Optional end time for the query
            limit: Optional limit for the number of trades to retrieve
            
        Returns:
            List[TradeData]: List of trade data objects
        """
        try:
            # Determine directory path
            exchange_dir = self.trades_dir / exchange
            symbol_dir = exchange_dir / symbol.replace("/", "_")
            
            # Check if directory exists
            if not symbol_dir.exists():
                self.logger.debug("No trade data directory found", 
                                exchange=exchange, 
                                symbol=symbol)
                return []
            
            # List all files in the directory
            all_files = list(symbol_dir.glob(f"*.{self.file_format}"))
            if not all_files:
                self.logger.debug("No trade data files found", 
                                exchange=exchange, 
                                symbol=symbol)
                return []
            
            # Filter files based on date range
            files_to_check = all_files
            if start_time or end_time:
                # Extract dates from filenames
                date_files = []
                for file_path in all_files:
                    try:
                        date_str = file_path.stem
                        file_date = datetime.strptime(date_str, "%Y%m%d")
                        date_files.append((file_date, file_path))
                    except ValueError:
                        continue
                
                # Filter by date range
                if start_time:
                    start_date = datetime(start_time.year, start_time.month, start_time.day)
                    date_files = [(d, f) for d, f in date_files if d >= start_date]
                
                if end_time:
                    end_date = datetime(end_time.year, end_time.month, end_time.day) + timedelta(days=1)
                    date_files = [(d, f) for d, f in date_files if d < end_date]
                
                files_to_check = [f for _, f in date_files]
            
            # Sort files by date (newest first)
            files_to_check.sort(reverse=True)
            
            # Read and combine data from all files
            dfs = []
            for file_path in files_to_check:
                # Acquire lock for this file
                file_lock = self._get_file_lock(str(file_path))
                async with file_lock:
                    # Read the dataframe
                    df = self._read_dataframe(file_path)
                    
                    if not df.empty:
                        # Convert timestamps to datetime for filtering
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        
                        # Apply time filters
                        if start_time:
                            df = df[df["timestamp"] >= start_time]
                        
                        if end_time:
                            df = df[df["timestamp"] <= end_time]
                        
                        if not df.empty:
                            dfs.append(df)
            
            if not dfs:
                self.logger.debug("No trade data found matching criteria", 
                                exchange=exchange, 
                                symbol=symbol)
                return []
            
            # Combine all dataframes
            combined_df = pd.concat(dfs)
            
            # Sort by timestamp
            combined_df = combined_df.sort_values("timestamp")
            
            # Apply limit
            if limit:
                combined_df = combined_df.tail(limit)
            
            # Convert back to trade objects
            trades = []
            for _, row in combined_df.iterrows():
                trade = TradeData(
                    symbol=symbol,
                    exchange=exchange,
                    timestamp=row["timestamp"].to_pydatetime(),
                    price=float(row["price"]),
                    size=float(row["size"]),
                    side=row["side"]
                )
                trades.append(trade)
            
            self.logger.debug("Retrieved trades", 
                            exchange=exchange, 
                            symbol=symbol, 
                            count=len(trades))
            return trades
            
        except Exception as e:
            self.logger.error("Failed to retrieve trades", 
                            exchange=exchange, 
                            symbol=symbol, 
                            error=str(e))
            return []
    
    async def get_latest_candle(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: TimeFrame
    ) -> Optional[CandleData]:
        """Retrieve the latest candle for a symbol, exchange, and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            exchange: The exchange identifier
            timeframe: The candle timeframe
            
        Returns:
            Optional[CandleData]: Latest candle data object, or None if not found
        """
        # Get the most recent candle from the candles storage
        candles = await self.get_candles(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            limit=1
        )
        
        if candles:
            return candles[0]
        
        return None
    
    async def purge_old_data(self, max_age_days: Dict[str, int]) -> bool:
        """Purge data older than the specified age.
        
        Args:
            max_age_days: Dictionary of data types to maximum age in days
            
        Returns:
            bool: True if purge was successful, False otherwise
        """
        success = True
        
        try:
            current_date = datetime.now()
            
            # Purge candles
            if "candles" in max_age_days:
                days = max_age_days["candles"]
                cutoff_date = current_date - timedelta(days=days)
                success = success and await self._purge_old_candles(cutoff_date)
            
            # Purge orderbooks
            if "orderbooks" in max_age_days:
                days = max_age_days["orderbooks"]
                cutoff_date = current_date - timedelta(days=days)
                success = success and await self._purge_old_orderbooks(cutoff_date)
            
            # Purge trades
            if "trades" in max_age_days:
                days = max_age_days["trades"]
                cutoff_date = current_date - timedelta(days=days)
                success = success and await self._purge_old_trades(cutoff_date)
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to purge old data", error=str(e))
            return False
    
    async def _purge_old_candles(self, cutoff_date: datetime) -> bool:
        """Purge candle data older than the cutoff date.
        
        Args:
            cutoff_date: The cutoff date for purging
            
        Returns:
            bool: True if purge was successful, False otherwise
        """
        success = True
        
        # This would involve reading all candle files, removing old data, and writing back
        # For simplicity, we'll just log this for now
        self.logger.info("Purge old candles not implemented yet", cutoff_date=cutoff_date.isoformat())
        
        return success
    
    async def _purge_old_orderbooks(self, cutoff_date: datetime) -> bool:
        """Purge orderbook data older than the cutoff date.
        
        Args:
            cutoff_date: The cutoff date for purging
            
        Returns:
            bool: True if purge was successful, False otherwise
        """
        success = True
        
        try:
            cutoff_date_str = cutoff_date.strftime("%Y%m%d")
            
            # Walk through the orderbooks directory and delete old files
            for exchange_dir in self.orderbooks_dir.iterdir():
                if not exchange_dir.is_dir():
                    continue
                
                for symbol_dir in exchange_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    
                    for file_path in symbol_dir.glob(f"*.{self.file_format}"):
                        try:
                            date_str = file_path.stem
                            if date_str < cutoff_date_str:
                                # Acquire lock for this file
                                file_lock = self._get_file_lock(str(file_path))
                                async with file_lock:
                                    # Delete the file
                                    file_path.unlink()
                                    self.logger.debug("Deleted old orderbook file", 
                                                    file=str(file_path))
                        except Exception as e:
                            self.logger.error("Failed to delete orderbook file", 
                                            file=str(file_path), 
                                            error=str(e))
                            success = False
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to purge old orderbooks", error=str(e))
            return False
    
    async def _purge_old_trades(self, cutoff_date: datetime) -> bool:
        """Purge trade data older than the cutoff date.
        
        Args:
            cutoff_date: The cutoff date for purging
            
        Returns:
            bool: True if purge was successful, False otherwise
        """
        success = True
        
        try:
            cutoff_date_str = cutoff_date.strftime("%Y%m%d")
            
            # Walk through the trades directory and delete old files
            for exchange_dir in self.trades_dir.iterdir():
                if not exchange_dir.is_dir():
                    continue
                
                for symbol_dir in exchange_dir.iterdir():
                    if not symbol_dir.is_dir():
                        continue
                    
                    for file_path in symbol_dir.glob(f"*.{self.file_format}"):
                        try:
                            date_str = file_path.stem
                            if date_str < cutoff_date_str:
                                # Acquire lock for this file
                                file_lock = self._get_file_lock(str(file_path))
                                async with file_lock:
                                    # Delete the file
                                    file_path.unlink()
                                    self.logger.debug("Deleted old trade file", 
                                                    file=str(file_path))
                        except Exception as e:
                            self.logger.error("Failed to delete trade file", 
                                            file=str(file_path), 
                                            error=str(e))
                            success = False
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to purge old trades", error=str(e))
            return False
    
    def _get_file_lock(self, file_path: str) -> asyncio.Lock:
        """Get a lock for a file path.
        
        Args:
            file_path: The file path to lock
            
        Returns:
            asyncio.Lock: The lock for the file
        """
        if file_path not in self.file_locks:
            self.file_locks[file_path] = asyncio.Lock()
        
        return self.file_locks[file_path]
    
    def _read_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Read a dataframe from a file.
        
        Args:
            file_path: The file path to read
            
        Returns:
            pd.DataFrame: The dataframe read from the file
        """
        if self.file_format == "csv":
            return pd.read_csv(file_path)
        elif self.file_format == "parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
    
    def _write_dataframe(self, df: pd.DataFrame, file_path: Path) -> None:
        """Write a dataframe to a file.
        
        Args:
            df: The dataframe to write
            file_path: The file path to write to
        """
        if self.file_format == "csv":
            df.to_csv(file_path, index=False)
        elif self.file_format == "parquet":
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}") 