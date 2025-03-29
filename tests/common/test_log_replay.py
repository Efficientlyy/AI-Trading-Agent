"""Unit tests for the log replay system."""

import gzip
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.common.log_replay import LogReplay


@pytest.fixture
def sample_log_entries():
    """Create sample log entries for testing."""
    return [
        {
            "timestamp": "2023-01-01T00:00:00Z",
            "level": "info",
            "component": "api",
            "request_id": "req1",
            "message": "Request started"
        },
        {
            "timestamp": "2023-01-01T00:00:01Z",
            "level": "error",
            "component": "api",
            "request_id": "req1",
            "message": "Request failed"
        },
        {
            "timestamp": "2023-01-01T00:00:02Z",
            "level": "info",
            "component": "db",
            "request_id": "req2",
            "message": "Query executed"
        }
    ]


@pytest.fixture
def temp_log_file(sample_log_entries):
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        for entry in sample_log_entries:
            f.write(json.dumps(entry) + "\n")
            
    yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def temp_gz_log_file(sample_log_entries):
    """Create a temporary compressed log file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as f:
        with gzip.open(f.name, 'wt') as gz:
            for entry in sample_log_entries:
                gz.write(json.dumps(entry) + "\n")
                
    yield Path(f.name)
    Path(f.name).unlink()


def test_basic_replay(temp_log_file):
    """Test basic log replay functionality."""
    processed_entries = []
    
    def handler(entry):
        processed_entries.append(entry)
        
    replay = LogReplay(
        log_file=str(temp_log_file),
        handlers={"info": handler}
    )
    
    count = replay.replay_from_file(str(temp_log_file))
    assert count == 3
    assert len(processed_entries) == 2  # Only info level entries


def test_compressed_replay(temp_gz_log_file):
    """Test replay from compressed log file."""
    processed_entries = []
    
    def handler(entry):
        processed_entries.append(entry)
        
    replay = LogReplay(
        log_file=str(temp_gz_log_file),
        handlers={"default": handler}
    )
    
    count = replay.replay_from_file(str(temp_gz_log_file))
    assert count == 3
    assert len(processed_entries) == 3


def test_filtered_replay(temp_log_file):
    """Test replay with filters."""
    processed_entries = []
    
    def handler(entry):
        processed_entries.append(entry)
        
    replay = LogReplay(
        log_file=str(temp_log_file),
        handlers={"default": handler},
        filters={"component": "api"}
    )
    
    count = replay.replay_from_file(str(temp_log_file))
    assert count == 2  # Only api component entries
    assert all(e["component"] = = "api" for e in processed_entries)


def test_time_range_replay(temp_log_file):
    """Test replay with time range filter."""
    processed_entries = []
    
    def handler(entry):
        processed_entries.append(entry)
        
    replay = LogReplay(handlers={"default": handler})
    
    start_time = datetime.fromisoformat("2023-01-01T00:00:01Z")
    count = replay.replay_from_file(
        str(temp_log_file),
        start_time=start_time
    )
    
    assert count == 2  # Only entries after start_time
    assert all(
        datetime.fromisoformat(e["timestamp"]) >= start_time
        for e in processed_entries
    )


def test_request_id_replay(temp_log_file, sample_log_entries):
    """Test replay by request ID."""
    processed_entries = []
    
    def handler(entry):
        processed_entries.append(entry)
        
    replay = LogReplay(handlers={"default": handler})
    
    count = replay.replay_by_request_id("req1")
    assert count == 2  # Only entries for req1
    assert all(e["request_id"] = = "req1" for e in processed_entries)


def test_component_replay(temp_log_file):
    """Test replay by component."""
    processed_entries = []
    
    def handler(entry):
        processed_entries.append(entry)
        
    replay = LogReplay(handlers={"default": handler})
    
    count = replay.replay_by_component("db")
    assert count == 1  # Only db component entries
    assert all(e["component"] = = "db" for e in processed_entries)


def test_speed_factor(temp_log_file):
    """Test replay speed factor."""
    processed_entries = []
    start_time = datetime.now()
    
    def handler(entry):
        processed_entries.append(entry)
        
    replay = LogReplay(handlers={"default": handler})
    
    # Speed up replay by 10x
    count = replay.replay_from_file(str(temp_log_file), speed_factor=10.0)
    duration = (datetime.now() - start_time).total_seconds()
    
    assert count == 3
    # Original log spans 2 seconds, so at 10x speed it should take ~0.2 seconds
    assert duration < 0.3


def test_invalid_log_file():
    """Test handling of invalid log file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Invalid JSON\n")
        f.write(json.dumps({"valid": "entry"}) + "\n")
        
    replay = LogReplay()
    count = replay.replay_from_file(f.name)
    
    assert count == 1  # Only valid entry processed
    Path(f.name).unlink()


def test_multiple_handlers():
    """Test multiple handlers for different event types."""
    info_entries = []
    error_entries = []
    
    def info_handler(entry):
        info_entries.append(entry)
        
    def error_handler(entry):
        error_entries.append(entry)
        
    handlers = {
        "info": info_handler,
        "error": error_handler
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        entries = [
            {"level": "info", "message": "test1"},
            {"level": "error", "message": "test2"},
            {"level": "warning", "message": "test3"}
        ]
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    replay = LogReplay(handlers=handlers)
    count = replay.replay_from_file(f.name)
    
    assert count == 3
    assert len(info_entries) == 1
    assert len(error_entries) == 1
    
    Path(f.name).unlink()
