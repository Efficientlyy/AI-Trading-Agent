"""Unit tests for the log query system."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.common.log_query import LogQuery, Lexer, Parser, Interpreter


def test_lexer():
    """Test the query lexer."""
    query = 'level = "error" AND (component = "api" OR timestamp > "2023-01-01")'
    lexer = Lexer(query)
    tokens = []
    
    while True:
        token = lexer.get_next_token()
        tokens.append(token)
        if token.type.value == "EOF":
            break
            
    assert len(tokens) == 13  # Including EOF
    assert tokens[0].value == "level"
    assert tokens[1].value == "="
    assert tokens[2].value == "error"
    assert tokens[3].value == "AND"


def test_parser():
    """Test the query parser."""
    query = 'level = "error" AND component = "api"'
    lexer = Lexer(query)
    parser = Parser(lexer)
    ast = parser.parse()
    
    assert ast["type"] == "binary_op"
    assert ast["operator"] == "AND"
    assert ast["left"]["type"] == "condition"
    assert ast["left"]["field"] == "level"
    assert ast["left"]["value"] == "error"


def test_interpreter():
    """Test the query interpreter."""
    query = 'level = "error" AND component = "api"'
    log_query = LogQuery(query)
    
    # Should match
    entry1 = {
        "level": "error",
        "component": "api",
        "message": "Test error"
    }
    assert log_query.matches(entry1) is True
    
    # Should not match
    entry2 = {
        "level": "info",
        "component": "api",
        "message": "Test info"
    }
    assert log_query.matches(entry2) is False


def test_complex_query():
    """Test complex query with multiple conditions."""
    query = ('(level = "error" OR level = "warning") AND '
             'component = "api" AND '
             'NOT message ~ "timeout"')
    log_query = LogQuery(query)
    
    # Should match
    entry1 = {
        "level": "error",
        "component": "api",
        "message": "Test error"
    }
    assert log_query.matches(entry1) is True
    
    # Should not match (has timeout)
    entry2 = {
        "level": "error",
        "component": "api",
        "message": "Connection timeout"
    }
    assert log_query.matches(entry2) is False
    
    # Should not match (wrong level)
    entry3 = {
        "level": "info",
        "component": "api",
        "message": "Test info"
    }
    assert log_query.matches(entry3) is False


def test_numeric_comparisons():
    """Test numeric comparisons in queries."""
    query = 'duration > 1000 AND status_code >= 500'
    log_query = LogQuery(query)
    
    # Should match
    entry1 = {
        "duration": 1500,
        "status_code": 500
    }
    assert log_query.matches(entry1) is True
    
    # Should not match
    entry2 = {
        "duration": 800,
        "status_code": 500
    }
    assert log_query.matches(entry2) is False


def test_contains_operator():
    """Test the contains operator."""
    query = 'tags ~ "production" AND NOT message ~ "debug"'
    log_query = LogQuery(query)
    
    # Should match
    entry1 = {
        "tags": ["production", "api"],
        "message": "Error in production"
    }
    assert log_query.matches(entry1) is True
    
    # Should not match (has debug)
    entry2 = {
        "tags": ["production", "api"],
        "message": "Debug: test message"
    }
    assert log_query.matches(entry2) is False


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        entries = [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "level": "error",
                "component": "api",
                "message": "Test error 1"
            },
            {
                "timestamp": "2023-01-01T00:00:01Z",
                "level": "info",
                "component": "api",
                "message": "Test info"
            },
            {
                "timestamp": "2023-01-01T00:00:02Z",
                "level": "error",
                "component": "db",
                "message": "Test error 2"
            }
        ]
        
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    yield Path(f.name)
    Path(f.name).unlink()


def test_file_search(temp_log_file):
    """Test searching within a log file."""
    query = 'level = "error"'
    log_query = LogQuery(query)
    
    results = log_query.search_file(str(temp_log_file))
    assert len(results) == 2
    assert all(r["level"] == "error" for r in results)


def test_invalid_query():
    """Test handling of invalid queries."""
    with pytest.raises(Exception):
        LogQuery("invalid = ")
        
    with pytest.raises(Exception):
        LogQuery("field > 'unclosed string")


def test_empty_query():
    """Test behavior with empty query."""
    log_query = LogQuery()
    
    entry = {
        "level": "info",
        "message": "test"
    }
    
    # Empty query should match everything
    assert log_query.matches(entry) is True
