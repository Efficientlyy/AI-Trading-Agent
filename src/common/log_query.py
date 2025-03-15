"""Structured Log Query Language for searching and filtering logs.

This module provides a simple query language to search and filter logs,
allowing complex queries with multiple conditions and operations.
"""

import json
import operator
import re
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import structlog

from src.common.logging import LOG_DIR


# Initialize logger
logger = structlog.get_logger("log_query")


class TokenType(Enum):
    """Token types for the query parser."""
    FIELD = "FIELD"
    OPERATOR = "OPERATOR"
    VALUE = "VALUE"
    LOGICAL = "LOGICAL"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"


class Token:
    """Token for the query parser."""
    
    def __init__(self, token_type: TokenType, value: str):
        """
        Initialize a token.
        
        Args:
            token_type: Type of token
            value: String value of token
        """
        self.type = token_type
        self.value = value
        
    def __repr__(self):
        """Return string representation."""
        return f"Token({self.type}, {self.value!r})"


class Lexer:
    """Lexer for the query language."""
    
    # Define token patterns
    OPERATORS = {
        "=": "=",
        "!=": "!=",
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
        "~": "~",  # contains
        "!~": "!~"  # does not contain
    }
    
    LOGICAL = {
        "AND": "AND",
        "OR": "OR",
        "NOT": "NOT"
    }
    
    def __init__(self, text: str):
        """
        Initialize the lexer.
        
        Args:
            text: Query text to tokenize
        """
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if text else None
        
    def error(self):
        """Raise an error."""
        raise Exception(f"Invalid character: {self.current_char}")
        
    def advance(self):
        """Advance the position pointer."""
        self.pos += 1
        if self.pos >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]
            
    def skip_whitespace(self):
        """Skip whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
            
    def field(self) -> str:
        """Parse a field name."""
        result = ""
        while (self.current_char is not None and 
               (self.current_char.isalnum() or self.current_char == '.' or self.current_char == '_')):
            result += self.current_char
            self.advance()
            
        return result
        
    def operator(self) -> str:
        """Parse an operator."""
        if self.current_char == '=':
            self.advance()
            return "="
        elif self.current_char == '!':
            self.advance()
            if self.current_char == '=':
                self.advance()
                return "!="
            elif self.current_char == '~':
                self.advance()
                return "!~"
            else:
                self.error()
        elif self.current_char == '>':
            self.advance()
            if self.current_char == '=':
                self.advance()
                return ">="
            return ">"
        elif self.current_char == '<':
            self.advance()
            if self.current_char == '=':
                self.advance()
                return "<="
            return "<"
        elif self.current_char == '~':
            self.advance()
            return "~"
        else:
            self.error()
            
    def string(self) -> str:
        """Parse a string value."""
        quote_char = self.current_char  # ' or "
        self.advance()
        result = ""
        
        while self.current_char is not None and self.current_char != quote_char:
            if self.current_char == '\\':
                self.advance()
                if self.current_char is None:
                    self.error()
                result += self.current_char
            else:
                result += self.current_char
            self.advance()
            
        if self.current_char is None:
            self.error()  # Unterminated string
            
        self.advance()  # Skip closing quote
        return result
        
    def number(self) -> Union[int, float]:
        """Parse a numeric value."""
        result = ""
        has_decimal = False
        
        while (self.current_char is not None and 
               (self.current_char.isdigit() or self.current_char == '.')):
            if self.current_char == '.':
                if has_decimal:
                    self.error()  # Multiple decimal points
                has_decimal = True
            result += self.current_char
            self.advance()
            
        return float(result) if has_decimal else int(result)
        
    def get_next_token(self) -> Token:
        """Get the next token from the input."""
        while self.current_char is not None:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
                
            # Parentheses
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(')
                
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')')
                
            # Identifiers (field names) or logical operators
            if self.current_char.isalpha() or self.current_char == '_':
                field_name = self.field()
                
                # Check if it's a logical operator
                upper_field = field_name.upper()
                if upper_field in self.LOGICAL:
                    return Token(TokenType.LOGICAL, upper_field)
                    
                return Token(TokenType.FIELD, field_name)
                
            # Operators
            if self.current_char in '=!><~':
                op = self.operator()
                return Token(TokenType.OPERATOR, op)
                
            # String values
            if self.current_char in '\'"':
                value = self.string()
                return Token(TokenType.VALUE, value)
                
            # Numeric values
            if self.current_char.isdigit() or self.current_char == '.':
                value = self.number()
                return Token(TokenType.VALUE, value)
                
            # Unknown character
            self.error()
            
        return Token(TokenType.EOF, "")


class Parser:
    """Parser for the query language."""
    
    def __init__(self, lexer: Lexer):
        """
        Initialize the parser.
        
        Args:
            lexer: Lexer to use for tokenization
        """
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
        
    def error(self):
        """Raise a syntax error."""
        raise Exception("Syntax error")
        
    def eat(self, token_type: TokenType):
        """
        Consume the current token if it matches the expected type.
        
        Args:
            token_type: Expected token type
        """
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()
            
    def condition(self) -> Dict[str, Any]:
        """
        Parse a condition (field operator value).
        
        Returns:
            Dict representing the condition
        """
        # Get field name
        field = self.current_token.value
        self.eat(TokenType.FIELD)
        
        # Get operator
        operator = self.current_token.value
        self.eat(TokenType.OPERATOR)
        
        # Get value
        value = self.current_token.value
        self.eat(TokenType.VALUE)
        
        return {
            "type": "condition",
            "field": field,
            "operator": operator,
            "value": value
        }
        
    def factor(self) -> Dict[str, Any]:
        """
        Parse a factor (condition, parenthesized expression, or NOT expression).
        
        Returns:
            Dict representing the parsed factor
        """
        token = self.current_token
        
        # Handle NOT expressions
        if token.type == TokenType.LOGICAL and token.value == "NOT":
            self.eat(TokenType.LOGICAL)
            return {
                "type": "unary_op",
                "operator": "NOT",
                "right": self.factor()
            }
            
        # Handle parenthesized expressions
        if token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
            
        # Handle basic conditions
        return self.condition()
        
    def term(self) -> Dict[str, Any]:
        """
        Parse a term (factor AND factor...).
        
        Returns:
            Dict representing the parsed term
        """
        node = self.factor()
        
        while (self.current_token.type == TokenType.LOGICAL and 
               self.current_token.value == "AND"):
            self.eat(TokenType.LOGICAL)
            node = {
                "type": "binary_op",
                "operator": "AND",
                "left": node,
                "right": self.factor()
            }
            
        return node
        
    def expr(self) -> Dict[str, Any]:
        """
        Parse an expression (term OR term...).
        
        Returns:
            Dict representing the parsed expression
        """
        node = self.term()
        
        while (self.current_token.type == TokenType.LOGICAL and 
               self.current_token.value == "OR"):
            self.eat(TokenType.LOGICAL)
            node = {
                "type": "binary_op",
                "operator": "OR",
                "left": node,
                "right": self.term()
            }
            
        return node
        
    def parse(self) -> Dict[str, Any]:
        """
        Parse the entire query.
        
        Returns:
            Dict representing the parsed query
        """
        return self.expr()


class Interpreter:
    """Interpreter for executing queries on log entries."""
    
    def __init__(self, ast: Dict[str, Any]):
        """
        Initialize the interpreter.
        
        Args:
            ast: Abstract syntax tree from parser
        """
        self.ast = ast
        
    @staticmethod
    def eval_condition(condition: Dict[str, Any], entry: Dict[str, Any]) -> bool:
        """
        Evaluate a condition against a log entry.
        
        Args:
            condition: Condition to evaluate
            entry: Log entry to evaluate against
            
        Returns:
            True if the condition is satisfied
        """
        field = condition["field"]
        op = condition["operator"]
        value = condition["value"]
        
        # Check if field exists in entry
        if field not in entry:
            return False
            
        entry_value = entry[field]
        
        # Handle different operators
        if op == "=":
            return entry_value == value
        elif op == "!=":
            return entry_value != value
        elif op == ">":
            return entry_value > value
        elif op == ">=":
            return entry_value >= value
        elif op == "<":
            return entry_value < value
        elif op == "<=":
            return entry_value <= value
        elif op == "~":  # contains
            # For string contains
            if isinstance(entry_value, str) and isinstance(value, str):
                return value in entry_value
            # For list contains
            elif isinstance(entry_value, list):
                return value in entry_value
            return False
        elif op == "!~":  # does not contain
            # For string not contains
            if isinstance(entry_value, str) and isinstance(value, str):
                return value not in entry_value
            # For list not contains
            elif isinstance(entry_value, list):
                return value not in entry_value
            return True
        else:
            raise Exception(f"Unknown operator: {op}")
            
    def eval(self, node: Dict[str, Any], entry: Dict[str, Any]) -> bool:
        """
        Evaluate an AST node against a log entry.
        
        Args:
            node: AST node to evaluate
            entry: Log entry to evaluate against
            
        Returns:
            True if the node condition is satisfied
        """
        node_type = node["type"]
        
        if node_type == "condition":
            return self.eval_condition(node, entry)
        elif node_type == "unary_op":
            op = node["operator"]
            if op == "NOT":
                return not self.eval(node["right"], entry)
        elif node_type == "binary_op":
            op = node["operator"]
            if op == "AND":
                return self.eval(node["left"], entry) and self.eval(node["right"], entry)
            elif op == "OR":
                return self.eval(node["left"], entry) or self.eval(node["right"], entry)
                
        raise Exception(f"Unknown node type: {node_type}")
        
    def interpret(self, entry: Dict[str, Any]) -> bool:
        """
        Interpret the query AST against a log entry.
        
        Args:
            entry: Log entry to evaluate
            
        Returns:
            True if the entry matches the query
        """
        return self.eval(self.ast, entry)


class LogQuery:
    """Query system for searching and filtering logs."""
    
    def __init__(self, query: str = None):
        """
        Initialize the log query system.
        
        Args:
            query: Optional query string
        """
        self.query_str = query
        self.interpreter = None
        
        if query:
            self.compile(query)
            
    def compile(self, query: str) -> None:
        """
        Compile a query string.
        
        Args:
            query: Query string to compile
        """
        try:
            lexer = Lexer(query)
            parser = Parser(lexer)
            ast = parser.parse()
            self.interpreter = Interpreter(ast)
            self.query_str = query
        except Exception as e:
            logger.error(f"Error compiling query: {query}", error=str(e))
            raise
            
    def matches(self, entry: Dict[str, Any]) -> bool:
        """
        Check if a log entry matches the query.
        
        Args:
            entry: Log entry to check
            
        Returns:
            True if the entry matches the query
        """
        if not self.interpreter:
            return True  # Empty query matches everything
            
        try:
            return self.interpreter.interpret(entry)
        except Exception as e:
            logger.error(f"Error evaluating query against entry", error=str(e), entry=entry)
            return False
            
    @staticmethod
    def parse_log_file(file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a log file into a list of entries.
        
        Args:
            file_path: Path to log file
            
        Returns:
            List of parsed log entries
        """
        entries = []
        
        try:
            # Check file type
            if file_path.endswith('.gz'):
                import gzip
                opener = gzip.open
                mode = 'rt'
            else:
                opener = open
                mode = 'r'
                
            with opener(file_path, mode) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
        except Exception as e:
            logger.error(f"Error parsing log file: {file_path}", error=str(e))
            
        return entries
        
    def search_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Search a log file for entries matching the query.
        
        Args:
            file_path: Path to log file
            
        Returns:
            List of matching log entries
        """
        results = []
        
        # Parse log file
        entries = self.parse_log_file(file_path)
        
        # Filter entries
        for entry in entries:
            if self.matches(entry):
                results.append(entry)
                
        return results
        
    def search_directory(
        self,
        directory: str = None,
        pattern: str = "*.log*",
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Search all log files in a directory for entries matching the query.
        
        Args:
            directory: Directory to search (defaults to LOG_DIR)
            pattern: File pattern to match
            limit: Maximum number of results
            
        Returns:
            List of matching log entries
        """
        results = []
        
        # Use default log directory if none provided
        if directory is None:
            directory = LOG_DIR
            
        # Get all matching files
        dir_path = Path(directory)
        log_files = list(dir_path.glob(pattern))
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Search each file
        for file_path in log_files:
            file_results = self.search_file(str(file_path))
            results.extend(file_results)
            
            if len(results) >= limit:
                results = results[:limit]
                break
                
        return results


def quick_search(
    query: str,
    directory: str = None,
    pattern: str = "*.log*",
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Quick search for log entries matching a query.
    
    Args:
        query: Query string
        directory: Directory to search
        pattern: File pattern to match
        limit: Maximum number of results
        
    Returns:
        List of matching log entries
    """
    log_query = LogQuery(query)
    return log_query.search_directory(directory, pattern, limit)


# Example query language usage:
# level = "error" AND component = "api"
# timestamp > "2023-01-01" AND (level = "error" OR level = "warning")
# message ~ "timeout" AND NOT component = "scheduler"

if __name__ == "__main__":
    # Simple example of using the query language
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python log_query.py 'query'")
        sys.exit(1)
        
    query_str = sys.argv[1]
    results = quick_search(query_str)
    
    print(f"Found {len(results)} matching entries")
    for i, entry in enumerate(results[:10]):  # Show top 10
        print(f"{i+1}. {json.dumps(entry)}")
        
    if len(results) > 10:
        print(f"... and {len(results) - 10} more")
