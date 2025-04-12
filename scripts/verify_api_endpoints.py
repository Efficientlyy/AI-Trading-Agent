#!/usr/bin/env python
"""
API endpoints verification script for the AI Trading Agent.

This script verifies that all API endpoints are using database repositories.
"""

import os
import sys
import re
import inspect
import logging
from typing import Dict, List, Set, Tuple, Any

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import the API module
try:
    from backend.api import app
    from fastapi import APIRouter, Depends
    from fastapi.routing import APIRoute
except ImportError as e:
    logger.error(f"Error importing API module: {e}")
    sys.exit(1)

# Repository classes to check for
REPOSITORY_CLASSES = [
    "UserRepository",
    "StrategyRepository",
    "OptimizationRepository",
    "BacktestRepository",
    "AssetRepository",
    "OHLCVRepository",
    "SentimentRepository",
]

# Repository instances to check for
REPOSITORY_INSTANCES = [
    "user_repository",
    "strategy_repository",
    "optimization_repository",
    "backtest_repository",
    "asset_repository",
    "ohlcv_repository",
    "sentiment_repository",
]

def get_api_file_content() -> str:
    """Get the content of the API file."""
    api_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "api.py")
    with open(api_file, "r") as f:
        return f.read()

def count_repository_usage(file_content: str) -> Dict[str, int]:
    """Count the number of times each repository is used in the API file."""
    repository_usage = {}
    for repo_class in REPOSITORY_CLASSES:
        # Count occurrences of repository class
        count = len(re.findall(r'\b' + repo_class + r'\b', file_content))
        repository_usage[repo_class] = count
    return repository_usage

def get_endpoint_handlers() -> List[Tuple[str, Any]]:
    """Get all endpoint handlers from the FastAPI app."""
    handlers = []
    
    # Get all routes from the app
    for route in app.routes:
        if isinstance(route, APIRoute):
            endpoint_path = route.path
            endpoint_handler = route.endpoint
            handlers.append((endpoint_path, endpoint_handler))
    
    return handlers

def check_handler_for_repositories(handler: Any) -> Tuple[bool, Set[str]]:
    """Check if a handler uses repositories."""
    # Get the source code of the handler
    try:
        source = inspect.getsource(handler)
    except (TypeError, OSError):
        # If we can't get the source, assume it doesn't use repositories
        return False, set()
    
    # Check if any repository instance is used in the handler
    used_repos = set()
    for repo_instance in REPOSITORY_INSTANCES:
        if re.search(r'\b' + repo_instance + r'\.', source):
            # Extract the repository class from the instance name
            repo_class = repo_instance.replace('_repository', 'Repository')
            repo_class = repo_class[0].upper() + repo_class[1:]
            used_repos.add(repo_class)
    
    return len(used_repos) > 0, used_repos

def verify_api_endpoints():
    """Verify that all API endpoints are using database repositories."""
    logger.info("Verifying API endpoints...")
    
    # Get the API file content
    api_content = get_api_file_content()
    
    # Count repository usage
    repo_usage = count_repository_usage(api_content)
    logger.info("Repository usage in API file:")
    for repo, count in repo_usage.items():
        logger.info(f"  {repo}: {count} occurrences")
    
    # Get all endpoint handlers
    try:
        handlers = get_endpoint_handlers()
        logger.info(f"Found {len(handlers)} API endpoints")
    except Exception as e:
        logger.error(f"Error checking endpoint handlers: {e}")
        handlers = []
    
    # Check each handler for repository usage
    endpoints_with_repos = []
    endpoints_without_repos = []
    
    for path, handler in handlers:
        try:
            uses_repos, used_repos = check_handler_for_repositories(handler)
            if uses_repos:
                endpoints_with_repos.append((path, used_repos))
            else:
                endpoints_without_repos.append(path)
        except Exception as e:
            logger.error(f"Error checking handler for {path}: {e}")
    
    # Print results
    logger.info(f"\n{len(endpoints_with_repos)} endpoints using repositories:")
    for path, repos in endpoints_with_repos:
        logger.info(f"  {path}: {', '.join(repos)}")
    
    logger.info(f"\n{len(endpoints_without_repos)} endpoints NOT using repositories:")
    for path in endpoints_without_repos:
        logger.info(f"  {path}")
    
    # Calculate percentage of endpoints using repositories
    if handlers:
        percentage = (len(endpoints_with_repos) / len(handlers)) * 100
        logger.info(f"\n{percentage:.2f}% of endpoints are using repositories")
    
    logger.info("\nAPI endpoints verification completed")

if __name__ == "__main__":
    verify_api_endpoints()
