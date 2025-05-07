"""
Debug script for paper trading sessions.

This script helps debug issues with paper trading sessions by:
1. Creating a paper trading session
2. Retrieving the sessions
3. Printing the exact data being returned
"""

import requests
import json
import uuid
from datetime import datetime

# API base URL
API_BASE_URL = "http://localhost:8000/api"

def create_session():
    """Create a paper trading session."""
    print("Creating a paper trading session...")
    
    session_data = {
        "name": f"Debug Session {uuid.uuid4().hex[:8]}",
        "description": "Session created for debugging",
        "exchange": "binance",
        "symbols": ["BTC/USDT"],
        "strategy": "default",
        "initial_capital": 10000.0
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/paper-trading/sessions",
            json=session_data
        )
        response.raise_for_status()
        session = response.json()
        print(f"Session created successfully: {session['session_id']}")
        print(f"Session data: {json.dumps(session, indent=2)}")
        return session
    except Exception as e:
        print(f"Error creating session: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def get_sessions():
    """Get all paper trading sessions."""
    print("\nRetrieving all paper trading sessions...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/paper-trading/sessions")
        response.raise_for_status()
        sessions = response.json()
        print(f"Retrieved {len(sessions)} sessions")
        print(f"Sessions data: {json.dumps(sessions, indent=2)}")
        return sessions
    except Exception as e:
        print(f"Error retrieving sessions: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def create_session_via_start():
    """Create a paper trading session via the /start endpoint."""
    print("\nCreating a paper trading session via /start endpoint...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/paper-trading/start")
        response.raise_for_status()
        session = response.json()
        print(f"Session created successfully: {session['session_id']}")
        print(f"Session data: {json.dumps(session, indent=2)}")
        return session
    except Exception as e:
        print(f"Error creating session via /start: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def main():
    """Main function."""
    print("=== Paper Trading Debug Script ===")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Current time: {datetime.now().isoformat()}")
    print("=" * 40)
    
    # Get existing sessions
    existing_sessions = get_sessions()
    
    # Create a new session
    new_session = create_session()
    
    # Get sessions again to see if the new session is included
    updated_sessions = get_sessions()
    
    # Create a session via the /start endpoint
    start_session = create_session_via_start()
    
    # Get sessions again to see if the start session is included
    final_sessions = get_sessions()
    
    print("\n=== Summary ===")
    print(f"Initial session count: {len(existing_sessions) if existing_sessions else 0}")
    print(f"Final session count: {len(final_sessions) if final_sessions else 0}")
    print("=" * 40)

if __name__ == "__main__":
    main()
