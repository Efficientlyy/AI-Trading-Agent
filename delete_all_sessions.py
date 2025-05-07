import requests
import json

# API base URL
API_BASE_URL = "http://localhost:8000/api"

def get_all_sessions():
    """Get all paper trading sessions."""
    url = f"{API_BASE_URL}/paper-trading/sessions"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        sessions = response.json()
        print(f"Found {len(sessions)} paper trading sessions")
        return sessions
    else:
        print(f"Failed to get sessions. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return []

def delete_session(session_id):
    """Delete a paper trading session."""
    url = f"{API_BASE_URL}/paper-trading/{session_id}"
    
    response = requests.delete(url)
    
    if response.status_code == 200:
        print(f"Successfully deleted session {session_id}")
        return True
    else:
        print(f"Failed to delete session {session_id}. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    # Get all sessions
    sessions = get_all_sessions()
    
    if not sessions:
        print("No sessions to delete")
        return
    
    # Delete each session
    for session in sessions:
        session_id = session.get("session_id")
        if session_id:
            delete_session(session_id)
    
    # Verify all sessions are deleted
    remaining_sessions = get_all_sessions()
    if not remaining_sessions:
        print("All sessions have been deleted successfully")
    else:
        print(f"There are still {len(remaining_sessions)} sessions remaining")

if __name__ == "__main__":
    main()
