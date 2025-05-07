import os
import pickle
import json
import shutil

# Path to the main sessions file used by the full implementation
sessions_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ai_trading_agent",
    "api",
    "routers",
    "paper_trading_sessions.pkl"
)

# Path to the simplified implementation's sessions file (to be removed)
simple_sessions_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ai_trading_agent",
    "api",
    "routers",
    "paper_sessions.json"
)

# Create an empty dictionary for sessions
empty_sessions = {}

# Clear the main sessions file
print(f"Clearing sessions in: {sessions_file}")
with open(sessions_file, 'wb') as f:
    pickle.dump(empty_sessions, f)

# Remove the simple sessions file if it exists
if os.path.exists(simple_sessions_file):
    print(f"Removing simplified sessions file: {simple_sessions_file}")
    # Create a backup before removing
    backup_file = f"{simple_sessions_file}.bak"
    shutil.copy2(simple_sessions_file, backup_file)
    print(f"Created backup at: {backup_file}")
    
    # Write an empty JSON object to the file
    with open(simple_sessions_file, 'w') as f:
        json.dump({}, f)
    print(f"Cleared simplified sessions file")

print("All paper trading sessions have been cleared.")
print("You'll need to restart the backend server for changes to take effect.")
print("Note: The system is now configured to use the full paper trading implementation.")

