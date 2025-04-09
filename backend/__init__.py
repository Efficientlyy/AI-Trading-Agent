# Initialize environment variables on backend startup

import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print("Warning: .env file not found, using system environment variables")