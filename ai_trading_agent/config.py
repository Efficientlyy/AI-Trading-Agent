import os
import logging
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
# If .env is not found, it will silently continue (useful for production where env vars are set differently)
load_dotenv()

# API Keys
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Other potential API keys (add as needed)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Add other configuration variables if needed
# Example: DEFAULT_CRYPTO_PAIRS = os.getenv("DEFAULT_CRYPTO_PAIRS", "BTC/USD,ETH/USD").split(',')

# Validate essential keys
def validate_config():
    missing_keys = []
    # Add keys here that are absolutely essential for the core functionality
    # Example: if not TWELVEDATA_API_KEY:
    #     missing_keys.append("TWELVEDATA_API_KEY")
    # Example: if not ALPHA_VANTAGE_API_KEY:
    #     missing_keys.append("ALPHA_VANTAGE_API_KEY")

    if missing_keys:
        logger.error(f"Missing essential environment variables: {', '.join(missing_keys)}")
        logger.error("Please ensure they are set in your .env file or system environment.")
        # Optionally, raise an exception or exit
        # raise ValueError(f"Missing essential environment variables: {', '.join(missing_keys)}")
    else:
        logger.info("Configuration loaded successfully.")

# Run validation on import
validate_config()

# Example usage:
# from ai_trading_agent.config import TWELVEDATA_API_KEY
# print(TWELVEDATA_API_KEY)
