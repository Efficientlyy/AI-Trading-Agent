# Main entry point to run the AI Trading Agent

import asyncio
import logging
import sys
import os

# Ensure the package directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_trading_agent.agent_orchestrator import start_agent

if __name__ == "__main__":
    try:
        asyncio.run(start_agent())
    except KeyboardInterrupt:
        logging.info("Agent run initiated shutdown via KeyboardInterrupt.")
    except Exception as e:
        logging.critical(f"Critical error in agent execution: {e}", exc_info=True)
