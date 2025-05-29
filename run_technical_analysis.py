# Trivial comment to force re-evaluation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import json

# --- Early Logging Setup ---
# Attempt to configure logging before any other project module can implicitly initialize it.
from ai_trading_agent.common.logging_config import setup_logging
# Temporarily get log level from default_config structure for early setup
# This is a bit of a hack, assuming default_config structure is known and stable here.
# A more robust solution might involve environment variables for initial log level if needed this early.
_DEFAULT_AGENT_LOG_LEVEL = {
    "technical_analysis_agent": {
        "log_level": "INFO"  # Default to INFO for this pre-setup
    }
}
temp_initial_log_level = _DEFAULT_AGENT_LOG_LEVEL.get("technical_analysis_agent", {}).get("log_level", "INFO")
setup_logging(log_level=temp_initial_log_level, colorize_console=False)

# --- Prime stdout and stderr ---
# Attempt to send a harmless initial character to satisfy PowerShell's redirection quirks.
# --- End Priming ---

# --- End Early Logging Setup ---

# Assuming your project structure allows these imports
from ai_trading_agent.agent.indicator_engine import IndicatorEngine
from ai_trading_agent.agent.strategy_manager import StrategyManager 
# from ai_trading_agent.agent.strategy_manager import SignalDirection 
from ai_trading_agent.agent.technical_analysis_agent import TechnicalAnalysisAgent
from ai_trading_agent.common.utils import get_logger # Now this will get the pre-configured logger

# --- Configuration (from Step 1.5) ---
default_config = {
    "symbols": ["AAPL", "MSFT"],
    "indicator_engine": {
        "cache_enabled": True,
        "trend": {
            "sma": {
                "enabled": True, 
                "periods": [20, 50]
            },
            "ema": {
                "enabled": True, 
                "periods": [12, 26]
            }
        },
        "momentum": {
            "rsi": {
                "enabled": True, 
                "period": 14  
            }
        },
        "volatility": {
            "bollinger_bands": {
                "enabled": True, 
                "periods": [20], 
                "std_devs": [2]
            }
        },
        "features": {
            "lag_features": {
                "enabled": True, 
                "lags": [1, 2, 3, 5]
            }
        }
    },
    "strategy_manager": {
        "strategies": {
            "ma_cross": {
                "enabled": True,
                "fast_ma": {"type": "ema", "period": 12},
                "slow_ma": {"type": "ema", "period": 26},
                "signal_threshold": 0.005, 
                "min_lookback": 30,
                "confirmation_period": 3,
                "max_volatility": 5.0, 
                "volatility_adjustment_factor": 1.5, 
                "volatility_exit_factor": 1.5 
            },
            "rsi_ob_os": {
                "enabled": True,
                "period": 14,
                "overbought": 70,
                "oversold": 30,
                "overbought_exit": 65, 
                "oversold_exit": 35,   
                "signal_threshold": 0.3, 
                "min_divergence": 5, 
                "confirmation_periods": 2,
                "max_volatility": 4.0, 
                "volatility_adjustment": 5 
            }
        }
    },
    "technical_analysis_agent": {
        "log_level": "DEBUG"
    }
}

# --- Mock Market Data Generation ---
def generate_mock_market_data(symbols, days=100):
    market_data = {}
    for symbol in symbols:
        dates = pd.to_datetime([datetime.now() - timedelta(days=i) for i in range(days)][::-1])
        data = {
            'open': np.random.rand(days) * 100 + 50,
            'high': np.random.rand(days) * 5 + 150, 
            'low': np.random.rand(days) * 5 + 45,   
            'close': np.random.rand(days) * 100 + 50,
            'volume': np.random.rand(days) * 100000 + 10000
        }
        df = pd.DataFrame(data, index=dates)
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(days) * 5 
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(days) * 5   
        market_data[symbol] = df
    return market_data

# --- Main Execution ---
if __name__ == "__main__":
    # Logging is already set up. We can get the logger instance directly.
    # The log level set by the early setup_logging will be used unless reconfigured by TAA itself.
    logger = get_logger("TechnicalAnalysisRun") # Level is already set by early setup_logging

    logger.debug("DEBUG_PRINT: run_technical_analysis.py - AFTER IMPORTS, START OF MAIN")
    logger.info("Starting Technical Analysis Agent test run...")

    symbols_to_process = default_config["symbols"]
    
    try:
        tech_agent = TechnicalAnalysisAgent(
            agent_id_suffix="test_run_001", 
            name="TestTechnicalAgent",
            symbols=symbols_to_process,
            config_details=default_config
        )
        logger.info("TechnicalAnalysisAgent initialized.")
    except Exception as e:
        logger.error(f"Error initializing TechnicalAnalysisAgent: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Generating mock market data for: {symbols_to_process}")
    mock_data = generate_mock_market_data(symbols_to_process, days=200) 
    
    signals_result = None
    processing_successful = False
    try:
        logger.info("Starting agent processing...")
        signals_result = tech_agent.process(data={"market_data": mock_data})
        if signals_result is not None:
            logger.info(f"Processing complete. Generated {len(signals_result)} signals.")
            if not signals_result:
                logger.info("No signals generated.")
            else:
                for signal in signals_result:
                    logger.info(f"Generated signal: {signal}")
            processing_successful = True # Mark as successful if process returned (even if no signals)
        else:
            logger.error("Agent processing returned None, indicating an issue.")

    except Exception as e:
        logger.error(f"Error during agent processing: {e}", exc_info=True)
    finally:
        logger.info("Technical Analysis Agent test run finished.")
        if processing_successful:
            logger.info("PYTHON_SCRIPT_SUCCESS") # Replaced print with logger

    if signals_result:
        logger.info("--- Generated Signals ---")
        for i, signal in enumerate(signals_result):
            logger.info(f"Signal {i+1}:")
            # Pretty print the signal dictionary
            # Assuming signal structure is {'type': 'technical_signal', 'payload': {...}}
            payload = signal.get('payload', signal) # Handle if payload is not a sub-dict
            for key, value in payload.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"    {sub_key}: {sub_value}")
                else:
                    logger.info(f"  {key}: {value}")
            logger.info("-" * 20)
    else:
        logger.info("No signals generated.")

    if processing_successful:
        sys.exit(0) # Explicit success exit
    else:
        sys.exit(1) # Explicit failure exit if processing was not successful
