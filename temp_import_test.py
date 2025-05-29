import sys
import os
import logging
import importlib
import pandas as pd

# Configure basic logging to capture output
log_file_path = os.path.join(os.getcwd(), "temp_import_test_output.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler(sys.stdout) # Also print to console for immediate feedback
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"--- temp_import_test.py (logging to {log_file_path}) ---")
logger.info(f"sys.executable: {sys.executable}")
logger.info(f"Initial sys.path: {sys.path}")

# Ensure the user's site-packages directory is in sys.path
user_site_packages = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Python', f"Python{sys.version_info.major}{sys.version_info.minor}", 'site-packages')
logger.info(f"Expected user site-packages: {user_site_packages}")

if user_site_packages not in sys.path:
    sys.path.append(user_site_packages)
    logger.info(f"Added user site-packages '{user_site_packages}' to sys.path.")
    logger.info(f"Updated sys.path: {sys.path}")
else:
    logger.info(f"User site-packages '{user_site_packages}' IS ALREADY in sys.path.")

# Create a sample pandas Series for testing
sample_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

logger.info("\nAttempting to import ai_trading_agent_rs directly...")
try:
    ai_trading_agent_rs = importlib.import_module('ai_trading_agent_rs')
    logger.info(f"Successfully imported ai_trading_agent_rs: {ai_trading_agent_rs}")
    logger.info(f"Attributes of ai_trading_agent_rs: {dir(ai_trading_agent_rs)}")

    logger.info("\nAttempting to call calculate_sma_rust_direct...")
    try:
        sma_result = ai_trading_agent_rs.calculate_sma_rust_direct(sample_series.tolist(), [10]) # Pass list
        logger.info(f"SMA (Rust direct) result: {sma_result}")
    except Exception as e:
        logger.error(f"Error calling calculate_sma_rust_direct: {e}")

    logger.info("\nAttempting to call create_ema_features_rs...")
    try:
        ema_result = ai_trading_agent_rs.create_ema_features_rs(sample_series.tolist(), [10], None) # Pass list, Alpha = None
        logger.info(f"EMA (Rust direct) result: {ema_result}")
    except Exception as e:
        logger.error(f"Error calling create_ema_features_rs: {e}")

except ImportError as e_direct_import:
    logger.error(f"Failed to import ai_trading_agent_rs directly: {e_direct_import}")
    ai_trading_agent_rs = None # Ensure it's None if import fails
except Exception as e_attrs: # Catch other potential errors during initial setup/calls
    logger.error(f"Error during direct import or initial testing of ai_trading_agent_rs: {e_attrs}")
    ai_trading_agent_rs = None # Ensure it's None

# Proceed with IndicatorEngine checks only if ai_trading_agent_rs was successfully imported and tested
if ai_trading_agent_rs:
    logger.info("\nAttempting to import IndicatorEngine module and check its Rust integration points...")
    indicator_engine_imported_successfully = False
    try:
        from ai_trading_agent.agent.indicator_engine import IndicatorEngine
        from ai_trading_agent.rust_integration import indicators as rust_indicators_module
        from ai_trading_agent.rust_integration import features as rust_features_module
        indicator_engine_imported_successfully = True
    except ImportError as e:
        logger.error(f"Failed to import IndicatorEngine or its Rust integration modules: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during import of IndicatorEngine or rust modules: {e}", exc_info=True)

    if indicator_engine_imported_successfully:
        logger.info("Successfully imported IndicatorEngine and rust_integration modules.")

        # Check attributes on the ai_trading_agent_rs module directly (as seen by this test script)
        logger.info(f"Direct check on ai_trading_agent_rs for 'calculate_sma_rust_direct': {hasattr(ai_trading_agent_rs, 'calculate_sma_rust_direct')}")
        logger.info(f"Direct check on ai_trading_agent_rs for 'create_ema_features_rs': {hasattr(ai_trading_agent_rs, 'create_ema_features_rs')}")

        # Check RUST_AVAILABLE from the features module
        logger.info(f"rust_integration.features.RUST_FEATURES_AVAILABLE: {getattr(rust_features_module, 'RUST_FEATURES_AVAILABLE', 'Not Defined')}")
        if hasattr(rust_features_module, '_create_lag_features_rs_rust'):
            logger.info(f"rust_integration.features._create_lag_features_rs_rust is callable: {callable(rust_features_module._create_lag_features_rs_rust)}")
        else:
            logger.warning("rust_integration.features does not have _create_lag_features_rs_rust attribute.")

        # THIS IS THE MOST IMPORTANT CHECK NOW for indicators
        logger.info(f"rust_integration.indicators.RUST_AVAILABLE: {getattr(rust_indicators_module, 'RUST_AVAILABLE', 'Not Defined')}")

        if hasattr(rust_indicators_module, 'RUST_AVAILABLE'):
            if rust_indicators_module.RUST_AVAILABLE.get("sma"):
                logger.info("SUCCESS: rust_integration.indicators reports SMA is available via Rust.")
            else:
                logger.warning("FAILURE: rust_integration.indicators reports SMA is NOT available via Rust.")

            if rust_indicators_module.RUST_AVAILABLE.get("ema"):
                logger.info("SUCCESS: rust_integration.indicators reports EMA is available via Rust.")
            else:
                logger.warning("FAILURE: rust_integration.indicators reports EMA is NOT available via Rust.")
            
            # You can add checks for other indicators like MACD, RSI etc. here if needed
            if rust_indicators_module.RUST_AVAILABLE.get("macd"):
                logger.info("SUCCESS: rust_integration.indicators reports MACD is available via Rust.")
            else:
                logger.warning("FAILURE: rust_integration.indicators reports MACD is NOT available via Rust.")

        logger.info("\nInstantiating IndicatorEngine...")
        try:
            engine = IndicatorEngine(config={}) # Pass an empty config dictionary
            logger.info("IndicatorEngine instantiated.")
            # Further tests on the engine instance could go here
        except Exception as e:
            logger.error(f"Error instantiating IndicatorEngine: {e}", exc_info=True)
    else:
        logger.error("Skipping IndicatorEngine related checks due to import failure.")
else:
    logger.error("Skipping all IndicatorEngine and rust_integration module checks because ai_trading_agent_rs failed to import or had errors during initial setup.")

logger.info("\n--- End of temp_import_test.py ---")

if __name__ == "__main__":
    pass
