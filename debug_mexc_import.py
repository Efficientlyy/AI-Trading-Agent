"""
Debug script to identify MEXC import errors
"""
import sys
import traceback

def test_imports():
    print("Testing imports individually to find the failing module...")
    
    try:
        print("Importing dotenv...")
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ dotenv imported successfully")
    except Exception as e:
        print(f"✗ Error importing dotenv: {e}")
        traceback.print_exc()
    
    try:
        print("\nImporting MEXC config...")
        from ai_trading_agent.config.mexc_config import MEXC_CONFIG, API_ENDPOINTS
        print(f"✓ MEXC config imported successfully")
        print(f"API Key available: {'Yes' if MEXC_CONFIG.get('API_KEY') else 'No'}")
        print(f"API Secret available: {'Yes' if MEXC_CONFIG.get('API_SECRET') else 'No'}")
    except Exception as e:
        print(f"✗ Error importing MEXC config: {e}")
        traceback.print_exc()
    
    try:
        print("\nImporting MexcConnector...")
        from ai_trading_agent.data_acquisition.mexc_connector import MexcConnector
        print("✓ MexcConnector imported successfully")
    except Exception as e:
        print(f"✗ Error importing MexcConnector: {e}")
        traceback.print_exc()
    
    try:
        print("\nImporting MockMexcConnector...")
        from ai_trading_agent.data_acquisition.mock_mexc_connector import MockMexcConnector
        print("✓ MockMexcConnector imported successfully")
    except Exception as e:
        print(f"✗ Error importing MockMexcConnector: {e}")
        traceback.print_exc()
    
    try:
        print("\nImporting connector factory...")
        from ai_trading_agent.data_acquisition.mexc_connector_factory import create_mexc_connector
        print("✓ MEXC connector factory imported successfully")
    except Exception as e:
        print(f"✗ Error importing connector factory: {e}")
        traceback.print_exc()
    
    try:
        print("\nImporting MEXC WebSocket API...")
        from ai_trading_agent.api.mexc_websocket import router as mexc_ws_router
        print("✓ MEXC WebSocket API imported successfully")
    except Exception as e:
        print(f"✗ Error importing MEXC WebSocket API: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()
    print("\nDebug complete.")