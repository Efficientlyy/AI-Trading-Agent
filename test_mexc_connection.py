"""
Test script to verify MEXC API credentials and connectivity
"""
import os
import asyncio
from dotenv import load_dotenv
from ai_trading_agent.data_acquisition.mexc_spot_v3_client import MexcSpotV3Client

# Load environment variables
load_dotenv()

# Get API credentials
api_key = os.getenv("MEXC_API_KEY")
api_secret = os.getenv("MEXC_API_SECRET")

print(f"MEXC API Key found: {'Yes' if api_key else 'No'}")
print(f"MEXC API Secret found: {'Yes' if api_secret else 'No'}")

async def test_mexc_connection():
    print("\nInitializing MEXC API client...")
    client = MexcSpotV3Client(api_key=api_key, api_secret=api_secret)
    
    try:
        print("Testing MEXC system status...")
        status = await client.get_system_status()
        print(f"System status: {status}")
        
        print("\nTesting ticker data for BTC/USDT...")
        ticker = await client.get_ticker("BTC/USDT")
        print(f"Ticker data: {ticker}")
        
        print("\nSuccess! Your MEXC API credentials are working.")
    except Exception as e:
        print(f"Error connecting to MEXC API: {e}")
    finally:
        print("\nClosing all connections...")
        await client.close_session()
        print("All connections closed.")

if __name__ == "__main__":
    asyncio.run(test_mexc_connection())