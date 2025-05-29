"""
WebSocket Client Test Tool

This script provides a WebSocket client that can connect to the AI Trading Agent
WebSocket server and send test commands, particularly for testing the 'set_data_mode' action.
"""

import sys
import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import websockets
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WSClientTest")

# Default WebSocket URL
DEFAULT_WS_URL = "ws://localhost:8000/ws/test-client"

async def test_websocket_set_data_mode(ws_url, session_id=None):
    """
    Test the WebSocket 'set_data_mode' action.
    
    Args:
        ws_url: WebSocket server URL
        session_id: Optional session ID to use (will be appended to URL)
    """
    url = ws_url
    if session_id:
        # Replace any existing session ID in the URL
        if '/ws/' in url:
            url = url.split('/ws/')[0] + f'/ws/{session_id}'
        else:
            url = url + f'/{session_id}'
    
    logger.info(f"Connecting to WebSocket server at {url}")
    
    try:
        async with websockets.connect(url) as websocket:
            logger.info("Connected to WebSocket server")
            
            # Wait for welcome message
            response = await websocket.recv()
            logger.info(f"Received: {response}")
            
            # Test sequence: Toggle between mock and real data modes
            test_sequence = [
                {"action": "set_data_mode", "mode": "mock"},
                {"action": "set_data_mode", "mode": "real"},
                {"action": "set_data_mode", "mode": "mock"}
            ]
            
            for i, test_msg in enumerate(test_sequence):
                # Send message
                logger.info(f"Sending message {i+1}/{len(test_sequence)}: {test_msg}")
                await websocket.send(json.dumps(test_msg))
                
                # Wait for response
                response = await websocket.recv()
                try:
                    response_data = json.loads(response)
                    logger.info(f"Received response: {json.dumps(response_data, indent=2)}")
                    
                    # Verify response matches expected format
                    if response_data.get("type") == "data_mode_update":
                        logger.info(f"✅ Received data_mode_update response with mode: {response_data.get('mode')}")
                        # Verify mode matches what we set
                        if response_data.get("mode") == test_msg.get("mode"):
                            logger.info(f"✅ Mode matches requested mode: {test_msg.get('mode')}")
                        else:
                            logger.warning(f"❌ Mode mismatch! Requested: {test_msg.get('mode')}, Received: {response_data.get('mode')}")
                    else:
                        logger.warning(f"❌ Unexpected response type: {response_data.get('type')}")
                    
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON response: {response}")
                
                # Wait between tests
                await asyncio.sleep(2)
            
            # Send a ping to verify connection is still alive
            ping_msg = {"action": "ping"}
            logger.info(f"Sending ping: {ping_msg}")
            await websocket.send(json.dumps(ping_msg))
            
            # Wait for pong
            response = await websocket.recv()
            try:
                response_data = json.loads(response)
                if response_data.get("type") == "pong":
                    logger.info("✅ Received pong response")
                else:
                    logger.warning(f"❌ Unexpected response to ping: {response_data}")
            except json.JSONDecodeError:
                logger.error(f"Received invalid JSON response: {response}")
            
            logger.info("WebSocket test completed successfully")
    
    except Exception as e:
        logger.error(f"Error in WebSocket test: {str(e)}")
        return False
    
    return True


async def interactive_websocket_client(ws_url, session_id=None):
    """
    Interactive WebSocket client for manual testing.
    
    Args:
        ws_url: WebSocket server URL
        session_id: Optional session ID to use (will be appended to URL)
    """
    url = ws_url
    if session_id:
        # Replace any existing session ID in the URL
        if '/ws/' in url:
            url = url.split('/ws/')[0] + f'/ws/{session_id}'
        else:
            url = url + f'/{session_id}'
    
    logger.info(f"Connecting to WebSocket server at {url}")
    
    try:
        async with websockets.connect(url) as websocket:
            logger.info("Connected to WebSocket server")
            
            # Wait for welcome message
            response = await websocket.recv()
            logger.info(f"Received: {response}")
            
            print("\n=== Interactive WebSocket Client ===")
            print("Available commands:")
            print("  mock - Set data mode to mock")
            print("  real - Set data mode to real")
            print("  ping - Send ping message")
            print("  quit - Exit client")
            print("  custom - Send custom JSON message")
            
            while True:
                command = input("\nEnter command: ").strip().lower()
                
                if command == "quit":
                    break
                
                if command == "mock":
                    msg = {"action": "set_data_mode", "mode": "mock"}
                elif command == "real":
                    msg = {"action": "set_data_mode", "mode": "real"}
                elif command == "ping":
                    msg = {"action": "ping"}
                elif command == "custom":
                    custom_json = input("Enter custom JSON: ")
                    try:
                        msg = json.loads(custom_json)
                    except json.JSONDecodeError:
                        print("Invalid JSON. Try again.")
                        continue
                else:
                    print(f"Unknown command: {command}")
                    continue
                
                # Send message
                print(f"Sending: {json.dumps(msg)}")
                await websocket.send(json.dumps(msg))
                
                # Wait for response
                response = await websocket.recv()
                try:
                    response_data = json.loads(response)
                    print(f"Received: {json.dumps(response_data, indent=2)}")
                except json.JSONDecodeError:
                    print(f"Received non-JSON response: {response}")
            
            print("WebSocket client closed")
    
    except Exception as e:
        logger.error(f"Error in WebSocket client: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="WebSocket Client Test Tool")
    parser.add_argument("--url", type=str, default=DEFAULT_WS_URL, help="WebSocket server URL")
    parser.add_argument("--session", type=str, default="test-client", help="Session ID to use")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--autotest", action="store_true", help="Run automated test sequence")
    args = parser.parse_args()
    
    # Run client
    if args.interactive:
        asyncio.run(interactive_websocket_client(args.url, args.session))
    elif args.autotest:
        asyncio.run(test_websocket_set_data_mode(args.url, args.session))
    else:
        print("Please specify either --interactive or --autotest mode")
