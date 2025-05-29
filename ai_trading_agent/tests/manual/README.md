# Technical Analysis Agent Verification Tools

This directory contains tools for verifying the integration and operation of the Technical Analysis Agent in the AI Trading Agent system.

## Verification Tasks

These tools help complete the following verification tasks:

1. **Verify WebSocket Backend Handler**:
   - Check if the WebSocket server properly processes the 'set_data_mode' action
   - Verify the TechnicalAnalysisAgent's data mode is updated correctly

2. **Check Decision Agent Integration**:
   - Verify that the decision-making agent correctly consumes signals from TechnicalAnalysisAgent
   - Confirm the signal flow through the orchestration layer

3. **Test the Complete Flow**:
   - Test with mock data (predictable patterns)
   - Verify signals are generated correctly
   - Confirm decision-making agent receives and processes these signals

4. **Monitor Execution**:
   - Track processing times and signal generation
   - Monitor metrics for performance analysis

## Tools Included

### 1. Verification Test Script (`verify_ta_agent_flow.py`)

This script performs a complete end-to-end test of the Technical Analysis Agent flow:
- Tests data mode toggling
- Captures and verifies signal generation
- Confirms signals reach the decision components
- Checks metrics at each stage

**Usage:**
```bash
# Run the verification test
python verify_ta_agent_flow.py
```

### 2. Metrics Monitor (`monitor_ta_agent_metrics.py`)

This script continuously monitors and records the Technical Analysis Agent's performance metrics:
- Tracks signal generation counts
- Records processing times
- Monitors data source changes
- Exports data to CSV and JSON for analysis

**Usage:**
```bash
# Run with default settings (10 minutes monitoring, 10 second intervals)
python monitor_ta_agent_metrics.py

# Custom duration and interval
python monitor_ta_agent_metrics.py --duration 600 --interval 5
```

### 3. WebSocket Client Test Tool (`websocket_client_test.py`)

This tool provides a WebSocket client to test the WebSocket server implementation:
- Tests the 'set_data_mode' action
- Verifies proper responses from the server
- Provides both automated and interactive testing modes

**Usage:**
```bash
# Run automated test sequence
python websocket_client_test.py --autotest --url ws://localhost:8000/ws/test-client

# Run interactive client for manual testing
python websocket_client_test.py --interactive
```

## Running the Complete Verification

To fully verify the Technical Analysis Agent:

1. Start the AI Trading Agent application
2. Run the WebSocket test to verify the backend handler:
   ```
   python websocket_client_test.py --autotest
   ```
3. Run the verification test to check the complete flow:
   ```
   python verify_ta_agent_flow.py
   ```
4. Use the metrics monitor during normal operation:
   ```
   python monitor_ta_agent_metrics.py
   ```

## Interpreting Results

- **WebSocket Handler Verification**: The WebSocket test will display ✅ for successful responses and ❌ for failures.
- **Signal Flow Verification**: The verification test will show decision signal counts and a final VERIFICATION SUCCESSFUL message if signals are properly routed.
- **Metrics Monitoring**: The metrics monitor will create CSV and JSON files with performance data for further analysis.

## Dependencies

These tools require:
- Python 3.8+
- websockets library (`pip install websockets`)
- Access to the running AI Trading Agent application
