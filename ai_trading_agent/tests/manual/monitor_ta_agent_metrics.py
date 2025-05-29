"""
Technical Analysis Agent Metrics Monitor

This script continuously monitors the performance metrics of the Technical Analysis Agent
and records them for analysis. It can be run alongside the main application to track
signal generation, processing times, and overall performance.
"""

import sys
import os
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import csv

# Add project root to Python path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TAAgentMonitor")

# Import project modules
from ai_trading_agent.agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ai_trading_agent.api.data_source_api import get_ta_agent

# Output file for metrics
METRICS_FILE = Path(__file__).parent / "ta_agent_metrics.csv"
METRICS_JSON = Path(__file__).parent / "ta_agent_metrics.json"

# Monitor settings
MONITORING_INTERVAL = 10  # seconds
MONITORING_DURATION = 60 * 10  # 10 minutes


async def monitor_metrics(duration_seconds=MONITORING_DURATION, interval_seconds=MONITORING_INTERVAL):
    """
    Monitor Technical Analysis Agent metrics for a specified duration.
    
    Args:
        duration_seconds: Total monitoring duration in seconds
        interval_seconds: Interval between metric checks in seconds
    """
    logger.info(f"Starting Technical Analysis Agent metrics monitoring for {duration_seconds} seconds")
    logger.info(f"Metrics will be recorded every {interval_seconds} seconds")
    
    # Initialize CSV file with headers
    if not METRICS_FILE.exists():
        with open(METRICS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'data_source', 'signals_generated', 'signals_validated', 
                'signals_rejected', 'avg_signal_confidence', 'regime_changes',
                'parameter_adaptations', 'last_execution_time_ms'
            ])
    
    # Store metrics history for JSON output
    metrics_history = []
    
    # Get TA agent
    ta_agent = get_ta_agent()
    logger.info(f"Connected to Technical Analysis Agent (data mode: {ta_agent.get_data_source_type()})")
    
    # Monitor metrics until duration expires
    end_time = datetime.now() + timedelta(seconds=duration_seconds)
    
    while datetime.now() < end_time:
        try:
            # Get metrics
            metrics = ta_agent.get_metrics()
            
            # Log metrics
            logger.info(f"Current data source: {metrics.get('data_source', 'unknown')}")
            logger.info(f"Signals generated: {metrics.get('signals_generated', 0)}")
            logger.info(f"Signals validated: {metrics.get('signals_validated', 0)}")
            logger.info(f"Signals rejected: {metrics.get('signals_rejected', 0)}")
            logger.info(f"Avg signal confidence: {metrics.get('avg_signal_confidence', 0):.4f}")
            logger.info(f"Last execution time: {metrics.get('last_execution_time_ms', 0)} ms")
            
            # Add timestamp to metrics
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Record metrics to history
            metrics_history.append(metrics)
            
            # Write to CSV
            with open(METRICS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics['timestamp'],
                    metrics.get('data_source', 'unknown'),
                    metrics.get('signals_generated', 0),
                    metrics.get('signals_validated', 0),
                    metrics.get('signals_rejected', 0),
                    metrics.get('avg_signal_confidence', 0),
                    metrics.get('regime_changes', 0),
                    metrics.get('parameter_adaptations', 0),
                    metrics.get('last_execution_time_ms', 0)
                ])
            
            # Write to JSON
            with open(METRICS_JSON, 'w') as f:
                json.dump(metrics_history, f, indent=2)
            
            # Wait for next interval
            await asyncio.sleep(interval_seconds)
            
        except Exception as e:
            logger.error(f"Error monitoring metrics: {str(e)}")
            await asyncio.sleep(interval_seconds)
    
    logger.info(f"Monitoring complete. Metrics recorded: {len(metrics_history)}")
    logger.info(f"Metrics saved to CSV: {METRICS_FILE}")
    logger.info(f"Metrics saved to JSON: {METRICS_JSON}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Monitor Technical Analysis Agent metrics")
    parser.add_argument("--duration", type=int, default=MONITORING_DURATION, help="Monitoring duration in seconds")
    parser.add_argument("--interval", type=int, default=MONITORING_INTERVAL, help="Monitoring interval in seconds")
    args = parser.parse_args()
    
    # Run monitoring
    asyncio.run(monitor_metrics(args.duration, args.interval))
