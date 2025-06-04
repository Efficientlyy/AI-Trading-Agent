#!/usr/bin/env python
"""
End-to-End Pipeline Test with Fixed Flash Trading Signals

This script tests the complete trading pipeline from market data collection
to signal generation, paper trading, and notifications, focusing on BTCUSDC only.
Uses the fixed FlashTradingSignals class with explicit type checks.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("end_to_end_test")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
try:
    # Use enhanced version of MultiAssetDataService
    from enhanced_multi_asset_data_service import MultiAssetDataService
    from enhanced_market_data_pipeline import MarketDataPipeline
    from fixed_flash_trading_signals import FlashTradingSignals as SignalGenerator
    from fixed_paper_trading import PaperTradingSystem
    from enhanced_telegram_notifications import TelegramNotifier
    
    logger.info("Successfully imported required modules")
except Exception as e:
    logger.error(f"Error importing required modules: {str(e)}")
    sys.exit(1)

def run_end_to_end_test():
    """Run end-to-end pipeline test"""
    logger.info("=== Starting End-to-End Pipeline Test ===")
    
    # Test configuration
    config = {
        'symbols': ['BTCUSDC'],  # Focus on BTCUSDC only
        'timeframes': ['1m', '5m', '15m', '1h'],
        'signal_threshold': 0.6,
        'position_size': 0.1,
        'test_mode': True,
        'mock_data': False,  # Use real data only
        'fallback_to_mock': False  # Never fall back to mock data
    }
    
    # Initialize components
    logger.info("Initializing pipeline components")
    
    # Initialize data service
    data_service = MultiAssetDataService(supported_assets=[s.replace('USDC', '/USDC') for s in config['symbols']])
    
    # Initialize market data pipeline
    market_data_pipeline = MarketDataPipeline(
        data_service=data_service,
        symbols=config['symbols'],
        timeframes=config['timeframes'],
        fallback_to_mock=config['fallback_to_mock']
    )
    
    # Initialize signal generator with strict dependency injection
    signal_generator = SignalGenerator(
        client_instance=data_service
    )
    
    # Initialize paper trading system
    paper_trading = PaperTradingSystem(
        client=data_service,
        config={
            'position_size': config['position_size'],
            'initial_balance': {
                'USDC': 10000.0,
                'BTC': 0.1
            }
        }
    )
    
    # Initialize notification system
    notifier = TelegramNotifier()
    
    # Set notification callback for paper trading
    paper_trading.set_notification_callback(notifier.send_signal_notification)
    
    # Start components
    logger.info("Starting pipeline components")
    market_data_pipeline.start()
    paper_trading.start()
    notifier.start()
    
    # Collect market data
    logger.info("Collecting market data")
    market_data = {}
    for symbol in config['symbols']:
        market_data[symbol] = {}
        for timeframe in config['timeframes']:
            try:
                # Get candles
                candles = market_data_pipeline.get_candles(symbol, timeframe)
                market_data[symbol][timeframe] = candles
                logger.info(f"Collected {len(candles)} candles for {timeframe}")
                
                # Log first and last candle for verification
                if candles:
                    logger.info(f"First candle: {candles[0]}")
                    logger.info(f"Last candle: {candles[-1]}")
            except Exception as e:
                logger.error(f"Error collecting market data for {symbol} {timeframe}: {str(e)}")
                market_data[symbol][timeframe] = []
    
    # Check if we have enough data
    has_data = False
    for symbol in config['symbols']:
        for timeframe in config['timeframes']:
            if len(market_data[symbol].get(timeframe, [])) > 0:
                has_data = True
                break
    
    if not has_data:
        logger.error("Market data collection failed for all timeframes")
        notifier.send_error_notification("Market data collection failed for all timeframes")
        
        # Save results
        save_results({
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'market_data': {tf: len(market_data[config['symbols'][0]].get(tf, [])) for tf in config['timeframes']},
            'signals': [],
            'trades': [],
            'errors': []
        })
        
        # Stop components
        stop_components(market_data_pipeline, paper_trading, notifier)
        return False
    
    # Generate signals
    logger.info("Generating signals")
    signals = []
    for symbol in config['symbols']:
        try:
            # Use 15m timeframe for signal generation if available, otherwise use the first available timeframe
            timeframe = '15m' if len(market_data[symbol].get('15m', [])) > 0 else next((tf for tf in config['timeframes'] if len(market_data[symbol].get(tf, [])) > 0), None)
            
            if timeframe:
                candles = market_data[symbol][timeframe]
                if len(candles) > 0:
                    # Start signal generator for this symbol
                    signal_generator.start([symbol])
                    
                    # Wait a moment for market state to update
                    time.sleep(2.0)
                    
                    # Generate signal
                    signal = signal_generator.generate_signals(symbol)
                    
                    # Stop signal generator
                    signal_generator.stop()
                    
                    if signal:
                        # Take the first signal if multiple are returned
                        if isinstance(signal, list) and len(signal) > 0:
                            signal = signal[0]
                            
                        # Format signal for compatibility with paper trading
                        formatted_signal = {
                            'id': f"SIG-{int(time.time())}",
                            'symbol': symbol,
                            'direction': signal.get('type', 'UNKNOWN'),
                            'strength': signal.get('strength', 0.5),
                            'price': signal.get('price', 0.0),
                            'source': signal.get('source', 'technical'),
                            'timestamp': signal.get('timestamp', int(time.time() * 1000))
                        }
                        
                        signals.append(formatted_signal)
                        logger.info(f"Generated signal: {formatted_signal}")
                        
                        # Send notification
                        notifier.send_signal_notification(formatted_signal)
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
    
    # Execute trades based on signals
    logger.info("Executing trades")
    trades = []
    for signal in signals:
        try:
            trade = paper_trading.execute_trade(signal)
            if trade:
                trades.append(trade)
                logger.info(f"Executed trade: {trade}")
                
                # Send notification
                notifier.send_trade_notification(trade)
        except Exception as e:
            logger.error(f"Error executing trade for signal {signal}: {str(e)}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'status': 'success' if len(signals) > 0 else 'no_signals',
        'market_data': {tf: len(market_data[config['symbols'][0]].get(tf, [])) for tf in config['timeframes']},
        'signals': signals,
        'trades': trades,
        'errors': []
    }
    
    save_results(results)
    
    # Stop components
    stop_components(market_data_pipeline, paper_trading, notifier)
    
    return True

def stop_components(market_data_pipeline, paper_trading, notifier):
    """Stop all pipeline components"""
    logger.info("Stopping pipeline components")
    try:
        market_data_pipeline.stop()
    except Exception as e:
        logger.error(f"Error stopping market data pipeline: {str(e)}")
    
    try:
        paper_trading.stop()
    except Exception as e:
        logger.error(f"Error stopping paper trading system: {str(e)}")
    
    try:
        notifier.stop()
    except Exception as e:
        logger.error(f"Error stopping notification system: {str(e)}")

def save_results(results):
    """Save test results to file"""
    # Create results directory if it doesn't exist
    os.makedirs("pipeline_results", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./pipeline_results/pipeline_results_{timestamp}.json"
    
    # Save results
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Pipeline results saved to {filename}")
    
    # Print summary
    print("=== END-TO-END PIPELINE SUMMARY ===")
    print(f"Symbol: {results['signals'][0]['symbol'] if results['signals'] else 'BTCUSDC'}")
    print(f"Test Time: {results['timestamp']}")
    print(f"Status: {results['status']}")
    print("Market Data:")
    for tf, count in results['market_data'].items():
        print(f"{tf}: {'✅' if count > 0 else '❌'} {count} candles")
    print("Signals:")
    if results['signals']:
        for signal in results['signals']:
            print(f"- {signal['symbol']} {signal['direction']} (strength: {signal['strength']:.2f})")
    else:
        print("- No signals generated")
    print("Trades:")
    if results['trades']:
        for trade in results['trades']:
            print(f"- {trade['symbol']} {trade['action']} {trade['quantity']} @ {trade['price']}")
    else:
        print("- No trades executed")
    print("Errors:")
    if results['errors']:
        for error in results['errors']:
            print(f"- {error}")
    else:
        print("- No errors")
    print("Detailed results saved to JSON file in pipeline_results directory")

if __name__ == "__main__":
    run_end_to_end_test()
